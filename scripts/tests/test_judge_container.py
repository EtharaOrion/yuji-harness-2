"""The judge container: where every channel is graded, and what it can see.

Grading used to happen in `main`, the container the agent had root in, and the
rubric on the host, where `codex exec --sandbox read-only` can still READ every
other run on disk (measured: a canary outside the working directory was cat'ed
straight back). All four channels now run in the bundle's `judge` service
(tools/judge/codexbridge.py), which runs services/scoring/tests/evaluate.sh --
the harness's one copy for every bundle, unchanged from what main ran -- so the
numbers cannot drift. These tests pin what makes that true, cheapest first:

  1. codexbridge in-process, against a fake evaluate.sh: token, readiness,
     reports, and that the command is fixed rather than taken from the caller.
  2. every bundle: the judge service and its mounts, the token, the split
     between test.sh (build the trajectory) and evaluate.sh (grade).
  3. the judge's own allowlist and the network shape overlay-judge.yaml builds.
  4. run_task.sh: builds the image, exports a fresh token, adds the overlay.
  5. the host fallback reuses container verdicts instead of re-buying them.
  6. docker, skipped without the image: what the running judge can reach, and
     that it carries pytest and the mcp client. JUDGE_LIVE=1 adds a real codex
     grade (spends quota).
"""
from __future__ import annotations

import ast
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import tomllib
import uuid
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

import pytest
import yaml

from test_network_policy import OVERLAY, PROXY_DIR, _run_harbor_stage, fake_zbridge  # noqa: F401

REPO = Path(__file__).resolve().parents[2]
JUDGE_DIR = REPO / "tools" / "judge"
BRIDGE = JUDGE_DIR / "codexbridge.py"
OVERLAY_JUDGE = PROXY_DIR / "overlay-judge.yaml"
OVERLAY_JUDGE_HEADROOM = PROXY_DIR / "overlay-judge-headroom.yaml"
OVERLAY_ZBRIDGE = PROXY_DIR / "overlay-zbridge.yaml"
SQUID_JUDGE = PROXY_DIR / "squid-judge.conf"
RUN_TASK = REPO / "scripts" / "run_task.sh"
JUDGE_IMAGE = "codex-judge:latest"
AUTH_TARGET = "/run/codex-auth/auth.json"
BUNDLES = sorted(REPO.glob("tasks/*/task.toml"))

# The grading scripts are the harness's, one copy for every bundle, mounted into
# the judge with the rest of services/scoring. They used to be copied into each
# bundle's tests/ and drifted there: four different evaluate.sh, three bundles
# with no grade.py at all. Where they are in a container:
SCORING_IN_CONTAINER = "/harness/scoring"
SHARED_TESTS = REPO / "services" / "scoring" / "tests"
SHARED_EVALUATE = SHARED_TESTS / "evaluate.sh"
SHARED_GRADE = SHARED_TESTS / "grade.py"
SHARED_CONTAINER_CHECK = SHARED_TESTS / "test_judge_container.py"
CONTAINER_CHECK_PATH = f"{SCORING_IN_CONTAINER}/tests/test_judge_container.py"

_opener = build_opener(ProxyHandler({}))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _call(url: str, body: dict | None = None, token: str | None = None) -> tuple[int, dict]:
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"}
    if token is not None:
        headers["x-judge-token"] = token
    req = Request(url, data=data, method="POST" if data is not None else "GET", headers=headers)
    try:
        with _opener.open(req, timeout=30) as resp:
            return resp.status, json.loads(resp.read())
    except HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"{}")


# =============================================================================
# 1. codexbridge, in-process
# =============================================================================

FAKE_EVALUATE = r"""#!/bin/bash
# Stands in for the harness's evaluate.sh: writes the same reports, without
# codex, pytest or a world. In the container these paths are fixed at
# /logs/verifier and /tmp/agent_trajectory.json; here they follow the fixture,
# and the trajectory path is the one codexbridge exports to every child.
set -u
LOGS="${JUDGE_TEST_LOGS:-/logs/verifier}"
TRAJ="${COMPLEXMCP_TRAJECTORY:-/tmp/agent_trajectory.json}"
mkdir -p "$LOGS"
echo "evaluate.sh ran with trajectory: $(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['final_message'])" "$TRAJ")"
case "${FAKE_MODE:-ok}" in
  fail)     echo "grader exploded" >&2; exit 3 ;;
  noreward) echo '{"score": 1.0, "per_criterion": [{"number": "1", "satisfied": true}]}' > "$LOGS/rubric_breakdown.json" ;;
  *)        echo '{"score": 1.0, "per_criterion": [{"number": "1", "satisfied": true}]}' > "$LOGS/rubric_breakdown.json"
            echo '{"reward": 0.42, "completion_rate": 1.0, "misbehave_rate": 0.0}' > "$LOGS/reward.json"
            echo '{"producer": "judge_container"}' > "$LOGS/reward_producer.json"
            echo 'ctrf' > "$LOGS/ctrf.json" ;;
esac
"""

TRAJ = {"steps": [{"tool": "refund", "arguments": {}, "response": "ok"}], "final_message": "refunded"}
RUBRIC = {"criteria": [{"number": "1", "criterion": "Refunds Kelso.", "is_positive": True}]}


@pytest.fixture
def bridge(tmp_path, monkeypatch):
    """codexbridge with a fake evaluate.sh, a fake codex, and a real log dir."""
    mod = _load("codexbridge_under_test", BRIDGE)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    codex = bin_dir / "codex"
    codex.write_text("#!/bin/sh\necho 'codex-cli 0.0.0-test'\n")
    codex.chmod(0o755)
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "evaluate.sh").write_text(FAKE_EVALUATE)
    logs = tmp_path / "verifier"
    logs.mkdir()
    auth = tmp_path / "mounted-auth.json"
    auth.write_text('{"tokens": "host copy"}')
    # mcp is the state dump's client; the harness venv need not carry it, so a
    # stub on sys.path is enough for the readiness check under test.
    stub = tmp_path / "stubs"
    (stub / "mcp").mkdir(parents=True)
    (stub / "mcp" / "__init__.py").write_text("")
    monkeypatch.syspath_prepend(str(stub))

    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    monkeypatch.setenv("JUDGE_TOKEN", "s3cret")
    # The rubric grader is baked into the image at /judge; out here the repo copy
    # is the same file, and readiness only asks whether it is present.
    monkeypatch.setenv("JUDGE_CLI", str(REPO / "services" / "scoring" / "rubric_judge_cli.py"))
    monkeypatch.setenv("CODEX_AUTH_SRC", str(auth))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex-home"))
    monkeypatch.setenv("JUDGE_TEST_LOGS", str(logs))
    monkeypatch.setattr(mod, "EVALUATE_SH", tests / "evaluate.sh")
    monkeypatch.setattr(mod, "VERIFIER_DIR", logs)
    monkeypatch.setattr(mod, "TRAJECTORY_PATH", tmp_path / "agent_trajectory.json")
    mod._state["credential_error"] = mod.install_credential()

    srv = mod.make_server("127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    class B:
        module = mod
        url = f"http://127.0.0.1:{srv.server_address[1]}"
        home = tmp_path / "codex-home"
        mounted = auth
        logs_dir = logs
        script = tests / "evaluate.sh"
    yield B
    srv.shutdown()
    srv.server_close()


def test_health_is_ok_when_an_evaluation_could_run(bridge):
    status, doc = _call(f"{bridge.url}/healthz")
    assert status == 200, doc
    assert doc["status"] == "ok"


def test_health_names_a_missing_token(bridge, monkeypatch):
    monkeypatch.delenv("JUDGE_TOKEN")
    status, doc = _call(f"{bridge.url}/healthz")
    assert status == 503
    assert "JUDGE_TOKEN" in doc["reason"]


def test_health_names_a_missing_login(bridge, monkeypatch, tmp_path):
    """Compose's `up --wait` holds the trial on this, so a run with no login
    stops before the agent phase rather than after it."""
    monkeypatch.setenv("CODEX_AUTH_SRC", str(tmp_path / "nope.json"))
    bridge.module._state["credential_error"] = bridge.module.install_credential()
    status, doc = _call(f"{bridge.url}/healthz")
    assert status == 503
    assert "codex login not mounted" in doc["reason"]


def test_health_names_an_evaluate_sh_that_is_not_mounted(bridge, monkeypatch, tmp_path):
    """Without it there is nothing to grade with, and the trial must not start."""
    monkeypatch.setattr(bridge.module, "EVALUATE_SH", tmp_path / "missing" / "evaluate.sh")
    status, doc = _call(f"{bridge.url}/healthz")
    assert status == 503
    assert "evaluate.sh" in doc["reason"] and "mounted" in doc["reason"]


def test_the_judge_defaults_to_the_harness_copy_of_the_grading(monkeypatch):
    """The default path, not the fixture's. A judge that fell back to a bundle
    path would grade with whatever that bundle happened to ship."""
    monkeypatch.delenv("JUDGE_EVALUATE_SH", raising=False)
    mod = _load("codexbridge_defaults", BRIDGE)
    assert str(mod.EVALUATE_SH) == f"{SCORING_IN_CONTAINER}/tests/evaluate.sh"


def test_health_names_an_unwritable_report_dir(bridge, monkeypatch, tmp_path):
    """harbor's per-trial /logs/verifier is where the reports have to land."""
    ro = tmp_path / "readonly"
    ro.mkdir(mode=0o500)
    monkeypatch.setattr(bridge.module, "VERIFIER_DIR", ro)
    status, doc = _call(f"{bridge.url}/healthz")
    assert status == 503
    assert "not writable" in doc["reason"]


@pytest.mark.parametrize("token", [None, "", "wrong"])
def test_evaluating_needs_the_run_token(bridge, token):
    """The agent shares a network with the judge. The token is what keeps it out."""
    status, doc = _call(f"{bridge.url}/evaluate", {"trajectory": TRAJ}, token)
    assert status == 401, doc


def test_an_evaluation_runs_the_bundle_script_and_reports_what_it_wrote(bridge):
    status, doc = _call(f"{bridge.url}/evaluate", {"trajectory": TRAJ}, "s3cret")
    assert status == 200, doc
    assert doc["ok"] is True and doc["reason"] is None
    assert doc["graded_in"] == "judge-container"
    assert doc["reward"] == {"reward": 0.42, "completion_rate": 1.0, "misbehave_rate": 0.0}
    assert set(doc["written"]) == {"ctrf.json", "reward.json", "reward_producer.json",
                                   "rubric_breakdown.json"}
    assert doc["rubric_criteria"] == 1
    assert "refunded" in doc["log_tail"], "the script did not receive the posted trajectory"
    assert (bridge.logs_dir / "reward.json").is_file(), "reports must land in the mounted log dir"


def test_the_command_is_fixed_not_taken_from_the_caller(bridge, tmp_path):
    """`main` is the container the agent worked in. If it could name the command,
    the agent's leftovers could grade themselves."""
    pwned = tmp_path / "pwned"
    status, doc = _call(f"{bridge.url}/evaluate",
                        {"trajectory": TRAJ, "command": f"touch {pwned}",
                         "evaluate_sh": f"touch {pwned}"}, "s3cret")
    assert status == 200 and doc["ok"] is True
    assert not pwned.exists(), "the request steered what ran"


def test_a_failing_script_is_reported_with_its_log(bridge, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "fail")
    status, doc = _call(f"{bridge.url}/evaluate", {"trajectory": TRAJ}, "s3cret")
    assert status == 200
    assert doc["ok"] is False and doc["reason"] == "evaluation exited 3"
    assert "exploded" in doc["log_tail"]


def test_an_evaluation_that_writes_no_reward_is_not_ok(bridge, monkeypatch):
    """A rubric on its own is not a graded run; run_task.sh must fall back."""
    monkeypatch.setenv("FAKE_MODE", "noreward")
    status, doc = _call(f"{bridge.url}/evaluate", {"trajectory": TRAJ}, "s3cret")
    assert status == 200
    assert doc["ok"] is False and doc["reason"] == "no reward.json was written"


def test_a_request_without_a_trajectory_is_refused(bridge):
    status, _ = _call(f"{bridge.url}/evaluate", {"nope": 1}, "s3cret")
    assert status == 400


def test_an_oversized_body_is_refused(bridge, monkeypatch):
    monkeypatch.setenv("JUDGE_MAX_BODY_BYTES", "64")
    status, _ = _call(f"{bridge.url}/evaluate", {"trajectory": TRAJ}, "s3cret")
    assert status == 413


def test_the_login_is_a_private_copy(bridge):
    """codex rewrites auth.json when it refreshes. It rewrites the copy."""
    copy = bridge.home / "auth.json"
    assert copy.read_text() == bridge.mounted.read_text()
    assert oct(copy.stat().st_mode & 0o777) == "0o600"
    copy.write_text("refreshed in the container")
    assert bridge.mounted.read_text() == '{"tokens": "host copy"}'


# =============================================================================
# 2. every bundle
# =============================================================================

pytest_bundles = pytest.mark.skipif(not BUNDLES, reason="no task bundles in this checkout")


def _ids(paths):
    return [p.parent.name[:40] for p in paths]


def _code(script: str) -> str:
    """A shell script with its comment lines dropped.

    These scripts name, in their comments, the exact files the assertions below
    forbid in their commands -- which grader moved where, and which marker is no
    longer read. Grepping the raw text reads that documentation as behaviour."""
    return "\n".join(l for l in script.splitlines() if not l.lstrip().startswith("#"))


def _compose(task_toml: Path) -> dict:
    return yaml.safe_load((task_toml.parent / "environment" / "docker-compose.yaml").read_text())


def _volume(spec: str) -> tuple[str, str, str | None]:
    """(source, target, mode) for one compose volume string.

    Splitting on colons from the right does not work here: the sources are
    `${VAR:?message}` interpolations that carry colons of their own. The target
    is the last absolute path in the string."""
    mode = None
    if spec.endswith((":ro", ":rw")):
        spec, mode = spec[:-3], spec[-2:]
    i = spec.rfind(":/")
    return spec[:i], spec[i + 1:], mode


@pytest_bundles
@pytest.mark.parametrize("task_toml", BUNDLES, ids=_ids(BUNDLES))
def test_bundle_declares_the_judge(task_toml):
    judge = (_compose(task_toml).get("services") or {}).get("judge")
    assert judge, "no judge service: the rubric would fall back to the host"
    assert judge.get("image") == JUDGE_IMAGE
    assert "build" not in judge, "the judge image is built once by run_task.sh, never per bundle"
    assert "${JUDGE_TOKEN" in str((judge.get("environment") or {}).get("JUDGE_TOKEN"))
    assert "--health" in " ".join((judge.get("healthcheck") or {}).get("test") or [])


@pytest_bundles
@pytest.mark.parametrize("task_toml", BUNDLES, ids=_ids(BUNDLES))
def test_no_bundle_ships_its_own_grading_scripts(task_toml):
    """These three are the harness's. A copy left in a bundle is not read by
    anything any more, so it can only drift and mislead whoever finds it."""
    tests = task_toml.parent / "tests"
    for name in ("evaluate.sh", "grade.py", "test_judge_container.py"):
        assert not (tests / name).exists(), (
            f"tests/{name} belongs to the harness now: services/scoring/tests/{name}")


@pytest_bundles
@pytest.mark.parametrize("task_toml", BUNDLES, ids=_ids(BUNDLES))
def test_main_can_reach_the_judge_container_check(task_toml):
    """test.sh's last step runs it from /harness/scoring, inside main. Without
    the mount that step reports the grading never happened, on every run."""
    vols = [str(v) for v in (_compose(task_toml)["services"]["main"].get("volumes") or [])]
    targets = {target for _, target, _ in (_volume(v) for v in vols)}
    assert SCORING_IN_CONTAINER in targets, vols


@pytest_bundles
@pytest.mark.parametrize("task_toml", BUNDLES, ids=_ids(BUNDLES))
def test_the_judge_mounts_what_it_grades_and_nothing_more(task_toml):
    """Everything the graders read, read-only, plus the one place reports go.

    The output tree is not here: the judge sees this run's tests, this run's
    workspace and this run's log dir, so it cannot read another run at all."""
    vols = [str(v) for v in (_compose(task_toml)["services"]["judge"].get("volumes") or [])]
    parsed = [_volume(v) for v in vols]
    by_target = {target: (source, mode) for source, target, mode in parsed}
    assert set(by_target) == {AUTH_TARGET, "/tests", "/harness/scoring", "/workspace",
                              "/logs/verifier"}, vols
    for target in (AUTH_TARGET, "/tests", "/harness/scoring", "/workspace"):
        assert by_target[target][1] == "ro", f"{target} is writable: {by_target[target]}"
    assert by_target["/logs/verifier"][1] is None, "reports could not be written"
    assert "CODEX_AUTH_FILE" in by_target[AUTH_TARGET][0]
    assert "HOST_VERIFIER_LOGS_PATH" in by_target["/logs/verifier"][0], (
        "the report dir must be harbor's own per-trial mount, or the reports never "
        "reach the trial directory")


@pytest_bundles
@pytest.mark.parametrize("task_toml", BUNDLES, ids=_ids(BUNDLES))
def test_the_judge_waits_for_the_world_it_has_to_read(task_toml):
    """tests/state_dump.py reads the world back out of light-servers."""
    dep = ((_compose(task_toml)["services"]["judge"].get("depends_on") or {})
           .get("light-servers") or {})
    assert dep.get("condition") == "service_healthy", dep


@pytest_bundles
@pytest.mark.parametrize("task_toml", BUNDLES, ids=_ids(BUNDLES))
def test_no_other_service_gets_the_login(task_toml):
    services = _compose(task_toml).get("services") or {}
    for name, spec in services.items():
        if name == "judge":
            continue
        text = json.dumps(spec)
        assert "CODEX_AUTH_FILE" not in text and "auth.json" not in text, (
            f"{name} can read the codex login")


@pytest_bundles
@pytest.mark.parametrize("task_toml", BUNDLES, ids=_ids(BUNDLES))
def test_the_verifier_receives_the_token_and_room_to_grade(task_toml):
    cfg = tomllib.loads(task_toml.read_text())
    assert cfg["verifier"]["env"].get("JUDGE_TOKEN") == "${JUDGE_TOKEN}"
    assert cfg["verifier"].get("timeout_sec", 0) >= 1800, (
        "the grade now happens inside the verifier window; judge_client gives up at 1200s")


@pytest_bundles
@pytest.mark.parametrize("task_toml", BUNDLES, ids=_ids(BUNDLES))
def test_main_builds_the_trajectory_and_grades_nothing(task_toml):
    """The split: test.sh parses the agent stream and hands it over. Every scored
    step lives in evaluate.sh, which runs in the judge container."""
    tests = task_toml.parent / "tests"
    sh = (tests / "test.sh").read_text()
    assert "judge_client.py" in sh, "main never hands the run to the judge"
    assert CONTAINER_CHECK_PATH in sh, "nothing would report where grading ran"
    assert SHARED_CONTAINER_CHECK.is_file()
    assert not (tests / "test_judge_container.py").exists(), (
        "a bundle copy would shadow the harness's; there is one check, in one place")
    code = _code(sh)
    for grader in ("rubric_judge_cli.py", "state_dump.py", "/tests/test_outputs.py", "grade.py"):
        assert grader not in code, f"{grader} still runs in main, the container the agent had root in"


def test_evaluate_sh_carries_every_scored_channel():
    """One script, run by the judge for every bundle, so the published numbers
    cannot move because a bundle shipped an older copy of the grading."""
    assert SHARED_EVALUATE.is_file(), "no evaluate.sh: the judge has nothing to grade with"
    body = _code(SHARED_EVALUATE.read_text())
    assert f"{SCORING_IN_CONTAINER}/rubric_judge_cli.py" in body, "rubric channel missing"
    assert "/tests/test_outputs.py" in body, "Channel A missing"
    assert f"{SCORING_IN_CONTAINER}/tests/grade.py" in body, "the ledger never runs"
    assert "reward.json" in body, "the ledger never publishes a reward"
    assert "/tests/state_dump.py" in body, "state channel missing"
    assert "judge_client.py" not in body, "the judge would call itself over HTTP"


def test_evaluate_sh_reads_every_per_bundle_file_from_the_bundle():
    """What the harness owns is the grading; what the task owns is what to grade.
    A per-bundle file read out of /harness/scoring would grade every task against
    one bundle's answers."""
    body = _code(SHARED_EVALUATE.read_text())
    for name in ("test_outputs.py", "test_weights.json", "rubric.json", "state_dump.py"):
        assert f"{SCORING_IN_CONTAINER}/tests/{name}" not in body, (
            f"{name} is per-bundle; it must come from /tests")


def test_a_bundle_that_still_grades_itself_is_not_graded_twice():
    """Three bundles still write reward_channel_a.json from inside pytest, with
    assert-style checks. grade.py reads return values, so regrading them here
    would score every passing assert False (a passing assert returns None) and
    publish a reward that is far too low with nothing raised anywhere."""
    body = _code(SHARED_EVALUATE.read_text())
    assert '[ -s "$LOGS/reward_channel_a.json" ]' in body, (
        "grade.py would overwrite a reward the bundle's own pytest step produced")


@pytest_bundles
@pytest.mark.parametrize("task_toml", BUNDLES, ids=_ids(BUNDLES))
def test_the_container_check_is_never_weighted(task_toml):
    """A judge outage is infrastructure. It must not move the agent's score, and
    harbor_to_output counts a failing unweighted test in test_outputs.py as
    'missed' -- which is why the check lives in a file of its own."""
    tests = task_toml.parent / "tests"
    names = {n.name for n in ast.walk(ast.parse(SHARED_CONTAINER_CHECK.read_text()))
             if isinstance(n, ast.FunctionDef)}
    weights = json.loads((tests / "test_weights.json").read_text())
    weighted = set(((weights.get("components") or {}).get("traj_tests") or {}).get("tests") or {})
    assert not names & weighted, sorted(names & weighted)
    assert "judge_container" not in (tests / "test_outputs.py").read_text()


def test_reward_is_stamped_only_where_it_includes_the_rubric():
    """producer=judge_container tells the host its work is done. A bundle whose
    own reward leaves the rubric out (bull-street's binary traj_pytest) must not
    claim it, or the published reward silently loses the rubric channel.

    One script grades them all, so the test is on the reward it just wrote, not
    on which files the bundle ships."""
    body = _code(SHARED_EVALUATE.read_text())
    assert "reward_producer.json" in body, "the host would re-grade every run"
    assert 'isinstance(d.get("rubric"), (int, float))' in body, (
        "the stamp does not check that this reward carries the rubric")


def test_the_stamp_never_waits_on_a_marker_written_after_it():
    """judge_container.json is written by judge_client.py back in main, AFTER the
    evaluation returns. An evaluate.sh that checks it never writes the label at
    all: the host pass then re-grades every run, and reshape fails on the ones it
    cannot re-grade (measured on an oracle run with an empty trajectory)."""
    code = _code(SHARED_EVALUATE.read_text())
    assert "judge_container.json" not in code, (
        "evaluate.sh waits on a marker that cannot exist yet")


def test_step_5_keeps_reward_json_numeric():
    """harbor reads /logs/verifier/reward.json into VerifierResult.rewards,
    typed dict[str, float | int]. A string there fails the whole trial with a
    ValidationError and scores it 0 -- which is exactly what the first
    end-to-end run of the judge container did with a producer stamp. The label
    travels in reward_producer.json and run_task.sh adds it on the host."""
    sh = _code(SHARED_EVALUATE.read_text())
    assert 'doc["producer"]' not in sh and 'out["producer"]' not in sh


# =============================================================================
# 3. the judge's allowlist and network
# =============================================================================

def _directives(path: Path) -> list[str]:
    return [l.split("#", 1)[0].strip() for l in path.read_text().splitlines()
            if l.split("#", 1)[0].strip()]


def test_the_judge_allowlist_is_what_codex_needs_and_no_more():
    """Measured behind an allow-all squid: chatgpt.com for the model,
    auth.openai.com for a login refresh. ab.chatgpt.com and oaiusercontent were
    also contacted, denied, and grading still succeeded."""
    hosts = set()
    for line in _directives(SQUID_JUDGE):
        m = re.match(r"acl\s+\S+\s+dstdomain\s+(.+)$", line)
        if m:
            hosts.update(m.group(1).split())
    assert hosts == {"chatgpt.com", "auth.openai.com"}, hosts


def test_the_judge_allowlist_is_tls_only_and_ends_in_deny():
    access = [l for l in _directives(SQUID_JUDGE) if l.startswith("http_access")]
    assert access[0] == "http_access deny CONNECT !SSL_ports"
    assert access[-1] == "http_access deny all"
    for rule in access[1:-1]:
        assert rule.startswith("http_access allow CONNECT SSL_ports judge_hosts"), rule


def test_the_judge_reaches_the_sidecars_without_its_proxy():
    """The state dump reads the world back out of light-servers, by hostname.

    With HTTP(S)_PROXY set on the judge and only loopback exempt, those calls go
    to judge-proxy, whose allowlist is OpenAI and nothing else. Measured twice:
    at the proxy as `TCP_DENIED/403 POST http://light-servers:9142/mcp`, and in a
    real trial as every app's dump failing with an ExceptionGroup while the state
    channel reported unavailable -- a scored channel lost to a proxy setting.
    """
    env = yaml.safe_load(OVERLAY_JUDGE.read_text())["services"]["judge"]["environment"]
    assert env["NO_PROXY"] == env["no_proxy"], "tools split on case; both must agree"
    exempt = {h.strip() for h in env["NO_PROXY"].split(",")}
    assert {"light-servers", "main", "judge"} <= exempt, sorted(exempt)
    assert {"localhost", "127.0.0.1", "::1"} <= exempt, sorted(exempt)
    # The model call is the one thing that must still go through the proxy.
    assert not {"chatgpt.com", "auth.openai.com"} & exempt, sorted(exempt)


def test_both_squid_configs_are_baked_and_parsed_at_build():
    body = (PROXY_DIR / "Dockerfile").read_text()
    assert "COPY squid-judge.conf /etc/squid/squid-judge.conf" in body
    assert "squid -k parse -f /etc/squid/squid-judge.conf" in body


def test_the_agent_allowlist_is_untouched():
    hosts = set()
    for line in _directives(PROXY_DIR / "squid.conf"):
        m = re.match(r"acl\s+\S+\s+dstdomain\s+(.+)$", line)
        if m:
            hosts.update(m.group(1).split())
    assert hosts == {"api.anthropic.com"}


def _resolved(*files: Path, **extra_env) -> dict:
    env = {"PATH": "/usr/bin:/bin:/usr/local/bin",
           "SCORING_DIR": str(REPO / "services" / "scoring"),
           "HOST_AGENT_LOGS_PATH": "/tmp/egress-out-test",
           "JUDGE_TOKEN": "compose-config-test",
           "CODEX_AUTH_FILE": "/tmp/codex-auth-test.json",
           # harbor binds its per-trial verifier log dir (trial.py:798); the judge
           # writes its reports straight into it, so the compose file declares it `:?`.
           "HOST_VERIFIER_LOGS_PATH": "/tmp/verifier-logs-test", **extra_env}
    args = ["docker", "compose"]
    for f in files:
        args += ["-f", str(f)]
    proc = subprocess.run(args + ["config", "--format", "json"], capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        pytest.fail(f"compose config failed:\n{proc.stderr}")
    return json.loads(proc.stdout)


def _nets(cfg: dict, service: str) -> set[str]:
    return set(((cfg.get("services") or {}).get(service) or {}).get("networks") or {})


@pytest.mark.skipif(shutil.which("docker") is None, reason="docker not installed")
@pytest_bundles
@pytest.mark.parametrize("glm", [False, True], ids=["opus", "glm"])
def test_overlay_judge_gives_the_judge_its_own_way_out(glm, tmp_path):
    compose = BUNDLES[0].parent / "environment" / "docker-compose.yaml"
    files, env = [compose, OVERLAY], {}
    if glm:
        conf = tmp_path / "squid.conf"
        conf.write_text("")
        files.append(OVERLAY_ZBRIDGE)
        env["EGRESS_SQUID_CONF"] = str(conf)
    cfg = _resolved(*files, OVERLAY_JUDGE, **env)
    networks = cfg.get("networks") or {}
    assert (networks.get("judge-net") or {}).get("internal") is True
    assert _nets(cfg, "main") == {"default"}, "the agent must not reach the judge's proxy"
    assert _nets(cfg, "judge") == {"default", "judge-net"}, "the judge must not sit on egress"
    assert _nets(cfg, "judge-proxy") == {"judge-net", "egress"}
    assert _nets(cfg, "egress-proxy") == {"default", "egress"}
    on_egress = sorted(n for n in cfg["services"] if "egress" in _nets(cfg, n))
    assert on_egress == ["egress-proxy", "judge-proxy"], on_egress
    judge_env = cfg["services"]["judge"].get("environment") or {}
    assert judge_env.get("HTTPS_PROXY") == "http://judge-proxy:3128"
    assert "squid-judge.conf" in " ".join(cfg["services"]["judge-proxy"].get("command") or [])
    main_env = cfg["services"]["main"].get("environment") or {}
    assert main_env.get("HTTPS_PROXY") == "http://egress-proxy:3128"
    assert "judge" in main_env.get("NO_PROXY", "").split(",")


# =============================================================================
# 4. run_task.sh
# =============================================================================

JUDGE_BUNDLE_COMPOSE = """services:
  main:
    image: example/main:1
  judge:
    image: codex-judge:latest
"""


def test_run_task_knows_how_to_build_the_judge_image():
    body = RUN_TASK.read_text()
    m = re.search(r'^\s*codex-judge\)\s*echo\s+"([^"]+)"', body, re.M)
    assert m, "image_build_context has no codex-judge arm; ensure_image would try to pull it"
    ctx = Path(m.group(1).replace("$REPO", str(REPO)))
    assert (ctx / "Dockerfile").is_file()
    assert re.search(r'codex-judge\)\s*printf .*--build-context "scoring=\$REPO/services/scoring"', body), (
        "the judge Dockerfile COPYs --from=scoring; without the named context the build fails")


def test_makefile_builds_the_same_tag_with_the_same_context():
    body = (REPO / "Makefile").read_text()
    seg = body[body.index("build-codex-judge:"):]
    m = re.search(r"docker build --build-context scoring=(\S+) -t (\S+) (\S+)", seg)
    assert m, "build-codex-judge does not run docker build with the scoring context"
    assert (REPO / m.group(1)).resolve() == (REPO / "services" / "scoring").resolve()
    assert m.group(2) == JUDGE_IMAGE
    assert (REPO / m.group(3)).resolve() == JUDGE_DIR.resolve()


def _overlays(run) -> list[str]:
    return [run.argv[i + 1] for i, a in enumerate(run.argv) if a == "--extra-docker-compose"]


@pytest.fixture
def login(tmp_path):
    auth = tmp_path / "codex" / "auth.json"
    auth.parent.mkdir()
    # Credential-SHAPED, not just present. run_task.sh refuses an auth.json that
    # carries no tokens, because `codex login status` calls {} logged in.
    auth.write_text('{"auth_mode": "chatgpt", "tokens": {"access_token": "stub"}}')
    return auth


def test_a_judge_bundle_gets_a_token_the_login_and_its_overlay(tmp_path, login):
    run = _run_harbor_stage(tmp_path / "r", compose=JUDGE_BUNDLE_COMPOSE,
                            CODEX_AUTH_FILE=str(login), JUDGE_TOKEN="from-the-caller")
    assert run.invoked, run.stderr[-2000:]
    token = run.env.get("JUDGE_TOKEN", "")
    assert re.fullmatch(r"[0-9a-f]{64}", token), token
    assert token != "from-the-caller", "a token that outlives the run is a token someone else has"
    assert Path(run.env["CODEX_AUTH_FILE"]).resolve() == login.resolve()
    assert _overlays(run) == [str(OVERLAY), str(OVERLAY_JUDGE)]


def test_every_invocation_gets_a_fresh_token(tmp_path, login):
    a = _run_harbor_stage(tmp_path / "a", compose=JUDGE_BUNDLE_COMPOSE, CODEX_AUTH_FILE=str(login))
    b = _run_harbor_stage(tmp_path / "b", compose=JUDGE_BUNDLE_COMPOSE, CODEX_AUTH_FILE=str(login))
    assert a.invoked and b.invoked
    assert a.env["JUDGE_TOKEN"] != b.env["JUDGE_TOKEN"]


def test_an_open_run_still_gets_a_token_but_no_judge_overlay(tmp_path, login):
    run = _run_harbor_stage(tmp_path, compose=JUDGE_BUNDLE_COMPOSE,
                            CODEX_AUTH_FILE=str(login), NETWORK_ISOLATION_OFF="1")
    assert run.invoked, run.stderr[-2000:]
    assert re.fullmatch(r"[0-9a-f]{64}", run.env.get("JUDGE_TOKEN", ""))
    assert _overlays(run) == []


def test_glm_runs_add_the_judge_overlay_after_zbridges(tmp_path, login, fake_zbridge):
    run = _run_harbor_stage(tmp_path, compose=JUDGE_BUNDLE_COMPOSE, CODEX_AUTH_FILE=str(login),
                            CC_MODE="zbridge", **fake_zbridge)
    assert run.invoked, run.stderr[-2000:]
    assert _overlays(run) == [str(OVERLAY), str(OVERLAY_ZBRIDGE), str(OVERLAY_JUDGE)]


def test_a_missing_login_stops_before_harbor(tmp_path):
    run = _run_harbor_stage(tmp_path, compose=JUDGE_BUNDLE_COMPOSE,
                            CODEX_AUTH_FILE=str(tmp_path / "missing.json"))
    assert not run.invoked
    assert run.returncode != 0


def test_a_bundle_without_a_judge_is_unchanged(tmp_path):
    run = _run_harbor_stage(tmp_path)
    assert run.invoked, run.stderr[-2000:]
    assert str(OVERLAY_JUDGE) not in run.argv


# --- grader-path Headroom ----------------------------------------------------
# It used to be wired into `main` (enable_headroom.sh, now gone): a pip line in
# the task image and a flag on the service that stopped grading anything when
# the judge took over. So the flag was on in six bundles and the grader read it
# nowhere. It belongs on the judge, which is where the rubric is graded.

def test_grader_headroom_swaps_the_judge_image_and_sets_the_flag(tmp_path, login):
    run = _run_harbor_stage(tmp_path, compose=JUDGE_BUNDLE_COMPOSE,
                            CODEX_AUTH_FILE=str(login), GRADER_HEADROOM_ENABLED="true")
    assert run.invoked, run.stderr[-2000:]
    assert _overlays(run) == [str(OVERLAY), str(OVERLAY_JUDGE), str(OVERLAY_JUDGE_HEADROOM)]


def test_grader_headroom_is_off_by_default(tmp_path, login):
    run = _run_harbor_stage(tmp_path, compose=JUDGE_BUNDLE_COMPOSE, CODEX_AUTH_FILE=str(login))
    assert run.invoked, run.stderr[-2000:]
    assert str(OVERLAY_JUDGE_HEADROOM) not in run.argv


def test_the_grader_headroom_overlay_touches_nothing_but_the_judge():
    """Compression runs inside the judge process; the network is not involved."""
    cfg = yaml.safe_load(OVERLAY_JUDGE_HEADROOM.read_text())
    assert set(cfg) == {"services"}, "a grader-path overlay must not add networks"
    assert set(cfg["services"]) == {"judge"}, sorted(cfg["services"])
    judge = cfg["services"]["judge"]
    assert judge["image"] == "codex-judge-headroom:latest"
    assert judge["environment"]["GRADER_HEADROOM_ENABLED"] == "true"
    # Without the baked vocab the library cannot count tokens here: the judge's
    # squid allows chatgpt.com and auth.openai.com only.
    assert judge["environment"]["TIKTOKEN_CACHE_DIR"] == "/opt/tiktoken"
    assert "volumes" not in judge, "the judge's mounts are what it grades; leave them alone"


def test_run_task_builds_the_headroom_judge_from_the_same_dockerfile():
    body = RUN_TASK.read_text()
    assert re.search(r'codex-judge-headroom\)\s*echo\s+"\$REPO/tools/judge"', body), (
        "image_build_context has no codex-judge-headroom arm; ensure_image would pull it"
    )
    assert re.search(r"codex-judge-headroom\)\s*\n?\s*printf .*WITH_HEADROOM=1", body, re.S), (
        "the headroom judge is built from the same Dockerfile; without the build "
        "arg it is a second tag for the plain image and the flag does nothing"
    )
    assert "ARG WITH_HEADROOM" in (JUDGE_DIR / "Dockerfile").read_text()


# =============================================================================
# 5. the host fallback
# =============================================================================

AGENT_STREAM = "\n".join(json.dumps(e) for e in [
    {"type": "assistant", "message": {"content": [
        {"type": "tool_use", "id": "t1", "name": "LightStripe_create_refund", "input": {"charge": "ch_1"}}]}},
    {"type": "user", "message": {"content": [
        {"type": "tool_result", "tool_use_id": "t1", "content": '{"status": "ok"}'}]}},
    {"type": "result", "result": "Refunded ch_1."},
]) + "\n"


@pytest.fixture
def graded_trial(tmp_path):
    """A trial whose judge container graded the rubric but whose bundle reward
    left it out -- the bull-street shape."""
    task = tmp_path / "tasks" / "demo"
    (task / "tests").mkdir(parents=True)
    (task / "tests" / "rubric.json").write_text(json.dumps(RUBRIC))
    (task / "tests" / "test_weights.json").write_text(json.dumps({"components": {
        "traj_tests": {"weight": 5, "graded": True}, "rubric": {"weight": 3, "graded": True}}}))
    trial = tmp_path / "job" / "demo__abc"
    (trial / "agent").mkdir(parents=True)
    (trial / "agent" / "claude-code.txt").write_text(AGENT_STREAM)
    v = trial / "verifier"
    v.mkdir()
    (v / "judge_container.json").write_text(json.dumps({"ok": True, "graded_in": "judge-container",
                                                        "model": "gpt-5.6-sol"}))
    (v / "rubric_breakdown.json").write_text(json.dumps({"score": 0.5, "per_criterion": [
        {"number": "1", "satisfied": True, "justification": "seen"}]}))
    (v / "reward_channel_a.json").write_text(json.dumps({"channel_a": 1.0, "guards_tripped": []}))
    (v / "state_channel.json").write_text(json.dumps({"available": False}))
    return trial, task


def test_the_host_reuses_container_verdicts_instead_of_rejudging(graded_trial, monkeypatch):
    hrp = _load("host_rubric_pass_under_test", REPO / "scripts" / "host_rubric_pass.py")
    trial, task = graded_trial

    def no_judge(*a, **k):
        raise AssertionError("the host called the judge for a rubric the container already graded")
    monkeypatch.setattr(hrp.subprocess, "run", no_judge)
    monkeypatch.setattr(sys, "argv", ["host_rubric_pass.py", "--trial", str(trial), "--task", str(task)])
    assert hrp.main() == 0
    doc = json.loads((trial / "verifier" / "reward_channel_a.json").read_text())
    assert doc["rubric_graded_on"] == "judge-container"
    assert doc["reward"] == hrp.norm_reward((5 * 1.0 + 3 * 0.5) / 8)
    assert json.loads((trial / "verifier" / "reward.json").read_text())["producer"] == "host_rubric_pass"


def test_rejudge_calls_the_judge_on_the_host_anyway(graded_trial, monkeypatch):
    hrp = _load("host_rubric_pass_rejudge", REPO / "scripts" / "host_rubric_pass.py")
    trial, task = graded_trial
    calls = []

    def fake_run(cmd, *a, **k):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 1)
    monkeypatch.setattr(hrp.subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["host_rubric_pass.py", "--trial", str(trial),
                                      "--task", str(task), "--rejudge"])
    assert hrp.main() == 1
    assert calls and "rubric_judge_cli.py" in " ".join(map(str, calls[0]))


# =============================================================================
# 6. docker
# =============================================================================

def _image_present() -> bool:
    if shutil.which("docker") is None:
        return False
    return subprocess.run(["docker", "image", "inspect", JUDGE_IMAGE],
                          capture_output=True).returncode == 0


needs_image = pytest.mark.skipif(not _image_present(),
                                 reason=f"{JUDGE_IMAGE} not built (make build-codex-judge)")

# A bundle small enough to start the judge against: the readiness check wants an
# evaluate.sh and a writable report dir, so a container with neither never turns
# healthy and every assertion below would report the fixture rather than the image.
RUNTIME_EVALUATE_SH = """#!/bin/bash
set -u
echo '{"reward": 0.5, "completion_rate": 1.0, "misbehave_rate": 0.0}' > /logs/verifier/reward.json
"""


def _exec(name: str, *cmd: str, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", "exec", name, *cmd], capture_output=True, text=True, **kw)


def _start_judge(tmp_path: Path, token: str = "t0ken") -> str:
    auth = tmp_path / "auth.json"
    auth.write_text('{"auth_mode": "chatgpt", "tokens": {"access_token": "stub"}}')
    tests = tmp_path / "tests"
    tests.mkdir(exist_ok=True)
    # evaluate.sh moved from the bundle's tests/ to the shared graders, so the
    # readiness check now looks for it under the /harness/scoring mount and
    # codexbridge runs it from there (codexbridge.py EVALUATE_SH). Put it where
    # the container looks: a copy under /tests is never read, and the judge sits
    # at 503 "is not mounted" until its 20s of retries run out.
    scoring_tests = tmp_path / "scoring" / "tests"
    scoring_tests.mkdir(parents=True, exist_ok=True)
    if not (scoring_tests / "evaluate.sh").exists():
        (scoring_tests / "evaluate.sh").write_text(RUNTIME_EVALUATE_SH)
    logs = tmp_path / "verifier"
    logs.mkdir(exist_ok=True)
    name = f"judge-test-{uuid.uuid4().hex[:8]}"
    subprocess.run(["docker", "run", "-d", "--name", name, "-e", f"JUDGE_TOKEN={token}",
                    "-v", f"{auth}:{AUTH_TARGET}:ro", "-v", f"{tests}:/tests:ro",
                    "-v", f"{tmp_path / 'scoring'}:{SCORING_IN_CONTAINER}:ro",
                    "-v", f"{logs}:/logs/verifier", JUDGE_IMAGE],
                   check=True, capture_output=True)
    for _ in range(40):
        if _exec(name, "python3", "/judge/codexbridge.py", "--health").returncode == 0:
            return name
        time.sleep(0.5)
    logs_out = subprocess.run(["docker", "logs", name], capture_output=True, text=True)
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    pytest.fail(f"judge never became healthy:\n{logs_out.stdout}{logs_out.stderr}")


@pytest.fixture
def running_judge(tmp_path):
    canary = tmp_path / "canary.txt"
    canary.write_text("CANARY")
    name = _start_judge(tmp_path)
    yield name, tmp_path
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)


@needs_image
def test_the_image_carries_every_grader_the_channels_need(running_judge):
    """Channel A is pytest, the state dump speaks mcp, the rubric is codex. A
    missing one would surface as a channel silently going unscored."""
    name, _ = running_judge
    out = _exec(name, "python3", "-c",
                "import pytest, mcp, anyio, mcp.client.streamable_http as m;"
                "from mcp.client.session import ClientSession; print('ok')")
    assert out.stdout.strip() == "ok", out.stderr
    assert "codex-cli" in _exec(name, "codex", "--version").stdout


@needs_image
def test_the_running_judge_mounts_only_what_it_grades(running_judge):
    name, _ = running_judge
    mounts = json.loads(subprocess.run(["docker", "inspect", name, "--format", "{{json .Mounts}}"],
                                       capture_output=True, text=True, check=True).stdout)
    by_target = {m["Destination"]: m["RW"] for m in mounts}
    assert by_target == {AUTH_TARGET: False, "/tests": False,
                         SCORING_IN_CONTAINER: False, "/logs/verifier": True}, mounts


@needs_image
def test_host_files_are_not_visible_to_the_judge(running_judge):
    name, canary_dir = running_judge
    canary = canary_dir / "canary.txt"
    assert _exec(name, "test", "-e", str(canary)).returncode != 0, "the judge can see a host file"
    assert _exec(name, "sh", "-c", f"echo x >> {AUTH_TARGET}").returncode != 0, "the login mount is writable"
    mode = _exec(name, "sh", "-c", 'stat -c %a "$CODEX_HOME/auth.json"')
    assert mode.stdout.strip() == "600", mode


@needs_image
def test_the_running_judge_refuses_an_evaluation_without_the_token(running_judge):
    name, _ = running_judge
    probe = ("import urllib.request as u\n"
             "r=u.Request('http://127.0.0.1:8770/evaluate',data=b'{}',method='POST',"
             "headers={'Content-Type':'application/json'})\n"
             "try:\n    u.build_opener(u.ProxyHandler({})).open(r); print(200)\n"
             "except Exception as e:\n    print(getattr(e,'code',e))\n")
    out = _exec(name, "python3", "-c", probe)
    assert out.stdout.strip() == "401", out


@needs_image
def test_an_evaluation_runs_the_mounted_script_and_leaves_its_reports(running_judge):
    """End to end through the real container: the bundle's script writes into
    harbor's log dir, and the host sees the file without main touching it."""
    name, tmp_path = running_judge
    probe = ("import json,urllib.request as u\n"
             "r=u.Request('http://127.0.0.1:8770/evaluate',"
             "data=json.dumps({'trajectory':{'steps':[],'final_message':'x'}}).encode(),"
             "method='POST',headers={'Content-Type':'application/json','x-judge-token':'t0ken'})\n"
             "print(u.build_opener(u.ProxyHandler({})).open(r).read().decode())\n")
    out = _exec(name, "python3", "-c", probe, timeout=300)
    doc = json.loads(out.stdout)
    assert doc["ok"] is True, doc
    assert doc["written"] == ["reward.json"], doc
    assert json.loads((tmp_path / "verifier" / "reward.json").read_text())["reward"] == 0.5


@needs_image
@pytest.mark.skipif(os.environ.get("JUDGE_LIVE") != "1", reason="set JUDGE_LIVE=1 to spend quota on a real grade")
def test_live_rubric_grade_through_the_container(tmp_path):
    """The real thing: codex grades a one-criterion rubric inside the image."""
    auth = Path(os.environ.get("CODEX_AUTH_FILE", Path.home() / ".codex" / "auth.json"))
    if not auth.is_file():
        pytest.skip("no codex login on this machine")
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "rubric.json").write_text(json.dumps({"criteria": [
        {"number": "1", "criterion": "The agent refunds charge ch_1.",
         "evaluation_target": "trajectory", "is_positive": True,
         "importance": "critically_important", "score": 5, "weight": 5}]}))
    (tests / "evaluate.sh").write_text(
        "#!/bin/bash\nset -u\n"
        "python3 /judge/rubric_judge_cli.py --rubric /tests/rubric.json "
        "--trajectory /tmp/agent_trajectory.json --output /logs/verifier/rubric_breakdown.json "
        "--token-output /logs/verifier/judge_tokens.json\n"
        "echo '{\"reward\": 1.0}' > /logs/verifier/reward.json\n")
    (tmp_path / "auth.json").write_bytes(auth.read_bytes())
    name = _start_judge(tmp_path, token="live-token")
    try:
        probe = ("import json,urllib.request as u\n"
                 "r=u.Request('http://127.0.0.1:8770/evaluate',"
                 "data=json.dumps({'trajectory':{'steps':[{'tool':'refund','arguments':"
                 "{'charge':'ch_1'},'response':'ok'}],'final_message':'refunded ch_1'}}).encode(),"
                 "method='POST',headers={'Content-Type':'application/json','x-judge-token':'live-token'})\n"
                 "print(u.build_opener(u.ProxyHandler({})).open(r, timeout=900).read().decode())\n")
        out = _exec(name, "python3", "-c", probe, timeout=900)
        doc = json.loads(out.stdout)
        assert doc["ok"] is True, doc
        assert doc["rubric_criteria"] == 1, doc
        breakdown = json.loads((tmp_path / "verifier" / "rubric_breakdown.json").read_text())
        assert breakdown["per_criterion"][0]["justification"].strip()
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)


# ---------------------------------------------------------------------------
# A run the agent never made must not be graded.
#
# Nothing counted the steps before grading: the whole rubric was bought from
# codex against empty evidence and reward 0.0 published, which reads exactly
# like an agent that tried and failed. scripts/host_rubric_pass.py has always
# refused this case; these pin the judge container to the same rule.
# ---------------------------------------------------------------------------

def _judge_module(tmp_path, monkeypatch):
    import importlib.util
    monkeypatch.setenv("JUDGE_VERIFIER_DIR", str(tmp_path / "verifier"))
    monkeypatch.setenv("JUDGE_TRAJECTORY_PATH", str(tmp_path / "traj.json"))
    monkeypatch.setenv("JUDGE_EVALUATE_SH", str(tmp_path / "never-run.sh"))
    src = Path(__file__).resolve().parents[2] / "tools" / "judge" / "codexbridge.py"
    spec = importlib.util.spec_from_file_location("codexbridge_under_test", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("trajectory", [
    {},
    {"steps": [], "final_message": ""},
    {"steps": [], "final_message": "   "},
])
def test_a_run_with_no_agent_activity_is_refused_not_scored(tmp_path, monkeypatch, trajectory):
    mod = _judge_module(tmp_path, monkeypatch)
    doc = mod.evaluate(trajectory)
    assert doc["ok"] is False
    assert "no agent activity" in doc["reason"]
    assert doc["reward"] is None
    assert (tmp_path / "verifier" / "no_agent_activity.txt").is_file()
    # evaluate.sh must not have been reached: no rubric was bought.
    assert doc["rubric_criteria"] == 0


@pytest.mark.parametrize("trajectory", [
    {"steps": [{"tool": "list_alarms", "arguments": {}}], "final_message": ""},
    {"steps": [], "final_message": "the cause was a failing contact"},
])
def test_a_run_that_did_something_is_still_graded(tmp_path, monkeypatch, trajectory):
    """The guard must be narrow. A run that answered in prose without tools, or
    called a tool and said nothing, is a real attempt and has to reach the
    grader -- refusing either would turn a scored run into a missing one."""
    mod = _judge_module(tmp_path, monkeypatch)
    doc = mod.evaluate(trajectory)
    assert "no agent activity" not in (doc["reason"] or "")
