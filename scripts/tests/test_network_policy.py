"""Network policy: the seam between the design and an actual block.

The two sibling files each cover one end of the feature:

  test_network_isolation.py   the topology holds for every bundle
  test_egress_allowlist.py    the proxy's own config denies by default

This file covers what decides whether either of them is ever reached, and the
ways the block can be defeated without any of the three files changing:

  1. DISPATCH -- run_task.sh has to actually hand harbor the overlay, on the
     runs that should have it and not on the ones that should not, and hand it
     a *usable* path. The return channel there is stdout and the function calls
     a helper that narrates, so "the path" and "the path plus a build log" are
     one missing redirect apart.

  2. TOPOLOGY -- `internal: true` on the default network only blocks a service
     that is *on* the default network. A bundle that declares its own network,
     sets a network_mode, or joins `egress` is outside the overlay's reach and
     nothing in the overlay would notice.

  3. POLICY -- `http_access allow CONNECT SSL_ports allowed_hosts` is a host
     allowlist. Drop the last token and the same line is an open tunnel; drop
     the port acl and it is an open tunnel to any port. Both edits look like
     cleanups.

Everything here is config parsing, `docker compose config`, or run_task.sh run
against a stub harbor. No container is started.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import yaml

from conftest import mirror_harbor_package, requires_docker, requires_credentials

REPO = Path(__file__).resolve().parents[2]
PROXY_DIR = REPO / "tools" / "network" / "egress-proxy"
SQUID_CONF = PROXY_DIR / "squid.conf"
OVERLAY = PROXY_DIR / "overlay.yaml"
DOCKERFILE = PROXY_DIR / "Dockerfile"
ENTRYPOINT = PROXY_DIR / "entrypoint.sh"
RUN_TASK = REPO / "scripts" / "run_task.sh"
MAKEFILE = REPO / "Makefile"

PROXY_SERVICE = "egress-proxy"
PROXY_IMAGE = "egress-proxy:latest"
EGRESS_NETWORK = "egress"

BUNDLES = sorted(REPO.glob("tasks/*/task.toml"))


def _ids(paths):
    return [p.parent.name for p in paths]


# =============================================================================
# 1. DISPATCH -- run_task.sh -> harbor argv
# =============================================================================
#
# Driven by shadowing `harbor` on PATH with a stub that records the argv and
# environment it was handed, then reading those back. That checks the wiring
# rather than restating the constants: a test that greps run_task.sh for
# "--extra-docker-compose" passes even when the flag is built in a branch that
# never runs.

_HARBOR_STUB = """#!/usr/bin/env bash
printf '%s\\n' "$@" >> "$HARBOR_ARGS"
env >> "$HARBOR_ENV"
mkdir -p "$JOB_DIR"
touch "$JOB_DIR/result.json"
exit 0
"""


@dataclass
class HarborRun:
    """What run_task.sh's harbor stage did."""

    returncode: int
    stdout: str
    stderr: str
    argv: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)

    @property
    def invoked(self) -> bool:
        return bool(self.argv)

    def flag_value(self, flag: str) -> str | None:
        """The argv element immediately after `flag`, or None if absent."""
        if flag not in self.argv:
            return None
        return self.argv[self.argv.index(flag) + 1]


def _mirror_harbor_package(tmp_path: Path, bin_dir: Path) -> None:
    """Give patch_harbor.py a fake harbor install to rewrite.

    Delegates to the shared helper in scripts/tests/conftest.py -- this file and
    test_run_task_stages.py both stub `harbor` on PATH and hit the same problem,
    and two copies of the venv-layout knowledge would drift the moment harbor
    changes it. bin_dir is unused; kept in the signature so callers here read the
    same as before.

    Skips when harbor is absent: without a package to mirror, patch_harbor.py
    raises inside run_task.sh and the assertion that follows reports a dispatch
    bug that is not there.
    """
    if not mirror_harbor_package(tmp_path):
        pytest.skip("harbor is not installed; cannot mirror its package")


def _run_harbor_stage(tmp_path: Path, compose: str | None = None, **overrides) -> HarborRun:
    """Run `run_task.sh --stage harbor` against a stub harbor.

    Pass None as an override value to *unset* that variable rather than set it.
    `compose` writes the bundle's environment/docker-compose.yaml.
    """
    task = tmp_path / "tasks" / "alpha"
    task.mkdir(parents=True)
    (task / "task.toml").write_text('name = "acme/alpha"\n')
    if compose is not None:
        (task / "environment").mkdir()
        (task / "environment" / "docker-compose.yaml").write_text(compose)

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = bin_dir / "harbor"
    stub.write_text(_HARBOR_STUB)
    stub.chmod(0o755)
    _mirror_harbor_package(tmp_path, bin_dir)

    args_file = tmp_path / "harbor_args.txt"
    env_file = tmp_path / "harbor_env.txt"

    env = dict(os.environ)
    env.update({
        "PATH": f"{bin_dir}:{env['PATH']}",
        "OUTPUT_DIR": str(tmp_path / "output"),
        "JOB": "alpha",
        "JOB_DIR": str(tmp_path / "output" / "alpha"),
        "HARBOR_ARGS": str(args_file),
        "HARBOR_ENV": str(env_file),
        "RUN_OFFSET": "0",
        "MODEL": "m1",
        "N": "1",
        "AGENT": "claude-code",
    })
    # Deterministic starting point: these three decide the isolation branches,
    # and inheriting an operator's shell would make the result depend on it.
    for key in ("NETWORK_ISOLATION_OFF", "CC_MODE", "ANTHROPIC_BASE_URL"):
        env.pop(key, None)
    env["AGENT_HEADROOM_ENABLED"] = "false"
    # Same reason, and it bit: this repo's own .env carries
    # GRADER_HEADROOM_ENABLED=true, so every judge test inherited a third
    # overlay and read as a dispatch bug.
    env["GRADER_HEADROOM_ENABLED"] = "false"

    for key, value in overrides.items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = str(value)

    try:
        proc = subprocess.run(
            [str(RUN_TASK), "--stage", "harbor", str(task)],
            capture_output=True, text=True, env=env, cwd=str(REPO), timeout=240,
        )
    except subprocess.TimeoutExpired:
        pytest.skip("harbor stage timed out (host-side helper unavailable)")

    run = HarborRun(returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)
    if args_file.exists():
        run.argv = [a for a in args_file.read_text().split("\n") if a != ""]
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            key, sep, value = line.partition("=")
            if sep:
                run.env.setdefault(key, value)
    return run


@pytest.fixture(scope="module")
def isolated_run(tmp_path_factory) -> HarborRun:
    """One default run, shared: the stage is slow and most tests only read it."""
    run = _run_harbor_stage(tmp_path_factory.mktemp("isolated"))
    if not run.invoked:
        pytest.skip(f"harbor stage did not reach harbor:\n{run.stderr[-2000:]}")
    return run


def test_overlay_is_passed_to_harbor(isolated_run):
    """Nothing else in the tree asserts the flag is passed at all.

    Every topology test in test_network_isolation.py resolves the overlay by
    hand with an explicit -f. If run_task.sh stopped passing it, those tests
    would stay green and every run would be an open-network run.
    """
    assert "--extra-docker-compose" in isolated_run.argv, (
        "harbor was not given the isolation overlay; the run has full egress"
    )


def test_overlay_path_is_the_overlay_and_nothing_else(isolated_run):
    """The return channel is stdout, and the function calls ensure_image.

    ensure_image narrates its build to stdout. Without the `>&2` redirect on
    that call, its progress lines are captured by the command substitution and
    concatenated into the path, and harbor is handed a -f that does not exist.
    Comparing to the literal path is what catches that; an `endswith` would
    not, because the build log is prepended.
    """
    value = isolated_run.flag_value("--extra-docker-compose")
    assert value == str(OVERLAY), (
        f"--extra-docker-compose got {value!r}, expected {str(OVERLAY)!r}. "
        "A longer value usually means a helper narrated to stdout."
    )


def test_overlay_path_exists_on_disk(isolated_run):
    """docker compose fails late and unhelpfully on a missing -f."""
    assert Path(isolated_run.flag_value("--extra-docker-compose")).is_file()


def test_overlay_is_passed_once(isolated_run):
    """A second -f of the same file is harmless but signals a double-append."""
    assert isolated_run.argv.count("--extra-docker-compose") == 1


def test_no_overlay_when_isolation_is_off(tmp_path):
    """NETWORK_ISOLATION_OFF=1 must produce a genuinely open run."""
    run = _run_harbor_stage(tmp_path, NETWORK_ISOLATION_OFF="1")
    if not run.invoked:
        pytest.skip("harbor stage did not reach harbor")
    assert "--extra-docker-compose" not in run.argv


def test_block_is_not_agent_specific(tmp_path):
    """The routing block holds for any agent; only the tool deny is claude-code.

    stage_harbor builds --disallowedTools inside an `if AGENT = claude-code`
    branch. The overlay must be outside it: an oracle run on an open network is
    the same leak as an agent run on one.
    """
    run = _run_harbor_stage(tmp_path, AGENT="oracle")
    if not run.invoked:
        pytest.skip("harbor stage did not reach harbor")
    assert "--extra-docker-compose" in run.argv, (
        "AGENT=oracle skipped the network block"
    )
    assert not [a for a in run.argv if a.startswith("disallowed_tools=")], (
        "disallowed_tools is a claude-code agent kwarg; harbor rejects it for oracle"
    )


def test_disallowed_tools_can_be_emptied_without_dropping_the_block(tmp_path):
    """The two layers are independent, and the tool deny is the optional one."""
    run = _run_harbor_stage(tmp_path, DISALLOWED_TOOLS="")
    if not run.invoked:
        pytest.skip("harbor stage did not reach harbor")
    assert "--extra-docker-compose" in run.argv, (
        "emptying DISALLOWED_TOOLS also dropped the routing block"
    )
    assert not [a for a in run.argv if a.startswith("disallowed_tools=")]


def test_disallowed_tools_override_is_passed_verbatim(tmp_path):
    run = _run_harbor_stage(tmp_path, DISALLOWED_TOOLS="WebFetch")
    if not run.invoked:
        pytest.skip("harbor stage did not reach harbor")
    assert "disallowed_tools=WebFetch" in run.argv


def test_default_deny_list_is_the_two_web_tools(isolated_run):
    """Pinned because the default is what every unattended run gets."""
    denied = [a for a in isolated_run.argv if a.startswith("disallowed_tools=")]
    assert denied == ["disallowed_tools=WebSearch,WebFetch"], denied


def test_only_the_overlay_path_reaches_the_return_channel():
    """The cold-start bug no warm machine can reproduce.

    network_isolation_overlay() returns by echoing to stdout, and it calls
    ensure_image, which narrates. On a machine that already has the image that
    call is `docker build -q >/dev/null` and prints nothing, so a missing `>&2`
    is invisible -- every dispatch test above still passes. On a fresh clone the
    same call prints "not present locally -- building from ..." plus a full
    BuildKit log, all of it captured by the command substitution and
    concatenated into the path, and harbor is handed a -f that does not exist.

    So this is asserted statically: it is the only way to cover the first run on
    a new machine, which is precisely the run that would hit it.
    """
    body, seen = [], False
    for line in RUN_TASK.read_text().splitlines():
        if line.startswith("network_isolation_overlay()"):
            seen = True
            continue
        if seen:
            if line.startswith("}"):
                break
            body.append(line)
    assert body, "network_isolation_overlay() not found in run_task.sh"

    code = [l for l in body if l.strip() and not l.lstrip().startswith("#")]
    assert code[-1].strip() == 'echo "$overlay"', (
        f"last statement is {code[-1].strip()!r}; the function returns by echoing "
        "the overlay path, so that must be the final word on stdout"
    )
    for line in code[:-1]:
        if "echo" not in line and "ensure_image" not in line:
            continue
        assert ">&2" in line, (
            f"`{line.strip()}` writes to stdout, which is this function's return "
            "channel -- its output would be concatenated into the overlay path"
        )


# --- refusals ----------------------------------------------------------------
# A live headroom proxy points the agent at host.docker.internal, which an
# internal network has no route to. GLM runs are not refused: they reach zbridge
# through squid instead. The refusal has to actually stop the run:
# `exit 2` fires inside a command substitution, so it kills the subshell, and
# only reaches the parent because the assignment is a statement of its own
# under `set -e`. Writing it as `local _iso="$(...)"` would swallow the status
# and the run would continue with an empty overlay -- silently open.


@pytest.fixture
def fake_zbridge():
    """A health endpoint on a free port, so ensure_zbridge does not spawn one.

    Reaching the isolation guard on the zbridge path means getting past two
    things that come first: check_credentials, which needs the two ZB_ vars,
    and ensure_zbridge, which starts a real zbridge unless something already
    answers /health on ZB_PORT. Answering it here keeps the test from
    background-launching a process it cannot reliably reap.
    """
    import http.server
    import threading

    class _Health(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, *_):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), _Health)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield {
            "ZB_PORT": str(srv.server_address[1]),
            "ZB_ZAI_API_KEY": "test-key",
            "ZB_BRIDGE_SECRET": "test-secret",
        }
    finally:
        srv.shutdown()
        srv.server_close()


def test_zbridge_run_is_isolated_through_squid(tmp_path, fake_zbridge):
    """GLM runs keep the block and reach zbridge through squid, on its port only."""
    port = fake_zbridge["ZB_PORT"]
    run = _run_harbor_stage(tmp_path, CC_MODE="zbridge", **fake_zbridge)
    assert "REFUSING" not in run.stderr, run.stderr[-2000:]
    assert run.invoked, "the run never reached harbor:\n" + run.stderr[-2000:]

    overlays = [run.argv[i + 1] for i, a in enumerate(run.argv) if a == "--extra-docker-compose"]
    assert overlays == [str(OVERLAY), str(PROXY_DIR / "overlay-zbridge.yaml")], overlays

    conf = Path(run.env["EGRESS_SQUID_CONF"])
    assert f"acl zbridge_port port {port}" in conf.read_text().splitlines()
    assert run.env.get("ANTHROPIC_BASE_URL") == f"http://host.docker.internal:{port}"
    assert any(a.startswith("disallowed_tools=") for a in run.argv), run.argv


# --- agent-path Headroom -----------------------------------------------------
# What stood here was a REFUSAL: headroom ran on the host, so main was pointed
# at host.docker.internal, which an internal network cannot route to. The proxy
# is a container in the project now, so the run is expected to START -- with the
# block still on, and with the proxy's own egress fenced separately.

HEADROOM_OVERLAY = PROXY_DIR / "overlay-headroom.yaml"
HEADROOM_OVERLAY_ISOLATED = PROXY_DIR / "overlay-headroom-isolated.yaml"


@requires_docker
def test_headroom_runs_under_isolation_in_its_own_container(tmp_path):
    run = _run_harbor_stage(tmp_path, AGENT_HEADROOM_ENABLED="true")
    assert "REFUSING" not in run.stderr, run.stderr[-2000:]
    assert run.invoked, "the run never reached harbor:\n" + run.stderr[-2000:]

    overlays = [run.argv[i + 1] for i, a in enumerate(run.argv) if a == "--extra-docker-compose"]
    assert overlays == [str(OVERLAY), str(HEADROOM_OVERLAY), str(HEADROOM_OVERLAY_ISOLATED)], (
        "the headroom overlays must come last: they re-set main's NO_PROXY, and "
        "an earlier position would let overlay.yaml's own value win\n" + str(overlays)
    )
    assert run.env.get("ANTHROPIC_BASE_URL") == "http://headroom:8787", run.env.get("ANTHROPIC_BASE_URL")
    assert run.env.get("HEADROOM_UPSTREAM") == "https://api.anthropic.com"
    assert Path(run.env["HEADROOM_SQUID_CONF"]) == PROXY_DIR / "squid.conf"


@requires_docker
def test_headroom_on_a_glm_run_keeps_zbridge_behind_the_proxy(tmp_path, fake_zbridge):
    """The case the host chain gave up on: isolation ON and CC_MODE=zbridge.

    It used to clear AGENT_HEADROOM_ENABLED and route straight to zbridge, so a
    GLM run asked for compression and silently got none.
    """
    port = fake_zbridge["ZB_PORT"]
    run = _run_harbor_stage(tmp_path, AGENT_HEADROOM_ENABLED="true",
                            CC_MODE="zbridge", **fake_zbridge)
    assert run.invoked, run.stderr[-2000:]
    assert run.env.get("ANTHROPIC_BASE_URL") == "http://headroom:8787", (
        "the agent bypassed headroom; compression was dropped, not applied"
    )
    assert run.env.get("HEADROOM_UPSTREAM") == f"http://host.docker.internal:{port}"
    conf = Path(run.env["HEADROOM_SQUID_CONF"])
    assert f"acl zbridge_port port {port}" in conf.read_text().splitlines(), (
        "headroom's own squid does not allow zbridge, so every model call is denied"
    )


@requires_docker
def test_no_headroom_container_unless_the_flag_is_set(tmp_path):
    run = _run_harbor_stage(tmp_path)
    assert run.invoked, run.stderr[-2000:]
    assert str(HEADROOM_OVERLAY) not in run.argv, "compression is opt-in"
    assert not run.env.get("ANTHROPIC_BASE_URL"), (
        "a base URL was exported without the flag; harbor pins every model alias "
        "to $MODEL whenever it is set"
    )


def test_the_headroom_overlay_exempts_it_from_mains_proxy():
    """main reaches headroom directly, or squid denies a host not on its list."""
    cfg = yaml.safe_load(HEADROOM_OVERLAY.read_text())
    main_env = cfg["services"]["main"]["environment"]
    for key in ("NO_PROXY", "no_proxy"):
        assert "headroom" in main_env[key].split(","), main_env[key]
    # Re-setting the key replaces it, so everything overlay.yaml exempted has to
    # be repeated here -- light-servers above all, which is how main reads the world.
    isolated = yaml.safe_load(OVERLAY.read_text())["services"]["main"]["environment"]
    assert set(isolated["NO_PROXY"].split(",")) <= set(main_env["NO_PROXY"].split(",")), (
        "the headroom overlay drops an exemption overlay.yaml made"
    )
    assert cfg["services"]["headroom"]["image"] == "headroom-compress:latest"


def test_the_headroom_overlay_leaves_claude_code_alone():
    """Nothing about the agent's own behaviour is changed to suit compression.

    ENABLE_TOOL_SEARCH=false sat here for two runs and did not help: the flag is
    a mode selector rather than a switch, and the damage was on the proxy side
    (61 tools in, 60 out). compress_proxy.py relays `tools` untouched, so the
    agent is configured exactly as it is on a run without compression -- which
    is also what keeps the two comparable.
    """
    cfg = yaml.safe_load(HEADROOM_OVERLAY.read_text())
    env = cfg["services"]["main"]["environment"]
    assert "ENABLE_TOOL_SEARCH" not in env, (
        "the agent is being reconfigured for the compressor; a compressor that "
        "needs that is not safe to put in front of it")
    assert cfg["services"]["headroom"]["image"] == "headroom-compress:latest"


def test_a_correctly_sized_vm_is_not_rounded_down_into_a_refusal(tmp_path):
    """What docker reports for a VM set to 12 GB, measured: 12,304,840 kB.

    The kernel's own reservation comes off the top, so flooring the division
    called that machine 11 GB and refused it -- telling an operator to raise a
    limit they had just raised, which is how a guard ends up commented out.
    """
    out = _drive_memory_check(tmp_path, str(12_304_840 * 1024))
    assert "REACHED_THE_RUN" in out.stdout, out.stdout + out.stderr


def test_headroom_gets_its_own_proxy_and_stays_out_of_the_agents_log():
    cfg = yaml.safe_load(HEADROOM_OVERLAY_ISOLATED.read_text())
    assert cfg["networks"]["headroom-net"]["internal"] is True
    assert cfg["services"]["headroom"]["environment"]["HTTPS_PROXY"] == "http://headroom-proxy:3128"
    # The agent's squid tees its access log into harbor's agent log dir, and
    # detect_internet_use.py reads that as evidence of what the AGENT did.
    # headroom's own blocked startup fetches (huggingface, a price list, an
    # onnxruntime telemetry host) would be scored as the agent reaching for the
    # web, so this proxy must not write there.
    volumes = cfg["services"]["headroom-proxy"].get("volumes") or []
    assert not any("/egress-out" in str(v) for v in volumes), volumes


def _drive_memory_check(tmp_path: Path, mem_bytes: str | None, **env_over) -> subprocess.CompletedProcess:
    """Run headroom_memory_check() out of run_task.sh against a stub docker.

    Lifted rather than driven through a stage, the way test_network_isolation
    drives load_dotenv: the check runs inside stage_preflight, and reaching it
    for real would mean building images.
    """
    import re as _re
    body = RUN_TASK.read_text()
    m = _re.search(r"^headroom_memory_check\(\) \{.*?^\}", body, _re.S | _re.M)
    assert m, "headroom_memory_check() not found in run_task.sh"

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    docker = bin_dir / "docker"
    # `docker info` answers with the byte count; anything else exits non-zero,
    # so a check that shells out to something else fails loudly here.
    docker.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "info" ]; then ' + (f'echo "{mem_bytes}"; exit 0; ' if mem_bytes is not None else "exit 1; ")
        + "fi\nexit 9\n"
    )
    docker.chmod(0o755)

    driver = tmp_path / "drive.sh"
    driver.write_text(
        "set -u\n"
        f'PATH="{bin_dir}:$PATH"\n'
        'AGENT_HEADROOM_ENABLED="${AGENT_HEADROOM_ENABLED:-true}"\n'
        'HEADROOM_MIN_DOCKER_GB="${HEADROOM_MIN_DOCKER_GB:-12}"\n'
        + m.group(0) + "\n"
        "headroom_memory_check\n"
        'echo REACHED_THE_RUN\n'
    )
    env = dict(os.environ)
    # Pinned, not inherited: this repo's .env sets these, and a shell that
    # carries AGENT_HEADROOM_ENABLED=false turns the refusal tests green
    # without the refusal ever running.
    env.pop("HEADROOM_IGNORE_MEMORY", None)
    env.pop("HEADROOM_MIN_DOCKER_GB", None)
    env["AGENT_HEADROOM_ENABLED"] = "true"
    env.update({k: str(v) for k, v in env_over.items()})
    return subprocess.run(["bash", str(driver)], capture_output=True, text=True, env=env, timeout=60)


def test_a_docker_too_small_for_headroom_is_refused(tmp_path):
    """The 8 GB case that killed a run: refuse before the agent phase is paid for."""
    out = _drive_memory_check(tmp_path, str(8 * 1024**3))
    assert out.returncode == 2, out.stdout + out.stderr
    assert "REFUSING" in out.stderr, out.stderr
    assert "REACHED_THE_RUN" not in out.stdout


def test_enough_memory_runs(tmp_path):
    out = _drive_memory_check(tmp_path, str(16 * 1024**3))
    assert "REACHED_THE_RUN" in out.stdout, out.stdout + out.stderr


def test_the_check_is_only_about_headroom_runs(tmp_path):
    out = _drive_memory_check(tmp_path, str(4 * 1024**3), AGENT_HEADROOM_ENABLED="false")
    assert "REACHED_THE_RUN" in out.stdout, out.stderr


def test_an_unreadable_docker_does_not_block(tmp_path):
    """No answer is not evidence of a small machine, and blocking on it would
    make every CI box unable to run."""
    out = _drive_memory_check(tmp_path, None)
    assert "REACHED_THE_RUN" in out.stdout, out.stderr


def test_the_memory_check_can_be_overridden(tmp_path):
    out = _drive_memory_check(tmp_path, str(4 * 1024**3), HEADROOM_IGNORE_MEMORY="1")
    assert "REACHED_THE_RUN" in out.stdout, out.stderr


def test_run_task_knows_how_to_build_the_compressor():
    body = RUN_TASK.read_text()
    assert re.search(r'headroom-compress\)\s*echo "\$REPO/tools/headroom"', body), (
        "image_build_context has no headroom-compress arm; ensure_image would pull it"
    )
    assert re.search(r"headroom-compress\)\s*\n?\s*printf .*scoring=", body, re.S), (
        "the compressor image COPYs grader_compress.py --from=scoring; without the "
        "named context the build fails"
    )
    assert (REPO / "tools" / "headroom" / "Dockerfile").is_file()
    assert (REPO / "tools" / "headroom" / "compress_proxy.py").is_file()


def test_zbridge_is_allowed_when_isolation_is_off(tmp_path, fake_zbridge):
    """The conflict is with the block, not with the proxies."""
    run = _run_harbor_stage(tmp_path, NETWORK_ISOLATION_OFF="1", CC_MODE="zbridge", **fake_zbridge)
    assert "REFUSING" not in run.stderr, (
        "refused a run that had explicitly opted out of isolation"
    )


def test_headroom_is_allowed_when_isolation_is_off(tmp_path):
    run = _run_harbor_stage(tmp_path, NETWORK_ISOLATION_OFF="1", AGENT_HEADROOM_ENABLED="true")
    assert "REFUSING" not in run.stderr, (
        "refused a run that had explicitly opted out of isolation"
    )


def test_agent_is_not_pointed_at_the_host_under_isolation(isolated_run):
    """harbor forwards ANTHROPIC_BASE_URL from this environment into main.

    A default isolated run must leave it alone or leave it unset; anything on
    host.docker.internal is unreachable once the bridge has no gateway.
    """
    base = isolated_run.env.get("ANTHROPIC_BASE_URL", "")
    assert "host.docker.internal" not in base, base


@pytest.mark.parametrize(
    "base_url",
    [
        "http://host.docker.internal:4000",
        "http://127.0.0.1:4000",
        "http://localhost:8787",
    ],
    ids=["host-gateway", "loopback-ip", "loopback-name"],
)
def test_an_inherited_host_base_url_never_reaches_harbor(tmp_path, base_url):
    """harbor forwards ANTHROPIC_BASE_URL from this environment into main
    verbatim (claude_code.py:1291), and under isolation main can reach none of
    these addresses -- the trial would die on its first model call looking like
    an outage.

    The protection is NOT in the isolation guard: run_task.sh:43 unsets the
    variable at the top of the script, before any stage runs, because running
    this from inside a Claude Code session inherits that session's own proxy
    address. So the isolation guard only has to cover the two proxies run_task
    starts itself, which is what it checks -- a second check on the resolved
    value there would be unreachable code.

    Asserted against the environment harbor was actually handed, since that is
    the only thing that decides what lands in the container.
    """
    run = _run_harbor_stage(tmp_path, ANTHROPIC_BASE_URL=base_url)
    if not run.invoked:
        pytest.skip("harbor stage did not reach harbor")
    assert not run.env.get("ANTHROPIC_BASE_URL"), (
        f"harbor was handed ANTHROPIC_BASE_URL={run.env['ANTHROPIC_BASE_URL']!r}; "
        "main has no route to it"
    )


def test_the_scrub_does_not_take_the_credential_with_it(tmp_path):
    """The same line unsets ANTHROPIC_API_KEY, so the OAuth token is the only
    credential the run has left and it must survive."""
    run = _run_harbor_stage(tmp_path, CLAUDE_CODE_OAUTH_TOKEN="sk-ant-oat01-test")
    if not run.invoked:
        pytest.skip("harbor stage did not reach harbor")
    assert run.env.get("CLAUDE_CODE_OAUTH_TOKEN") == "sk-ant-oat01-test"


# =============================================================================
# 1b. PREFLIGHT -- the gate that fails before the image is built
# =============================================================================
#
# The checks in preflight_network.py all predate the egress block: they ask
# whether HARBOR can enforce task.toml's network_mode, which under isolation is
# deliberately "public" -- the value that appends no harbor overlay at all. So
# every one of them can pass on a bundle that cannot possibly run.
#
# Driven end-to-end through run_task.sh rather than by importing the module, so
# a check that stops being CALLED fails here too.

# Whatever bundle is on disk, not a name. Pinning "Input_1" meant every test
# below skipped in silence the moment the bundles were renamed, and the preflight
# went unexercised for as long as that lasted.
PREFLIGHT_TASK = BUNDLES[0].parent if BUNDLES else REPO / "tasks" / "__none__"


def _copy_bundle(tmp_path: Path, bundle_src: Path) -> Path:
    import shutil
    task = tmp_path / bundle_src.name
    shutil.copytree(bundle_src, task)
    return task


def _run_preflight(task: Path, **overrides):
    """Run run_task.sh's preflight stage against a bundle copy."""
    import os
    import subprocess

    env = dict(os.environ)
    # Both of these turn off the thing under test, and both are commonly left
    # set in a shell or a .env. Inheriting either would make every assertion
    # below pass against a gate that never ran.
    env.pop("NETWORK_ISOLATION_OFF", None)
    env.pop("PREFLIGHT_NETWORK_OFF", None)
    env.update({"OUTPUT_DIR": str(task.parent / "output"), "AGENT": "claude-code"})
    env.update({k: str(v) for k, v in overrides.items()})
    return subprocess.run(
        [str(RUN_TASK), "--stage", "preflight", str(task)],
        capture_output=True, text=True, env=env, cwd=str(REPO), timeout=300,
    )


@requires_docker
@requires_credentials
@pytest.mark.skipif(not PREFLIGHT_TASK.is_dir(), reason="reference bundle absent")
def test_preflight_passes_a_bundle_that_is_ready_for_isolation(tmp_path):
    proc = _run_preflight(_copy_bundle(tmp_path, PREFLIGHT_TASK))
    assert proc.returncode == 0, proc.stdout[-3000:]
    assert "pre-bakes the Claude Code CLI" in proc.stdout
    assert "reachable under network isolation" in proc.stdout
    # The capability flags were READ, not warned around. harbor moved them onto
    # an instance attribute __init__ computes; reading them off the bare class
    # started raising, the raise was reported as a task defect, and every
    # operator answered with PREFLIGHT_NETWORK_OFF=1 -- which turns the whole
    # gate off. A run that cannot answer this question must never refuse.
    assert "is enforceable by the docker provider" in proc.stdout
    assert "capability flags could not be read" not in proc.stdout


@pytest.mark.skipif(not PREFLIGHT_TASK.is_dir(), reason="reference bundle absent")
def test_preflight_refuses_a_bundle_that_cannot_install_the_cli(tmp_path):
    """The failure this replaces: harbor's installer curls downloads.claude.ai
    inside the container, the allowlist denies it, and the trial dies in agent
    setup with an empty /logs/agent -- after the environment image was built,
    which is the expensive part."""
    task = _copy_bundle(tmp_path, PREFLIGHT_TASK)
    df = task / "environment" / "Dockerfile"
    df.write_text(df.read_text().replace("downloads.claude.ai", "example.invalid"))
    proc = _run_preflight(task)
    assert proc.returncode != 0, proc.stdout[-3000:]
    assert "does not pre-bake" in proc.stdout


@pytest.mark.skipif(not PREFLIGHT_TASK.is_dir(), reason="reference bundle absent")
def test_preflight_refuses_an_mcp_host_the_proxy_would_deny(tmp_path):
    """sidecar_hosts() only ever excluded localhost, so an external MCP url
    counted as a reachable sidecar. Under isolation it is a 403, and the symptom
    is a tool that silently does not work rather than anything naming a proxy."""
    task = _copy_bundle(tmp_path, PREFLIGHT_TASK)
    toml = task / "task.toml"
    toml.write_text(toml.read_text().replace(
        "http://light-servers:", "https://mcp.example.com:", 1))
    proc = _run_preflight(task)
    assert proc.returncode != 0, proc.stdout[-3000:]
    assert "egress proxy will deny" in proc.stdout


@requires_docker
@requires_credentials
@pytest.mark.skipif(not PREFLIGHT_TASK.is_dir(), reason="reference bundle absent")
def test_preflight_does_not_apply_isolation_rules_to_an_open_run(tmp_path):
    """An operator who asked for an open network must not be blocked by a rule
    about a block that is not in force."""
    task = _copy_bundle(tmp_path, PREFLIGHT_TASK)
    df = task / "environment" / "Dockerfile"
    df.write_text(df.read_text().replace("downloads.claude.ai", "example.invalid"))
    proc = _run_preflight(task, NETWORK_ISOLATION_OFF="1")
    assert proc.returncode == 0, proc.stdout[-3000:]
    assert "network isolation OFF" in proc.stdout


# =============================================================================
# 2. TOPOLOGY -- what `docker compose config` actually resolves
# =============================================================================
#
# The overlay states a policy about the *default* network. These assert the
# policy over the whole resolved project, so a bundle that opts out of the
# default network fails here rather than silently keeping its route.

pytest_topology = pytest.mark.skipif(not BUNDLES, reason="no task bundles in this checkout")


@pytest.fixture(scope="module")
def overlay() -> dict:
    return yaml.safe_load(OVERLAY.read_text())


def _resolved(task_toml: Path) -> dict:
    compose = task_toml.parent / "environment" / "docker-compose.yaml"
    if not compose.is_file():
        pytest.skip(f"{task_toml.parent.name} has no compose file")
    proc = subprocess.run(
        ["docker", "compose", "-f", str(compose), "-f", str(OVERLAY),
         "config", "--format", "json"],
        capture_output=True, text=True,
        env={"PATH": "/usr/bin:/bin:/usr/local/bin",
             "SCORING_DIR": str(REPO / "services" / "scoring"),
             # harbor exports this per trial (compose_env.py::
             # legacy_log_mount_env_vars) as the host side of the dir it mounts
             # at /logs/agent; the overlay declares it `:?` so a proxy whose
             # access log goes nowhere fails at `compose up` rather than
             # producing an audit with no evidence in it. Value is irrelevant
             # here -- config resolution only, nothing is started.
             "HOST_AGENT_LOGS_PATH": "/tmp/egress-out-test",
             # Exported by scripts/run_task.sh for bundles with a judge service;
             # their compose file declares both `:?`. Values irrelevant here.
             "JUDGE_TOKEN": "compose-config-test",
             "CODEX_AUTH_FILE": "/tmp/codex-auth-test.json",
             # harbor binds its per-trial verifier log dir (trial.py:798); the judge
             # writes its reports straight into it, so the compose file declares it `:?`.
             "HOST_VERIFIER_LOGS_PATH": "/tmp/verifier-logs-test"},
    )
    if proc.returncode != 0:
        pytest.fail(f"compose config failed:\n{proc.stderr}")
    return json.loads(proc.stdout)


@pytest.mark.skipif(shutil.which("docker") is None, reason="docker not installed")
@pytest_topology
@pytest.mark.parametrize("task_toml", BUNDLES, ids=_ids(BUNDLES))
class TestResolvedTopology:
    """One resolved project per bundle, asserted from several angles."""

    def test_every_network_but_egress_is_internal(self, task_toml):
        """The overlay names `default`. A bundle may name others.

        `internal: true` is per-network. A bundle that declares `networks:
        {app: {}}` and puts main on it keeps a full route out, and every
        assertion phrased about `default` still passes.
        """
        networks = _resolved(task_toml).get("networks") or {}
        leaky = [
            name for name, spec in networks.items()
            if name != EGRESS_NETWORK and not (spec or {}).get("internal")
        ]
        assert not leaky, (
            f"networks {leaky} are not internal -- services on them keep a route "
            "off the bridge that the proxy never sees"
        )

    def test_the_egress_network_is_not_internal(self, task_toml):
        """The inverse mistake: making `egress` internal too.

        That is a total outage rather than a leak -- squid can no longer reach
        api.anthropic.com either -- and it presents as an agent that cannot
        make a single model call.
        """
        egress = (_resolved(task_toml).get("networks") or {}).get(EGRESS_NETWORK)
        assert egress is not None, "no egress network for the proxy to reach out through"
        assert not egress.get("internal"), (
            "the egress network is internal; the proxy has no route out either"
        )

    def test_only_the_proxy_is_attached_to_egress(self, task_toml):
        services = (_resolved(task_toml).get("services") or {})
        joined = sorted(
            name for name, spec in services.items()
            if EGRESS_NETWORK in ((spec or {}).get("networks") or {})
        )
        assert joined == [PROXY_SERVICE], (
            f"{joined} sit on the egress network; it must be the proxy alone"
        )

    def test_no_service_sets_a_network_mode(self, task_toml):
        """network_mode bypasses `networks:` entirely.

        `host` shares the host stack outright; `bridge` puts the service on the
        default docker bridge, which has a gateway. Either one is a route out
        that no amount of overlay fixes, and Harbor writes this key itself when
        task.toml asks for no-network.
        """
        services = (_resolved(task_toml).get("services") or {})
        modes = {n: s.get("network_mode") for n, s in services.items() if s.get("network_mode")}
        assert not modes, f"network_mode set on {modes}; the overlay does not reach these"

    def test_main_waits_for_the_proxy_in_the_resolved_project(self, task_toml):
        """The overlay contributes a depends_on *mapping*.

        Bundles write depends_on as a list. Compose normalises before merging,
        so this works -- but it works by a rule neither file states, and the
        failure (agent starts before squid listens; first call is a connection
        refused) reads like an outage. Assert the merge, not the intent.
        """
        main = (_resolved(task_toml).get("services") or {}).get("main") or {}
        dep = (main.get("depends_on") or {}).get(PROXY_SERVICE)
        assert dep, "main does not depend on the proxy in the resolved project"
        assert dep.get("condition") == "service_healthy", dep

    def test_main_keeps_its_own_dependencies(self, task_toml):
        """The merge must add, not replace.

        If the overlay's depends_on overwrote the bundle's, main would stop
        waiting for light-servers and the agent would start with no MCP tools --
        the exact failure no-network caused, reintroduced by the fix for it.
        """
        resolved = _resolved(task_toml)
        services = resolved.get("services") or {}
        if "light-servers" not in services:
            pytest.skip("bundle has no light-servers sidecar")
        deps = ((services.get("main") or {}).get("depends_on") or {})
        assert "light-servers" in deps, sorted(deps)

    def test_proxy_env_survives_the_merge(self, task_toml):
        """The overlay is applied after the bundle, so it should win.

        Should is doing work there: harbor appends a mounts overlay after ours,
        and a bundle could grow an env_file. Read the resolved value.
        """
        main = (_resolved(task_toml).get("services") or {}).get("main") or {}
        env = main.get("environment") or {}
        for var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
            assert env.get(var) == f"http://{PROXY_SERVICE}:3128", (
                f"{var} resolved to {env.get(var)!r}"
            )

    def test_every_sidecar_is_exempt_from_the_proxy(self, task_toml):
        """NO_PROXY is a hardcoded list; the service list is per bundle.

        A bundle that adds a sidecar gets its bridge-local traffic routed at
        squid, which denies it -- so the new sidecar is simply unreachable, and
        the symptom is a broken tool rather than anything mentioning the proxy.
        """
        resolved = _resolved(task_toml)
        main = (resolved.get("services") or {}).get("main") or {}
        no_proxy = {h.strip() for h in (main.get("environment") or {}).get("NO_PROXY", "").split(",")}
        missing = sorted(set(resolved.get("services") or {}) - no_proxy)
        assert not missing, (
            f"services {missing} are not in NO_PROXY; traffic to them would be "
            "sent to squid and denied"
        )


@pytest_topology
@pytest.mark.parametrize("task_toml", BUNDLES, ids=_ids(BUNDLES))
def test_bundle_declares_no_networks_of_its_own(task_toml):
    """Cheap static twin of the resolved check, and it runs without docker.

    Also catches the case the resolved check cannot: a bundle whose compose
    fails to resolve at all still gets read here.
    """
    compose = task_toml.parent / "environment" / "docker-compose.yaml"
    if not compose.is_file():
        pytest.skip("no compose file")
    spec = yaml.safe_load(compose.read_text()) or {}
    declared = set(spec.get("networks") or {})
    assert not declared - {"default", EGRESS_NETWORK}, (
        f"{task_toml.parent.name} declares networks {sorted(declared)}; the "
        "overlay only makes `default` internal"
    )
    for name, svc in (spec.get("services") or {}).items():
        assert not (svc or {}).get("network_mode"), (
            f"{name} sets network_mode; the overlay cannot reach it"
        )


# =============================================================================
# 3. POLICY -- squid's allow rules as rules, not as text
# =============================================================================


def _directives(text: str) -> list[str]:
    out = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            out.append(line)
    return out


@pytest.fixture(scope="module")
def squid_lines() -> list[str]:
    assert SQUID_CONF.is_file(), f"squid.conf missing at {SQUID_CONF}"
    return _directives(SQUID_CONF.read_text())


@pytest.fixture(scope="module")
def acls(squid_lines) -> dict[str, tuple[str, set[str]]]:
    """acl name -> (type, values). Squid appends on redefinition; so do we."""
    out: dict[str, tuple[str, set[str]]] = {}
    for line in squid_lines:
        m = re.match(r"acl\s+(\S+)\s+(\S+)\s*(.*)$", line)
        if not m:
            continue
        name, kind, rest = m.group(1), m.group(2), m.group(3)
        prev_kind, prev_vals = out.get(name, (kind, set()))
        out[name] = (prev_kind, prev_vals | set(rest.split()))
    return out


@pytest.fixture(scope="module")
def access_rules(squid_lines) -> list[tuple[str, list[str]]]:
    """http_access lines as (action, [acl tokens]), in file order."""
    rules = []
    for line in squid_lines:
        m = re.match(r"http_access\s+(allow|deny)\s+(.*)$", line)
        if m:
            rules.append((m.group(1), m.group(2).split()))
    return rules


def test_every_referenced_acl_is_defined(access_rules, acls):
    """A typo'd acl name is a squid parse error, i.e. a container that dies.

    `squid -k parse` in the Dockerfile catches it at build time; this catches it
    at commit time, which is where the edit was made.
    """
    known = set(acls) | {"all", "manager", "localhost", "to_localhost", "CONNECT"}
    for action, tokens in access_rules:
        for tok in tokens:
            assert tok.lstrip("!") in known, (
                f"http_access {action} references undefined acl {tok!r}"
            )


def test_every_allow_is_scoped_to_a_host_allowlist(access_rules, acls):
    """The load-bearing token is the dstdomain one.

    `http_access allow CONNECT SSL_ports` -- the same line with the allowlist
    dropped -- reads like a rule about TLS and is an open tunnel to every host
    on the internet.
    """
    for action, tokens in access_rules:
        if action != "allow":
            continue
        kinds = {acls.get(t.lstrip("!"), ("", set()))[0] for t in tokens}
        assert "dstdomain" in kinds, (
            f"`http_access allow {' '.join(tokens)}` is not scoped to a "
            "dstdomain allowlist; it permits hosts by something other than name"
        )


def test_the_connect_rule_names_only_the_tls_port(access_rules, acls):
    """Narrow check: the rule that mentions CONNECT is scoped to 443.

    This is NOT the same as "CONNECT is restricted to 443" -- squid takes the
    first matching rule, and a later rule that does not mention CONNECT can
    still match one. test_connect_to_a_non_tls_port_is_denied below asks that
    question properly. Kept because a widened SSL_ports is worth naming on its
    own.
    """
    for action, tokens in access_rules:
        if action != "allow" or "CONNECT" not in tokens:
            continue
        port_acls = [t for t in tokens if acls.get(t, ("", set()))[0] == "port"]
        assert port_acls, f"`http_access allow {' '.join(tokens)}` names no port acl"
        for name in port_acls:
            assert acls[name][1] == {"443"}, (
                f"acl {name} covers ports {sorted(acls[name][1])}, not 443 alone"
            )


# --- squid's first-match semantics, evaluated ---------------------------------
#
# Reading the rules one line at a time is what let the check above look
# sufficient. A request is matched against the whole list in order, so the
# guarantee a line appears to make is only real if no *later* line also matches.
# These evaluate a request the way squid does and assert on the verdict.


def _acl_matches(token: str, acls, method: str, host: str, port: int) -> bool:
    negated = token.startswith("!")
    name = token.lstrip("!")
    if name == "all":
        result = True
    else:
        kind, values = acls.get(name, ("", set()))
        if kind == "method":
            result = method in values
        elif kind == "port":
            result = str(port) in values
        elif kind == "dstdomain":
            result = any(
                host == v or (v.startswith(".") and host.endswith(v))
                for v in values
            )
        else:
            # An acl type this evaluator does not model. Treat it as matching so
            # the verdict errs toward "allowed" -- a test that silently decided
            # such a rule could never fire would under-report.
            result = True
    return result != negated


def _verdict(access_rules, acls, method: str, host: str, port: int) -> str:
    """allow/deny for one request, by squid's first-match rule."""
    for action, tokens in access_rules:
        if all(_acl_matches(t, acls, method, host, port) for t in tokens):
            return action
    return "deny"


ALLOWED_HOST = "api.anthropic.com"
DENIED_HOST = "example.com"


@pytest.mark.parametrize(
    "method,host,port,expected",
    [
        ("CONNECT", ALLOWED_HOST, 443, "allow"),   # the one thing that must work
        ("GET", ALLOWED_HOST, 80, "allow"),
        ("CONNECT", DENIED_HOST, 443, "deny"),
        ("GET", DENIED_HOST, 80, "deny"),
        ("CONNECT", "1.1.1.1", 443, "deny"),
        ("GET", "anthropic.com.evil.test", 80, "deny"),
    ],
    ids=lambda v: str(v),
)
def test_requests_get_the_verdict_the_policy_intends(
    access_rules, acls, method, host, port, expected
):
    assert _verdict(access_rules, acls, method, host, port) == expected


@pytest.mark.parametrize("port", [22, 25, 3306], ids=["ssh", "smtp", "mysql"])
def test_connect_to_a_non_tls_port_is_denied(access_rules, acls, port):
    assert _verdict(access_rules, acls, "CONNECT", ALLOWED_HOST, port) == "deny"


def test_no_allow_is_scoped_by_client_address(acls, access_rules):
    """`acl x src all` + allow is a rule about who asks, not about where to.

    Everything inside the compose project would match it, which is everything
    that can reach the proxy at all.
    """
    for action, tokens in access_rules:
        if action != "allow":
            continue
        for tok in tokens:
            kind = acls.get(tok, ("", set()))[0]
            assert kind != "src", (
                f"allow rule scoped by source address ({tok}); that permits by "
                "client, and every client here is inside the project"
            )


def test_there_is_exactly_one_terminal_deny(access_rules):
    """Ordering: squid takes the first match.

    An allow appended after `deny all` is dead and harmless. An allow inserted
    above it is live. Pinning the deny to the last position is what makes the
    difference visible in a diff.
    """
    assert access_rules, "no http_access rules; squid falls back to its built-in policy"
    assert access_rules[-1] == ("deny", ["all"]), access_rules[-1]
    assert [r for r in access_rules if r == ("deny", ["all"])] == [("deny", ["all"])], (
        "more than one `deny all`; the first wins and the rest are dead text"
    )


def test_the_connect_port_deny_precedes_every_allow(access_rules):
    """Ordering is the entire mechanism; a deny below the allows is dead text.

    The evaluator tests above would catch a reordering too, but they would
    report it as "CONNECT :22 is allowed" rather than as "the rule is in the
    wrong place", and the second is the thing to fix.
    """
    denies = [i for i, (a, t) in enumerate(access_rules) if a == "deny" and "CONNECT" in t]
    allows = [i for i, (a, _) in enumerate(access_rules) if a == "allow"]
    assert denies, "no CONNECT port deny; the 443 restriction is decorative"
    assert max(denies) < min(allows), (
        "the CONNECT deny sits below an allow, so squid matches the allow first"
    )


def test_optional_claude_traffic_is_turned_off_at_the_source(overlay):
    """Denying telemetry at the proxy works but costs a retry and a log line.

    Every one of those denials lands in access.log beside the ones an operator
    is actually reading, which is the practical cost: a real denied host gets
    harder to find.
    """
    env = overlay["services"]["main"]["environment"]
    assert env.get("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC") == "1", (
        "Claude Code will keep attempting telemetry and update checks that the "
        "allowlist denies"
    )


def test_no_second_route_out_of_the_proxy(squid_lines):
    """cache_peer forwards to another proxy, outside these rules entirely."""
    for line in squid_lines:
        assert not line.startswith("cache_peer"), (
            f"`{line}` gives the proxy an upstream the allowlist does not cover"
        )


def test_the_proxy_port_is_consistent_across_the_three_files(squid_lines, overlay):
    """squid.conf listens, overlay points, Dockerfile documents."""
    ports = [l.split()[1] for l in squid_lines if l.startswith("http_port")]
    assert ports, "no http_port directive"
    port = ports[0]
    env = overlay["services"]["main"]["environment"]
    for var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        assert env[var].endswith(f":{port}"), f"{var}={env[var]} but squid listens on {port}"
    exposed = re.findall(r"^EXPOSE\s+(\d+)", DOCKERFILE.read_text(), re.M)
    assert exposed == [port], f"Dockerfile EXPOSEs {exposed}, squid listens on {port}"


# =============================================================================
# 4. PACKAGING -- the image the overlay names has to be the one that gets built
# =============================================================================


def test_overlay_names_a_prebuilt_image(overlay):
    proxy = overlay["services"][PROXY_SERVICE]
    assert proxy.get("image") == PROXY_IMAGE, proxy.get("image")
    assert "build" not in proxy, (
        "`build:` here rebuilds the proxy inside every task project, per run"
    )


def test_run_task_knows_how_to_build_that_image():
    """image_build_context() is what turns the tag into a buildable context.

    Without the arm, ensure_image treats egress-proxy:latest as a registry
    image and tries to `docker pull` it, which fails as "repository does not
    exist" minutes into the run.
    """
    body = RUN_TASK.read_text()
    m = re.search(r"^\s*egress-proxy\)\s*echo\s+\"([^\"]+)\"", body, re.M)
    assert m, "run_task.sh image_build_context has no egress-proxy arm"
    ctx = m.group(1).replace("$REPO", str(REPO))
    assert Path(ctx).is_dir(), f"build context {ctx} does not exist"
    assert (Path(ctx) / "Dockerfile").is_file()


def test_makefile_builds_the_same_tag_from_the_same_context():
    body = MAKEFILE.read_text()
    assert "build-egress-proxy:" in body, "no make target to build the proxy"
    m = re.search(r"docker build -t (\S+) (\S+)", body[body.index("build-egress-proxy:"):])
    assert m, "build-egress-proxy target does not run docker build"
    assert m.group(1) == PROXY_IMAGE, m.group(1)
    assert (REPO / m.group(2)).resolve() == PROXY_DIR.resolve(), m.group(2)


def test_entrypoint_prepares_every_directory_squid_writes_to(squid_lines):
    """Squid drops to the `proxy` user before opening its log and pid files.

    Either directory missing or owned by root is a FATAL at startup, which
    presents from inside main as a network outage rather than as a proxy that
    failed to boot. Both paths come from squid.conf, so read them from there
    rather than restating them.
    """
    sh = ENTRYPOINT.read_text()
    paths = [
        l.split("stdio:")[1].split()[0] for l in squid_lines if l.startswith("access_log")
    ] + [
        l.split()[1] for l in squid_lines if l.startswith("pid_filename")
    ]
    for path in paths:
        parent = str(Path(path).parent)
        assert f"mkdir -p" in sh and parent in sh, f"entrypoint never creates {parent}"
        assert re.search(rf"chown[^\n]*{re.escape(parent)}", sh), (
            f"entrypoint never chowns {parent} to the proxy user"
        )


def test_entrypoint_is_strict_and_execs():
    """`exec` keeps squid as PID 1 so compose stop reaches it, not the shell."""
    sh = ENTRYPOINT.read_text()
    assert re.search(r"^set -e", sh, re.M), "entrypoint does not abort on setup failure"
    assert re.search(r"^exec ", sh, re.M), "entrypoint does not exec; squid is not PID 1"


def test_the_proxy_base_image_is_pinned():
    """This container is the enforcement boundary.

    On a floating tag the component deciding what the agent may reach can change
    between two runs of the same benchmark, with nothing in the repo recording
    that it did. squid.conf already makes the reproducibility argument for
    disabling the cache; it applies to the proxy itself at least as strongly.
    """
    froms = re.findall(r"^FROM\s+(\S+)", DOCKERFILE.read_text(), re.M)
    assert froms, "no FROM in the proxy Dockerfile"
    for ref in froms:
        assert "@sha256:" in ref, f"`FROM {ref}` is a floating reference"


def test_config_is_parsed_at_build_time():
    assert re.search(r"squid -k parse", DOCKERFILE.read_text()), (
        "Dockerfile does not parse the config; a malformed ACL would fail the "
        "run instead of the build"
    )


# =============================================================================
# 5. The bundles must not re-introduce the mode this replaced
# =============================================================================


@pytest_topology
@pytest.mark.parametrize("task_toml", BUNDLES, ids=_ids(BUNDLES))
def test_no_bundle_asks_harbor_for_a_network_mode_of_its_own(task_toml):
    """Restated from test_network_isolation.py in the negative.

    That file asserts the value is "public". This asserts the two values that
    break things are absent, including from sections it does not read -- a
    [agent.kwargs] or a future [environment.overrides] naming no-network would
    pass there and fail here.
    """
    cfg = tomllib.loads(task_toml.read_text())
    # Comment lines are excluded on purpose: every bundle here carries a comment
    # explaining why it is NOT no-network, and that prose is the documentation
    # this test exists to protect. Only live settings are searched.
    live = [
        l for l in task_toml.read_text().splitlines()
        if l.strip() and not l.lstrip().startswith("#")
    ]
    offenders = [l for l in live if "no-network" in l]
    assert not offenders, (
        f"{task_toml.parent.name} sets no-network ({offenders}); that detaches "
        "main from the bridge, takes light-servers with it, and grades the run 0"
    )
    assert (cfg.get("environment") or {}).get("network_mode") == "public"
