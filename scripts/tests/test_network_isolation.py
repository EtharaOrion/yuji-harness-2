"""The egress block is configuration, and configuration regresses silently.

These tests assert the three properties that make it real, all of them cheap and
none of them needing Docker to run a container:

  1. the overlay makes the project's default network internal, and puts main on
     it and nothing else -- this is what removes the route, and it is the only
     part the agent cannot defeat from inside;
  2. every task bundle stays on network_mode = "public", because any other value
     re-introduces the abort this replaced;
  3. every bundle pre-bakes the CLI, because harbor's own installer cannot reach
     downloads.claude.ai once the block is on.

A `docker compose config` is used for (1) rather than a YAML diff: the question
is what Compose *resolves*, and the implicit default network only appears there.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
OVERLAY = REPO / "tools" / "network" / "egress-proxy" / "overlay.yaml"
BUNDLES = sorted(REPO.glob("tasks/*/task.toml"))

pytestmark = pytest.mark.skipif(not BUNDLES, reason="no task bundles in this checkout")


def _ids(paths):
    return [p.parent.name for p in paths]


def test_overlay_exists():
    assert OVERLAY.is_file(), f"network isolation overlay missing at {OVERLAY}"


@pytest.mark.skipif(shutil.which("docker") is None, reason="docker not installed")
@pytest.mark.parametrize("task_toml", BUNDLES, ids=_ids(BUNDLES))
def test_overlay_isolates_main(task_toml: Path):
    """main must end up on an internal network, and off the egress one."""
    compose = task_toml.parent / "environment" / "docker-compose.yaml"
    if not compose.is_file():
        pytest.skip(f"{task_toml.parent.name} has no compose file")

    proc = subprocess.run(
        ["docker", "compose", "-f", str(compose), "-f", str(OVERLAY), "config", "--format", "json"],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin:/usr/local/bin",
             "SCORING_DIR": str(REPO / "services" / "scoring"),
             # Supplied by harbor per trial (compose_env.py::legacy_log_mount_env_vars);
             # the overlay declares it `:?` so an unmounted proxy log fails at
             # `compose up` instead of silently producing an empty audit. Any
             # path works here -- nothing is started.
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

    cfg = json.loads(proc.stdout)
    networks = cfg.get("networks") or {}
    services = cfg.get("services") or {}

    assert networks.get("default", {}).get("internal") is True, (
        "the default network is not internal -- main keeps a route to the open web"
    )
    assert "egress" in networks, "no egress network for the proxy to reach out through"

    main_nets = set((services.get("main") or {}).get("networks") or {})
    assert main_nets == {"default"}, (
        f"main must sit on the internal default network alone, got {sorted(main_nets)}"
    )

    proxy = services.get("egress-proxy")
    assert proxy, "overlay did not contribute the egress-proxy service"
    assert set(proxy.get("networks") or {}) == {"default", "egress"}, (
        "the proxy must span both networks; it is the only route out"
    )

    # The proxy's access log is the only per-run evidence that the block was
    # live. It has to leave the container while the container still exists:
    # harbor tears the project down before run_task.sh reaches stage_netaudit,
    # so `docker logs` is not available to the audit at any later point.
    targets = {v.get("target") for v in (proxy.get("volumes") or [])}
    assert "/egress-out" in targets, (
        "the proxy does not mount /egress-out, so its access log dies with the "
        "container and the audit degrades to trajectory inference with no "
        "ground truth. See tools/network/egress-proxy/overlay.yaml."
    )

    # The sidecars are the reason no-network was unusable. Keep them reachable.
    if "light-servers" in services:
        assert set(services["light-servers"].get("networks") or {}) == {"default"}


def test_harbor_still_exports_the_var_the_overlay_mounts():
    """The overlay's one dependency on harbor internals.

    tools/network/egress-proxy/overlay.yaml mounts ${HOST_AGENT_LOGS_PATH} to carry
    squid's access log off the container, and that variable is not ours -- harbor
    derives it in compose_env.py::legacy_log_mount_env_vars from the bind mount
    it makes at /logs/agent, by taking the target's BASENAME and looking it up in
    a private suffix table.

    Nothing upstream promises to keep doing that. If harbor renames the mount,
    drops the legacy export, or changes the table, the overlay's `:?` turns every
    run into a `compose up` failure -- correct, but the message would point at our
    file rather than at the harbor change that caused it. This test names the real
    cause up front.

    Skipped, not failed, when harbor is not importable from this interpreter: the
    repo venv and harbor's pipx venv have incompatible pydantic_core builds, the
    same condition preflight_network.py degrades on. Run it under harbor's python
    to actually exercise the assertions:

        $(dirname $(readlink -f $(command -v harbor)))/python -m pytest ...
    """
    pytest.importorskip("harbor", reason="harbor not importable from this interpreter")
    from harbor.environments.docker.compose_env import (
        _LEGACY_LOG_MOUNT_SUFFIXES,
        legacy_log_mount_env_vars,
    )

    assert _LEGACY_LOG_MOUNT_SUFFIXES.get("agent") == "AGENT_LOGS", (
        "harbor no longer maps an /logs/agent mount to the AGENT_LOGS legacy name; "
        f"got {_LEGACY_LOG_MOUNT_SUFFIXES!r}"
    )

    env = legacy_log_mount_env_vars(
        [{"type": "bind", "source": "/host/trial/agent", "target": "/logs/agent"}],
        host_value="source",
    )
    assert env.get("HOST_AGENT_LOGS_PATH") == "/host/trial/agent", (
        "harbor stopped exporting HOST_AGENT_LOGS_PATH as the HOST side of the "
        f"agent log mount; got {env!r}. tools/network/egress-proxy/overlay.yaml "
        "interpolates that name and will fail at `compose up`."
    )


@pytest.mark.parametrize("task_toml", BUNDLES, ids=_ids(BUNDLES))
def test_network_mode_stays_public(task_toml: Path):
    """no-network detaches the compose bridge and grades the run 0.

    It also differs from [environment], and the docker provider cannot switch
    policy mid-trial, so the trial aborts before the agent phase.
    """
    cfg = tomllib.loads(task_toml.read_text())
    env_mode = (cfg.get("environment") or {}).get("network_mode")
    agent_mode = (cfg.get("agent") or {}).get("network_mode")

    assert env_mode == "public", f"[environment].network_mode is {env_mode!r}, must be 'public'"
    if agent_mode is not None:
        assert agent_mode == "public", (
            f"[agent].network_mode is {agent_mode!r}; egress is blocked by the compose "
            "overlay, not by harbor, so this must stay 'public'"
        )


@pytest.mark.parametrize("task_toml", BUNDLES, ids=_ids(BUNDLES))
def test_bundle_prebakes_cli(task_toml: Path):
    """harbor's installer cannot reach downloads.claude.ai under isolation."""
    dockerfile = task_toml.parent / "environment" / "Dockerfile"
    if not dockerfile.is_file():
        pytest.skip(f"{task_toml.parent.name} has no Dockerfile")

    body = dockerfile.read_text()
    assert "downloads.claude.ai" in body, (
        "Dockerfile does not pre-bake the Claude Code CLI; agent setup will try to "
        "download it inside the container, where there is no route out"
    )
    assert "procps" in body, (
        "procps missing -- harbor's installer used to add it and is now a no-op, "
        "but claude's node-tree-kill still shells out to ps/pgrep"
    )


# --- tool-level deny ---------------------------------------------------------
# The second layer, borrowed from WildClawBench's tools.deny: the routing block
# already makes WebSearch/WebFetch fail, but a failing tool still costs a turn.
# Denying them removes them from the tool list entirely.
#
# These drive run_task.sh with a stub `harbor` on PATH and read the argv it
# built, so they check the wiring rather than re-stating the constant.

RUN_TASK = REPO / "scripts" / "run_task.sh"

_HARBOR_STUB = """#!/usr/bin/env bash
printf '%s\\n' "$@" >> "$HARBOR_ARGS"
mkdir -p "$JOB_DIR"
touch "$JOB_DIR/result.json"
exit 0
"""


def _harbor_argv(tmp_path, **overrides) -> list[str]:
    """Run the harbor stage against a stub harbor and return the argv it got."""
    import os
    import pathlib
    import shutil
    import subprocess

    task = tmp_path / "tasks" / "alpha"
    task.mkdir(parents=True)
    (task / "task.toml").write_text('name = "acme/alpha"\n')

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = bin_dir / "harbor"
    stub.write_text(_HARBOR_STUB)
    stub.chmod(0o755)

    # run_task.sh runs patch_harbor.py at dispatch, and
    # find_harbor_claude_code() locates harbor relative to whichever `harbor` is
    # on PATH: it globs <venv>/lib/python*/site-packages/harbor/... where <venv>
    # is the stub's grandparent. Shadowing PATH therefore points it at this tmp
    # dir, and it dies before harbor is ever invoked -- which is why the
    # pre-existing test_run_task_stages.py cases fail too.
    #
    # Mirror the two files it patches into a fake venv layout so the glob
    # resolves here. They are COPIES: patch_harbor rewrites them in place, and
    # a test must not mutate the real harbor install.
    real_harbor = shutil.which("harbor")
    if real_harbor:
        real_pkg = None
        venv_root = pathlib.Path(real_harbor).resolve().parent.parent
        for cc in venv_root.glob("lib/python*/site-packages/harbor/agents/installed/claude_code.py"):
            real_pkg = cc.parent.parent.parent
            rel = cc.relative_to(venv_root)
            break
        if real_pkg is not None:
            for src, dest_rel in (
                (real_pkg / "agents" / "installed" / "claude_code.py", rel),
                (real_pkg / "trial" / "trial.py",
                 rel.parent.parent.parent / "trial" / "trial.py"),
                # Same gap as conftest.mirror_harbor_package: patch_harbor.py now
                # rewrites cli/jobs.py too, and a MISS stops the stage before harbor.
                (real_pkg / "cli" / "jobs.py",
                 rel.parent.parent.parent / "cli" / "jobs.py"),
            ):
                dest = tmp_path / dest_rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                if src.exists():
                    dest.write_text(src.read_text())

    args_file = tmp_path / "harbor_args.txt"
    env = dict(os.environ)
    env.update({
        "PATH": f"{bin_dir}:{env['PATH']}",
        "OUTPUT_DIR": str(tmp_path / "output"),
        "JOB": "alpha",
        "JOB_DIR": str(tmp_path / "output" / "alpha"),
        "HARBOR_ARGS": str(args_file),
        "RUN_OFFSET": "0",
        "MODEL": "m1",
        "N": "1",
        "AGENT": "claude-code",
    })
    env.update({k: str(v) for k, v in overrides.items()})

    subprocess.run(
        [str(RUN_TASK), "--stage", "harbor", str(task)],
        capture_output=True, text=True, env=env, cwd=str(REPO), timeout=300,
    )
    if not args_file.exists():
        pytest.skip("harbor stage did not reach harbor (unrelated preflight failure)")
    return args_file.read_text().split("\n")


def test_web_tools_denied_when_isolated(tmp_path):
    argv = _harbor_argv(tmp_path)
    assert "disallowed_tools=WebSearch,WebFetch" in argv, (
        "web tools not denied; the agent will spend turns on tools that cannot work"
    )


def test_bash_is_not_denied(tmp_path):
    """Bash does real local work and its egress is already dead at the router."""
    argv = _harbor_argv(tmp_path)
    denied = [a for a in argv if a.startswith("disallowed_tools=")]
    assert denied, "expected a disallowed_tools flag"
    assert "Bash" not in denied[0], (
        "denying Bash breaks tasks to buy nothing -- egress is blocked by routing"
    )


def test_not_denied_when_isolation_is_off(tmp_path):
    """An operator who asked for an open run must get one, tools included."""
    argv = _harbor_argv(tmp_path, NETWORK_ISOLATION_OFF="1")
    assert not [a for a in argv if a.startswith("disallowed_tools=")], (
        "web tools still denied with NETWORK_ISOLATION_OFF=1; that run would not "
        "mean what the operator asked for"
    )


# --- the third layer: the PreToolUse egress guard ---------------------------
#
# The routing table removes the route and DISALLOWED_TOOLS removes the web
# tools. Neither says anything back when the model reaches for `apt-get`, so it
# reaches again: one recorded run spent seven consecutive Bash calls on npm,
# apt-get, chromium and puppeteer before giving up. The guard refuses in the
# same turn and names what to use instead.

def _guard_settings_path(argv: list[str]) -> str | None:
    for a in argv:
        if a.startswith("config="):
            return a[len("config="):]
    return None


def test_egress_guard_is_passed_to_harbor(tmp_path):
    argv = _harbor_argv(tmp_path)
    path = _guard_settings_path(argv)
    assert path, (
        "no `--ak config=` in harbor's argv; the agent runs with no PreToolUse "
        "guard and burns turns on commands that cannot work"
    )

    import json
    doc = json.loads(Path(path).read_text())
    hooks = doc["hooks"]["PreToolUse"]
    assert hooks and "Bash" in hooks[0]["matcher"], doc


def test_egress_guard_is_not_passed_when_isolation_is_off(tmp_path):
    """An operator who asked for an open run must get one, Bash included."""
    argv = _harbor_argv(tmp_path, NETWORK_ISOLATION_OFF="1")
    assert _guard_settings_path(argv) is None, (
        "Bash egress still guarded with NETWORK_ISOLATION_OFF=1; that run would "
        "not mean what the operator asked for"
    )


# --- "delivery withheld" has to be true -------------------------------------

# A trajectory shaped to produce each verdict. The stage re-runs the detector,
# so seeding the audit JSON by hand would be overwritten -- and seeding the
# TRAJECTORY instead means these tests exercise the real path end to end.
_HOOK_REFUSAL = ("PreToolUse:Bash hook error: BLOCKED: this command reaches the "
                 "public internet, and this task is closed-world.")
_TRAJECTORIES = {
    # it got out: the command's own output proves the index answered
    "reached_internet": ("pip install pandas", "Successfully installed pandas-2.2.3"),
    # it tried and the guard refused it before it ran
    "attempt_blocked": ("pip install pandas", _HOOK_REFUSAL),
    # it tried and nothing witnessed the outcome
    "attempt_unverified": ("pip install pandas", None),
    # it never reached for the web
    "no_attempt": ("ls -la /workspace/data", "data"),
}


def _drive_netaudit(tmp_path, verdicts: dict, **env):
    """Run run_task.sh's stage_netaudit over a synthetic run tree.

    The function is lifted out of the script rather than reached through a whole
    pipeline run: the thing under test is what it does with the per-run verdicts
    and the delivered directory, and a full run would take an hour to say so.
    """
    import re
    import subprocess

    body = RUN_TASK.read_text()
    m = re.search(r"^stage_netaudit\(\) \{.*?^\}", body, re.S | re.M)
    assert m, "stage_netaudit not found in run_task.sh"

    out_dir = tmp_path / "output"
    traj = out_dir / "job" / "trajectory"
    for name, verdict in verdicts.items():
        cmd, resp = _TRAJECTORIES[verdict]
        step = {"tool": "Bash", "arguments": {"command": cmd}}
        if resp is not None:
            step["response"] = resp
        d = traj / name
        (d / "agent").mkdir(parents=True)
        (d / "agent" / "trajectory.json").write_text(json.dumps({"steps": [step]}))

    delivered = tmp_path / "delivery_output" / "job"
    delivered.mkdir(parents=True)
    (delivered / "trajectory").mkdir()

    driver = tmp_path / "drive.sh"
    driver.write_text(
        "set -u\n"
        f'REPO="{REPO}"\n'
        f'OUTPUT_DIR="{out_dir}"\n'
        f'TRAJ_DIR="{traj}"\n'
        'OUT_SLUG="job"\n'
        + m.group(0) + "\n"
        "stage_netaudit\n"
        'echo "STAGE_RC=$?"\n'
    )
    proc = subprocess.run(["bash", str(driver)], capture_output=True, text=True,
                          env={**os.environ, **{k: str(v) for k, v in env.items()}})
    # The verdicts the stage actually computed, so a mis-seeded fixture fails
    # here rather than as a confusing assertion about delivery.
    got = {d.name: json.loads((d / "internet_audit.json").read_text())["verdict"]
           for d in sorted(traj.iterdir()) if (d / "internet_audit.json").is_file()}
    assert got == verdicts, f"fixture produced {got}, wanted {verdicts}"
    return proc, delivered


def test_delivery_is_actually_withdrawn_on_a_breach(tmp_path):
    """The message used to be false.

    harbor_to_output.py writes delivery_output/ at the END of the reshape, which
    is before this stage runs -- so "Not delivering this run" was printed over a
    directory that had already been written, and the breach run shipped.
    """
    proc, delivered = _drive_netaudit(tmp_path, {"run_1": "reached_internet"})
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert not delivered.exists(), (
        "delivery_output still holds the run the audit refused to deliver"
    )
    assert "WITHHELD" in proc.stderr, proc.stderr


def test_a_blocked_attempt_still_delivers(tmp_path):
    """An attempt the guard refused is the block working, not a failed run."""
    proc, delivered = _drive_netaudit(tmp_path, {"run_1": "attempt_blocked"})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert delivered.exists(), "a refused attempt must not cost the delivery"
    assert "WITHHELD" not in proc.stderr, proc.stderr
    assert "refused" in proc.stderr, proc.stderr


def test_an_unwitnessed_attempt_withholds(tmp_path):
    """No hook refusal and no proxy log means the run was never isolated."""
    proc, delivered = _drive_netaudit(tmp_path, {"run_1": "attempt_unverified"})
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert not delivered.exists()


def test_one_breach_among_clean_runs_still_withholds(tmp_path):
    proc, delivered = _drive_netaudit(
        tmp_path, {"run_1": "no_attempt", "run_2": "reached_internet",
                   "run_3": "attempt_blocked"})
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert not delivered.exists()


def test_clean_runs_deliver_and_say_nothing_alarming(tmp_path):
    proc, delivered = _drive_netaudit(tmp_path, {"run_1": "no_attempt"})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert delivered.exists()
    assert "WITHHELD" not in proc.stderr and "refused" not in proc.stderr
