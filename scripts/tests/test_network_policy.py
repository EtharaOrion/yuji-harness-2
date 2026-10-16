"""Network policy: the task declares it, Harbor enforces it, the harness adds hosts.

Each bundle's task.toml declares network_mode for [environment], [agent] and
[verifier], and Harbor's docker egress-control sidecar enforces each phase,
refusing a trial whose policy it cannot enforce. The harness keeps no proxy of
its own. What is left for it, and what these tests pin:

  1. DISPATCH -- the model route this run adds on the host (a bridge, or the
     headroom container) reaches Harbor as --allow-agent-host, and nothing else
     about the network does: no compose overlay of the harness's touches egress.
  2. AGENT-SIDE LAYERS -- --disallowedTools and the egress-guard hook follow the
     task's resolved agent-phase policy, read through Harbor's own resolver.
  3. PREFLIGHT -- tools/network/preflight_network.py answers the same questions
     the runner will, before anything is built.

Driven against a stub `harbor` on PATH that records its argv and environment.
The stub's bin/ also carries a `python` that is the real harbor's interpreter,
so run_task.sh resolves each task's policy through Harbor itself.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import yaml

from conftest import mirror_harbor_package

REPO = Path(__file__).resolve().parents[2]
RUN_TASK = REPO / "scripts" / "run_task.sh"
PREFLIGHT = REPO / "tools" / "network" / "preflight_network.py"
HEADROOM_OVERLAY = REPO / "tools" / "headroom" / "overlay.yaml"
OPENHANDS_DIR = REPO / "tools" / "openhands_agent"


def _real_harbor_python() -> Path | None:
    harbor = shutil.which("harbor")
    if not harbor:
        return None
    py = Path(os.path.realpath(harbor)).parent / "python"
    return py if py.exists() else None


HARBOR_PY = _real_harbor_python()
requires_harbor_python = pytest.mark.skipif(
    HARBOR_PY is None, reason="needs an installed harbor to resolve task policy")

CLOSED_TASK = """\
[task]
name = "acme/alpha"

[environment]
network_mode = "public"

[agent]
network_mode = "allowlist"
allowed_hosts = ["api.anthropic.com"]

[verifier]
network_mode = "public"
"""

PUBLIC_TASK = """\
[task]
name = "acme/alpha"

[environment]
network_mode = "public"

[agent]
network_mode = "public"
"""

_HARBOR_STUB = """#!/usr/bin/env bash
printf '%s\\n' "$@" >> "$HARBOR_ARGS"
env >> "$HARBOR_ENV"
mkdir -p "$JOB_DIR"
touch "$JOB_DIR/result.json"
exit 0
"""


@dataclass
class HarborRun:
    returncode: int
    stdout: str
    stderr: str
    argv: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)

    @property
    def invoked(self) -> bool:
        return bool(self.argv)

    def values(self, flag: str) -> list[str]:
        return [self.argv[i + 1] for i, a in enumerate(self.argv) if a == flag]


def _run_harbor_stage(tmp_path: Path, task_toml: str = CLOSED_TASK, **overrides) -> HarborRun:
    """Run `run_task.sh --stage harbor` against a stub harbor. None unsets a var."""
    if HARBOR_PY is None or not mirror_harbor_package(tmp_path):
        pytest.skip("harbor is not installed; cannot mirror its package")
    task = tmp_path / "tasks" / "alpha"
    task.mkdir(parents=True)
    (task / "task.toml").write_text(task_toml)

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = bin_dir / "harbor"
    stub.write_text(_HARBOR_STUB)
    stub.chmod(0o755)
    # A wrapper, not a symlink: a venv's python finds its pyvenv.cfg beside the
    # path it was invoked by, so a link from here would lose harbor.
    shim = bin_dir / "python"
    shim.write_text(f'#!/bin/sh\nexec "{HARBOR_PY}" "$@"\n')
    shim.chmod(0o755)

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
        "AGENT_HEADROOM_ENABLED": "false",
    })
    for key in ("NETWORK_ISOLATION_OFF", "GRADER_HEADROOM_ENABLED", "CC_MODE",
                "ANTHROPIC_BASE_URL", "DISALLOWED_TOOLS"):
        env.pop(key, None)
    for key, value in overrides.items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = str(value)

    proc = subprocess.run([str(RUN_TASK), "--stage", "harbor", str(task)],
                          capture_output=True, text=True, env=env, cwd=str(REPO), timeout=240)
    run = HarborRun(returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)
    if args_file.exists():
        run.argv = [a for a in args_file.read_text().split("\n") if a != ""]
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            key, sep, value = line.partition("=")
            if sep:
                run.env.setdefault(key, value)
    return run


@pytest.fixture
def fake_zbridge():
    """A /healthz on a free port, so ensure_zbridge does not spawn a real one."""
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
        yield {"ZB_PORT": str(srv.server_address[1]), "ZB_ZAI_API_KEY": "test-key"}
    finally:
        srv.shutdown()
        srv.server_close()


# =============================================================================
# 1. DISPATCH
# =============================================================================

@requires_harbor_python
def test_no_harness_overlay_touches_the_network(tmp_path):
    run = _run_harbor_stage(tmp_path)
    assert run.invoked, run.stderr[-2000:]
    assert run.values("--extra-docker-compose") == [], run.argv
    assert run.values("--allow-agent-host") == [], (
        "claude-code calls the API directly; the task's own allowlist names it")
    assert not any("egress-proxy" in a or "squid" in a for a in run.argv), run.argv


@requires_harbor_python
def test_a_glm_run_allows_zbridge_on_the_host(tmp_path, fake_zbridge):
    run = _run_harbor_stage(tmp_path, CC_MODE="zbridge", **fake_zbridge)
    assert run.invoked, run.stderr[-2000:]
    assert run.values("--allow-agent-host") == ["host.docker.internal"], run.argv
    assert run.env.get("ANTHROPIC_BASE_URL") == f"http://host.docker.internal:{fake_zbridge['ZB_PORT']}"


@requires_harbor_python
def test_headroom_is_added_by_its_own_overlay_and_allowed_by_name(tmp_path):
    if subprocess.run(["docker", "image", "inspect", "headroom-compress:latest"],
                      capture_output=True).returncode != 0 and shutil.which("docker"):
        pytest.skip("headroom-compress:latest is not built (make build-headroom-compress)")
    run = _run_harbor_stage(tmp_path, AGENT_HEADROOM_ENABLED="true")
    assert run.invoked, run.stderr[-2000:]
    assert run.values("--extra-docker-compose") == [str(HEADROOM_OVERLAY)], run.argv
    assert run.values("--allow-agent-host") == ["headroom"], run.argv
    assert run.env.get("ANTHROPIC_BASE_URL") == "http://headroom:8787"
    assert run.env.get("HEADROOM_UPSTREAM") == "https://api.anthropic.com"


@requires_harbor_python
def test_no_headroom_unless_the_flag_is_set(tmp_path):
    run = _run_harbor_stage(tmp_path)
    assert run.invoked, run.stderr[-2000:]
    assert str(HEADROOM_OVERLAY) not in run.argv
    assert not run.env.get("ANTHROPIC_BASE_URL"), (
        "a base URL was exported without the flag; harbor pins every model alias "
        "to $MODEL whenever it is set")


@pytest.mark.parametrize("base_url", [
    "http://host.docker.internal:4000", "http://127.0.0.1:4000", "http://localhost:8787",
], ids=["host-gateway", "loopback-ip", "loopback-name"])
@requires_harbor_python
def test_an_inherited_host_base_url_never_reaches_harbor(tmp_path, base_url):
    """Running from inside a Claude Code session inherits that session's own
    proxy address; harbor would forward it into main verbatim."""
    run = _run_harbor_stage(tmp_path, ANTHROPIC_BASE_URL=base_url)
    assert run.invoked, run.stderr[-2000:]
    assert not run.env.get("ANTHROPIC_BASE_URL"), run.env.get("ANTHROPIC_BASE_URL")


@requires_harbor_python
def test_the_scrub_does_not_take_the_credential_with_it(tmp_path):
    run = _run_harbor_stage(tmp_path, CLAUDE_CODE_OAUTH_TOKEN="<stub-token>")
    assert run.invoked, run.stderr[-2000:]
    assert run.env.get("CLAUDE_CODE_OAUTH_TOKEN") == "<stub-token>"


# =============================================================================
# 2. AGENT-SIDE LAYERS follow the task's agent-phase policy
# =============================================================================

@requires_harbor_python
def test_a_closed_agent_phase_withholds_the_web_tools(tmp_path):
    run = _run_harbor_stage(tmp_path, task_toml=CLOSED_TASK)
    assert run.invoked, run.stderr[-2000:]
    assert "disallowed_tools=WebSearch,WebFetch" in run.values("--ak"), run.argv


@requires_harbor_python
def test_a_public_agent_phase_keeps_them(tmp_path):
    """A task that declares an open agent phase gets an open run."""
    run = _run_harbor_stage(tmp_path, task_toml=PUBLIC_TASK)
    assert run.invoked, run.stderr[-2000:]
    assert not any(a.startswith(("disallowed_tools=", "config=")) for a in run.values("--ak")), run.argv


@requires_harbor_python
def test_disallowed_tools_can_be_emptied(tmp_path):
    run = _run_harbor_stage(tmp_path, DISALLOWED_TOOLS="")
    assert run.invoked, run.stderr[-2000:]
    assert not any(a.startswith("disallowed_tools=") for a in run.values("--ak")), run.argv


# =============================================================================
# 3. PREFLIGHT
# =============================================================================

def _preflight(task: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([str(HARBOR_PY), str(PREFLIGHT), str(task), *args],
                          capture_output=True, text=True, timeout=300)


def _task(tmp_path: Path, toml: str) -> Path:
    task = tmp_path / "task"
    task.mkdir()
    (task / "task.toml").write_text(toml)
    return task


@requires_harbor_python
@pytest.mark.parametrize("toml,mode", [(CLOSED_TASK, "allowlist"), (PUBLIC_TASK, "public")])
def test_agent_mode_is_harbors_resolution(tmp_path, toml, mode):
    out = _preflight(_task(tmp_path, toml), "--agent-mode")
    assert out.returncode == 0 and out.stdout.strip() == mode, out.stdout + out.stderr


@requires_harbor_python
def test_an_allowed_agent_host_closes_nothing_that_was_open(tmp_path):
    """Harbor ignores run-specific hosts against a public policy."""
    out = _preflight(_task(tmp_path, PUBLIC_TASK), "--agent-mode",
                     "--allow-agent-host", "host.docker.internal")
    assert out.stdout.strip() == "public", out.stdout + out.stderr


@requires_harbor_python
@pytest.mark.parametrize("toml,state", [(CLOSED_TASK, "on"), (PUBLIC_TASK, "off")])
def test_egress_control_matches_harbors_rule(tmp_path, toml, state):
    out = _preflight(_task(tmp_path, toml), "--egress-control")
    assert out.stdout.strip() == state, out.stdout + out.stderr


@requires_harbor_python
def test_an_undeclared_phase_is_reported_not_filled_in(tmp_path):
    out = _preflight(_task(tmp_path, '[task]\nname = "acme/alpha"\n'))
    assert "[agent].network_mode is not declared" in out.stdout, out.stdout
    assert "public\" (matches" not in out.stdout, "the harness substituted a policy of its own"


@requires_harbor_python
def test_allowed_hosts_beside_a_public_mode_is_refused(tmp_path):
    out = _preflight(_task(tmp_path, PUBLIC_TASK.replace(
        '[agent]\nnetwork_mode = "public"\n',
        '[agent]\nnetwork_mode = "public"\nallowed_hosts = ["x.example"]\n')))
    assert out.returncode == 2, out.stdout


# =============================================================================
# 4. The overlays' promises, read from the files
# =============================================================================

def test_the_headroom_overlay_declares_no_networking():
    """Harbor routes only services without network_mode/networks through its
    egress sidecar; declaring either would take the proxy out of the policy."""
    cfg = yaml.safe_load(HEADROOM_OVERLAY.read_text())
    for name, svc in cfg["services"].items():
        assert "network_mode" not in svc and "networks" not in svc, name
    assert "networks" not in cfg
    assert cfg["services"]["headroom"]["image"] == "headroom-compress:latest"


def test_the_headroom_overlay_leaves_claude_code_alone():
    """compress_proxy.py relays `tools` untouched, so the agent is configured
    exactly as it is on a run without compression."""
    cfg = yaml.safe_load(HEADROOM_OVERLAY.read_text())
    assert "environment" not in cfg["services"]["main"]


def test_no_harness_file_names_a_proxy_of_its_own():
    assert not (REPO / "tools" / "network" / "egress-proxy").exists()
    body = RUN_TASK.read_text()
    assert not re.search(r"--extra-docker-compose[^\n]*egress", body)


# --- agent-path Headroom memory gate -----------------------------------------

def _drive_memory_check(tmp_path: Path, mem_bytes: str | None, **env_over) -> subprocess.CompletedProcess:
    """Run headroom_memory_check() out of run_task.sh against a stub docker."""
    body = RUN_TASK.read_text()
    m = re.search(r"^headroom_memory_check\(\) \{.*?^\}", body, re.S | re.M)
    assert m, "headroom_memory_check() not found in run_task.sh"

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    docker = bin_dir / "docker"
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
    env.pop("HEADROOM_IGNORE_MEMORY", None)
    env.pop("HEADROOM_MIN_DOCKER_GB", None)
    env["AGENT_HEADROOM_ENABLED"] = "true"
    env.update({k: str(v) for k, v in env_over.items()})
    return subprocess.run(["bash", str(driver)], capture_output=True, text=True, env=env, timeout=60)


def test_a_docker_too_small_for_headroom_is_refused(tmp_path):
    out = _drive_memory_check(tmp_path, str(8 * 1024**3))
    assert out.returncode == 2, out.stdout + out.stderr
    assert "REFUSING" in out.stderr, out.stderr
    assert "REACHED_THE_RUN" not in out.stdout


def test_a_correctly_sized_vm_is_not_rounded_down_into_a_refusal(tmp_path):
    """What docker reports for a VM set to 12 GB, measured: 12,304,840 kB."""
    out = _drive_memory_check(tmp_path, str(12_304_840 * 1024))
    assert "REACHED_THE_RUN" in out.stdout, out.stdout + out.stderr


def test_enough_memory_runs(tmp_path):
    out = _drive_memory_check(tmp_path, str(16 * 1024**3))
    assert "REACHED_THE_RUN" in out.stdout, out.stdout + out.stderr


def test_the_check_is_only_about_headroom_runs(tmp_path):
    out = _drive_memory_check(tmp_path, str(4 * 1024**3), AGENT_HEADROOM_ENABLED="false")
    assert "REACHED_THE_RUN" in out.stdout, out.stderr


def test_an_unreadable_docker_does_not_block(tmp_path):
    out = _drive_memory_check(tmp_path, None)
    assert "REACHED_THE_RUN" in out.stdout, out.stderr


def test_the_memory_check_can_be_overridden(tmp_path):
    out = _drive_memory_check(tmp_path, str(4 * 1024**3), HEADROOM_IGNORE_MEMORY="1")
    assert "REACHED_THE_RUN" in out.stdout, out.stderr


def test_run_task_knows_how_to_build_the_compressor():
    body = RUN_TASK.read_text()
    assert re.search(r'headroom-compress\)\s*echo "\$REPO/tools/headroom"', body)
    assert re.search(r"headroom-compress\)\s*\n?\s*printf .*scoring=", body, re.S), (
        "the compressor image COPYs grader_compress.py --from=scoring")
    assert "build-headroom-compress:" in (REPO / "Makefile").read_text()
