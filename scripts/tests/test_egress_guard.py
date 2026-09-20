"""The PreToolUse egress guard -- the layer the model can actually read.

Three properties, and the third is the one that makes the other two worth
having:

  1. the hook refuses egress and lets local work through;
  2. it fails OPEN, because the routing table is the enforcement boundary and a
     bug here must not be able to kill every Bash call in a run;
  3. it and tools/network/detect_internet_use.py agree, on every command, always
     -- they import the same rules, and a test says so out loud, because a
     command the hook allows and the audit later blocks costs a whole graded run.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
RULES = REPO / "tools" / "network" / "egress_rules.py"
MAKE_SETTINGS = REPO / "tools" / "network" / "make_guard_settings.py"
DETECT = REPO / "tools" / "network" / "detect_internet_use.py"


def hook(tool_name: str, tool_input: dict) -> subprocess.CompletedProcess:
    """Run the guard exactly as Claude Code does: payload on stdin."""
    return subprocess.run(
        [sys.executable, str(RULES)],
        input=json.dumps({"tool_name": tool_name, "tool_input": tool_input}),
        capture_output=True, text=True,
    )


def hook_bash(cmd: str) -> subprocess.CompletedProcess:
    return hook("Bash", {"command": cmd})


# --- what it refuses --------------------------------------------------------

@pytest.mark.parametrize("cmd", [
    "pip install pandas",
    "timeout 600 npm i puppeteer@23 --no-audit --no-fund 2>&1 | tail -5",
    "sudo apt-get install -y chromium",
    "dpkg --print-architecture; (apt-get install -y -q chromium | tail -3)",
    "curl -s https://api.github.com/repos/x",
    "wget https://example.com/data.csv",
    "git clone https://github.com/a/b",
    "python3 -c 'import urllib.request; urllib.request.urlopen(\"http://x\")'",
])
def test_egress_is_refused(cmd):
    r = hook_bash(cmd)
    assert r.returncode == 2, f"{cmd!r} was allowed:\n{r.stderr}"


@pytest.mark.parametrize("tool,payload", [
    ("WebFetch", {"url": "https://example.com/x"}),
    ("WebSearch", {"query": "norman general fund fye26"}),
])
def test_web_tools_are_refused(tool, payload):
    assert hook(tool, payload).returncode == 2


def test_the_refusal_says_what_to_use_instead():
    """The whole point. A silent block costs the same turns a timeout does."""
    r = hook_bash("pip install pandas")
    assert "BLOCKED" in r.stderr
    assert "MCP tools" in r.stderr
    assert "/workspace/data" in r.stderr
    # It must also say retrying is pointless, or the model tries the next
    # package manager -- one recorded run spent seven Bash calls doing exactly
    # that (npm, apt-get, chromium, puppeteer) before giving up.
    assert "will not work" in r.stderr


# --- what it must NOT refuse ------------------------------------------------
#
# A false positive here breaks a run that did nothing wrong, so these are as
# load-bearing as the cases above.

@pytest.mark.parametrize("cmd", [
    "curl -s http://light-servers:9142/mcp",          # sidecar health probe
    "curl -s http://localhost:8000/health",
    "curl -s http://127.0.0.1:9142/mcp",
    "pip install --no-index ./wheels/x.whl",          # pinned to disk
    "git status && git log --oneline -5",
    "grep -rn 'pip install' /workspace",
    "timeout 30 python3 /tmp/build.py",
    "(cd /workspace && python3 build.py)",
    "ls -la /workspace/data && cat /workspace/data/x.csv",
    "echo '<svg xmlns=\"http://www.w3.org/2000/svg\"/>' > /tmp/a.svg",
    "python3 - <<'PY'\nimport json\nprint(json.dumps({'a': 1}))\nPY",
])
def test_local_work_is_allowed(cmd):
    r = hook_bash(cmd)
    assert r.returncode == 0, f"{cmd!r} was refused:\n{r.stderr}"


def test_mcp_tools_are_never_egress():
    """The closed world is served over MCP; it is the answer, not the problem."""
    assert hook("mcp__LightGmail__list_messages", {"limit": 10}).returncode == 0
    assert hook("mcp__LightBudget__update_transaction", {"id": 1}).returncode == 0


# --- failure modes ----------------------------------------------------------

@pytest.mark.parametrize("payload", [
    "not json at all",
    "",
    "{}",
    '{"tool_name": "Bash"}',                 # no tool_input
    '{"tool_input": {"command": "ls"}}',     # no tool_name
])
def test_the_guard_fails_open(payload):
    """The router is the enforcement boundary; this layer buys turns.

    A hook that exits non-zero on a payload it did not expect would block every
    Bash call in the run, which is a far more expensive way to be wrong than
    letting one command through to a network that has no gateway anyway.
    """
    r = subprocess.run([sys.executable, str(RULES)], input=payload,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


# --- the generated settings -------------------------------------------------

@pytest.fixture(scope="module")
def settings(tmp_path_factory) -> dict:
    out = tmp_path_factory.mktemp("guard") / "claude-settings.json"
    r = subprocess.run([sys.executable, str(MAKE_SETTINGS), str(RULES), str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return json.loads(out.read_text())


def test_settings_declare_a_pretooluse_hook(settings):
    entries = settings["hooks"]["PreToolUse"]
    assert len(entries) == 1
    matcher = entries[0]["matcher"]
    for tool in ("Bash", "WebFetch", "WebSearch"):
        assert tool in matcher, matcher


def test_the_hook_command_carries_the_rules_and_runs(settings, tmp_path):
    """End-to-end: the command string as harbor ships it, run under /bin/sh.

    This is the test that would have caught the first draft, which piped the
    decoded script into `python3 -` and so consumed the hook's own stdin.
    """
    command = settings["hooks"]["PreToolUse"][0]["hooks"][0]["command"]

    denied = subprocess.run(
        ["/bin/sh", "-c", command],
        input=json.dumps({"tool_name": "Bash",
                          "tool_input": {"command": "pip install pandas"}}),
        capture_output=True, text=True,
    )
    assert denied.returncode == 2, denied.stderr
    assert "BLOCKED" in denied.stderr

    allowed = subprocess.run(
        ["/bin/sh", "-c", command],
        input=json.dumps({"tool_name": "Bash",
                          "tool_input": {"command": "ls /workspace/data"}}),
        capture_output=True, text=True,
    )
    assert allowed.returncode == 0, allowed.stderr


def test_the_settings_are_regenerated_from_the_live_rules(settings):
    """No cached copy: the shipped bytes must be today's rules.

    A stale settings.json would enforce whatever the rules were the last time
    somebody looked, which is the drift this whole arrangement exists to stop.
    """
    import base64
    command = settings["hooks"]["PreToolUse"][0]["hooks"][0]["command"]
    blob = command.split("'")[1]
    assert base64.b64decode(blob) == RULES.read_bytes()


# --- the parity that makes one rule set worth having ------------------------

PARITY_CASES = [
    "pip install pandas",
    "timeout 600 npm i puppeteer@23 | tail -5",
    "sudo -u root apt-get install -y chromium",
    "curl -s https://api.github.com/x",
    "git clone https://github.com/a/b",
    "(cd /tmp && pip install foo)",
    "curl -s http://light-servers:9142/mcp",
    "pip install --no-index ./wheels/x.whl",
    "grep -rn 'pip install' /workspace",
    "timeout 30 python3 /tmp/build.py",
    "ls /workspace/data",
]


@pytest.mark.parametrize("cmd", PARITY_CASES)
def test_hook_and_audit_agree(cmd, tmp_path):
    """The two callers of egress_rules must never disagree about a command.

    Disagreement in one direction is expensive and silent: the hook lets a
    command run, the model builds on it, and the audit discards the finished run
    hours later. This test is what keeps the two from drifting even though they
    execute in different processes, on different machines, at different times.
    """
    traj = tmp_path / "t.json"
    traj.write_text(json.dumps(
        {"steps": [{"tool": "Bash", "arguments": {"command": cmd}}]}))
    # --strict, because the question here is "did both recognise this as
    # egress?", not "did both withhold delivery?". By default the audit only
    # fails a run that actually REACHED the internet, so without the flag every
    # blocked-attempt case would read as a disagreement it is not.
    audit = subprocess.run([sys.executable, str(DETECT), str(traj), "--strict"],
                           capture_output=True, text=True)

    hook_blocked = hook_bash(cmd).returncode == 2
    audit_blocked = audit.returncode == 2
    assert hook_blocked == audit_blocked, (
        f"{cmd!r}: hook {'blocked' if hook_blocked else 'allowed'} but audit "
        f"{'blocked' if audit_blocked else 'allowed'}"
    )


# --- MCP over raw HTTP is the closed world, not egress -----------------------
#
# An agent that drives the sidecars over HTTP instead of through the tool list
# is doing the task, not escaping it. The hint list is a substring match and
# used to fire on the words alone: one delivered run audited as seven internet
# attempts, all seven of them urllib pointed at http://light-servers:9015/mcp.

@pytest.mark.parametrize("cmd", [
    "python3 -c \"import urllib.request; urllib.request.urlopen('http://light-servers:9015/mcp')\"",
    "python3 - <<'PY'\nimport urllib.request\nurllib.request.urlopen('http://light-servers:9015/mcp')\nPY",
    "python3 -c \"import requests; requests.post('http://127.0.0.1:9020/mcp', json={})\"",
])
def test_an_interpreter_talking_to_a_sidecar_is_allowed(cmd):
    r = hook_bash(cmd)
    assert r.returncode == 0, f"{cmd!r} was refused:\n{r.stderr}"


@pytest.mark.parametrize("cmd", [
    # one external host among the internal ones is still egress
    "python3 -c \"import urllib.request; urllib.request.urlopen('http://light-servers:9015/mcp'); urllib.request.urlopen('https://pypi.org/x')\"",
    # no host visible at all: the target could be built at runtime, so the
    # hint stands. This is the fail-closed branch and it must stay closed.
    "python3 -c 'import urllib.request as u; u.urlopen(target)'",
])
def test_an_interpreter_with_any_reachable_target_is_still_refused(cmd):
    assert hook_bash(cmd).returncode == 2, f"{cmd!r} was allowed"


# --- searched-for text is not a payload --------------------------------------
#
# One bundle asks for a page that reaches out for nothing to draw itself, so the
# model audits its own HTML for exactly the words the hint list carries. The
# guard answered that with "there is no route out of this container" -- true,
# unrelated, and it costs the check.

@pytest.mark.parametrize("cmd", [
    "grep -Eo 'fetch\\(|XMLHttpRequest' page.html",
    "grep -rn 'urlopen' /workspace",
    "(grep -Eo 'fetch\\(|XMLHttpRequest' page.html)",
    "echo '== external refs =='\ngrep -nEio 'https?://|src *=|fetch\\(|XMLHttpRequest' $F | grep -v example",
])
def test_grepping_for_the_hint_words_is_allowed(cmd):
    r = hook_bash(cmd)
    assert r.returncode == 0, f"{cmd!r} was refused:\n{r.stderr}"


@pytest.mark.parametrize("cmd", [
    # A separator glued to the word before it must still end the grep, or the
    # command after it is collected as one of grep's operands and its hints are
    # lifted out of the scan along with the pattern.
    "grep -Eo 'fetch\\(' a.html; python3 -c 'import urllib.request as u; u.urlopen(t)'",
    "grep -Eo 'fetch\\(' a.html && python3 -c 'import urllib.request as u; u.urlopen(t)'",
    "grep -Eo 'fetch\\(' a.html | python3 -c 'import urllib.request as u; u.urlopen(t)'",
    "grep -Eo 'XMLHttpRequest' a.html && curl https://evil.test/x",
    # awk and sed take PROGRAMS, not patterns, and an awk program can shell out.
    "awk '/XMLHttpRequest/{print}' a.html",
    "sed -n '/urlopen/p' a.py",
])
def test_a_search_does_not_launder_what_follows_it(cmd):
    assert hook_bash(cmd).returncode == 2, f"{cmd!r} was allowed"


# --- harbor has to actually deliver the guard --------------------------------

def _fake_harbor(root: Path, *, config_source: bool, native_config: bool,
                 options_config: bool) -> Path:
    """A harbor package shaped like the version under test, and nothing more."""
    pkg = root / "harbor" / "agents" / "installed"
    pkg.mkdir(parents=True)
    for d in (root / "harbor", root / "harbor" / "agents", pkg):
        (d / "__init__.py").write_text("")
    body = ["class _Caps:", f"    native_config = {native_config!r}", "",
            "class _Field: pass", "", "class _Options:",
            "    model_fields = {%s}" % ("'config': _Field()" if options_config else ""),
            "", "class ClaudeCode:", "    capabilities = _Caps()",
            "    options_model = _Options"]
    if config_source:
        body += ["    @property", "    def config_source(self): return None"]
    (pkg / "claude_code.py").write_text("\n".join(body) + "\n")
    return root


def _probe_with(path: Path) -> tuple[str, str]:
    """Run --probe-here under an interpreter that sees only this fake harbor."""
    import os
    env = dict(os.environ, PYTHONPATH=str(path))
    r = subprocess.run([sys.executable, str(MAKE_SETTINGS), "--probe-here"],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    verdict, _, detail = r.stdout.strip().split("\n")[-1].partition("\t")
    return verdict, detail


def test_a_harbor_that_delivers_reads_as_delivering(tmp_path):
    p = _fake_harbor(tmp_path / "new", config_source=True, native_config=True,
                     options_config=True)
    assert _probe_with(p)[0] == "delivers"


def test_a_harbor_without_the_mechanism_reads_as_absent(tmp_path):
    """harbor 0.20's shape: no capabilities, no options model, no config_source.

    `--ak config=` falls through to **kwargs there and is dropped in silence,
    which is how a delivered EC2 run came back with no --settings on the claude
    command line and no PreToolUse event in the whole session.
    """
    p = _fake_harbor(tmp_path / "old", config_source=False, native_config=False,
                     options_config=False)
    assert _probe_with(p)[0] == "absent"


def test_a_harbor_that_half_matches_is_reported_as_unreadable(tmp_path):
    """Drift this cannot read must not be ruled on in either direction."""
    p = _fake_harbor(tmp_path / "part", config_source=True, native_config=False,
                     options_config=False)
    verdict, detail = _probe_with(p)
    assert verdict == "unreadable"
    assert "config_source" in detail


def test_the_probe_answers_for_the_harbor_actually_installed():
    """Whatever this machine has, the probe must return one of the three words.

    Not an assertion about which: a laptop with no harbor is a legal state and
    the honest answer there is "cannot tell". What is being pinned is that the
    delegation into harbor's own pipx interpreter runs to an answer rather than
    raising, since that path is the one every real run takes.
    """
    sys.path.insert(0, str(MAKE_SETTINGS.parent))
    try:
        import make_guard_settings as m
    finally:
        sys.path.pop(0)
    verdict, detail = m.harbor_native_config_support()
    assert verdict in (m.DELIVERS, m.ABSENT, m.UNREADABLE)
    assert detail
