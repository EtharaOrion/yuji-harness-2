"""tools/network/detect_internet_use.py -- the closed-world guarantee.

Two things must hold, and the second is the one that costs real money if it
breaks: every form of egress is caught, and NO legitimate bundle traffic is
flagged. A false positive here blocks a run that did nothing wrong, so the
sidecar/offline cases below are as load-bearing as the detection cases.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent.parent
DETECT = REPO / "tools" / "network" / "detect_internet_use.py"


def run(steps, *flags, tmp_path, strict=True):
    """Audit a synthetic trajectory.

    `strict` by default, because most tests here ask "is this recognised as
    egress?" and --strict is the flag that turns that question back into an exit
    code. Without it only a run that actually REACHED the internet exits 2 --
    an attempt the guard refused is a warning that still delivers. Tests for
    that policy pass strict=False and assert on the verdict.
    """
    traj = tmp_path / "t.json"
    traj.write_text(json.dumps({"steps": steps}))
    argv = [sys.executable, str(DETECT), str(traj), *flags]
    if strict:
        argv.append("--strict")
    return subprocess.run(argv, capture_output=True, text=True)


def bash(cmd):
    return {"tool": "Bash", "arguments": {"command": cmd}}


# --- caught -----------------------------------------------------------------

@pytest.mark.parametrize("step", [
    {"tool": "WebSearch", "arguments": {"query": "cherry side table price"}},
    {"tool": "WebFetch", "arguments": {"url": "https://example.com/x"}},
    bash("curl -s https://api.github.com/repos/x"),
    bash("wget https://example.com/data.csv"),
    bash("pip install pandas"),
    bash("npm install left-pad"),
    bash("apt-get install -y jq"),
    bash("git clone https://github.com/a/b"),
    bash("python3 -c \"import urllib.request; urllib.request.urlopen('http://x.com')\""),
    bash("ssh user@example.com ls"),
    bash("ls /workspace && curl https://evil.test/x"),
])
def test_egress_is_blocked(step, tmp_path):
    r = run([step], tmp_path=tmp_path)
    assert r.returncode == 2, r.stdout


# --- not caught -------------------------------------------------------------

@pytest.mark.parametrize("step", [
    # The sidecars are the point of the bundle, not egress.
    bash("curl -s http://light-servers:9142/mcp"),
    bash("curl -s http://localhost:9110/mcp"),
    bash("curl -s http://127.0.0.1:9067/mcp"),
    # Ordinary local work, including the docx/zip unpacking these tasks do.
    bash("ls -la /workspace/data"),
    bash("unzip -o -q /workspace/data/policy.docx -d /tmp/x"),
    bash("git status"),
    bash("git log --oneline -5"),
    bash("pip install --no-index ./wheels/x.whl"),
    bash("python3 -c \"import json; print(json.load(open('/tmp/a.json')))\""),
    {"tool": "mcp__LightEtsy__update_listing", "arguments": {"listing_id": 1020}},
    {"tool": "Read", "arguments": {"file_path": "/workspace/data/img_20.jpg"}},
])
def test_local_work_is_not_flagged(step, tmp_path):
    r = run([step], tmp_path=tmp_path)
    assert r.returncode == 0, r.stdout


def test_clean_run_passes(tmp_path):
    steps = [bash("ls /workspace/data"),
             {"tool": "mcp__LightGmail__list_messages", "arguments": {}}]
    r = run(steps, tmp_path=tmp_path)
    assert r.returncode == 0
    assert "never reached for the internet" in r.stdout


# --- shapes, flags, edges ---------------------------------------------------

def test_raw_harbor_trajectory_shape_is_understood(tmp_path):
    """Harbor publishes tool_calls[].function_name; tests/test.sh writes tool/arguments.

    Both must audit identically or the check silently covers only one of the two
    places a trajectory is read from.
    """
    steps = [{"tool_calls": [
        {"function_name": "Bash", "arguments": {"command": "curl https://evil.test"}}]}]
    r = run(steps, tmp_path=tmp_path)
    assert r.returncode == 2, r.stdout


def test_warn_only_reports_without_blocking(tmp_path):
    r = run([bash("curl https://evil.test")], "--warn-only", tmp_path=tmp_path)
    assert r.returncode == 0
    assert "FAIL" in r.stdout


def test_findings_are_written_for_grading(tmp_path):
    out = tmp_path / "audit.json"
    run([bash("curl https://evil.test")], "--json", str(out), tmp_path=tmp_path)
    data = json.loads(out.read_text())
    assert data["attempted_internet"] is True
    assert data["findings"] and data["findings"][0]["kind"] == "fetch"


def test_clean_run_records_a_negative_result(tmp_path):
    out = tmp_path / "audit.json"
    run([bash("ls /workspace")], "--json", str(out), tmp_path=tmp_path)
    assert json.loads(out.read_text())["attempted_internet"] is False


def test_one_command_yields_one_finding(tmp_path):
    """`curl <url>` trips both the URL sweep and the token walk; the report
    should name the act once, not name every rule that matched it."""
    out = tmp_path / "audit.json"
    run([bash("curl https://evil.test/x")], "--json", str(out), tmp_path=tmp_path)
    kinds = [f["kind"] for f in json.loads(out.read_text())["findings"]]
    assert kinds == ["fetch"]


def test_unlexable_command_is_reported_not_skipped(tmp_path):
    """An unbalanced quote must not become a silent pass -- that is the one
    hole worth having none of."""
    r = run([bash('curl "https://evil.test')], tmp_path=tmp_path)
    assert r.returncode == 2, r.stdout


HEREDOC_EDIT = """python3 - << 'PYEOF'
s = open('/tmp/page.py', encoding='utf-8').read()
s = s.replace(\"\"\" --neutral:#383835;\"\"\", \"\"\" --neutral:#898781;\"\"\")
# tooltips: the value leads, the label follows
open('/tmp/page.py', 'w', encoding='utf-8').write(s)
PYEOF
python3 /tmp/page.py"""


def test_a_heredoc_of_local_python_is_not_a_finding(tmp_path):
    """shlex has no heredoc rule, so it lexed the body as shell words and the
    first apostrophe in it raised "No closing quotation" -- and the fail-closed
    branch turned that parser limit into a blocked run. `python3 - <<'PY'` is
    how the agent writes most of its multi-line edits; this one only touches
    /tmp."""
    r = run([bash(HEREDOC_EDIT)], tmp_path=tmp_path)
    assert r.returncode == 0, r.stdout


def test_lifting_the_body_out_does_not_blind_the_lexer(tmp_path):
    """Only the body is lifted. Commands after the terminator still get walked,
    or the heredoc becomes a place to hide the next line."""
    r = run([bash(HEREDOC_EDIT.replace("python3 /tmp/page.py",
                                       "curl https://evil.test/x"))],
            tmp_path=tmp_path)
    assert r.returncode == 2, r.stdout


@pytest.mark.parametrize("body", [
    "curl https://evil.test/x",             # also caught by the raw URL sweep
    "curl evil.test/x",                     # no scheme: only the verb walk sees it
    "pip install pandas",                   # no host at all: flags read it
    "git clone https://github.com/a/b",
    "import urllib.request as u; u.urlopen('x')",
])
def test_egress_inside_a_heredoc_is_still_caught(body, tmp_path):
    """A body lifted out for lexing is audited, not excused -- otherwise the
    heredoc becomes the place to keep an install where nothing looks."""
    r = run([bash(f"bash << 'EOF'\n{body}\nEOF")], tmp_path=tmp_path)
    assert r.returncode == 2, r.stdout


def test_every_line_of_a_heredoc_body_is_walked(tmp_path):
    """shlex does not treat a newline as a separator, so a body lexed whole
    collapses into one run-on segment and only its first word is ever read as a
    verb. The install on line two has to be found too."""
    r = run([bash("bash << 'EOF'\nls /workspace\npip install pandas\nEOF")],
            tmp_path=tmp_path)
    assert r.returncode == 2, r.stdout


def test_prose_written_through_a_heredoc_is_not_egress(tmp_path):
    """The walk reads verbs, not words. A note that mentions curl is not a run
    of curl -- and writing notes to /workspace is the job."""
    r = run([bash("cat > /workspace/NOTES.md << 'EOF'\n"
                  "The data was not fetched with curl -- it ships in the bundle.\n"
                  "EOF")], tmp_path=tmp_path)
    assert r.returncode == 0, r.stdout


def test_a_heredoc_fetch_is_named_once(tmp_path):
    """The body walk and the raw URL sweep both see the same act. _collapse
    keys on (step, evidence), so the body's findings must carry the whole
    command as evidence, exactly as the command-line walk does."""
    out = tmp_path / "audit.json"
    run([bash("bash << 'EOF'\ncurl https://evil.test/x\nEOF")],
        "--json", str(out), tmp_path=tmp_path)
    kinds = [f["kind"] for f in json.loads(out.read_text())["findings"]]
    assert kinds == ["fetch"], kinds


def test_an_unterminated_heredoc_is_still_audited(tmp_path):
    """A body that never meets its delimiter runs to the end of the command.
    Nothing may fall off that edge unread."""
    r = run([bash("bash << 'EOF'\npip install pandas")], tmp_path=tmp_path)
    assert r.returncode == 2, r.stdout


def test_missing_trajectory_does_not_block(tmp_path):
    """A run that produced nothing is the aborted-trial guard's business, not
    this one's; blocking here would double-report it under the wrong name."""
    r = subprocess.run(
        [sys.executable, str(DETECT), str(tmp_path / "nope.json")],
        capture_output=True, text=True)
    assert r.returncode == 0
    assert "warn" in r.stdout


# --- install attempt vs install that landed ---------------------------------
#
# The command line says the model reached for an index; the response says
# whether it got there. Both block, but only the second is a breach, and the
# response is the only witness that survives a run with no proxy log.

def bash_out(cmd, out):
    return {"tool": "Bash", "arguments": {"command": cmd}, "response": out}


PIP_DENIED = (
    "WARNING: Retrying (Retry(total=4)) after connection broken by "
    "'ProxyError('Cannot connect to proxy.', ...)': /simple/pandas/\n"
    "ERROR: Could not find a version that satisfies the requirement pandas"
)
PIP_LANDED = (
    "Collecting pandas\n  Downloading pandas-2.2.3-cp312-manylinux.whl (12 MB)\n"
    "Installing collected packages: pandas\n"
    "Successfully installed pandas-2.2.3"
)


@pytest.mark.parametrize("out", [
    PIP_LANDED,
    "added 1 package in 812ms",                                    # npm
    "Get:1 http://deb.debian.org/debian bookworm/main jq amd64 1.6\n"
    "Setting up jq (1.6-2.1) ...",                                 # apt
    "Installed 3 packages in 41ms\n + pandas==2.2.3",              # uv
    "(1/2) Installing jq (1.7.1-r0)",                              # apk
])
def test_successful_install_is_a_breach(out, tmp_path):
    audit = tmp_path / "audit.json"
    r = run([bash_out("pip install pandas", out)], "--json", str(audit),
            tmp_path=tmp_path)
    assert r.returncode == 2, r.stdout
    data = json.loads(audit.read_text())
    assert [f["kind"] for f in data["findings"]] == ["package-installed"]
    assert data["verdict"] == "reached_internet", data


def test_denied_install_is_an_attempt_not_a_breach(tmp_path):
    """The proxy refusing pip is the block WORKING. It still blocks the run --
    reaching for an index is disqualifying here -- but calling it a breach
    sends an operator hunting a leak that never happened."""
    audit = tmp_path / "audit.json"
    r = run([bash_out("pip install pandas", PIP_DENIED)], "--json", str(audit),
            tmp_path=tmp_path)
    assert r.returncode == 2, r.stdout
    data = json.loads(audit.read_text())
    assert [f["kind"] for f in data["findings"]] == ["package-install"]
    assert data["verdict"] == "attempt_unverified", data


@pytest.mark.parametrize("out", [
    "Requirement already satisfied: pandas in /usr/lib/python3/dist-packages",
    "up to date, audited 1 package in 190ms",
    "jq is already the newest version (1.6-2.1).\n"
    "0 upgraded, 0 newly installed, 0 to remove and 0 not upgraded.",
])
def test_already_on_disk_is_not_a_breach(out, tmp_path):
    """A warm cache reaches no index. Reading these as success would turn the
    most ordinary install output there is into a false breach."""
    audit = tmp_path / "audit.json"
    run([bash_out("pip install pandas", out)], "--json", str(audit),
        tmp_path=tmp_path)
    data = json.loads(audit.read_text())
    assert [f["kind"] for f in data["findings"]] == ["package-install"]
    assert data["verdict"] != "reached_internet", data


def test_offline_install_stays_clean_whatever_it_prints(tmp_path):
    """`pip install --no-index ./x.whl` prints "Successfully installed" too. It
    is pinned to disk, so the success marker must not resurrect it."""
    r = run([bash_out("pip install --no-index ./wheels/x.whl",
                      "Successfully installed x-1.0.0")], tmp_path=tmp_path)
    assert r.returncode == 0, r.stdout


def test_missing_response_falls_back_to_the_attempt(tmp_path):
    """A truncated trajectory is not evidence the install failed, so absence
    must read as the weaker claim, not the stronger one."""
    audit = tmp_path / "audit.json"
    run([bash("pip install pandas")], "--json", str(audit), tmp_path=tmp_path)
    assert [f["kind"] for f in json.loads(audit.read_text())["findings"]] \
        == ["package-install"]


def test_harbor_shape_carries_the_response_too(tmp_path):
    """Harbor keeps the result in observation.results, joined to the call by id.
    Miss that and the stronger finding only ever fires on verifier-shape
    trajectories -- half the places this scanner runs."""
    steps = [{"tool_calls": [{"tool_call_id": "t1", "function_name": "Bash",
                              "arguments": {"command": "pip install pandas"}}],
              "observation": {"results": [{"source_call_id": "t1",
                                           "content": PIP_LANDED}]}}]
    audit = tmp_path / "audit.json"
    r = run(steps, "--json", str(audit), tmp_path=tmp_path)
    assert r.returncode == 2, r.stdout
    assert json.loads(audit.read_text())["verdict"] == "reached_internet"


def test_harbor_results_are_matched_by_id_not_position(tmp_path):
    """One step, two calls, results in the other order: a positional zip would
    hang the install output on the `ls` and clear the real one."""
    steps = [{"tool_calls": [
        {"tool_call_id": "t1", "function_name": "Bash",
         "arguments": {"command": "ls /workspace"}},
        {"tool_call_id": "t2", "function_name": "Bash",
         "arguments": {"command": "pip install pandas"}}],
        "observation": {"results": [{"source_call_id": "t2", "content": PIP_LANDED},
                                    {"source_call_id": "t1", "content": "data"}]}}]
    audit = tmp_path / "audit.json"
    run(steps, "--json", str(audit), tmp_path=tmp_path)
    assert json.loads(audit.read_text())["verdict"] == "reached_internet"


# --- namespace URIs are identifiers, not addresses --------------------------

@pytest.mark.parametrize("cmd", [
    # The exact command that blocked a clean run: the agent CHECKING its own
    # output had no external references.
    "python3 -c \"print('ok', 'http://' not in html.replace('http://www.w3.org/2000/svg',''))\"",
    'echo \'<svg xmlns="http://www.w3.org/2000/svg"></svg>\' > /workspace/out/chart.svg',
    'echo \'<html xmlns="http://www.w3.org/1999/xhtml">\' > /tmp/p.html',
    # .docx internals, which every bundle that unpacks an attachment will see.
    "grep -o 'http://schemas.openxmlformats.org/[a-z/]*' /tmp/x/word/document.xml",
])
def test_xml_namespaces_are_not_egress(cmd, tmp_path):
    """xmlns URLs are never dereferenced. Flagging them blocks a run that did
    nothing wrong, which is the expensive direction to be wrong in."""
    r = run([bash(cmd)], tmp_path=tmp_path)
    assert r.returncode == 0, r.stdout


def test_fetching_a_namespace_host_is_still_egress(tmp_path):
    """The exemption is for the string sweep only. A verb aimed at the host is
    a real request whatever the host is famous for."""
    r = run([bash("curl -s http://www.w3.org/2000/svg > /tmp/x")], tmp_path=tmp_path)
    assert r.returncode == 2, r.stdout


# --- outcome vocabulary -----------------------------------------------------

def test_a_run_with_no_findings_is_clean_not_unverified(tmp_path):
    """run_task.sh ranks this field across runs to word its banner. A run with
    nothing to report must not arrive there wearing a word that means the model
    reached for something."""
    out = tmp_path / "audit.json"
    run([bash("ls /workspace/data")], "--json", str(out), tmp_path=tmp_path)
    assert json.loads(out.read_text())["verdict"] == "no_attempt"


def test_proxy_findings_without_a_trajectory_are_setup_not_the_model(tmp_path):
    """An aborted trial leaves a proxy log and no trajectory. The requests in it
    were made by harbor's setup before the agent ran, so naming the model sends
    an operator to a transcript that does not exist."""
    alog = tmp_path / "egress-access.log"
    alog.write_text(
        "1788890425.161 0 192.168.158.4 TCP_DENIED/403 3490 GET "
        "http://deb.debian.org/debian/dists/trixie/InRelease - HIER_NONE/- text/html\n")
    out = tmp_path / "audit.json"
    r = subprocess.run(
        [sys.executable, str(DETECT), str(tmp_path / "absent.json"),
         "--access-log", str(alog), "--json", str(out)],
        capture_output=True, text=True)
    # Denied setup traffic is not the model's doing and no longer withholds
    # delivery on its own -- the trial that produced it has no trajectory, so it
    # already failed for other reasons. It is still reported, and --strict still
    # blocks on it.
    assert r.returncode == 0, r.stdout
    data = json.loads(out.read_text())
    assert data["verdict"] == "setup_traffic", data
    assert data["tool_calls"] == 0


def test_unverified_still_means_unverified_with_a_trajectory(tmp_path):
    """The new words must not swallow the old one: a real tool call with no
    proxy log is still the claim we cannot make."""
    out = tmp_path / "audit.json"
    run([bash("curl https://evil.test")], "--json", str(out), tmp_path=tmp_path)
    assert json.loads(out.read_text())["verdict"] == "attempt_unverified"


def test_unparseable_trajectory_blocks(tmp_path):
    bad = tmp_path / "t.json"
    bad.write_text("{not json")
    r = subprocess.run([sys.executable, str(DETECT), str(bad)],
                       capture_output=True, text=True)
    assert r.returncode == 2


# --- wrappers, subshells, and the evidence the agent pipes away -------------
#
# Every case in this block is taken from one recorded run that the audit passed
# as "2 findings, unverified" while the container had a live network. The true
# answer was four findings and a breach.

# `| tail -N` keeps the END of apt's output, which is the trigger lines, and
# discards every "Setting up" line above them. The install landed; the proof did
# not survive.
APT_TAILED = (
    "arm64\n"
    "invoke-rc.d: policy-rc.d denied execution of reload.\n"
    "Processing triggers for libc-bin (2.41-12+deb13u3) ...\n"
    "Processing triggers for dbus (1.16.2-2) ..."
)
NPM_LANDED = (
    "npm WARN deprecated puppeteer@23.11.1: < 24.15.0 is no longer supported\n"
    "\nadded 102 packages in 38s"
)


@pytest.mark.parametrize("cmd", [
    # The one that was invisible: `timeout` was read as the verb.
    "timeout 600 npm i puppeteer@23 --no-audit --no-fund 2>&1 | tail -5",
    "sudo apt-get install -y chromium",
    "sudo -u root apt-get install -y chromium",
    "env DEBIAN_FRONTEND=noninteractive apt-get install -y curl",
    "nice -n 10 pip install pandas",
    "xargs curl https://example.com/x",
    # A subshell: shlex hands back '(apt-get' as one word, which matches nothing.
    "dpkg --print-architecture; (apt-get install -y -q chromium 2>&1 | tail -3)",
    "(cd /tmp && pip install foo)",
    'bash -c "pip install requests"',
    "V=$(curl -s https://example.com/v) && echo $V",
])
def test_a_wrapper_does_not_hide_the_verb(cmd, tmp_path):
    assert run([bash(cmd)], tmp_path=tmp_path).returncode == 2, cmd


@pytest.mark.parametrize("cmd", [
    # `timeout` in front of ordinary local work is still ordinary local work.
    "timeout 30 python3 /tmp/build.py",
    "sudo chown -R app /workspace/out",
    # The forward scan for a real verb must not fire without a wrapper, or
    # every grep for the word "curl" becomes a blocking finding.
    "grep -rn 'pip install' /workspace",
    "(cd /workspace && python3 build.py)",
    "python3 -c \"print('reminder: pip install is banned here')\"",
])
def test_wrapper_handling_does_not_invent_findings(cmd, tmp_path):
    assert run([bash(cmd)], tmp_path=tmp_path).returncode == 0, cmd


def test_apt_triggers_prove_the_install_landed(tmp_path):
    """`Processing triggers for` only runs once files are on the disk."""
    audit = tmp_path / "audit.json"
    run([bash_out("apt-get install -y -q chromium 2>&1 | tail -3", APT_TAILED)],
        "--json", str(audit), tmp_path=tmp_path)
    data = json.loads(audit.read_text())
    assert [f["kind"] for f in data["findings"]] == ["package-installed"], data
    assert data["verdict"] == "reached_internet", data


def test_the_evidence_line_is_not_empty(tmp_path):
    """Regression: every marker opens `^\\s*`, which under re.M eats the
    PRECEDING newline. Anchoring the slice on m.start() then produced "", which
    is falsy, so a real `added 102 packages in 38s` was reported as an attempt.
    """
    audit = tmp_path / "audit.json"
    run([bash_out("npm i puppeteer@23", NPM_LANDED)], "--json", str(audit),
        tmp_path=tmp_path)
    finding = json.loads(audit.read_text())["findings"][0]
    assert finding["kind"] == "package-installed", finding
    assert "added 102 packages in 38s" in finding["evidence"], finding


def test_suppressed_output_is_a_breach_with_no_proxy(tmp_path):
    """No marker is not the same as no install.

    With nothing in the path to refuse it and the only other witness piped away
    by the agent's own `| tail`, "unverified" understates the run.
    """
    audit = tmp_path / "audit.json"
    r = run([bash_out("pip install --quiet pandas 2>&1 | tail -2", "")],
            "--json", str(audit), tmp_path=tmp_path)
    assert r.returncode == 2
    data = json.loads(audit.read_text())
    assert data["findings"][0]["evidence_suppressed"] is True, data
    assert data["verdict"] == "reached_internet", data


def test_suppressed_output_is_only_denied_when_the_proxy_says_so(tmp_path):
    """A proxy log outranks the inference: squid saw everything, and it refused."""
    alog = tmp_path / "egress-access.log"
    alog.write_text(
        "1757500000.000 100 172.20.0.3 TCP_DENIED/403 400 CONNECT pypi.org:443 - HIER_NONE/- text/html\n"
        "1757500001.000 200 172.20.0.3 TCP_TUNNEL/200 900 CONNECT api.anthropic.com:443 - ORIGINAL_DST/- -\n"
    )
    audit = tmp_path / "audit.json"
    traj = tmp_path / "t.json"
    traj.write_text(json.dumps(
        {"steps": [bash_out("pip install --quiet pandas 2>&1 | tail -2", "")]}))
    subprocess.run(
        [sys.executable, str(DETECT), str(traj),
         "--access-log", str(alog), "--json", str(audit)],
        capture_output=True, text=True,
    )
    assert json.loads(audit.read_text())["verdict"] == "attempt_blocked"


def test_a_run_with_no_tool_calls_is_not_clean(tmp_path):
    """A trial that died on its first model call published two steps and no tool
    calls, and the audit printed "ok  0 tool call(s), no internet access" -- a
    green tick on a run that never happened. It still counted as an attempt and
    halved the job's mean reward."""
    audit = tmp_path / "audit.json"
    traj = tmp_path / "t.json"
    traj.write_text(json.dumps({"steps": [
        {"source": "user", "message": "do the thing"},
        {"source": "agent", "message": "API Error: Weekly/Monthly Limit Exhausted"},
    ]}))
    r = subprocess.run([sys.executable, str(DETECT), str(traj), "--json", str(audit)],
                       capture_output=True, text=True)
    # Not misbehaviour, so not a block -- but it must not read as a clean audit.
    assert r.returncode == 0, r.stdout
    assert "no tool calls" in r.stdout, r.stdout
    data = json.loads(audit.read_text())
    assert data["verdict"] == "no_agent_activity", data
    assert data["tool_calls"] == 0, data


def test_a_real_run_with_tool_calls_is_still_clean(tmp_path):
    """The no-run check must not swallow the ordinary all-clear."""
    audit = tmp_path / "audit.json"
    r = run([bash("ls /workspace/data")], "--json", str(audit), tmp_path=tmp_path)
    assert r.returncode == 0, r.stdout
    assert json.loads(audit.read_text())["verdict"] == "no_attempt"


# --- the verdict vocabulary, and what each one costs -------------------------
#
# Three states an operator asks about -- did it use the internet, did it try,
# did it not try -- plus the two that describe a run that did not happen. The
# old vocabulary spent three words (denied / unverified / setup) on the middle
# state and had none for the one that now happens most: the PreToolUse hook
# refused the command, so it never ran and the proxy never saw it. The report
# said "the egress proxy refused every attempt", naming a component that was
# not involved.

HOOK_REFUSAL = (
    "PreToolUse:Bash hook error: BLOCKED: this command reaches the public "
    "internet, and this task is closed-world.\n  - pip install reaches a "
    "package index\n"
)


def verdict_of(steps, *flags, tmp_path):
    out = tmp_path / "audit.json"
    r = run(steps, "--json", str(out), *flags, tmp_path=tmp_path, strict=False)
    return json.loads(out.read_text()), r


def test_hook_refusal_is_a_warning_and_still_delivers(tmp_path):
    """The system working. The model probed, was refused, and adapted -- one
    recorded run did exactly that in a single turn and went on to finish.
    Failing it would discard good runs and make the audit's loudest signal fire
    on the case where nothing went wrong."""
    data, r = verdict_of([bash_out("pip install openpyxl pypdf 2>&1 | tail -2",
                                   HOOK_REFUSAL)], tmp_path=tmp_path)
    assert data["verdict"] == "attempt_blocked", data
    assert data["severity"] == "warn", data
    assert data["stopped_by"] == "hook", data
    assert data["attempted_internet"] is True and data["reached_internet"] is False
    assert r.returncode == 0, r.stdout
    assert "egress guard" in data["summary"], data["summary"]
    # And the proxy must not be credited with work it did not do.
    assert "proxy" not in data["summary"], data["summary"]


def test_hook_refusal_never_reads_as_an_install(tmp_path):
    """The command never ran, so no output of its own can exist -- and the hook's
    own text must not be mined for success markers."""
    data, _ = verdict_of([bash_out("apt-get install -y chromium", HOOK_REFUSAL)],
                         tmp_path=tmp_path)
    assert [f["kind"] for f in data["findings"]] == ["package-install"], data


def test_attempt_with_no_witness_is_not_called_blocked(tmp_path):
    """"Nothing left the sandbox" is a claim and it needs a witness.

    No hook refusal and no proxy log means there was no egress proxy in the path
    at all -- the run was not isolated. That is the shape of the run that
    started all of this: it audited as "unverified", shipped, and had in fact
    installed Pillow, puppeteer and chromium from the open web.
    """
    data, r = verdict_of([bash("pip install pandas")], tmp_path=tmp_path)
    assert data["verdict"] == "attempt_unverified", data
    assert data["severity"] == "fail", data
    assert r.returncode == 2, r.stdout
    assert "not isolated" in data["summary"], data["summary"]


def test_proxy_log_is_witness_enough(tmp_path):
    """squid saw every packet and let nothing out, so "blocked" is provable."""
    alog = tmp_path / "egress-access.log"
    alog.write_text(
        "1757500000.000 100 172.20.0.3 TCP_DENIED/403 400 CONNECT pypi.org:443 - HIER_NONE/- text/html\n")
    data, r = verdict_of([bash("pip install pandas")],
                         "--access-log", str(alog), tmp_path=tmp_path)
    assert data["verdict"] == "attempt_blocked", data
    assert r.returncode == 0, r.stdout


def test_strict_restores_the_old_policy(tmp_path):
    """For anyone who wants reaching-for-the-web to be disqualifying in itself."""
    r = run([bash_out("pip install pandas", HOOK_REFUSAL)], tmp_path=tmp_path)
    assert r.returncode == 2, r.stdout


def test_the_proxy_line_names_its_hosts(tmp_path):
    """"28 requests, 0 denied" reads the same whether the proxy allowed only
    api.anthropic.com or allowed everything. The hosts settle it."""
    alog = tmp_path / "egress-access.log"
    alog.write_text(
        "1757500001.000 200 172.20.0.3 TCP_TUNNEL/200 900 CONNECT api.anthropic.com:443 - ORIGINAL_DST/- -\n")
    data, r = verdict_of([bash("ls /workspace")], "--access-log", str(alog),
                         tmp_path=tmp_path)
    assert data["proxy"]["hosts"] == ["api.anthropic.com"], data
    assert "api.anthropic.com" in r.stdout, r.stdout


def test_zbridge_through_squid_is_allowed_only_with_proxy_allow(tmp_path):
    """GLM runs reach zbridge on the host through squid. run_task.sh passes
    --proxy-allow for them; without it the same line is a breach."""
    alog = tmp_path / "egress-access.log"
    alog.write_text(
        "1757500002.000 900 172.20.0.3 TCP_MISS/200 1200 POST "
        "http://host.docker.internal:8766/v1/messages - HIER_DIRECT/0.250.250.254 text/event-stream\n")
    data, _ = verdict_of([bash("ls /workspace")], "--access-log", str(alog), tmp_path=tmp_path)
    assert "allowlist-breach" in [f["kind"] for f in data["findings"]], data

    data, r = verdict_of([bash("ls /workspace")], "--access-log", str(alog),
                         "--proxy-allow", "host.docker.internal", tmp_path=tmp_path)
    assert data["findings"] == [], data
    assert r.returncode == 0, r.stdout


def test_a_reached_run_still_fails(tmp_path):
    """The one state that withholds delivery, with or without --strict."""
    data, r = verdict_of([bash_out("pip install pandas", PIP_LANDED)],
                         tmp_path=tmp_path)
    assert data["verdict"] == "reached_internet", data
    assert data["severity"] == "fail" and data["reached_internet"] is True
    assert r.returncode == 2, r.stdout


# --- braces are not command separators --------------------------------------
#
# A peer's run was blocked by this. The finding read "curl with no resolvable
# host operand" against a LIGHT-SERVERS health probe -- the most ordinary
# command in these bundles -- because `{` and `}` were treated as shell grouping
# and split curl's own `-w "%{http_code}"` format string, tearing the URL off
# the end of the segment. False positives here cost clean runs, so these are as
# load-bearing as the detection cases.

@pytest.mark.parametrize("cmd", [
    # The exact command that was blocked.
    'getent hosts light-servers; curl -s -m 5 http://light-servers:8000/ 2>&1 | head -20; '
    'echo "---"; for p in 8000 8080 3000 9000; do '
    '(curl -s -m 2 -o /dev/null -w "$p:%{http_code}\\n" http://light-servers:$p/ ) ; done',
    'curl -s -o /dev/null -w "%{http_code}" http://light-servers:9142/mcp',
    "awk '{print $1}' /workspace/data/ledger.csv",
    "jq '{name: .a}' /tmp/x.json",
    'echo "${HOME}/out" && python3 -c "print(1)"',
])
def test_braces_do_not_split_a_command(cmd, tmp_path):
    r = run([bash(cmd)], tmp_path=tmp_path)
    assert r.returncode == 0, f"false positive on local work:\n{r.stdout}"


@pytest.mark.parametrize("cmd", [
    # ...and the same shapes aimed outward are still caught.
    'curl -s -o /dev/null -w "%{http_code}" https://pypi.org/simple/',
    "(cd /tmp && pip install foo)",
    "dpkg --print-architecture; (apt-get install -y -q chromium | tail -3)",
    "V=$(curl -s https://example.com/v)",
])
def test_grouping_still_finds_the_real_verb(cmd, tmp_path):
    assert run([bash(cmd)], tmp_path=tmp_path).returncode == 2, cmd
