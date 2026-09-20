#!/usr/bin/env python3
"""Detect whether the agent reached the public internet, and block the run if it did.

    tools/network/detect_internet_use.py <trajectory.json> [--json OUT] [--warn-only]
                                  [--access-log run_N/logs/egress-access.log]

Exit 0 = clean. Exit 2 = the model reached for the internet; the run is
blocked. Whether it got there is a separate question -- see _verdict().

WHY THIS EXISTS

The tasks in this repo are closed-world: every fact the agent needs is served by
the light-servers MCP sidecars or sits under the read-only /workspace/data mount.
A run that answers from the open web has not solved the task, it has looked up
something adjacent to it -- and it grades as though it had, because nothing
downstream can tell the two apart.

Harbor cannot prevent this on the docker provider, and it is worth being precise
about why, because the obvious fix does not work:

  network_mode = "no-network"   sets `network_mode: none` on the `main` service
                                (harbor/environments/docker/docker-compose-no-network.yaml),
                                which detaches the compose bridge too. The MCP
                                sidecars become unreachable, the agent starts
                                with ZERO tools, and the run grades 0.

  network_mode = "allowlist"    the docker provider declares
                                network_allowlist=False -- Harbor cannot express
                                a host allowlist on docker at all.

And the agent is Claude Code driven by CLAUDE_CODE_OAUTH_TOKEN, so it must keep
reaching api.anthropic.com regardless. There is no Harbor setting that means
"sidecars and the API, nothing else".

There is now a block, but it lives BELOW Harbor rather than in it:
tools/network/egress-proxy/overlay.yaml is passed as --extra-docker-compose and makes
the compose project's default network `internal: true`, leaving a single squid
sidecar as the only route out with api.anthropic.com as its whole allowlist.
That is the distinction Harbor's network_mode cannot draw -- network_mode says
whether the container has a network, the overlay says where that network may go.

This scanner stays anyway, and stays blocking. Prevention is configuration and
configuration regresses quietly: an overlay that stopped being passed,
NETWORK_ISOLATION_OFF exported in a shell weeks ago, an allowlist widened to get
one run unstuck. The trajectory is the only artifact that records what the model
actually reached, so it remains the thing that decides whether a run ships. The
posture is PREVENT AND DETECT:
the container keeps working network, and any use of it by the MODEL is caught
from the recorded trajectory and blocks the run.

WHAT COUNTS

Only the agent's own tool calls are scanned. Claude Code's API traffic never
appears in a trajectory, so it is out of scope by construction rather than by
allowlist -- there is no rule here that could accidentally start permitting it.

Two families are caught:

  web tools       WebSearch / WebFetch, straight off the tool name.
  shell egress    Bash commands that fetch (curl, wget, nc, ssh, git clone) or
                  install (pip, npm, apt-get, ...), which cannot work without
                  the network.

Traffic aimed at the sidecars is NOT egress and must not be flagged: health
probes like `curl -s http://light-servers:9142/mcp` are a normal part of these
bundles. Fetchers are therefore judged by their TARGET HOST, and only hosts
outside INTERNAL_HOSTS count. Installers carry no host, so they are judged by
whether they are pinned to something local (--no-index, a path operand).

An installer is judged twice. The command line says the model REACHED for a
package index; the command's OUTPUT says whether it got there. A denied
`pip install pandas` prints a proxy 403 and installs nothing, while the same
command on a leaked network prints "Successfully installed pandas-2.2.3" -- and
that difference is the difference between a block that worked and a benchmark
result that was never closed-world. Both block the run; only the second is
reported as a breach (see INSTALL_SUCCESS_MARKERS and _outcome).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import textwrap
from pathlib import Path

# --------------------------------------------------------------------------
# THE RULES THEMSELVES LIVE IN egress_rules.py
#
# They used to live here, and that was correct while this audit was the only
# thing reading them. It stopped being correct when the run began DENYING a
# command as well as reporting it: the PreToolUse hook shipped into the
# container (tools/network/egress_rules.py, hook mode) has to agree with this
# file exactly, and two copies of "what counts as egress" drift in the
# expensive direction -- a command the hook allows and this audit later blocks
# costs a whole graded run, discarded after the fact for something that could
# have been refused in the turn it was typed.
#
# So the classification is imported, never redefined. What stays here is the
# half that needs an OUTCOME rather than an intention: the proxy log, the
# install-success markers, and the reporting.
#
# sys.path, not a package import: this file is run as a script by
# scripts/run_task.sh and by its tests, so `tools.network` is not importable.
# --------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).resolve().parent))

from egress_rules import (            # noqa: E402
    DENIAL_MARKER,
    GIT_NETWORK_SUBCOMMANDS,
    HEREDOC_RE,
    INLINE_NETWORK_HINTS,
    INSTALLERS,
    INTERNAL_HOSTS,
    NAMESPACE_URI_PREFIXES,
    OFFLINE_FLAGS,
    URL_RE,
    WEB_TOOLS,
    classify_tool,
    host_of,
    is_internal,
)

# Lines an installer prints only after it has actually pulled something from an
# index. These are read out of the STEP'S RESPONSE, which is the one place a
# trajectory records an outcome rather than an intention, and they are what
# lets this scanner say "the install landed" instead of "the model tried".
#
# Their weight: a confirmed install is ground truth of reach, in the same class
# as an allowlist breach in the proxy log, and unlike the proxy log it survives
# a NETWORK_ISOLATION_OFF=1 run where there is nothing to corroborate against.
#
# POSITIVE MARKERS ONLY, and the omissions are deliberate. pip's "Requirement
# already satisfied", npm's "up to date", apt's "0 newly installed" all mean the
# resolver found the package ALREADY ON DISK. Nothing left the container, so
# none of them may read as egress -- matching them would turn every warm-cache
# install into a false breach, which is the expensive direction to be wrong in.
INSTALL_SUCCESS_MARKERS = (
    re.compile(r"^\s*Successfully installed\s+\S", re.M),            # pip, gem
    re.compile(r"^\s*(?:Collecting|Downloading)\s+\S", re.M),        # pip, mid-install
    re.compile(r"^\s*(?:Installed|Prepared)\s+\d+\s+packages?", re.M),  # uv
    re.compile(r"^\s*added\s+\d+\s+packages?", re.M),               # npm
    re.compile(r"^\s*\+\s+\S+(?:@|==)\d", re.M),                     # npm/yarn/pnpm/uv per package
    re.compile(r"^\s*Setting up\s+\S+\s+\(", re.M),                  # apt / apt-get / dpkg
    # apt again, and the reason it is here: `apt-get install -y -q chromium
    # 2>&1 | tail -3` keeps exactly these trigger lines and cuts every "Setting
    # up" line above them. A real run installed chromium that way and the audit
    # reported only an attempt. dpkg runs triggers when a package's files have
    # actually been unpacked onto the disk, so the line means the same thing.
    re.compile(r"^\s*Processing triggers for\s+\S", re.M),
    re.compile(r"^\s*Unpacking\s+\S+\s+\(", re.M),                   # apt, mid-install
    re.compile(r"^\s*Get:\d+\s+https?://", re.M),                    # apt, fetching from a mirror
    # pip's self-check asks PyPI which version of pip is current and prints
    # this. It is not an install marker and it is deliberately weaker than the
    # others -- but install_landed() is only ever consulted on a command that
    # ALREADY produced an installer finding, so this can strengthen a finding
    # and can never invent one. `pip install --quiet Pillow 2>&1 | tail -2`
    # printed nothing else, and this was the only surviving proof of reach.
    re.compile(r"^\s*\[notice\].*new release of pip is available", re.M),
    re.compile(r"^\(\d+/\d+\)\s+Installing\s+\S", re.M),            # apk
    re.compile(r"^\s*Downloaded\s+\S+\s+v?\d", re.M),               # cargo
    re.compile(r"^\s*go: downloading\s+\S", re.M),                   # go get
    re.compile(r"^==>\s+Pouring\s+\S", re.M),                        # brew
)


FINDINGS: list[dict] = []

# Whether this audit saw an agent trajectory at all. An aborted trial -- the
# environment died, agent setup failed -- leaves a proxy log and no trajectory,
# and traffic in that log was made by harbor's own setup, before the model ran.
# Blaming the MODEL for it reads as agent misbehaviour and sends an operator
# looking at a transcript that does not exist.
HAD_TRAJECTORY = True

# How many tool calls the trajectory held. Separate from HAD_TRAJECTORY because
# a trajectory can exist, parse, and still record nothing the agent did: a
# recorded run died on its first model call with "Weekly/Monthly Limit
# Exhausted" and published two steps, a prompt and an error. The audit printed
# "ok  0 tool call(s), no internet access" -- a green tick on a run that never
# happened -- and the trial still counted as an attempt, halving the job's mean
# reward. Neither is an internet finding, so neither blocks; both need saying.
TOOL_CALLS = 0

# Whether squid's access.log was there to read, SEPARATELY from whether it had
# anything in it. The two used to be one question, answered by counting lines,
# and the answer was wrong the moment agent-path Headroom was switched on.
#
# What the line count was really measuring: Claude Code's own model calls
# (CONNECT api.anthropic.com) happen to pass through the agent's squid, so the
# log was never empty on a healthy run and "has lines" stood in for "a proxy
# existed". overlay-headroom.yaml puts `headroom` in NO_PROXY and gives it a
# squid of its own -- deliberately, so headroom's startup fetches stay out of
# the agent's log -- and the model calls stop passing through here. The log is
# then correctly empty, and the old test read that as "no egress proxy was in
# the path, so the run was not isolated": a FAIL verdict on a run whose
# isolation was fully intact. That happened, on a delivered run.
#
# The file's existence is the better witness anyway, and it is cheap: only
# tools/network/egress-proxy/entrypoint.sh creates it, with `tee -a` at
# container start, and tools/delivery/harbor_to_output.py:429 copies nothing
# that does not exist. So a file here -- even a zero-byte one -- means the
# egress-proxy container came up with its mount attached, which means
# overlay.yaml was applied, which is what puts `internal: true` on the agent's
# network and takes the gateway away. That is the enforcement boundary; squid
# is the part of it that keeps a record.
#
# Read it for what it proves and no more: the block was live. It does NOT say
# the model's traffic was inspected, which is why an empty log gets its own
# wording in _summary() rather than borrowing the denial sentence.
PROXY_LOG_SEEN = False


def _c(code: str, s: str) -> str:
    return f"\033[{code}m{s}\033[0m" if sys.stdout.isatty() else s


def flag(step: int | None, tool: str, kind: str, detail: str, evidence: str,
         *, suppressed: bool = False, stopped_by: str | None = None) -> None:
    """step is None for findings that come from the proxy log rather than a
    trajectory step -- they are real findings and must block, but they have no
    step number to point at.

    `suppressed` marks a command that piped its own output away, so the absence
    of an install-success marker proves nothing. `stopped_by` names what refused
    the command, when something did. _verdict() and _stopped_by() read both.
    """
    record = {"step": step, "tool": tool, "kind": kind, "detail": detail,
              "evidence": evidence[:400]}
    if suppressed:
        record["evidence_suppressed"] = True
    if stopped_by:
        record["stopped_by"] = stopped_by
    FINDINGS.append(record)



# --------------------------------------------------------------------------
# PROXY GROUND TRUTH
#
# Everything above infers egress from the TRAJECTORY: tool names and shell verbs
# in the transcript. That inference has a floor. `requests.get(...)` inside a
# python heredoc carries no verb this scanner knows, and a trajectory that was
# truncated or never written carries nothing at all.
#
# squid's access.log is the other half: not what the model said it would do, but
# what actually arrived at the proxy and what the proxy did about it. It is
# written per attempt into the trial's agent-log dir (tools/network/egress-proxy/
# entrypoint.sh) and reaches the run dir via tools/delivery/harbor_to_output.py.
# --------------------------------------------------------------------------

# The one host squid lets out. tools/network/egress-proxy/squid.conf is the source of
# truth and scripts/tests/test_egress_allowlist.py::EXPECTED_ALLOWLIST pins it
# there; this is the third copy, so change one and look at the other two.
# GLM runs add host.docker.internal (zbridge) with --proxy-allow.
PROXY_ALLOWLIST = {"api.anthropic.com"}

# Hosts the Claude Code CLI reaches on its own initiative -- update checks,
# feature flags, telemetry, error reporting. squid denies all of them, which is
# correct, but the MODEL did not ask for them and a run must not be blocked for
# the CLI clearing its throat. Enumerated in squid.conf's allowlist comment.
#
# overlay.yaml sets CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1, which stops most
# of this at the source, so in practice these lines are rare. They are listed
# anyway because a version bump can add one, and the failure mode of NOT listing
# it is a benchmark that refuses to deliver a clean run.
#
# TELEMETRY ONLY. Every host here is infrastructure that carries no content: an
# update check, a feature-flag fetch, a crash report. A denial to one of them
# says nothing about the model, so it is recorded and does not block.
#
# raw.githubusercontent.com is deliberately NOT here, though squid.conf lists it
# among the hosts the CLI reaches. It serves CONTENT -- a place to fetch
# instructions from or park data at -- and a denial there is exactly the event a
# closed-world benchmark wants to hear about. Classifying it as infrastructure
# would make the audit silent on the most useful denial it could ever show.
#
# The cost is a possible false positive, if the CLI fetches it unprompted
# despite CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1. That is the right way
# round: a false positive is loud, lands in internet_audit.json with the
# offending line attached, and is cleared with INTERNET_AUDIT_WARN=1 while
# someone decides. A false negative is silent and ships a benchmark result that
# was never actually checked. Same posture as squid.conf's "add a host here only
# after seeing it denied in access.log".
CLI_INFRA_HOSTS = {
    "platform.claude.com", "claude.ai", "statsig.anthropic.com",
    "downloads.claude.ai",
}

CLI_INFRA_SUFFIXES = (".datadoghq.com", ".statsig.com", ".sentry.io")

# squid native format, whitespace separated:
#   ts elapsed client CODE/STATUS bytes METHOD URL rfc931 hierarchy type
# Field 3 is the result code, 5 the method, 6 the URL (host:port for CONNECT).
_ACCESS_MIN_FIELDS = 7


def _access_host(url: str) -> str:
    """Host from an access.log URL field. CONNECT logs host:port, GET logs a URL."""
    if "://" in url:
        url = url.split("://", 1)[1]
    return url.split("/", 1)[0].rsplit(":", 1)[0].strip("[]").lower()


def _is_cli_infra(host: str) -> bool:
    return host in CLI_INFRA_HOSTS or host.endswith(CLI_INFRA_SUFFIXES)


def scan_access_log(path: Path) -> list[dict]:
    """Parse squid's log; flag the attempts the model is answerable for.

    Returned records go into the audit JSON whole -- including the allowed ones,
    because "api.anthropic.com was reached N times and nothing else was" is the
    positive evidence that the block was live for this run, which no amount of
    config assertion can supply.
    """
    attempts: list[dict] = []
    for line in path.read_text(errors="replace").splitlines():
        f = line.split()
        if len(f) < _ACCESS_MIN_FIELDS or "/" not in f[3]:
            continue
        code = f[3].split("/", 1)[0]
        status = f[3].split("/", 1)[1]
        method, url = f[5], f[6]
        host = _access_host(url)
        if not host:
            continue
        denied = code.endswith("_DENIED") or status in ("403", "407")
        rec = {"host": host, "method": method, "code": f[3], "denied": denied}
        attempts.append(rec)

        if _is_cli_infra(host):
            rec["verdict"] = "cli_infrastructure"
            continue
        if not denied and host not in PROXY_ALLOWLIST:
            # The allowlist did not hold. Worse than a denial: something left.
            rec["verdict"] = "ALLOWLIST_BREACH"
            flag(None, "egress-proxy", "allowlist-breach",
                 f"{host} was NOT denied by the proxy but is not on the allowlist",
                 line)
            continue
        if denied:
            # The model tried. The auditor already treats attempts as findings
            # regardless of outcome -- a WebFetch call counts whether or not it
            # returned -- so a denial is a finding, not an all-clear.
            rec["verdict"] = "blocked_attempt"
            flag(None, "egress-proxy", "proxy-denied",
                 f"{method} {host} was attempted and denied by the egress proxy",
                 line)
            continue
        rec["verdict"] = "allowed"
    return attempts


def response_text(resp) -> str:
    """A step's response as searchable text.

    Trajectories carry it three ways: a plain string (Bash stdout, the case that
    matters here), a parsed JSON object (agent_log_to_trajectory.py json.loads
    the tool_result when it can), or nothing at all. Nested containers are
    flattened by joining their string leaves on newlines rather than dumping
    them -- json.dumps would escape every newline and defeat the line anchors
    in INSTALL_SUCCESS_MARKERS, which is what keeps them from matching mid-line
    prose.
    """
    if resp is None:
        return ""
    if isinstance(resp, str):
        return resp
    if isinstance(resp, dict):
        return "\n".join(response_text(v) for v in resp.values())
    if isinstance(resp, (list, tuple)):
        return "\n".join(response_text(v) for v in resp)
    return str(resp)


def install_landed(resp) -> str | None:
    """The line proving a package was actually fetched and installed, or None.

    Returns the evidence rather than a bool: a finding that makes the stronger
    claim has to be able to show the line it made it from, or an operator
    cannot tell a real breach from a marker that matched something else.

    No response is NOT a failed install. A trajectory can be truncated, and a
    tool result can be missing for reasons that have nothing to do with the
    network, so absence falls back to the weaker attempt finding.
    """
    text = response_text(resp)
    if not text:
        return None
    for pat in INSTALL_SUCCESS_MARKERS:
        m = pat.search(text)
        if not m:
            continue
        # Anchor on m.end(), never m.start(). Every marker opens with `^\s*`,
        # and under re.M that `\s*` happily consumes the NEWLINE that ended the
        # previous line -- so m.start() points at that newline, `find("\n",
        # m.start())` returns the very same index, and the slice is "". Empty
        # is falsy, the caller read it as "no proof", and a real
        # `added 102 packages in 38s` was reported as a mere attempt. m.end()
        # is always inside the matched text, so the line it sits on is the line
        # that actually carries the evidence.
        line_start = text.rfind("\n", 0, m.end()) + 1
        line_end = text.find("\n", m.end())
        return text[line_start:line_end if line_end != -1 else len(text)].strip()
    return None



def normalise(traj: dict) -> list[tuple[int, str, dict, object]]:
    """Both trajectory shapes the repo produces, flattened to (step, tool, args, response).

    tests/test.sh writes {"steps":[{"tool","arguments","response"}]} for the
    verifier, while Harbor publishes agent/trajectory.json as
    {"steps":[{"tool_calls":[...],"observation":{"results":[...]}}]}. Accepting
    both means a run audits identically inside the verifier and on the host
    against a finished trial.

    The RESPONSE is carried because it is the only record of what a command
    achieved rather than what it asked for -- it is what tells a denied
    `pip install` from one that landed. It is whatever the trajectory holds
    (string, parsed JSON, or None); response_text() does the flattening.
    """
    out: list[tuple[int, str, dict, object]] = []
    for i, step in enumerate(traj.get("steps") or [], start=1):
        if not isinstance(step, dict):
            continue
        if step.get("tool"):
            out.append((i, str(step["tool"]), step.get("arguments") or {},
                        step.get("response")))
        # Harbor keeps the result out of the call and joins the two by id:
        # observation.results[].source_call_id == tool_calls[].tool_call_id.
        # A step can carry several calls and their results in either order, so
        # index first and look up second rather than zipping positionally.
        results: dict = {}
        obs = step.get("observation")
        if isinstance(obs, dict):
            for r in obs.get("results") or []:
                if isinstance(r, dict):
                    results[r.get("source_call_id")] = r.get("content")
        for call in step.get("tool_calls") or []:
            if isinstance(call, dict):
                name = call.get("function_name") or call.get("name") or ""
                out.append((i, str(name), call.get("arguments") or {},
                            results.get(call.get("tool_call_id"))))
    return out


def scan(traj: dict) -> None:
    """Turn every tool call in the trajectory into findings.

    The rules are egress_rules.classify_tool()'s, unchanged -- what this adds is
    the OUTCOME, which only the audit is in a position to know. An installer
    finding arrives here as "the model reached for an index"; the step's own
    output is then read to decide whether it got there:

      package-installed   the output proves it landed. Ground truth of reach,
                          in the same class as an allowlist breach in the proxy
                          log, and unlike that log it survives a run with no
                          proxy to corroborate against.
      package-install     no proof either way. The weaker claim, and the honest
                          one when a denied install printed a 403 and stopped.

    A SUPPRESSED install is the third case, and it is the one that cost a real
    run. `apt-get install -y -q chromium 2>&1 | tail -3` keeps the last three
    lines -- "Processing triggers for libc-bin" -- and drops every "Setting up"
    line above them, so the markers find nothing and the strongest available
    evidence reads as absent. Absent is not negative, so the flag is carried
    onto the finding and _verdict() decides what it means: with a proxy log,
    squid is the witness and settles it; without one, nothing could have stopped
    the install and the agent removed the only other record of it.
    """
    for step, tool, args, response in normalise(traj):
        cmd = str(args.get("command") or "") if tool.split("__")[-1] == "Bash" else ""
        for f in classify_tool(tool, args):
            evidence = cmd or str(args.get("url") or args.get("query") or "")
            out = response_text(response)
            # The PreToolUse hook writes its refusal where the command's output
            # would have been, so the step's own response says whether the
            # command ran at all. Nothing else can tell a hook refusal from a
            # proxy denial, and calling one the other names a component that
            # was never involved.
            by = "hook" if DENIAL_MARKER in out else None
            if f.kind != "package-install":
                flag(step, tool, f.kind, f.detail, evidence, stopped_by=by)
                continue
            landed = None if by else install_landed(response)
            if landed:
                flag(step, tool, "package-installed",
                     f.detail.replace("reaches", "reached")
                     + " and the install SUCCEEDED",
                     f"{evidence}  ->  {landed}")
            else:
                flag(step, tool, "package-install", f.detail, evidence,
                     suppressed=f.suppressed, stopped_by=by)


def main(argv=None) -> int:
    global HAD_TRAJECTORY, TOOL_CALLS, PROXY_LOG_SEEN
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("trajectory", type=Path)
    ap.add_argument("--json", type=Path, help="write findings here for grading")
    ap.add_argument("--warn-only", action="store_true",
                    help="report findings but exit 0 (does not block the run)")
    ap.add_argument("--access-log", type=Path,
                    help="squid access.log for this run; adds proxy ground truth "
                         "to the trajectory inference")
    ap.add_argument("--strict", action="store_true",
                    help="fail on a BLOCKED attempt too, not just on one that "
                         "reached the internet")
    ap.add_argument("--proxy-allow", action="append", default=[], metavar="HOST",
                    help="one more host squid may let out for this run "
                         "(GLM runs: host.docker.internal, where zbridge listens)")
    a = ap.parse_args(argv)
    PROXY_ALLOWLIST.update(h.lower() for h in a.proxy_allow)

    if not a.trajectory.is_file():
        # No trajectory is not evidence of good behaviour, but it is also not
        # evidence of bad. The aborted-trial guard in run_task.sh already fails
        # loudly on a run that produced nothing, so this stays out of its way.
        # A missing trajectory is not evidence of good behaviour -- and if the
        # proxy log survived, it is the better witness anyway. Audit it alone
        # rather than returning a clean bill for a run nobody can see.
        HAD_TRAJECTORY = False
        print(f"  {_c('33', 'warn')}  no trajectory at {a.trajectory}")
        if not (a.access_log and a.access_log.is_file()):
            return 0
        PROXY_LOG_SEEN = True
        attempts = scan_access_log(a.access_log)
        return _exit_code(_report(a, total=0, attempts=attempts), a)

    try:
        traj = json.loads(a.trajectory.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"  {_c('31', 'FAIL')}  trajectory does not parse: {exc}")
        return 2

    scan(traj)
    total = TOOL_CALLS = len(normalise(traj))

    # Deduplication happens per-command inside egress_rules.classify(), so proxy
    # findings are never folded into trajectory ones: a curl the scanner already
    # flagged AND a matching denial in the log are two independent observations
    # of the same attempt, and losing the second would cost the corroboration
    # the proxy log exists to add.
    attempts = []
    if a.access_log:
        if a.access_log.is_file():
            # Set before the scan, not after: an empty file is the witness, and
            # it counts whether or not a single line comes back.
            PROXY_LOG_SEEN = True
            attempts = scan_access_log(a.access_log)
        else:
            print(f"  {_c('33', 'warn')}  no proxy log at {a.access_log}; "
                  f"trajectory-only audit")

    verdict = _report(a, total=total, attempts=attempts)
    return _exit_code(verdict, a)


# --------------------------------------------------------------------------
# VERDICT
#
# Three states an operator actually cares about, plus two that describe a run
# that did not happen. The old vocabulary had five words for the middle state
# alone (denied / unverified / setup) and no word at all for the case that now
# happens most: the PreToolUse hook refused the command, so it never ran and the
# proxy never saw it. The report said "the egress proxy refused every attempt",
# which named a component that was not involved.
# --------------------------------------------------------------------------

REACHED = "reached_internet"        # it got out
ATTEMPTED = "attempt_blocked"       # it reached for the web and was PROVABLY stopped
UNVERIFIED = "attempt_unverified"   # it reached for the web; nothing witnessed the outcome
NO_ATTEMPT = "no_attempt"           # it never reached for the web
NO_ACTIVITY = "no_agent_activity"   # a trajectory with zero tool calls
SETUP = "setup_traffic"             # traffic before the agent ran; not the model

# Severity drives the exit code, and the split is the point: only REACHED is a
# failure. An attempt that was refused is the system WORKING -- the model probed,
# was told no, and adapted, which is the behaviour the guard was built to
# produce. Failing the run for it would discard good runs and teach nobody
# anything. --strict restores the older, harsher policy for anyone who wants it.
# ATTEMPTED is a warning and UNVERIFIED is a failure, and the gap between them
# is the whole reason both exist. "Nothing left the sandbox" is a claim, and it
# needs a witness: the hook's refusal in the step's own output, or a proxy log.
# With neither, there was no egress proxy in the path at all -- which means the
# run was not isolated and the attempt most likely succeeded. That is the shape
# of the run that started all of this: it audited as "unverified", shipped, and
# had in fact installed Pillow, puppeteer and chromium from the open web.
SEVERITY = {
    REACHED: "fail",
    UNVERIFIED: "fail",
    ATTEMPTED: "warn",
    NO_ATTEMPT: "ok",
    NO_ACTIVITY: "warn",
    SETUP: "warn",
}


def _stopped_by() -> str | None:
    """What actually stopped the attempts, in the words of whatever did it.

    Per-finding, `stopped_by` is recorded at scan time from the step's own
    response. Here it is collapsed to one word for the run: "hook" and "proxy"
    only when EVERY finding agrees, so a mixed run reads "mixed" rather than
    crediting one layer with the other's work.
    """
    marks = {f.get("stopped_by") for f in FINDINGS if f["step"] is not None}
    marks.discard(None)
    if not marks:
        return None
    return marks.pop() if len(marks) == 1 else "mixed"


def _proxy_was_in_the_path(attempts: list[dict]) -> bool:
    """Was there an egress proxy between the agent and the internet?

    Two independent witnesses, either of which settles it:

      attempts        lines in squid's log. Traffic arrived and squid ruled on
                      it, so it was plainly in the path.
      PROXY_LOG_SEEN  the log file itself. Only the proxy container creates it,
                      so it is standing proof the container ran even when it
                      recorded nothing -- which is the normal state of the
                      agent's squid once agent-path Headroom takes the model
                      calls elsewhere. See PROXY_LOG_SEEN for the run this cost.

    Counting only the first is what made a fully isolated run audit as "not
    isolated", so both are asked here and nowhere else.
    """
    return bool(attempts) or PROXY_LOG_SEEN


def _verdict(attempts: list[dict]) -> str:
    if not HAD_TRAJECTORY:
        return SETUP if FINDINGS else NO_ATTEMPT
    if not FINDINGS:
        return NO_ACTIVITY if not TOOL_CALLS else NO_ATTEMPT

    # Ground truth of reach, in order of strength: the proxy let a non-allowlist
    # host through, or a command's own output shows an index was reached.
    if any(f["kind"] in ("allowlist-breach", "package-installed") for f in FINDINGS):
        return REACHED

    proxied = _proxy_was_in_the_path(attempts)

    # An install whose output was piped away proves nothing by itself. What
    # decides it is whether anything was in a position to stop it:
    #   hook refused it      the command never ran; there is nothing to verify
    #   a proxy was there    the run had no gateway, so the install had nowhere
    #                        to reach -- and had it reached anyway, the rule
    #                        above would already have returned REACHED
    #   neither              nothing could have stopped it and the agent removed
    #                        the only other witness
    unproven = [f for f in FINDINGS if f.get("evidence_suppressed")
                and f.get("stopped_by") != "hook"]
    if unproven and not proxied:
        return REACHED

    # Can every attempt be SHOWN to have been stopped? Either the hook refused
    # it before it ran, or a proxy was in the path and (per the rule above)
    # carries no breach. Otherwise nothing witnessed the outcome and saying
    # "blocked" would be inventing the evidence.
    steps = [f for f in FINDINGS if f["step"] is not None]
    if proxied or (steps and all(f.get("stopped_by") for f in steps)):
        return ATTEMPTED
    return UNVERIFIED


def _summary(verdict: str, attempts: list[dict]) -> str:
    """One line of plain English. This is what gets read; everything else is
    evidence for it."""
    n = len([f for f in FINDINGS if f["step"] is not None]) or len(FINDINGS)
    by = _stopped_by()
    if verdict == REACHED:
        landed = sum(1 for f in FINDINGS if f["kind"] == "package-installed")
        if landed:
            return (f"the model reached the open internet -- {landed} package "
                    f"install(s) completed from a public index")
        return "the model reached the open internet -- traffic left the sandbox"
    if verdict == ATTEMPTED:
        how = {"hook": "refused by the egress guard before it ran",
               "proxy": "denied by the egress proxy",
               "mixed": "refused by the egress guard and the proxy"}.get(by)
        if how is None and not attempts:
            # The proxy was up (PROXY_LOG_SEEN) and logged nothing, so the run
            # was isolated and nothing got out -- but no component refused
            # anything in so many words, and borrowing the denial sentence here
            # would credit a refusal that never happened.
            return (f"the model reached for the internet {n} time(s); the egress "
                    f"proxy was up for this run and recorded no traffic at all, "
                    f"so nothing left the sandbox")
        return (f"the model reached for the internet {n} time(s); every attempt "
                f"was {how or 'refused'}, and nothing left the sandbox")
    if verdict == UNVERIFIED:
        return (f"the model reached for the internet {n} time(s) and nothing "
                f"witnessed the outcome -- no egress proxy was in the path, so "
                f"the run was not isolated and the attempts most likely succeeded")
    if verdict == NO_ATTEMPT:
        return "the model never reached for the internet"
    if verdict == NO_ACTIVITY:
        return "the agent made no tool calls -- there was nothing to audit"
    return ("traffic was attempted before the agent ran (harbor's own setup) "
            "and refused; the model made no tool calls")


def _proxy_summary(attempts: list[dict]) -> dict:
    """Counts plus the host list.

    The hosts matter more than the counts. "28 requests, 0 denied" is
    ambiguous -- it reads the same whether the proxy allowed only
    api.anthropic.com or allowed everything. Naming them settles it.
    """
    hosts = sorted({r["host"] for r in attempts})
    return {
        # Whether squid's log was there to read. "requests: 0" alone cannot say
        # whether the proxy saw nothing or was never there, and those two grade
        # differently -- see PROXY_LOG_SEEN.
        "log_present": PROXY_LOG_SEEN,
        "requests": len(attempts),
        "allowed": sum(1 for r in attempts if not r["denied"]),
        "denied": sum(1 for r in attempts if r["denied"]),
        "hosts": hosts,
        "attempts": attempts,
    }


def _exit_code(verdict: str, a) -> int:
    """2 blocks the run, 0 does not.

    Only REACHED blocks by default. An attempt that was refused is the block
    working: the model probed, the guard said no, and the model adapted -- one
    recorded run did exactly that in a single turn and went on to finish. Failing
    it would throw away a good run and, worse, would make the audit's loudest
    signal fire on the case where nothing went wrong, which is how an operator
    learns to ignore it.

    --strict (INTERNET_AUDIT_STRICT=1 in run_task.sh) restores the older policy
    for anyone who wants reaching-for-the-web to be disqualifying in itself.
    """
    if a.warn_only:
        return 0
    if SEVERITY[verdict] == "fail":
        return 2
    if a.strict and verdict in (ATTEMPTED, UNVERIFIED, SETUP):
        return 2
    return 0


_LABEL = {"ok": ("32", "OK  "), "warn": ("33", "WARN"), "fail": ("31", "FAIL")}


def _report(a, *, total: int, attempts: list[dict]) -> str:
    """Print the audit, write its JSON, return the verdict.

    One block per run, and every line earns its place: the verdict, then the
    evidence for it, then what the proxy saw. No banner, no repetition of the
    verdict in three different wordings -- that is what made the old output
    unreadable and, on one run, wrong.
    """
    verdict = _verdict(attempts)
    severity = SEVERITY[verdict]
    colour, label = _LABEL[severity]

    name = a.trajectory.parent.parent.name or a.trajectory.name
    print(f"== internet audit: {name} ==")
    # Wrapped, and continuation lines align under the first word rather than
    # under the label. An unwrapped verdict ran to 180 characters and folded
    # wherever the terminal happened to be, which buried the one line that
    # matters in the one place it is always read.
    for i, line in enumerate(textwrap.wrap(_summary(verdict, attempts), 72)):
        lead = f"  {_c(colour, label)}  " if i == 0 else " " * 8
        print(f"{lead}{line}")

    for f in FINDINGS:
        where = "proxy" if f["step"] is None else f"step {f['step']}"
        print(f"        {where:>9}  {f['detail']}")
        print(f"        {'':>9}  {f['evidence'].splitlines()[0][:150]}")

    if attempts:
        p = _proxy_summary(attempts)
        shown = ", ".join(p["hosts"][:3]) + (", ..." if len(p["hosts"]) > 3 else "")
        print(f"        {'proxy':>9}  {p['requests']} request(s) to {shown}"
              f" -- {p['denied']} denied")
    elif PROXY_LOG_SEEN:
        # Up, and silent. Normal once agent-path Headroom carries the model
        # calls: nothing else in the run has a reason to talk to this squid.
        # Said out loud so the empty log reads as evidence rather than as a
        # gap someone has to go and check.
        print(f"        {'proxy':>9}  log present and empty -- the proxy was up "
              f"and no traffic reached it")
    elif HAD_TRAJECTORY and verdict != NO_ACTIVITY:
        # Absence of a proxy log is itself a finding about the RUN's setup: it
        # means no egress proxy was in the path, so nothing could have been
        # denied and no claim about what left the container is verifiable.
        print(f"        {'proxy':>9}  no log -- the run had no egress proxy in "
              f"the path")

    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps({
            "verdict": verdict,
            "summary": _summary(verdict, attempts),
            "severity": severity,
            "attempted_internet": verdict in (REACHED, ATTEMPTED, UNVERIFIED, SETUP),
            "reached_internet": verdict == REACHED,
            "stopped_by": _stopped_by(),
            "tool_calls": total,
            "findings": FINDINGS,
            "proxy": _proxy_summary(attempts),
        }, indent=2))
    return verdict


if __name__ == "__main__":
    sys.exit(main())
