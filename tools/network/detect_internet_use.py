#!/usr/bin/env python3
"""Detect whether the agent reached the public internet, and block the run if it did.

    tools/network/detect_internet_use.py <trajectory.json> [--json OUT] [--warn-only]
                                  [--enforced-policy allowlist|no-network|public]

Exit 0 = clean. Exit 2 = the model reached for the internet; the run is
blocked. Whether it got there is a separate question -- see _verdict().

WHY THIS EXISTS

The tasks in this repo are closed-world: every fact the agent needs is served by
the light-servers MCP sidecars or sits under the read-only /workspace/data mount.
A run that answers from the open web has not solved the task, it has looked up
something adjacent to it -- and it grades as though it had, because nothing
downstream can tell the two apart.

The block itself is Harbor's, not this file's. Each bundle's task.toml declares
the agent phase's network_mode ("allowlist" with its allowed_hosts, or
"no-network"), and Harbor's docker egress-control sidecar enforces it during
agent.run(), refusing the trial outright when the host cannot. scripts/run_task.sh
passes that resolved mode here as --enforced-policy, which is what lets an
attempt be called "blocked" rather than merely "attempted".

This scanner stays anyway, and stays blocking. Prevention is configuration and
configuration regresses quietly: a bundle whose agent phase was left public, an
allowlist widened to get one run unstuck. The trajectory is the only artifact
that records what the model actually reached, so it remains the thing that
decides whether a run ships. The posture is PREVENT AND DETECT: any use of the
network by the MODEL is caught from the recorded trajectory and blocks the run.

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
`pip install pandas` fails to connect and installs nothing, while the same
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
# half that needs an OUTCOME rather than an intention: the enforced policy, the
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
    SHELL_TOOLS,
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
# Their weight: a confirmed install is ground truth of reach, and it holds
# whatever policy the run was under.
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
# environment died, agent setup failed -- leaves no trajectory, and blaming the
# MODEL for it reads as agent misbehaviour and sends an operator looking at a
# transcript that does not exist.
HAD_TRAJECTORY = True

# How many tool calls the trajectory held. Separate from HAD_TRAJECTORY because
# a trajectory can exist, parse, and still record nothing the agent did: a
# recorded run died on its first model call with "Weekly/Monthly Limit
# Exhausted" and published two steps, a prompt and an error. The audit printed
# "ok  0 tool call(s), no internet access" -- a green tick on a run that never
# happened -- and the trial still counted as an attempt, halving the job's mean
# reward. Neither is an internet finding, so neither blocks; both need saying.
TOOL_CALLS = 0

# The agent-phase network_mode Harbor enforced for this run, as scripts/run_task.sh
# resolved it with Harbor's own resolver: "allowlist", "no-network", "public",
# or None when the caller did not say. Only the first two put an enforcement
# boundary in the model's path; Harbor refuses a trial whose policy it cannot
# enforce, so a run that happened under one of them was bounded by it.
ENFORCED_POLICY: str | None = None
CLOSED_POLICIES = ("allowlist", "no-network")


def _c(code: str, s: str) -> str:
    return f"\033[{code}m{s}\033[0m" if sys.stdout.isatty() else s


def flag(step: int | None, tool: str, kind: str, detail: str, evidence: str,
         *, suppressed: bool = False, stopped_by: str | None = None) -> None:
    """Record one finding against trajectory step `step`.

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
                          whatever policy the run was under.
      package-install     no proof either way. The weaker claim, and the honest
                          one when a denied install printed a 403 and stopped.

    A SUPPRESSED install is the third case, and it is the one that cost a real
    run. `apt-get install -y -q chromium 2>&1 | tail -3` keeps the last three
    lines -- "Processing triggers for libc-bin" -- and drops every "Setting up"
    line above them, so the markers find nothing and the strongest available
    evidence reads as absent. Absent is not negative, so the flag is carried
    onto the finding and _verdict() decides what it means: under a closed
    enforced policy the install had nowhere to reach; without one, nothing could
    have stopped it and the agent removed the only other record of it.
    """
    for step, tool, args, response in normalise(traj):
        cmd = str(args.get("command") or "") if tool.split("__")[-1] in SHELL_TOOLS else ""
        for f in classify_tool(tool, args):
            evidence = cmd or str(args.get("url") or args.get("query") or "")
            out = response_text(response)
            # The PreToolUse hook writes its refusal where the command's output
            # would have been, so the step's own response says whether the
            # command ran at all. Nothing else can tell a hook refusal from a
            # network refusal, and calling one the other names a component that
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
    global HAD_TRAJECTORY, TOOL_CALLS, ENFORCED_POLICY
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("trajectory", type=Path)
    ap.add_argument("--json", type=Path, help="write findings here for grading")
    ap.add_argument("--warn-only", action="store_true",
                    help="report findings but exit 0 (does not block the run)")
    ap.add_argument("--strict", action="store_true",
                    help="fail on a BLOCKED attempt too, not just on one that "
                         "reached the internet")
    ap.add_argument("--enforced-policy", choices=("allowlist", "no-network", "public"),
                    help="the agent-phase network_mode Harbor enforced for this run")
    a = ap.parse_args(argv)
    ENFORCED_POLICY = a.enforced_policy

    if not a.trajectory.is_file():
        # No trajectory is not evidence of good behaviour, but it is also not
        # evidence of bad. The aborted-trial guard in run_task.sh already fails
        # loudly on a run that produced nothing, so this stays out of its way.
        HAD_TRAJECTORY = False
        print(f"  {_c('33', 'warn')}  no trajectory at {a.trajectory}")
        return 0

    try:
        traj = json.loads(a.trajectory.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"  {_c('31', 'FAIL')}  trajectory does not parse: {exc}")
        return 2

    scan(traj)
    total = TOOL_CALLS = len(normalise(traj))
    verdict = _report(a, total=total)
    return _exit_code(verdict, a)


# --------------------------------------------------------------------------
# VERDICT
#
# Three states an operator actually cares about, plus two that describe a run
# that did not happen. The old vocabulary had five words for the middle state
# alone (denied / unverified / setup) and no word at all for the case that now
# happens most: the PreToolUse hook refused the command, so it never ran and the
# network never saw it. A report that credits the network for it names a
# component that was not involved.
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
# needs a witness: the hook's refusal in the step's own output, or a closed
# agent-phase policy Harbor enforced. With neither, nothing bounded the run and
# the attempt most likely succeeded. That is the shape of the run that started
# all of this: it audited as "unverified", shipped, and had in fact installed
# Pillow, puppeteer and chromium from the open web.
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
    response. Here it is collapsed to one word for the run: "hook" only when
    EVERY finding agrees, so a mixed run reads "mixed" rather than crediting
    one layer with the other's work.
    """
    marks = {f.get("stopped_by") for f in FINDINGS if f["step"] is not None}
    marks.discard(None)
    if not marks:
        return None
    return marks.pop() if len(marks) == 1 else "mixed"


def _boundary_in_the_path() -> bool:
    """Was an enforced network boundary between the agent and the internet?"""
    return ENFORCED_POLICY in CLOSED_POLICIES


def _verdict() -> str:
    if not HAD_TRAJECTORY:
        return SETUP if FINDINGS else NO_ATTEMPT
    if not FINDINGS:
        return NO_ACTIVITY if not TOOL_CALLS else NO_ATTEMPT

    # Ground truth of reach: a command's own output shows an index was reached.
    if any(f["kind"] == "package-installed" for f in FINDINGS):
        return REACHED

    bounded = _boundary_in_the_path()

    # An install whose output was piped away proves nothing by itself. What
    # decides it is whether anything was in a position to stop it:
    #   hook refused it      the command never ran; there is nothing to verify
    #   a closed policy      Harbor refused the connection, so the install had
    #                        nowhere to reach -- and had it reached anyway, the
    #                        rule above would already have returned REACHED
    #   neither              nothing could have stopped it and the agent removed
    #                        the only other witness
    unproven = [f for f in FINDINGS if f.get("evidence_suppressed")
                and f.get("stopped_by") != "hook"]
    if unproven and not bounded:
        return REACHED

    # Can every attempt be SHOWN to have been stopped? Either the hook refused
    # it before it ran, or a closed policy was enforced and (per the rule above)
    # nothing landed. Otherwise nothing witnessed the outcome and saying
    # "blocked" would be inventing the evidence.
    steps = [f for f in FINDINGS if f["step"] is not None]
    if bounded or (steps and all(f.get("stopped_by") for f in steps)):
        return ATTEMPTED
    return UNVERIFIED


def _summary(verdict: str) -> str:
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
               "mixed": "refused by the egress guard or the enforced network policy"}.get(by)
        if how is None:
            # No step recorded a refusal in so many words; what bounded the
            # attempts was the policy, and the sentence says so rather than
            # crediting a refusal nobody recorded.
            return (f"the model reached for the internet {n} time(s) under the "
                    f"enforced '{ENFORCED_POLICY}' agent policy, so nothing left "
                    f"the sandbox")
        return (f"the model reached for the internet {n} time(s); every attempt "
                f"was {how or 'refused'}, and nothing left the sandbox")
    if verdict == UNVERIFIED:
        return (f"the model reached for the internet {n} time(s) and nothing "
                f"witnessed the outcome -- the agent phase ran under no enforced "
                f"closed policy, so the attempts most likely succeeded")
    if verdict == NO_ATTEMPT:
        return "the model never reached for the internet"
    if verdict == NO_ACTIVITY:
        return "the agent made no tool calls -- there was nothing to audit"
    return ("traffic was attempted before the agent ran (harbor's own setup) "
            "and refused; the model made no tool calls")


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


def _report(a, *, total: int) -> str:
    """Print the audit, write its JSON, return the verdict.

    One block per run, and every line earns its place: the verdict, then the
    evidence for it, then the policy that bounded the run. No banner, no repetition of the
    verdict in three different wordings -- that is what made the old output
    unreadable and, on one run, wrong.
    """
    verdict = _verdict()
    severity = SEVERITY[verdict]
    colour, label = _LABEL[severity]

    name = a.trajectory.parent.parent.name or a.trajectory.name
    print(f"== internet audit: {name} ==")
    # Wrapped, and continuation lines align under the first word rather than
    # under the label. An unwrapped verdict ran to 180 characters and folded
    # wherever the terminal happened to be, which buried the one line that
    # matters in the one place it is always read.
    for i, line in enumerate(textwrap.wrap(_summary(verdict), 72)):
        lead = f"  {_c(colour, label)}  " if i == 0 else " " * 8
        print(f"{lead}{line}")

    for f in FINDINGS:
        print(f"        {'step ' + str(f['step']):>9}  {f['detail']}")
        print(f"        {'':>9}  {f['evidence'].splitlines()[0][:150]}")

    if HAD_TRAJECTORY and verdict != NO_ACTIVITY:
        print(f"        {'policy':>9}  agent phase: {ENFORCED_POLICY or 'not stated'}"
              + ("" if _boundary_in_the_path() else " -- no enforced boundary"))

    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps({
            "verdict": verdict,
            "summary": _summary(verdict),
            "severity": severity,
            "attempted_internet": verdict in (REACHED, ATTEMPTED, UNVERIFIED, SETUP),
            "reached_internet": verdict == REACHED,
            "stopped_by": _stopped_by(),
            "tool_calls": total,
            "findings": FINDINGS,
            "enforced_policy": ENFORCED_POLICY,
        }, indent=2))
    return verdict


if __name__ == "__main__":
    sys.exit(main())
