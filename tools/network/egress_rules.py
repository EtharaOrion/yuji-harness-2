#!/usr/bin/env python3
"""Does this shell command reach the internet? One rule set, two callers.

    tools/network/detect_internet_use.py   post-hoc audit of a finished trajectory
    this file's __main__                   a Claude Code PreToolUse hook, live,
                                           inside the container

WHY THE RULES LIVE HERE AND NOT IN THE AUDITOR

The audit used to own these rules outright. That was fine while the audit was
the only thing reading them, and stopped being fine the moment the run started
DENYING a command as well as reporting it: two copies of "what counts as
egress" drift, and the direction they drift in is the expensive one -- a
command the hook lets through and the audit later blocks costs a whole graded
run, discarded after the fact for something that could have been refused in
the turn it was typed.

So the classification is a pure function of the command string, kept in one
place, imported by the auditor on the host and shipped into the container as
the hook. Neither caller may add a rule of its own.

WHAT THE TWO CALLERS DO DIFFERENTLY

Only the OUTCOME half differs, and it differs because only one of them has an
outcome to read. classify() sees a command and can say the model REACHED for a
package index; it cannot say whether the index answered. The auditor knows,
because it holds the command's output (see INSTALL_SUCCESS_MARKERS in
detect_internet_use.py) -- and the hook does not need to know, because it runs
before the command does.

FAIL-OPEN IS DELIBERATE (hook mode only)

The routing table is the enforcement boundary: tools/network/egress-proxy/
overlay.yaml leaves the agent's network with no gateway, so a command this hook
misses still cannot reach anything. The hook exists to turn a 30-second
connection timeout into an immediate, legible refusal, which is worth several
turns per run but is not worth a benchmark that dies whenever this file has a
bug. Anything unexpected in hook mode therefore exits 0 and lets the command
run into the wall it was always going to hit.
"""

from __future__ import annotations

import json
import re
import shlex
import sys
from pathlib import Path

# --------------------------------------------------------------------------
# WHAT IS INTERNAL
# --------------------------------------------------------------------------

# Reachable over the compose bridge, not over the internet. A curl at any of
# these is the bundle working as designed.
#
# Peer definition: tools/network/egress-proxy/overlay.yaml's NO_PROXY is the same
# set applied to the run rather than to the transcript, and squid.conf's comment
# makes keeping the two in agreement a standing pact.
# scripts/tests/test_egress_allowlist.py enforces it.
#
# host.docker.internal is deliberately absent -- `internal: true` leaves the
# bridge with no gateway, so the host-gateway address has no route no matter
# what the proxy settings say, and a call to it is a real failed egress attempt.
# 0.0.0.0 IS here: as a DESTINATION it is a local bind address, and
# `curl 0.0.0.0:8000` against a server the agent just started is ordinary local
# work. is_internal() does not otherwise cover it (it matches 127.* and
# localhost, not 0.0.0.0).
# judge is the bundle's rubric judge container (tools/judge): a sibling on the
# same bridge that test.sh calls. Its own way out goes through judge-proxy, on a
# network main never joins, so a curl at it leaves nothing.
INTERNAL_HOSTS = {
    "light-servers", "localhost", "127.0.0.1", "0.0.0.0", "::1", "main", "judge",
}

# --------------------------------------------------------------------------
# VERBS
# --------------------------------------------------------------------------

# Tool names that are internet access by definition -- no argument inspection
# can make them local.
WEB_TOOLS = {"WebSearch", "WebFetch"}

# Shell verbs that move bytes to or from a host named on the command line.
FETCHERS = {
    "curl", "wget", "nc", "ncat", "netcat", "telnet",
    "ssh", "scp", "sftp", "rsync", "ftp", "svn",
}

# Shell verbs that reach a package index. No host on the command line, so these
# are judged by their flags instead.
INSTALLERS = {
    ("pip", "install"), ("pip3", "install"), ("uv", "pip"), ("uv", "add"),
    ("npm", "install"), ("npm", "i"), ("npm", "ci"), ("yarn", "add"),
    ("pnpm", "add"), ("pnpm", "install"), ("npx", ""),
    ("apt", "install"), ("apt-get", "install"), ("apk", "add"),
    ("yum", "install"), ("dnf", "install"), ("brew", "install"),
    ("gem", "install"), ("cargo", "install"), ("go", "get"),
}

# Flags that pin an installer to something already on disk. `pip install
# --no-index ./wheel` touches no index and is not egress.
OFFLINE_FLAGS = {"--no-index", "--offline", "--frozen", "--cached", "--no-download"}

# git only reaches the network for these; `git status` and `git log` do not.
GIT_NETWORK_SUBCOMMANDS = {
    "clone", "fetch", "pull", "push", "remote", "ls-remote", "submodule",
}

# Commands that RUN another command. Their own name says nothing about the
# network; the verb that matters is somewhere to their right.
#
# This set is the fix for the hole that let `timeout 600 npm i puppeteer@23`
# through a live audit: the scan read `timeout` as the verb, found it in no rule
# table, and reported nothing while 102 packages came down from the registry.
# Anything that can prefix a command belongs here, because the cost of a missing
# entry is exactly that -- silence.
WRAPPERS = {
    "sudo", "doas", "timeout", "env", "nohup", "nice", "ionice", "stdbuf",
    "time", "command", "builtin", "exec", "setsid", "unbuffer", "xargs",
    "script", "chrt", "eatmydata", "proxychains", "proxychains4",
}

# Shells that take a command as a STRING argument, which shlex hands back as a
# single opaque token. `bash -c "pip install x"` carries no `pip` token at the
# top level, so the payload is re-classified as a command in its own right.
SHELLS = {"sh", "bash", "zsh", "dash", "ksh", "ash", "busybox"}

# Verbs whose operands are text to MATCH, not code to run.
#
# Only INLINE_NETWORK_HINTS consults this, and only to stop a command being
# refused for SPELLING a hint. `grep -Eo 'fetch\(|XMLHttpRequest' page.html` is
# a page being checked for exactly the thing the closed world forbids -- one
# bundle's brief asks for a page that "reaches out for nothing to draw itself",
# so auditing the output for those two words is the model doing the task -- and
# the guard's answer to it was "there is no route out of this container", which
# is true, unrelated, and costs the check.
#
# A pattern cannot open a socket. Every other rule in this file still reads the
# whole command, so a grep that also runs a fetcher is caught by the fetcher
# rule exactly as before.
#
# sed and awk are deliberately absent: their operands are PROGRAMS, and an awk
# program can call system() and shell out. Only the pure matchers are here.
SEARCH_VERBS = {"grep", "egrep", "fgrep", "rg", "ag", "ack"}

# Network access smuggled through an interpreter. Matched on the source text of
# a `python3 -c` / `node -e` payload rather than on the command name, which is
# why these are substrings and not verbs.
INLINE_NETWORK_HINTS = (
    "urllib.request", "urllib2", "requests.get", "requests.post", "requests.request",
    "httpx.", "aiohttp", "socket.create_connection", "http.client",
    "urlopen", "fetch(", "XMLHttpRequest",
)

# URLs that are IDENTIFIERS rather than addresses. XML namespaces are spelled
# as URLs by the spec and are never dereferenced: an SVG generator writes
# xmlns="http://www.w3.org/2000/svg" without a socket ever opening, and every
# .docx these bundles unpack is full of schemas.openxmlformats.org.
#
# Exempt from the bare URL SWEEP ONLY. A fetcher verb aimed at one of these
# hosts still trips the `fetch` rule, because `curl http://www.w3.org/x` is a
# real request whatever the host is famous for. That split is the whole point:
# the sweep is a string match and can afford to be wrong in the quiet
# direction; the verb rules cannot.
#
# Seen in the wild: a run was blocked because the agent printed
# `'http://www.w3.org/2000/svg'` while CHECKING ITS OWN OUTPUT had no external
# references. A false positive there costs a clean run; the miss it risks is a
# curl the verb rules catch anyway.
NAMESPACE_URI_PREFIXES = (
    "www.w3.org/1999/",
    "www.w3.org/2000/svg",
    "www.w3.org/2001/XMLSchema",
    "www.w3.org/XML/1998/",
    "schemas.openxmlformats.org/",
    "schemas.microsoft.com/",
    "purl.org/dc/",
)

_NAMESPACE_RE = re.compile(
    r"https?://(?:" + "|".join(re.escape(x) for x in NAMESPACE_URI_PREFIXES) + ")"
)

URL_RE = re.compile(r"\b(?:https?|ftp|ssh)://([^\s/'\"\\)>;|]+)", re.I)

# `cmd << EOF` / `cmd <<-'EOF'` / `cmd <<"EOF"`. The delimiter is group 2; the
# quoting around it only decides whether the shell expands the body, which is
# not this module's business.
HEREDOC_RE = re.compile(r"<<[-~]?\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\1")

# `FOO=bar cmd ...` -- a shell assignment prefix, not the command.
_ASSIGN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")

# A bare operand a wrapper takes before the command it runs: `timeout 600 ...`,
# `nice -n 10 ...`, `timeout 5m ...`.
_DURATION_RE = re.compile(r"\d+(?:\.\d+)?[smhd]?$")

# Shell grouping that starts a new command. Kept as a capture group so
# re.split returns the delimiters and _segments can tell them from words.
#
# BRACES ARE DELIBERATELY ABSENT, and the omission is load-bearing. `{` and `}`
# were here for shell brace groups (`{ cmd; }`) and cost a real run: a sidecar
# health probe,
#
#   curl -s -m 2 -o /dev/null -w "$p:%{http_code}\n" http://light-servers:$p/
#
# split at the braces of curl's own format string, which tore the URL off the
# end of the curl segment. The command was then reported as "curl with no
# resolvable host operand" -- a BLOCKING finding, on the most ordinary thing an
# agent does in these bundles. ${VAR}, awk '{print}' and jq '{a:.b}' are the
# same shape. A brace group is rare in agent-written shell and its inner `;`
# already separates the commands inside it, so nothing is lost by leaving it out.
_GROUPING_RE = re.compile(r"(\$\(|[()`])")

# Output-suppressing tails. These do not change what a command REACHES, only
# what it can be proven to have reached afterwards -- `apt-get install chromium
# | tail -3` keeps the "Processing triggers" lines and drops every "Setting up"
# line the auditor reads as proof. Recorded on the finding so the auditor can
# refuse to downgrade a suppressed install to a mere attempt.
_SUPPRESSOR_RE = re.compile(
    r"\|\s*(?:tail|head)\b"          # ... | tail -3
    r"|(?:^|\s)-q(?:q)?\b"           # -q, -qq
    r"|--quiet\b|--silent\b"
    r"|>\s*/dev/null"
)


class Finding:
    """One reason a command is not local. Ordered fields, cheap to construct."""

    __slots__ = ("kind", "detail", "suppressed")

    def __init__(self, kind: str, detail: str, suppressed: bool = False):
        self.kind = kind
        self.detail = detail
        # True when the command pipes its own output away. Only meaningful for
        # the installer kinds, where the auditor otherwise reads a missing
        # success marker as "the install did not land".
        self.suppressed = suppressed

    def __eq__(self, other):
        return (isinstance(other, Finding)
                and (self.kind, self.detail) == (other.kind, other.detail))

    def __hash__(self):
        return hash((self.kind, self.detail))

    def __repr__(self):
        return f"Finding({self.kind!r}, {self.detail!r})"


# --------------------------------------------------------------------------
# HOSTS
# --------------------------------------------------------------------------

def host_of(token: str) -> str | None:
    """Hostname a fetcher operand points at, or None if it names no host."""
    m = URL_RE.search(token)
    if m:
        return m.group(1).split("@")[-1].split(":")[0].lower()
    # scp/ssh shorthand: user@host:/path, or a bare host operand.
    if "@" in token and ":" in token.split("@", 1)[1]:
        return token.split("@", 1)[1].split(":", 1)[0].lower()
    return None


def is_internal(host: str) -> bool:
    if host in INTERNAL_HOSTS:
        return True
    # Compose service aliases and loopback ranges are internal by construction.
    return (host.startswith("127.")
            or host.endswith(".local")
            or host.endswith(".internal"))


def _addressed_hosts(cmd: str) -> list[str]:
    """Every host an absolute URL in `cmd` names, namespace URIs excluded.

    The same sweep classify() runs, factored out so the inline-network rule can
    ask what the command is actually pointed at instead of matching a word.
    """
    out = []
    for m in URL_RE.finditer(cmd):
        if _NAMESPACE_RE.match(cmd, m.start()):
            continue
        out.append(m.group(1).split("@")[-1].split(":")[0].lower())
    return out


# --------------------------------------------------------------------------
# LEXING
# --------------------------------------------------------------------------

def split_heredocs(cmd: str) -> tuple[str, list[str]]:
    """Split `cmd` into (the shell to lex, the heredoc bodies lifted out of it).

    A heredoc body is data being fed to a program, not shell words. shlex is a
    POSIX word lexer with no heredoc rule, so it reads the body as ordinary
    shell text and the first apostrophe or triple-quote in it raises "No
    closing quotation" -- which classify()'s fail-closed branch then turns into
    a blocking finding. `python3 - <<'PY' ... PY` is how an agent writes most
    of its multi-line edits, so that fired on ordinary local work.

    The bodies are returned rather than dropped: they are still audited, by the
    raw-command sweeps in classify() and by _scan_body().
    """
    if "<<" not in cmd:
        return cmd, []
    lines = cmd.split("\n")
    kept: list[str] = []
    bodies: list[str] = []
    i = 0
    while i < len(lines):
        kept.append(lines[i])
        line = lines[i]
        i += 1
        # One line can open several bodies (`cmd <<A <<B`); they arrive in the
        # order the redirections were written.
        for m in HEREDOC_RE.finditer(line):
            delim = m.group(2)
            body: list[str] = []
            while i < len(lines) and lines[i].strip() != delim:
                body.append(lines[i])
                i += 1
            i += 1          # the terminator line, or one past the last line
            bodies.append("\n".join(body))
    return "\n".join(kept), bodies


def _known_verb(word: str) -> bool:
    """Is this token a verb any rule below would act on?"""
    return (word in FETCHERS
            or word == "git"
            or word in SHELLS
            or any(word == tool for tool, _ in INSTALLERS))


def real_verb(seg: list[str]) -> tuple[str, list[str]]:
    """The command a segment actually runs, seen through wrappers.

    Returns (verb, args-after-verb). ("", []) when the segment is empty.

    Two passes, and the second is the one that earns its keep:

      1. Skip assignment prefixes (`DEBIAN_FRONTEND=noninteractive ...`) and
         WRAPPERS along with the options and bare operands they take, so
         `timeout 600 npm i x` yields ("npm", ["i", "x"]).

      2. If wrappers were stripped and what they left is NOT a verb this module
         knows, scan on to the first token that is. `sudo -u root apt-get
         install x` stops pass 1 at `root` -- sudo's operand, not a command --
         and reimplementing each wrapper's option grammar to know that is not
         worth it. The scan is gated on a wrapper having been present precisely
         so it cannot fire on an ordinary command: `grep -rn pip install .`
         keeps `grep` as its verb and reports nothing.
    """
    i, n = 0, len(seg)
    saw_wrapper = False
    while i < n:
        token = seg[i]
        if _ASSIGN_RE.match(token) and not token.startswith("-"):
            i += 1
            continue
        if Path(token).name.lower() not in WRAPPERS:
            break
        saw_wrapper = True
        i += 1
        # The wrapper's own options and any bare numeric operand it takes
        # before the command (`timeout -k 5 600 cmd`, `nice -n 10 cmd`).
        while i < n and (seg[i].startswith("-") or _DURATION_RE.fullmatch(seg[i])):
            i += 1

    if i >= n:
        return "", []

    verb = Path(seg[i]).name.lower()
    if saw_wrapper and not _known_verb(verb):
        for j in range(i + 1, n):
            candidate = Path(seg[j]).name.lower()
            if _known_verb(candidate):
                return candidate, seg[j + 1:]
    return verb, seg[i + 1:]


# --------------------------------------------------------------------------
# CLASSIFICATION
# --------------------------------------------------------------------------

def _judge_segment(seg: list[str], suppressed: bool, out: list[Finding],
                   depth: int) -> None:
    """Read verbs and their flags out of one lexed command segment."""
    verb, args = real_verb(seg)
    if not verb:
        return

    if verb in FETCHERS:
        hosts = [h for h in (host_of(a) for a in args) if h]
        external = [h for h in hosts if not is_internal(h)]
        if external:
            out.append(Finding("fetch", f"{verb} to {', '.join(sorted(set(external)))}"))
        elif not hosts:
            # A fetcher with no parseable host operand still ran a fetcher;
            # report it rather than assume it was local.
            out.append(Finding("fetch", f"{verb} with no resolvable host operand"))

    if verb == "git" and args and args[0].lower() in GIT_NETWORK_SUBCOMMANDS:
        out.append(Finding("vcs-network", f"git {args[0]}"))

    for tool, sub in INSTALLERS:
        if verb != tool:
            continue
        if sub and not (args and args[0].lower() == sub):
            continue
        if any(f in args for f in OFFLINE_FLAGS):
            continue
        name = f"{verb} {sub}".strip()
        out.append(Finding("package-install",
                           f"{name} reaches a package index", suppressed))
        break

    # `bash -c "<command>"` hides a whole command inside one token. Recurse
    # once: deeper nesting is not a pattern an agent produces by accident, and
    # an unbounded recursion here would be a denial-of-service on the hook.
    if verb in SHELLS and depth < 1:
        for k, arg in enumerate(args):
            if arg == "-c" and k + 1 < len(args):
                out.extend(classify(args[k + 1], _depth=depth + 1))
                break


def _scan_body(body: str, suppressed: bool, out: list[Finding], depth: int) -> None:
    """Audit a lifted heredoc body with the same verb walk as the command line.

    A body fed to `bash` is shell, and dropping it would hand the model a place
    to keep a `pip install` where nothing looks. So each line is lexed on its
    own -- line by line because shlex does not treat a newline as a separator,
    and a whole body lexed at once collapses into a single run-on segment whose
    first word is the only verb anyone reads.

    A line that will not lex is skipped rather than flagged. Bodies are usually
    not shell at all (a Python payload, a file being written), and there the
    quoting that defeats shlex is just the payload's own syntax. That is not the
    hole it looks like: URL_RE and INLINE_NETWORK_HINTS already ran over the raw
    command with its bodies intact, so an `http://` or a `urlopen` in there is
    caught whatever the body turns out to be.
    """
    for line in body.split("\n"):
        try:
            tokens = shlex.split(line, comments=True)
        except ValueError:
            continue
        for seg in _segments(tokens):
            _judge_segment(seg, suppressed, out, depth)


def _segments(tokens: list[str]) -> list[list[str]]:
    """Split on shell separators so `ls && curl x` is seen as two commands.

    Grouping punctuation is a separator too, and it has to be prised off the
    word it clings to. shlex has no grammar for `(` and `)`, so
    `(apt-get install -y chromium)` lexes as ['(apt-get', ..., 'chromium)'] and
    `(apt-get` matches no verb -- which is how a real run installed chromium
    inside a subshell while the audit reported nothing. `$(...)` and backticks
    hide a command the same way.
    """
    segments: list[list[str]] = [[]]
    for tok in tokens:
        if tok in ("&&", "||", ";", "|", "&"):
            segments.append([])
            continue
        for part in _GROUPING_RE.split(tok):
            if not part:
                continue
            if _GROUPING_RE.fullmatch(part):
                segments.append([])
            else:
                segments[-1].append(part)
    return [s for s in segments if s]


def _search_operands(cmd: str) -> list[str]:
    r"""Operands handed to a text-search verb: patterns and the files to read.

    Only INLINE_NETWORK_HINTS reads this. Every other rule still sees the whole
    command, so the worst a mistake here can do is fail to raise ONE finding on
    a command that named no host anyway.

    LEXED WITH punctuation_chars, which the rest of the module does not use.
    Two properties are needed at once and only this mode has both: a quoted
    regex stays a single token, parentheses and all, so `'fetch\(|XMLHttpRequest'`
    is not torn into pieces; and a separator glued to the word before it comes
    out on its own, so `a.html; python3 -c ...` ends the grep rather than
    feeding the python payload in as one of its operands. Plain
    shlex.split gives the first and not the second, and collecting a command as
    if it were a search pattern is the one way this helper could go quiet on
    real egress.

    A line that will not lex yields nothing, which leaves the hint rule fully
    closed for that line.
    """
    _SEPARATORS = {"&&", "||", ";", ";;", "|", "&", "(", ")", "<", ">", ">>", "<<"}
    lex_src, _ = split_heredocs(cmd)
    out: list[str] = []
    # LINE BY LINE, for the reason _scan_body gives: shlex reads a newline as
    # plain whitespace, so a multi-line command lexes into one run-on segment
    # and only its first verb is ever seen. The grep on line three of a
    # four-line self-check is invisible to a whole-string walk.
    for line in lex_src.split("\n"):
        try:
            lexer = shlex.shlex(line, posix=True, punctuation_chars=True)
            lexer.whitespace_split = True
            tokens = list(lexer)
        except ValueError:
            continue
        seg: list[str] = []
        for tok in [*tokens, ";"]:
            if tok in _SEPARATORS:
                _collect_search_operands(seg, out)
                seg = []
            else:
                seg.append(tok)
    return out


def _collect_search_operands(seg: list[str], out: list[str]) -> None:
    """Append `seg`'s operands to `out` when `seg` runs a text-search verb."""
    if not seg:
        return
    verb, args = real_verb(seg)
    if verb and Path(verb).name.lower() in SEARCH_VERBS:
        out.extend(a for a in args if not a.startswith("-"))


def classify(cmd: str, *, _depth: int = 0) -> list[Finding]:
    """Every reason this shell command is not local work. Empty == local.

    Pure: the command string is the only input, so the hook and the auditor
    cannot disagree about a command they both see.
    """
    out: list[Finding] = []
    if not cmd.strip():
        return out

    suppressed = bool(_SUPPRESSOR_RE.search(cmd))

    # Any absolute URL in the command is the strongest signal available, and it
    # survives quoting that would defeat the token walk below.
    hosts = _addressed_hosts(cmd)
    for host in hosts:
        if not is_internal(host):
            out.append(Finding("external-url", f"command references {host}"))

    # An interpreter payload that speaks HTTP, judged by what it is POINTED AT
    # rather than by the fact that it speaks HTTP.
    #
    # The hint list is a substring match, and on its own it cannot tell
    # `urlopen("https://pypi.org/...")` from `urlopen("http://light-servers:9015/mcp")`.
    # The second is the closed world working exactly as designed -- the sidecars
    # ARE the task's data, and an agent that finds the MCP client easier to
    # drive over raw HTTP than through the tool list is doing nothing wrong.
    # Blocking it cost a delivered run seven "internet attempts", every one of
    # them a call to a compose service on the bridge.
    #
    # So: when the command names hosts and every one of them is internal, the
    # payload has nowhere external to go and the hint is not evidence. When it
    # names an external host, the `external-url` rule above has already fired
    # and this adds the detail. When it names NO host at all -- a URL built at
    # runtime, a variable, a base64 blob -- nothing here can see the target, and
    # the hint stands.
    #
    # That last branch is the fail-closed one and it stays fail-closed on
    # purpose. The direction of the trade is the same one NAMESPACE_URI_PREFIXES
    # makes just above: this sweep is a string match over a whole command and
    # can afford to go quiet where the target is visible and harmless, because
    # the routing table -- not this file -- is what actually removes the route.
    #
    # Searched-for text is lifted out first, for the reason SEARCH_VERBS gives.
    # The removal touches this rule and nothing else: `hosts` above and every
    # verb walk below still read the command whole.
    hinted = cmd
    for operand in _search_operands(cmd):
        if any(h in operand for h in INLINE_NETWORK_HINTS):
            hinted = hinted.replace(operand, " ")

    if not (hosts and all(is_internal(h) for h in hosts)):
        for hint in INLINE_NETWORK_HINTS:
            if hint in hinted:
                out.append(Finding("inline-network",
                                   f"interpreter payload uses {hint}"))

    # Walk the command as tokens so we can read verbs and their flags. A command
    # we cannot lex is reported rather than skipped: silently passing an
    # unparseable command would be the one hole worth having none of.
    #
    # Heredoc bodies come out first -- see split_heredocs for why leaving them
    # in made that fail-closed branch fire on benign local edits -- and are
    # audited on their own terms.
    lex_src, bodies = split_heredocs(cmd)
    for body in bodies:
        _scan_body(body, suppressed, out, _depth)
    try:
        tokens = shlex.split(lex_src, comments=True)
    except ValueError:
        out.append(Finding("unparseable",
                           "command could not be lexed; not provably local"))
        return _collapse(out)

    for seg in _segments(tokens):
        _judge_segment(seg, suppressed, out, _depth)

    return _collapse(out)


def _collapse(out: list[Finding]) -> list[Finding]:
    """One command, one finding per distinct act.

    A `curl https://host/x` legitimately trips both the URL sweep and the token
    walk, and `git clone <url>` trips the URL sweep as well as the vcs rule.
    Both describe the same act, so the generic `external-url` finding is dropped
    when a more specific rule already fired -- the report should read as a list
    of things the model did, not a list of rules that matched.
    """
    specific = any(f.kind != "external-url" for f in out)
    kept = [f for f in out if not (specific and f.kind == "external-url")]
    # Preserve order, drop exact duplicates.
    seen: set[Finding] = set()
    unique = []
    for f in kept:
        if f not in seen:
            seen.add(f)
            unique.append(f)
    return unique


def classify_tool(tool_name: str, tool_input: dict) -> list[Finding]:
    """Findings for one tool call, whatever the tool.

    MCP tools are the closed world and are never egress, however they are
    named. Everything else is judged by name (the web tools) or by its command
    (Bash).
    """
    if tool_name.startswith("mcp__"):
        return []
    base = tool_name.split("__")[-1]
    if tool_name in WEB_TOOLS or base in WEB_TOOLS:
        target = tool_input.get("url") or tool_input.get("query") or ""
        return [Finding("web-tool", f"{tool_name} called on {target}"[:200])]
    if base == "Bash":
        return classify(str(tool_input.get("command") or ""))
    return []


# --------------------------------------------------------------------------
# HOOK MODE
#
# Shipped into the container by scripts/run_task.sh, which base64s this file
# into the `command` of a PreToolUse hook in Claude Code's --settings JSON.
# There is no mount and no second copy on disk: the bytes the hook runs are the
# bytes the auditor imports.
# --------------------------------------------------------------------------

# The first line of DENIAL, split out so the audit can recognise its own work.
#
# When this hook refuses a command, the command never runs and the step's
# response is this text instead of any program's output. That is the difference
# between "the proxy denied it" and "it never reached the proxy", and without a
# marker the audit cannot tell them apart -- it reported a hook refusal as
# "the egress proxy refused every attempt", which is a sentence about a
# component that never saw the request.
DENIAL_MARKER = "BLOCKED: this command reaches the public internet"

DENIAL = """\
{marker}, and this task is closed-world.
{reasons}

There is no route out of this container. Retrying, or reaching for a different
fetcher or package manager, will not work -- the network has no gateway, not a
filter you can talk your way past.

Everything the task needs is already here:
  * the MCP tools (LightGmail, LightDrive, LightBudget, LightSlack, ...) hold
    the world state -- list, search and read through them
  * /workspace/data holds the attachments, read-only
  * the Python stdlib and the packages already installed in this image
  * your own knowledge, for anything that is general rather than task-specific

Solve it with those. Do not install anything and do not fetch anything.\
"""


def _hook(stdin_text: str) -> int:
    """PreToolUse: exit 2 blocks the call and shows stderr to the model.

    Exit 2 rather than a JSON permissionDecision because it is the older and
    more widely supported of the two contracts, and this hook has to keep
    working across CLI versions the bundles pre-bake independently.
    """
    payload = json.loads(stdin_text)
    findings = classify_tool(
        str(payload.get("tool_name") or ""),
        payload.get("tool_input") or {},
    )
    if not findings:
        return 0
    reasons = "\n".join(f"  - {f.detail}" for f in findings)
    sys.stderr.write(DENIAL.format(marker=DENIAL_MARKER, reasons=reasons) + "\n")
    return 2


def main(argv=None) -> int:
    try:
        return _hook(sys.stdin.read())
    except Exception as exc:                                  # noqa: BLE001
        # Fail open -- see the module docstring. The router already blocks the
        # command; a broken hook must not also block the run.
        sys.stderr.write(f"[egress-guard] disabled for this call: {exc}\n")
        return 0


if __name__ == "__main__":
    sys.exit(main())
