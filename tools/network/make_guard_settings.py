#!/usr/bin/env python3
"""Build the Claude Code --settings file that carries the egress guard.

    tools/network/make_guard_settings.py <egress_rules.py> <out.json>

Called by scripts/run_task.sh, which passes the result to harbor as
`--ak config=<out.json>`. Harbor uploads it to
/tmp/claude-code-settings/settings.json inside the container and runs the CLI
with `--settings` pointed at it (harbor/agents/installed/claude_code.py:86,
:1875).

WHAT IT PRODUCES

A PreToolUse hook whose `command` is the whole of egress_rules.py, base64'd and
piped into python3. That is deliberate and the alternatives are worse:

  a bind mount     every bundle's docker-compose.yaml would need the same mount
                   added, and the overlay cannot supply one without knowing the
                   repo's host path.
  a file written   an agent running as root could overwrite it. The hook command
  into the image   is read out of settings.json at CLI startup, so once the run
                   is going there is nothing left on disk to tamper with.
  a second copy    two files spelling out "what counts as egress" drift, and the
  of the rules     drift is silent until a run is discarded for something the
                   hook let through.

base64 rather than a heredoc or an escaped literal: the payload then contains no
quote, newline, or backslash, so it survives JSON encoding, the shell, and
harbor's own upload without a single escaping question.

WHY THE MATCHER LISTS THE WEB TOOLS TOO

They are usually already gone -- run_task.sh passes --disallowedTools
WebSearch,WebFetch under isolation, so they never reach the tool list. The
matcher names them anyway so that a run configured without that flag (a hand
`harbor run`, DISALLOWED_TOOLS set empty) still refuses them rather than
silently allowing what the other layer was carrying.
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

# --------------------------------------------------------------------------
# DOES THIS HARBOR ACTUALLY DELIVER THE FILE
#
# Building the settings is half the job. The other half is harbor carrying it
# into the container, and harbor's agent kwargs are a mapping with a permissive
# floor: run_task.sh passes `--ak config=<path>`, and an older harbor whose
# ClaudeCode has no `config` parameter lets it fall through to `**kwargs` in
# BaseAgent.__init__ and drops it. No warning, no error, no `--settings` on the
# claude command line, and the run proceeds with no PreToolUse hook at all
# while run_task.sh reports "egress guard ON".
#
# That is not hypothetical. A delivered EC2 run of task
# 83d7e97e-2aed-4b23-a906-45f5a6a6a4da ran on harbor 0.20, which has no
# agents/options.py and no agents/capabilities.py; harbor invoked
#
#   claude --verbose --output-format=stream-json --disallowedTools ... --print
#
# with no `--settings`, the guard never loaded, and the session recorded zero
# PreToolUse events while the same commands exit 2 against these rules on the
# host. The local run of the same shape, on 0.23, shows `--settings
# /tmp/claude-code-settings/settings.json` in exactly that position.
#
# setup.sh pins harbor and that is the real fix. This is the check that says so
# out loud when the pin is not what is installed -- on a fresh box, a stale
# pipx, or an EC2 image built before the pin landed.
#
# WHAT IS PROBED, AND WHY THREE THINGS
#
# One attribute could be renamed by an upgrade and turn this into a guard that
# refuses every run. Three, each from a different layer, make the two ends
# unambiguous: all present is a harbor that delivers, none present is a harbor
# that predates the mechanism, and anything between is drift this cannot read
# and must not rule on. Only the middle answer is a guess, and it is reported
# as one.
# --------------------------------------------------------------------------

DELIVERS, ABSENT, UNREADABLE = "delivers", "absent", "unreadable"


def _probe_here() -> tuple[str, str]:
    """The three markers, read out of whatever harbor THIS interpreter can see."""
    try:
        from harbor.agents.installed.claude_code import ClaudeCode
    except Exception as exc:                                  # noqa: BLE001
        return UNREADABLE, f"harbor's claude-code agent will not import ({exc})"

    marks = {}
    # The agent exposes a native config source at all (BaseInstalledAgent).
    marks["ClaudeCode.config_source"] = hasattr(ClaudeCode, "config_source")
    # It declares that it accepts one (agents/capabilities.py). Harbor itself
    # raises on a config passed to an agent where this is False, so reading it
    # is reading harbor's own answer rather than guessing at one.
    caps = getattr(ClaudeCode, "capabilities", None)
    marks["capabilities.native_config"] = getattr(caps, "native_config", False) is True
    # `config` is a real option key, so --ak config= is validated rather than
    # swallowed (agents/options.py, which also sets extra="forbid").
    try:
        marks["options.config"] = "config" in ClaudeCode.options_model.model_fields
    except Exception:                                         # noqa: BLE001
        marks["options.config"] = False

    have = sorted(k for k, v in marks.items() if v)
    missing = sorted(k for k, v in marks.items() if not v)
    if not missing:
        return DELIVERS, "config_source, native_config and the config option are all present"
    if not have:
        return ABSENT, ("none of config_source, capabilities.native_config or the "
                        "config option exist on this harbor")
    return UNREADABLE, f"harbor has {', '.join(have)} but not {', '.join(missing)}"


def harbor_native_config_support() -> tuple[str, str]:
    """Whether the installed harbor will carry `--ak config=` into the agent.

    Returns (verdict, one-line detail). Pure inspection: nothing is imported
    that harbor does not import itself at startup, and nothing is run.

    HARBOR IS USUALLY NOT IMPORTABLE FROM HERE, and that is the normal case
    rather than an error. setup.sh installs it with pipx, so it lives in a venv
    of its own and run_task.sh only ever calls it as a command on PATH. Probing
    with this interpreter would then report "no module named harbor" on a
    perfectly healthy machine -- which reads as drift and is nothing of the
    kind. So when the import fails, the question is put to harbor's OWN
    interpreter, found the same way scripts/patch_harbor.py finds it: from the
    `harbor` entry point on PATH back to its venv's python.

    That interpreter imports this file and calls _probe_here(), so there is one
    copy of the marker logic and it is this one. `--probe-here` is the entry
    point it uses and the reason the delegation cannot recurse.
    """
    verdict, detail = _probe_here()
    if verdict != UNREADABLE or "will not import" not in detail:
        return verdict, detail

    import shutil
    import subprocess

    harbor_bin = shutil.which("harbor")
    if not harbor_bin:
        return UNREADABLE, "harbor is not on PATH and not importable here"

    venv_bin = Path(harbor_bin).resolve().parent
    python = next((c for c in (venv_bin / "python3", venv_bin / "python")
                   if c.exists()), None)
    if python is None:
        return UNREADABLE, f"no interpreter beside the harbor entry point at {venv_bin}"

    try:
        r = subprocess.run(
            [str(python), str(Path(__file__).resolve()), "--probe-here"],
            capture_output=True, text=True, timeout=30,
        )
    except Exception as exc:                                  # noqa: BLE001
        return UNREADABLE, f"harbor's interpreter could not be asked ({exc})"

    line = (r.stdout or "").strip().split("\n")[-1]
    head, _, tail = line.partition("\t")
    if head in (DELIVERS, ABSENT, UNREADABLE):
        return head, tail or "(no detail)"
    return UNREADABLE, f"harbor's interpreter answered nothing usable ({line[:120]!r})"


def check_harbor() -> int:
    """`--check-harbor`: 0 delivers, 1 positively does not, 3 cannot tell."""
    verdict, detail = harbor_native_config_support()
    if verdict == DELIVERS:
        return 0
    if verdict == ABSENT:
        print(f"[egress-guard] harbor will NOT deliver the guard: {detail}",
              file=sys.stderr)
        return 1
    print(f"[egress-guard] cannot confirm harbor delivers the guard: {detail}",
          file=sys.stderr)
    return 3

# Tools the hook is asked to judge. A regex over tool names, matched by Claude
# Code against each call before it runs.
MATCHER = "Bash|WebFetch|WebSearch"

# Generous, because the cost of the two failure modes is not symmetric: a hook
# that times out is skipped, and a skipped hook is a command that runs. The
# guard itself is a few milliseconds of regex over one string, so this only ever
# has to cover a cold interpreter start.
TIMEOUT_SEC = 15


def build(guard: Path) -> dict:
    blob = base64.b64encode(guard.read_bytes()).decode("ascii")
    return {
        "hooks": {
            "PreToolUse": [
                {
                    "matcher": MATCHER,
                    "hooks": [
                        {
                            "type": "command",
                            # Command substitution, NOT a pipe into `python3 -`.
                            # The hook's own JSON payload arrives on stdin and
                            # egress_rules.main() reads it there, so stdin is
                            # not available to carry the program as well;
                            # `python3 -c "$(...)"` puts the program in argv and
                            # leaves fd 0 alone.
                            #
                            # No temp file either, so there is nothing on disk
                            # for an agent running as root to rewrite between
                            # one Bash call and the next.
                            #
                            # The single quotes are safe unconditionally:
                            # base64's alphabet is [A-Za-z0-9+/=] and contains
                            # no quote, backslash or newline.
                            "command": (
                                f"python3 -c \"$(printf %s '{blob}' | base64 -d)\""
                            ),
                            "timeout": TIMEOUT_SEC,
                        }
                    ],
                }
            ]
        }
    }


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["--check-harbor"]:
        return check_harbor()
    if argv[:1] == ["--probe-here"]:
        # Run BY harbor's interpreter, read by harbor_native_config_support().
        # One tab-separated line on stdout; never delegates onward.
        print("\t".join(_probe_here()))
        return 0
    if len(argv) != 2:
        print(__doc__.strip().split("\n\n")[1], file=sys.stderr)
        return 2
    guard, out = Path(argv[0]), Path(argv[1])
    if not guard.is_file():
        print(f"egress guard not found: {guard}", file=sys.stderr)
        return 2
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(build(guard), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
