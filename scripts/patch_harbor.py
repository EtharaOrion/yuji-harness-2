#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ANCHOR = '            format="--permission-mode={value}",\n        ),'
PATCH = """
        CliFlag(
            "thinking",
            cli="--thinking",
            type="str",
        ),
        CliFlag(
            "thinking_display",
            cli="--thinking-display",
            type="str",
        ),"""
ALREADY_PATCHED_MARKER = '"thinking_display"'

# harbor 0.23.0 moved agent flags off CLI_FLAGS lists onto pydantic fields of
# ClaudeCodeOptions, so ANCHOR above can never match there. It also set
# extra="forbid" (agents/options.py), which turns an unknown --ak key from
# "ignored" into a hard validation error, so run_task.sh:1027's
# `--ak thinking=adaptive` fails the run up front unless this field exists.
# thinking_display became NATIVE in 0.23.0 and is deliberately not re-added.
ANCHOR_THINKING_PYD = '''    thinking_display: Annotated[
        Literal["summarized", "omitted"] | None,
        Cli("--thinking-display"),
    ] = Field(default=None, description="How thinking is displayed.")'''

REPLACEMENT_THINKING_PYD = ANCHOR_THINKING_PYD + '''
    # harbor-patch: thinking flag
    thinking: Annotated[str | None, Cli("--thinking")] = Field(
        default=None, description="Thinking mode."
    )'''

ALREADY_PATCHED_MARKER_THINKING_PYD = "# harbor-patch: thinking flag"


# Indentation is load-bearing. All three anchors sit inside a `try:` inside an
# `async def`, at 12 and 20 spaces; they used to be written 4 short. Anchor 1 is
# one line, so it still matched as a SUBSTRING (the missing spaces sit to its
# left) and the patch reported "applied", but the replacement's remaining lines
# landed a level out, and anchors 2 and 3 never matched at all. On a freshly
# installed harbor that compiled to
#   SyntaxError: expected 'except' or 'finally' block
# i.e. an unimportable claude_code.py. Verified against pristine 0.22.0 AND
# 0.23.0, so this also repairs the downgrade path. Re-measure after an upgrade.
ANCHOR_ARGMAX_1 = "            run_env = {**env, instruction_env_var: instruction}"
REPLACEMENT_ARGMAX_1 = """            import base64 as _base64
            _instr_id = uuid.uuid4().hex
            _instr_file = f"/tmp/harbor_instruction_{_instr_id}"
            _instr_b64 = _base64.b64encode(instruction.encode("utf-8")).decode("ascii")
            _chunks = [_instr_b64[i:i+4000] for i in range(0, len(_instr_b64), 4000)]
            _wparts = (
                [f"> {_instr_file}.b64"]
                + [f'printf "%s" {shlex.quote(c)} >> {_instr_file}.b64' for c in _chunks]
                + [f"base64 -d {_instr_file}.b64 > {_instr_file} && rm -f {_instr_file}.b64"]
            )
            await self.exec_as_agent(
                environment,
                command=" && ".join(_wparts),
                env=env,
            )

            run_env = {**env}"""

ANCHOR_ARGMAX_2 = """\
                    f'{instruction_shell_var}="${instruction_env_var}"; '
                    f"unset {instruction_env_var}; "
                    f'printf "%s" "${instruction_shell_var}" | '"""
REPLACEMENT_ARGMAX_2 = "                    f'cat {_instr_file} | '"

# Harbor builds the log path from environment_logs_dir; the old anchor expected
# a literal /logs/agent/claude-code.txt that no longer appears here.
ANCHOR_ARGMAX_3 = (
    "                    f\"{(self.environment_logs_dir / 'claude-code.txt').as_posix()}\"\n"
    "                ),\n"
    "                env=run_env,"
)
REPLACEMENT_ARGMAX_3 = (
    "                    f\"{(self.environment_logs_dir / 'claude-code.txt').as_posix()}\"\n"
    '                    f"; rm -f {_instr_file}"\n'
    "                ),\n"
    "                env=run_env,"
)

ALREADY_PATCHED_MARKER_ARGMAX = "_instr_file"


ANCHOR_COLLECT = """\
        if step_cfg is not None:
            hooks.extend(step_cfg.verifier.collect)
        return hooks"""

# v1's own output, so an already-patched harbor upgrades in place. Without it
# the anchor search misses (v1 consumed it), and the run dies: run_task.sh:1510
# calls this script under `set -e`.
ANCHOR_COLLECT_V1 = """\
        # harbor-patch: builtin collect
        _BUILTIN_CMD = "python3 /harness/scoring/collect_artifacts.py"
        if not any(_BUILTIN_CMD in h.command for h in hooks):
            from harbor.models.task.config import VerifierCollectConfig as _VCC
            hooks.append(_VCC(command=_BUILTIN_CMD))
        return hooks"""

# Plain shell, not `sh -c '...'`: harbor already wraps it (docker.py,
# `exec_command.extend(["sh", "-c", cmd])`), so a second shell buys only nested
# quoting. Guarded because this failure is otherwise invisible -- harbor treats
# a failed collect hook as non-fatal and collect_artifacts.py exits 0 by
# contract, so an unmounted /harness/scoring loses every artifact in silence.
#
# Dedup by equality, not containment: a task hook that merely MENTIONED the path
# suppressed the builtin and lost the artifacts with it. Collecting twice is
# harmless (copy2 overwrites, exit 0 always), so that is the safe side.
_COLLECT_BLOCK = """\
        # harbor-patch: builtin collect v2
        _BUILTIN_CMD = (
            'if [ -f /harness/scoring/collect_artifacts.py ]; then '
            'python3 /harness/scoring/collect_artifacts.py; else '
            'echo "[artifacts] FATAL: /harness/scoring is not mounted -- '
            'check the ../ depth in environment/docker-compose.yaml; '
            'NO agent artifacts were collected" >&2; fi'
        )
        _BUILTIN_PRIOR = "python3 /harness/scoring/collect_artifacts.py"
        if not any(h.command.strip() in (_BUILTIN_CMD, _BUILTIN_PRIOR) for h in hooks):
            from harbor.models.task.config import VerifierCollectConfig as _VCC
            hooks.append(_VCC(command=_BUILTIN_CMD))
        return hooks"""

REPLACEMENT_COLLECT = """\
        if step_cfg is not None:
            hooks.extend(step_cfg.verifier.collect)
""" + _COLLECT_BLOCK
REPLACEMENT_COLLECT_V1 = _COLLECT_BLOCK
ALREADY_PATCHED_MARKER_COLLECT = "harbor-patch: builtin collect v2"


# --- Agent failure classification ---------------------------------------------
# harbor names the cause of a failed agent run by regex-searching the WHOLE run
# output for r"rate.?limit" (base.py:174) and never looks at the exit code, which
# it formats into a string and then discards. Claude Code emits
#   {"type":"rate_limit_event","rate_limit_info":{"status":"allowed"}}
# as ordinary telemetry after successful calls, and ".?" matches the underscore,
# so that pattern hits on essentially every run: measured 55, 45 and 2 times
# across three runs of one task -- including the run that exited 0. The upshot is
# that ApiRateLimitError is harbor's de facto name for ANY non-zero agent exit.
#
# Measured here: an OOM kill (exit 137, mid-OCR) and a genuine Anthropic session
# limit (exit 1) came back with the identical label. The OOM's real cause
# survived nowhere -- SIGKILL writes no result event, the container is deleted
# before State.OOMKilled can be read, and the exit code is the only evidence
# left, which is precisely what the classifier ignores.
#
# The override below does two things harbor's does not:
#   1. Triages on HOW the process died before reading what its log mentions.
#      A signal death is a fact; a substring in 12 MB of output is a guess.
#   2. Requires a real rate-limit signal rather than the telemetry line.
#      Verified discriminator: '"error":"rate_limit"' appears once in the
#      genuinely rate-limited run and zero times in the OOM run, while
#      "rate_limit_event" appears in both.
#
# Retry semantics are unchanged except for the misfiled cases: ApiRateLimitError
# keeps meaning "waiting may help", which is exactly what an OOM does not.
ANCHOR_CLASSIFY = '''    @override
    def get_version_command(self) -> str | None:'''

REPLACEMENT_CLASSIFY = r'''    # harbor-patch: classify agent failures by how the process died
    _FATAL_SIGNALS = {
        137: "killed by SIGKILL (128+9) -- in a container this is normally the "
             "OOM killer; compare the task's memory_mb with what docker has",
        139: "killed by SIGSEGV (128+11) -- agent process crashed",
        143: "killed by SIGTERM (128+15) -- stopped by an external signal",
        124: "exit 124 -- command timed out (coreutils timeout)",
    }

    # A real provider rate limit, not Claude Code's "status: allowed" telemetry.
    _REAL_RATE_LIMIT = _re.compile(
        r'"error"\s*:\s*"rate_limit"'
        r'|"type"\s*:\s*"rate_limit_error"'
        r'|too many requests'
        r'|\b(?:session|usage|weekly|monthly)\s+limit\b'
        r'|Limit Exhausted',
        _re.IGNORECASE,
    )

    @override
    def _classify_exec_error(self, command, result):
        detail = (
            f"Command failed (exit {result.return_code}): {command}\n"
            f"stdout: {self._truncate_output(result.stdout)}\n"
            f"stderr: {self._truncate_output(result.stderr)}"
        )
        note = self._FATAL_SIGNALS.get(getattr(result, "return_code", None))
        if note:
            # How it died outranks what the log happens to mention.
            return NonZeroAgentExitCodeError(note + "\n" + detail)
        output = f"{result.stdout or ''}\n{result.stderr or ''}"
        if self._REAL_RATE_LIMIT.search(output):
            return ApiRateLimitError(detail)
        return NonZeroAgentExitCodeError(detail)

    @override
    def get_version_command(self) -> str | None:'''

ALREADY_PATCHED_MARKER_CLASSIFY = "harbor-patch: classify agent failures by how the process died"

# The override needs three names claude_code.py does not import today.
ANCHOR_CLASSIFY_IMPORT = '''from harbor.agents.installed.base import (
    BaseInstalledAgent,'''

REPLACEMENT_CLASSIFY_IMPORT = '''import re as _re  # harbor-patch: agent failure classification

from harbor.agents.installed.base import (
    ApiRateLimitError,
    BaseInstalledAgent,
    NonZeroAgentExitCodeError,'''

ALREADY_PATCHED_MARKER_CLASSIFY_IMPORT = "import re as _re  # harbor-patch"


ANCHOR_JUDGE_MODEL_1 = """\
        with self.agent_environment.with_default_user(user):
            verifier = VerifierFactory.create_verifier_from_config(
                self.config.verifier,
                task=self.task,
                trial_paths=self.paths,
                environment=self.agent_environment,
                override_env=self.config.verifier.env or None,"""
REPLACEMENT_JUDGE_MODEL_1 = """\
        with self.agent_environment.with_default_user(user):
            _ov_env = dict(self.config.verifier.env or {})
            _ov_env.setdefault("JUDGE_MODEL", "gpt-5.6-sol")
            verifier = VerifierFactory.create_verifier_from_config(
                self.config.verifier,
                task=self.task,
                trial_paths=self.paths,
                environment=self.agent_environment,
                override_env=_ov_env or None,"""

ANCHOR_JUDGE_MODEL_2 = """\
                verifier = VerifierFactory.create_verifier_from_config(
                    self.config.verifier,
                    task=self.task,
                    trial_paths=self.paths,
                    environment=target_env,
                    override_env=self.config.verifier.env or None,"""
REPLACEMENT_JUDGE_MODEL_2 = """\
                _ov_env = dict(self.config.verifier.env or {})
                _ov_env.setdefault("JUDGE_MODEL", "gpt-5.6-sol")
                verifier = VerifierFactory.create_verifier_from_config(
                    self.config.verifier,
                    task=self.task,
                    trial_paths=self.paths,
                    environment=target_env,
                    override_env=_ov_env or None,"""
ALREADY_PATCHED_MARKER_JUDGE_MODEL = '_ov_env.setdefault("JUDGE_MODEL"'


# --- Pre-baked Claude Code CLI ------------------------------------------------
# ClaudeCode.install() reaches the network twice inside the container: apt-get
# for curl/procps, then a bootstrap.sh download from downloads.claude.ai. Both
# run BEFORE the agent phase, and both die once the container's default network
# is `internal: true` (tools/network/egress-proxy/overlay.yaml).
#
# The bundles pre-bake the CLI at build time instead, where the network is still
# open, so these two commands have nothing left to do. They are made no-ops
# rather than deleted: an image WITHOUT a baked CLI still installs normally, so
# a bundle that forgets the Dockerfile line degrades to the old behaviour rather
# than failing to start an agent.
#
# The guard is plain shell on purpose. Probing from Python would mean parsing an
# exec result whose shape is not part of harbor's contract.
_Q = chr(34)

ANCHOR_PREBAKE_ROOT = (
    '                ' + _Q + 'if command -v apk &> /dev/null; then' + _Q + '\n'
    '                ' + _Q + '  apk add --no-cache curl bash nodejs npm procps;' + _Q
)

REPLACEMENT_PREBAKE_ROOT = (
    '                ' + _Q + 'if command -v claude &> /dev/null; then' + _Q + '\n'
    "                '  echo " + _Q + "harbor-patch: claude pre-baked; skipping apt" + _Q + ";'\n"
    '                ' + _Q + ' elif command -v apk &> /dev/null; then' + _Q + '\n'
    '                ' + _Q + '  apk add --no-cache curl bash nodejs npm procps;' + _Q
)

ANCHOR_PREBAKE_AGENT = (
    '                ' + _Q + 'set -euo pipefail; ' + _Q + '\n'
    '                ' + _Q + 'if command -v apk &> /dev/null; then' + _Q
)

REPLACEMENT_PREBAKE_AGENT = (
    '                ' + _Q + 'set -euo pipefail; ' + _Q + '\n'
    '                ' + _Q + 'if command -v claude &> /dev/null; then' + _Q + '\n'
    "                '  echo " + _Q + "harbor-patch: claude pre-baked; skipping bootstrap" + _Q + ";'\n"
    '                ' + _Q + ' elif command -v apk &> /dev/null; then' + _Q
)

ALREADY_PATCHED_MARKER_PREBAKE = "harbor-patch: claude pre-baked"

# Harbor has its own early return at the top of install():
#
#     if await self._installed_claude_satisfies_version(environment):
#         return
#
# CORRECTION: that method does NOT exist in 0.13.2, the version this harness
# actually runs. Checked the whole installed package: the only match is
# `_installed_codex_satisfies_version` (agents/installed/codex.py:85), which is
# Codex's own, on Codex's class. `ClaudeCode.install()` has no version-satisfies
# early return at all. The earlier "verified present in 0.13.2" was almost
# certainly the codex method read as the claude one; 0.20.0 and 0.21.0 are
# UNVERIFIED here, so treat the snippet above as a possibility, not a fact.
#
# This patch is therefore not merely belt-and-braces on 0.13.2 -- it is the only
# protection there is. (Independently, harbor's probe would run through
# environment.exec rather than exec_as_agent, so a claude pre-baked for one user
# can be invisible to the other even where the method does exist.)
#
# The constant is used only as a FALLBACK: consulted when the anchors are gone,
# to tell "harbor restructured but still has some protection" from "no
# protection at all". Where the anchors still match, the patch is applied as
# before, so on 0.13.2 this is never reached. Order matters -- see the prebake
# block in main().
#
# Matched as a tuple because the name is version-dependent and, on 0.13.2
# evidence, harbor's convention is `_installed_<agent>_satisfies_version`. A
# rename that follows that convention should land on the fallback rather than on
# the hard failure below, which would otherwise block every run after a harbor
# upgrade for no reason. Kept as exact names, not a loose "satisfies_version"
# substring: a false positive here lets a doomed run start and fail during agent
# setup, whereas the hard failure refuses it up front with a clear message.
#
# What changed in 0.21.0: the inline `apk add --no-cache curl bash nodejs npm
# procps` root block that ANCHOR_PREBAKE_ROOT targets was replaced by a call to
# ensure_system_dependencies(), so that anchor can never match again there.
NATIVE_PREBAKE_GUARD = (
    "_installed_claude_satisfies_version",
    "_installed_claude_code_satisfies_version",
)


# --- Suppress harbor's own score tables ---------------------------------------
# print_job_results_tables() is the single choke point for every score table
# harbor prints -- five call sites across cli/jobs.py and cli/exec.py. Those
# tables are rendered from the in-container verifier result, which is produced
# BEFORE the host rubric pass, so their numbers are always stale. Reading them
# as the run's result is the easiest way to be misled about a trial; the real
# reward is printed afterwards by this harness.
#
# Exceptions are still printed. Suppressing a table must not also swallow the
# fact that a trial errored -- that is the one number in it that is not stale.
# HARBOR_SHOW_SCORES=1 restores the tables verbatim.
#
# Anchored on the def line plus the loop that follows it: both are unique in
# jobs.py, and in an unpatched file they are adjacent.
ANCHOR_SUPPRESS_SCORES = (
    "def print_job_results_tables(job_result) -> None:\n"
    "    for evals_key, dataset_stats in job_result.stats.evals.items():"
)
REPLACEMENT_SUPPRESS_SCORES = (
    "def print_job_results_tables(job_result) -> None:\n"
    "    # harbor-patch: suppress pre-rubric scores\n"
    "    import os as _os\n"
    '    if _os.getenv("HARBOR_SHOW_SCORES") != "1":\n'
    "        for _key, _stats in job_result.stats.evals.items():\n"
    "            if _stats.n_errors:\n"
    "                console.print(\n"
    '                    f"[yellow]{_key}: {_stats.n_errors} exception(s)[/yellow]"\n'
    "                )\n"
    "        console.print(\n"
    '            "[dim]harbor\'s score tables are suppressed: they are printed before "\n'
    '            "the host rubric pass and are always stale. The published reward "\n'
    '            "follows below. HARBOR_SHOW_SCORES=1 restores them.[/dim]"\n'
    "        )\n"
    "        return\n"
    "    for evals_key, dataset_stats in job_result.stats.evals.items():"
)
ALREADY_PATCHED_MARKER_SUPPRESS_SCORES = "harbor-patch: suppress pre-rubric scores"


# --- Fallback model removal ---------------------------------------------------
# A fallback model silently swaps the model mid-run, which is exactly what a
# pinned-model trial must not do: the reward would be attributed to the model in
# --model while some of the turns came from another one. Removing the option is
# what makes that unrepresentable rather than merely unset.
#
# 0.23.0 moved the declaration from a quoted CliFlag entry to a pydantic field,
# so the old `'"fallback_model"' not in text` probe stopped matching and passed
# VACUOUSLY: the patch reported "already done" on a harbor that still carried the
# option. Both shapes are handled below, and the presence probe is now the
# unquoted name so it cannot go quietly true again.
#
# Deleting the field is safe here: the declaration is the only reference to it in
# the whole 0.23.0 package (verified), so nothing reads the attribute. It is also
# strictly stronger than on 0.22.0 -- with extra="forbid" the option is now a
# hard validation error rather than a silently ignored kwarg.
ANCHOR_FALLBACK = """\
        CliFlag(
            "fallback_model",
            cli="--fallback-model",
            type="str",
        ),"""

ANCHOR_FALLBACK_PYD = """\
    fallback_model: Annotated[str | None, Cli("--fallback-model")] = Field(
        default=None, description="Fallback model name."
    )
"""


def find_harbor_claude_code() -> Path:
    import shutil
    import subprocess

    try:
        spec = importlib.util.find_spec("harbor.agents.installed.claude_code")
    except (ModuleNotFoundError, ValueError):
        spec = None
    if spec and spec.origin:
        return Path(spec.origin)

    harbor_bin = shutil.which("harbor")
    if not harbor_bin:
        raise RuntimeError(
            "harbor not found in PATH. Install it first: pipx install harbor"
        )

    venv_bin = Path(harbor_bin).resolve().parent
    venv_root = venv_bin.parent
    candidates = sorted(venv_root.glob("lib/python*/site-packages/harbor/agents/installed/claude_code.py"))
    if candidates:
        return candidates[0]

    # Last resort: ask harbor's own interpreter where the module lives. Both
    # candidates can be absent -- a `harbor` shim on PATH that is not inside a
    # venv at all, which is exactly what a test stub looks like. Calling
    # subprocess.run on a path that does not exist raises FileNotFoundError from
    # deep inside subprocess, burying the RuntimeError below that actually says
    # what to do about it. Check first, and let that message be the one the
    # operator sees.
    venv_python = venv_bin / "python3"
    if not venv_python.exists():
        venv_python = venv_bin / "python"
    if venv_python.exists():
        result = subprocess.run(
            [str(venv_python), "-c",
             "import harbor.agents.installed.claude_code as m; print(m.__file__)"],
            capture_output=True, text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            return Path(result.stdout.strip())

    raise RuntimeError(
        f"Could not locate harbor/agents/installed/claude_code.py in pipx venv at {venv_root}"
    )


def find_harbor_trial() -> Path:
    claude_code = find_harbor_claude_code()
    harbor_pkg_dir = claude_code.parent.parent.parent
    trial = harbor_pkg_dir / "trial" / "trial.py"
    if trial.exists():
        return trial
    raise RuntimeError(f"Could not locate harbor/trial/trial.py (tried {trial})")


def find_harbor_jobs() -> Path:
    claude_code = find_harbor_claude_code()
    harbor_pkg_dir = claude_code.parent.parent.parent
    jobs = harbor_pkg_dir / "cli" / "jobs.py"
    if jobs.exists():
        return jobs
    raise RuntimeError(f"Could not locate harbor/cli/jobs.py (tried {jobs})")


def main() -> None:
    # --audit reports every patch's status without writing anything, and always
    # exits 0. Use it after a harbor upgrade: a normal run stops at the first
    # unapplicable patch, so drift is discovered one patch at a time.
    audit = "--audit" in sys.argv or "--check" in sys.argv

    # Anchors that could not be applied. Collected rather than exited on, so one
    # invocation reports all of them. Same reasoning as the ARG_MAX warning
    # below, extended to the rest: run_task.sh calls this unconditionally under
    # `set -e`, so an early sys.exit(1) blocks every stage of every run AND
    # hides whatever else drifted.
    failures: list[str] = []

    # Harbor absent is a setup fact, not a drifted anchor. Uncaught it surfaced
    # as a bare traceback with the one useful line buried inside it.
    try:
        target = find_harbor_claude_code()
    except RuntimeError as exc:
        print(f"[patch_harbor] {exc}", file=sys.stderr)
        sys.exit(0 if audit else 1)
    text = target.read_text(encoding="utf-8")
    changed = False

    if ALREADY_PATCHED_MARKER_THINKING_PYD in text or ALREADY_PATCHED_MARKER in text:
        print(f"[patch_harbor] Thinking flags: already patched")
    elif ANCHOR_THINKING_PYD in text:
        text = text.replace(ANCHOR_THINKING_PYD, REPLACEMENT_THINKING_PYD, 1)
        changed = True
        print(f"[patch_harbor] Thinking flags: patched (pydantic options, harbor >= 0.23.0)")
    elif ANCHOR in text:
        text = text.replace(ANCHOR, ANCHOR + PATCH, 1)
        changed = True
        print(f"[patch_harbor] Thinking flags: patched (CliFlag list, harbor < 0.23.0)")
    else:
        print(
            f"[patch_harbor] Thinking flags: NOT applied -- anchor not found in {target}",
            file=sys.stderr,
        )
        failures.append(f"thinking flags  ({target.name})")

    if ALREADY_PATCHED_MARKER_ARGMAX in text:
        print(f"[patch_harbor] ARG_MAX fix: already applied")
    elif ANCHOR_ARGMAX_1 not in text:
        # A drifted anchor is not a reason to take the whole harness down.
        #
        # This used to sys.exit(1), which meant a patch that no longer applies
        # blocked every stage of every run -- run_task.sh calls this script
        # unconditionally at dispatch, under `set -e`. Harbor 0.13.2 passes the
        # instruction inline as shlex.quote(instruction) (claude_code.py:1258,
        # :1414), while these anchors target an instruction_env_var form from a
        # different harbor release, so on this install the patch cannot apply at
        # all and the harness could not run anything.
        #
        # What is lost by continuing: the instruction goes on the command line,
        # so a bundle whose instruction.md approaches ARG_MAX (1 MiB on Linux)
        # would fail with "Argument list too long". Bundles here are ~1.5 KB, so
        # the warning is the proportionate response -- but it is printed loudly
        # rather than swallowed, because the day a bundle does get large this is
        # the only notice anyone gets.
        print(
            f"[patch_harbor] WARNING: ARG_MAX fix NOT applied -- anchor not found in {target}\n"
            "  This harbor passes the instruction on the command line. Fine for the\n"
            "  bundles in this repo (~1.5 KB); a bundle approaching ARG_MAX (1 MiB)\n"
            "  would fail with 'Argument list too long'. Re-anchor this patch if that\n"
            "  ever happens.",
            file=sys.stderr,
        )
    else:
        text = text.replace(ANCHOR_ARGMAX_1, REPLACEMENT_ARGMAX_1, 1)
        text = text.replace(ANCHOR_ARGMAX_2, REPLACEMENT_ARGMAX_2, 1)
        text = text.replace(ANCHOR_ARGMAX_3, REPLACEMENT_ARGMAX_3, 1)
        changed = True
        print(f"[patch_harbor] ARG_MAX fix: applied")

    if "fallback_model" not in text:
        print(f"[patch_harbor] Fallback model removal: already done")
    elif ANCHOR_FALLBACK_PYD in text:
        text = text.replace(ANCHOR_FALLBACK_PYD, "", 1)
        changed = True
        print(f"[patch_harbor] Fallback model removal: applied (pydantic field, harbor >= 0.23.0)")
    elif ANCHOR_FALLBACK in text:
        text = text.replace(ANCHOR_FALLBACK, "", 1)
        changed = True
        print(f"[patch_harbor] Fallback model removal: applied (CliFlag list, harbor < 0.23.0)")
    else:
        print(
            f"[patch_harbor] Fallback model removal: NOT applied -- anchor not found in {target}",
            file=sys.stderr,
        )
        failures.append(f"fallback_model removal  ({target.name})")

    if ALREADY_PATCHED_MARKER_PREBAKE in text:
        print(f"[patch_harbor] Pre-baked CLI guard: already applied")
    elif ANCHOR_PREBAKE_ROOT in text and ANCHOR_PREBAKE_AGENT in text:
        text = text.replace(ANCHOR_PREBAKE_ROOT, REPLACEMENT_PREBAKE_ROOT, 1)
        text = text.replace(ANCHOR_PREBAKE_AGENT, REPLACEMENT_PREBAKE_AGENT, 1)
        changed = True
        print(f"[patch_harbor] Pre-baked CLI guard: applied")
    elif (_native := next((g for g in NATIVE_PREBAKE_GUARD if g in text), None)):
        # Anchors gone (harbor >= 0.21.0 restructured install()), but harbor has
        # a version-satisfies early return of its own. Non-fatal: blocking every
        # run over a patch that has no place left to apply is worse than
        # proceeding on harbor's own protection. Loud because that protection is
        # not identical -- harbor probes via environment.exec, so if a run now
        # fails during agent setup trying to reach the network, this line is the
        # first place to look. The guard is NAMED in the message: on 0.13.2 no
        # such method existed at all, so which one matched is the fact worth
        # having when this fires.
        print(
            f"[patch_harbor] Pre-baked CLI guard: NOT applied -- anchors gone from {target.name};\n"
            f"  relying on harbor's own {_native} early return.\n"
            "  If agent setup starts failing on network access, re-anchor this patch.",
            file=sys.stderr,
        )
    else:
        print(
            f"[patch_harbor] Pre-baked CLI guard: NOT applied -- anchor not found in {target}",
            file=sys.stderr,
        )
        failures.append(f"pre-baked CLI guard  ({target.name})")

    if ALREADY_PATCHED_MARKER_CLASSIFY in text:
        print(f"[patch_harbor] Failure classification: already applied")
    elif ANCHOR_CLASSIFY not in text or ANCHOR_CLASSIFY_IMPORT not in text:
        # Non-fatal. Without it, a crashed or OOM-killed run is still recorded --
        # just under harbor's default name, ApiRateLimitError. That is a wrong
        # label on a real result, not a lost result, so it must not block a run.
        print(
            f"[patch_harbor] Failure classification: NOT applied -- anchor not found in {target.name}\n"
            "  Agent failures will keep being labelled ApiRateLimitError regardless of\n"
            "  what actually killed them (harbor base.py:174 greps the whole log for\n"
            "  r'rate.?limit', which Claude Code's own telemetry always matches).\n"
            "  Re-anchor against ClaudeCode.get_version_command if this matters.",
            file=sys.stderr,
        )
    else:
        if ALREADY_PATCHED_MARKER_CLASSIFY_IMPORT not in text:
            text = text.replace(ANCHOR_CLASSIFY_IMPORT, REPLACEMENT_CLASSIFY_IMPORT, 1)
        text = text.replace(ANCHOR_CLASSIFY, REPLACEMENT_CLASSIFY, 1)
        changed = True
        print(f"[patch_harbor] Failure classification: applied")

    if changed and not audit:
        target.write_text(text, encoding="utf-8")
        print(f"[patch_harbor] Written: {target}")
    elif changed:
        print(f"[patch_harbor] Would write (audit): {target}")
    else:
        print(f"[patch_harbor] Nothing to do: {target}")

    # A missing trial.py is itself drift worth reporting, not a traceback.
    try:
        trial = find_harbor_trial()
    except RuntimeError as exc:
        print(f"[patch_harbor] trial.py: NOT found -- {exc}", file=sys.stderr)
        failures.append("trial.py not found (collect hook + JUDGE_MODEL inject unapplied)")
        _report(failures, audit)
        return
    trial_text = trial.read_text(encoding="utf-8")
    trial_changed = False

    if ALREADY_PATCHED_MARKER_COLLECT in trial_text:
        print(f"[patch_harbor] Collect hook: already applied")
    elif ANCHOR_COLLECT_V1 in trial_text:
        trial_text = trial_text.replace(ANCHOR_COLLECT_V1, REPLACEMENT_COLLECT_V1, 1)
        trial_changed = True
        print(f"[patch_harbor] Collect hook: upgraded from v1")
    elif ANCHOR_COLLECT not in trial_text:
        print(
            f"[patch_harbor] Collect hook: NOT applied -- anchor not found in {trial}",
            file=sys.stderr,
        )
        failures.append(f"collect hook  ({trial.name})")
    else:
        trial_text = trial_text.replace(ANCHOR_COLLECT, REPLACEMENT_COLLECT, 1)
        trial_changed = True
        print(f"[patch_harbor] Collect hook: applied")

    if ALREADY_PATCHED_MARKER_JUDGE_MODEL in trial_text:
        print(f"[patch_harbor] JUDGE_MODEL inject: already applied")
    elif ANCHOR_JUDGE_MODEL_1 not in trial_text:
        print(
            f"[patch_harbor] JUDGE_MODEL inject: NOT applied -- anchor not found in {trial}",
            file=sys.stderr,
        )
        failures.append(f"JUDGE_MODEL inject  ({trial.name})")
    else:
        trial_text = trial_text.replace(ANCHOR_JUDGE_MODEL_1, REPLACEMENT_JUDGE_MODEL_1, 1)
        trial_text = trial_text.replace(ANCHOR_JUDGE_MODEL_2, REPLACEMENT_JUDGE_MODEL_2, 1)
        trial_changed = True
        print(f"[patch_harbor] JUDGE_MODEL inject: applied")

    if trial_changed and not audit:
        trial.write_text(trial_text, encoding="utf-8")
        print(f"[patch_harbor] Written: {trial}")
    elif trial_changed:
        print(f"[patch_harbor] Would write (audit): {trial}")

    # A missing jobs.py is itself drift worth reporting, not a traceback.
    try:
        jobs = find_harbor_jobs()
    except RuntimeError as exc:
        print(f"[patch_harbor] jobs.py: NOT found -- {exc}", file=sys.stderr)
        failures.append("jobs.py not found (score-table suppression unapplied)")
        _report(failures, audit)
        return
    jobs_text = jobs.read_text(encoding="utf-8")
    jobs_changed = False

    if ALREADY_PATCHED_MARKER_SUPPRESS_SCORES in jobs_text:
        print(f"[patch_harbor] Score-table suppression: already applied")
    elif ANCHOR_SUPPRESS_SCORES not in jobs_text:
        print(
            f"[patch_harbor] Score-table suppression: NOT applied -- anchor not found in {jobs}",
            file=sys.stderr,
        )
        failures.append(f"score-table suppression  ({jobs.name})")
    else:
        jobs_text = jobs_text.replace(
            ANCHOR_SUPPRESS_SCORES, REPLACEMENT_SUPPRESS_SCORES, 1
        )
        jobs_changed = True
        print(f"[patch_harbor] Score-table suppression: applied")

    if jobs_changed and not audit:
        jobs.write_text(jobs_text, encoding="utf-8")
        print(f"[patch_harbor] Written: {jobs}")
    elif jobs_changed:
        print(f"[patch_harbor] Would write (audit): {jobs}")

    _report(failures, audit)


def _report(failures: list[str], audit: bool) -> None:
    """Print one consolidated verdict and set the exit code.

    Every patch is attempted before this runs, so a harbor upgrade yields the
    full list of drifted anchors in one go instead of one per invocation.
    """
    if not failures:
        print("[patch_harbor] All patches accounted for.")
        return

    print(f"\n[patch_harbor] ---- {len(failures)} patch(es) could not be applied ----",
          file=sys.stderr)
    for name in failures:
        print(f"  MISS  {name}", file=sys.stderr)
    print(
        "\n  Harbor's source has drifted from these anchors -- most likely it was\n"
        "  upgraded. For each one, either re-anchor it against the new source, or\n"
        "  confirm harbor now provides the behaviour natively and detect that\n"
        "  instead (see NATIVE_PREBAKE_GUARD for the worked example).\n"
        "  Re-run with --audit to re-check without writing.",
        file=sys.stderr,
    )
    if not audit:
        sys.exit(1)


if __name__ == "__main__":
    main()
