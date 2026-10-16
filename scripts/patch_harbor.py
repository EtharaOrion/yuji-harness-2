#!/usr/bin/env python3
"""Patch the installed harbor for this harness, or refuse loudly.

    python3 scripts/patch_harbor.py           apply, exit 1 on any drift
    python3 scripts/patch_harbor.py --audit   report only, write nothing,
                                              still exit 1 on any drift

Every patch is anchored on the exact source of the harbor release pinned in
scripts/lib/harbor.sh (HARBOR_VERSION). Each one either applies, is already
applied (idempotent: the marker is checked first), or is reported as a MISS --
there is no third outcome where a patch is skipped and the run carries on.
A harbor whose version is not the pin is refused before any anchor is read,
because an anchor that happens to match on another release proves nothing.

Patches dropped when the pin moved to 0.24.0, and why:

  failure classification   NATIVE. 0.24.0's ClaudeCode._classify_exec_error
                           classifies a stream-json launch from the terminal
                           `result`/`error` events only (filter_jsonl_events),
                           so `rate_limit_event` telemetry no longer reads as
                           ApiRateLimitError. The old override also sat ABOVE the
                           native method in the class body, so the native one won
                           and the patch was dead code -- and its detail string
                           bypassed 0.24.0's command redaction.
  pre-baked CLI guard      NATIVE. install() returns early through
                           _installed_claude_satisfies_version.
  builtin collect hook     DUPLICATE. It copied /workspace minus data/ into the
                           artifact dir; task.toml `artifacts` does that natively
                           and the bundle owns that declaration.
  JUDGE_MODEL inject       GRADING. The judge model is the bundle's
                           [verifier.env] (tests/judge.toml under trinity); the
                           harness carries no grading configuration.
  score-table suppression  GRADING. The tables were "stale" only because a host
                           rubric pass rewrote the reward afterwards. That pass is
                           gone, so harbor's tables are the bundle's own reward.

NATIVE_REQUIRED below asserts the native behaviour the dropped patches now rely
on, so a harbor that lost it fails here rather than mid-trial.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HARBOR_LIB = REPO / "scripts" / "lib" / "harbor.sh"


def pinned_version() -> str:
    """HARBOR_VERSION from scripts/lib/harbor.sh, the one place the pin lives."""
    m = re.search(r'^HARBOR_VERSION="([^"]+)"', HARBOR_LIB.read_text(), re.M)
    if not m:
        raise RuntimeError(f"no HARBOR_VERSION=\"...\" line in {HARBOR_LIB}")
    return m.group(1)


@dataclass(frozen=True)
class Patch:
    """One source edit: every (anchor, replacement) pair applies, or none does."""

    name: str
    module: str
    marker: str
    edits: tuple[tuple[str, str], ...]
    # For a removal: the needle whose total absence means the goal already
    # holds (an earlier patcher removed it without leaving this marker).
    absent: str | None = None
    # A data file beside `module` (a package) instead of the module's own source.
    file: str | None = None


# --- thinking flag -------------------------------------------------------------
# ClaudeCodeOptions is extra="forbid", so `--ak thinking=adaptive` (run_task.sh)
# is a validation error unless the field exists. thinking_display is native and
# is the anchor; `thinking` is still absent in 0.24.0.
_THINKING_ANCHOR = '''    thinking_display: Annotated[
        Literal["summarized", "omitted"] | None,
        Cli("--thinking-display"),
    ] = Field(default=None, description="How thinking is displayed.")'''

THINKING = Patch(
    name="thinking flag",
    module="harbor.agents.installed.claude_code",
    marker="# harbor-patch: thinking flag",
    edits=((
        _THINKING_ANCHOR,
        _THINKING_ANCHOR + '''
    # harbor-patch: thinking flag
    thinking: Annotated[str | None, Cli("--thinking")] = Field(
        default=None, description="Thinking mode."
    )''',
    ),),
)

# --- fallback model removal ----------------------------------------------------
# A fallback model swaps the model mid-run, so the reward would be attributed to
# --model while some turns came from another. Removing the field makes it
# unrepresentable (extra="forbid" turns it into a validation error). The
# declaration is the only reference to it in the 0.24.0 package.
FALLBACK = Patch(
    name="fallback_model removal",
    module="harbor.agents.installed.claude_code",
    marker="# harbor-patch: fallback_model removed",
    edits=((
        '''    fallback_model: Annotated[str | None, Cli("--fallback-model")] = Field(
        default=None, description="Fallback model name."
    )
''',
        "    # harbor-patch: fallback_model removed\n",
    ),),
    absent="fallback_model",
)

# --- ARG_MAX ---------------------------------------------------------------------
# 0.24.0 hands the instruction to `claude` through an env var, and the docker
# provider passes every env var as one host-side argv string (`-e KEY=VAL`,
# environments/docker/docker.py). Linux caps a single argv string at 128 KiB
# (MAX_ARG_STRLEN), so a long instruction fails with "Argument list too long".
# The patch writes it to a file inside the container in 4000-byte base64 chunks
# instead. Indentation is load-bearing: the anchors sit in a `try:` inside an
# `async def`, at 12 and 20 spaces.
ARGMAX = Patch(
    name="ARG_MAX instruction file",
    module="harbor.agents.installed.claude_code",
    marker="_instr_file",
    edits=(
        (
            "            run_env = {**env, instruction_env_var: instruction}",
            """            import base64 as _base64
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

            run_env = {**env}""",
        ),
        (
            """\
                    f'{instruction_shell_var}="${instruction_env_var}"; '
                    f"unset {instruction_env_var}; "
                    f'printf "%s" "${instruction_shell_var}" | '""",
            "                    f'cat {_instr_file} | '",
        ),
        (
            "                    f\"{(self.environment_logs_dir / 'claude-code.txt').as_posix()}\"\n"
            "                ),\n"
            "                env=run_env,",
            "                    f\"{(self.environment_logs_dir / 'claude-code.txt').as_posix()}\"\n"
            '                    f"; rm -f {_instr_file}"\n'
            "                ),\n"
            "                env=run_env,",
        ),
    ),
)

# --- egress sidecar read timeout -------------------------------------------------
# Under an allowlist agent phase every agent connection, the ccbridge included,
# goes through the sidecar's gost `red` handler with HTTP sniffing on. That path
# drops a request whose response has not started within gost's short default
# read timeout (measured: a 90s host call is cut at ~16s, the same call direct
# succeeds), and a long model turn sends nothing until it is done, so the agent
# sees "Server disconnected without sending a response" and dies after five
# retries. 1800s matches the agent's LLM_TIMEOUT and what squid allowed. Harbor
# names the sidecar image by a hash of this directory, so the edit rebuilds it.
SIDECAR_TIMEOUT = Patch(
    name="egress sidecar read timeout",
    module="harbor.environments.docker",
    file="harbor-docker-egress-control-sidecar/gost.yaml",
    marker="# harbor-patch: sidecar read timeout",
    edits=((
        "        sniffing.fallback: true\n",
        "        sniffing.fallback: true\n"
        "        # harbor-patch: sidecar read timeout\n"
        "        readTimeout: 1800s\n",
    ),),
)

# --- egress sidecar service-name aliases ----------------------------------------
# With egress control on, harbor moves every compose service that declares no
# networking of its own into the sidecar's network namespace. They share it, so
# they reach each other on 127.0.0.1, but their compose service names stop
# resolving, and a task whose MCP URLs name a service (http://<service>:<port>)
# loses every tool (measured: all 22 MCP servers "MCP Connection Failure" while
# localhost:<port> answered). Containers that join a namespace share its owner's
# /etc/hosts, so aliasing each moved service to 127.0.0.1 on the sidecar restores
# name resolution for all of them. Built from the task's own compose; no names
# are hardcoded.
SIDECAR_ALIASES = Patch(
    name="egress sidecar service-name aliases",
    module="harbor.environments.docker.docker",
    marker="# harbor-patch: sidecar service-name aliases",
    edits=((
        """            for service_name in self._egress_controlled_service_names()
        }
        if not services:
            return None
""",
        """            for service_name in self._egress_controlled_service_names()
        }
        if not services:
            return None
        # harbor-patch: sidecar service-name aliases
        services[self._EGRESS_CONTROL_SERVICE_NAME] = {
            "extra_hosts": [f"{name}:127.0.0.1" for name in sorted(services)]
        }
""",
    ),),
)

PATCHES = (THINKING, FALLBACK, ARGMAX, SIDECAR_TIMEOUT, SIDECAR_ALIASES)

# Native behaviour the dropped patches now rely on: (module, needle, what).
NATIVE_REQUIRED = (
    ("harbor.agents.installed.claude_code", "def _classify_exec_error(",
     "ClaudeCode classifies failures from terminal stream events"),
    ("harbor.agents.installed.claude_code", "filter_jsonl_events(result.stdout",
     "rate_limit_event telemetry is filtered out of classification"),
    ("harbor.agents.installed.claude_code",
     "if await self._installed_claude_satisfies_version(environment):",
     "install() skips the network when claude is pre-baked"),
)


def installed_version() -> str | None:
    try:
        return importlib.metadata.version("harbor")
    except importlib.metadata.PackageNotFoundError:
        return None


def module_path(module: str) -> Path:
    """Where the harbor module lives: this interpreter, else harbor's own venv."""
    try:
        spec = importlib.util.find_spec(module)
    except (ModuleNotFoundError, ValueError):
        spec = None
    if spec and spec.origin:
        return Path(spec.origin)
    harbor_bin = shutil.which("harbor")
    if not harbor_bin:
        raise RuntimeError("harbor not found in PATH or this interpreter; "
                           "install it with scripts/lib/harbor.sh harbor_install")
    venv_bin = Path(harbor_bin).resolve().parent
    rel = Path(*module.split(".")).with_suffix(".py")
    found = sorted(venv_bin.parent.glob(f"lib/python*/site-packages/{rel.as_posix()}"))
    if found:
        return found[0]
    python = next((p for p in (venv_bin / "python3", venv_bin / "python") if p.exists()), None)
    if python is None:
        raise RuntimeError(f"no python beside {harbor_bin}; cannot locate {module}")
    result = subprocess.run(
        [str(python), "-c", f"import {module} as m; print(m.__file__)"],
        capture_output=True, text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError(f"harbor's interpreter cannot import {module}")
    return Path(result.stdout.strip())


def harbor_version_via(path: Path) -> str | None:
    """The version of the harbor that owns `path`, read from its own dist-info."""
    site = path
    while site.name != "harbor" and site != site.parent:
        site = site.parent
    for dist in site.parent.glob("harbor-*.dist-info"):
        meta = dist / "METADATA"
        if meta.is_file():
            m = re.search(r"^Version: (\S+)", meta.read_text(), re.M)
            if m:
                return m.group(1)
    return None


def apply(patch: Patch, text: str) -> tuple[str, str]:
    """(new text, status). status is 'already', 'applied' or 'MISS: ...'."""
    if patch.marker in text:
        return text, "already"
    if patch.absent is not None and patch.absent not in text:
        return text, f"already ({patch.absent!r} is absent from the module)"
    missing = [i + 1 for i, (anchor, _) in enumerate(patch.edits) if text.count(anchor) != 1]
    if missing:
        return text, f"MISS: anchor(s) {missing} of {len(patch.edits)} not found exactly once"
    for anchor, replacement in patch.edits:
        text = text.replace(anchor, replacement, 1)
    if patch.marker not in text:
        return text, "MISS: applied but its marker is absent, so it would re-apply"
    return text, "applied"


def main(argv: list[str]) -> int:
    audit = "--audit" in argv or "--check" in argv
    failures: list[str] = []
    pin = pinned_version()

    try:
        paths = {m: module_path(m) for m in {p.module for p in PATCHES}
                 | {m for m, _, _ in NATIVE_REQUIRED}}
    except RuntimeError as exc:
        print(f"[patch_harbor] {exc}", file=sys.stderr)
        return 1
    # A file patch targets a data file in the module's package directory.
    targets = {p.name: (paths[p.module].parent / p.file if p.file else paths[p.module])
               for p in PATCHES}
    for name, target in targets.items():
        if not target.is_file():
            print(f"[patch_harbor] {name}: MISS: {target} does not exist", file=sys.stderr)
            return 1

    found = harbor_version_via(next(iter(paths.values()))) or installed_version()
    if found != pin:
        print(f"[patch_harbor] REFUSING: installed harbor is {found or 'unknown'}, "
              f"the harness pins {pin} (scripts/lib/harbor.sh). Every anchor here "
              f"is measured against {pin}; reinstall the pin.", file=sys.stderr)
        return 1

    texts = {path: path.read_text(encoding="utf-8")
             for path in set(paths.values()) | set(targets.values())}
    original = dict(texts)

    for patch in PATCHES:
        path = targets[patch.name]
        texts[path], status = apply(patch, texts[path])
        line = f"[patch_harbor] {patch.name}: {status}"
        if status.startswith("MISS"):
            print(line, file=sys.stderr)
            failures.append(f"{patch.name}  ({path.name})")
        else:
            print(line)

    for module, needle, what in NATIVE_REQUIRED:
        if needle in original[paths[module]]:
            print(f"[patch_harbor] native: {what}")
        else:
            print(f"[patch_harbor] native: MISSING -- {what}", file=sys.stderr)
            failures.append(f"native: {what}")

    if failures:
        print(f"\n[patch_harbor] ---- {len(failures)} check(s) failed on harbor {found} ----",
              file=sys.stderr)
        for name in failures:
            print(f"  MISS  {name}", file=sys.stderr)
        print("  Nothing was written. Re-anchor against the pinned source, or confirm the\n"
              "  behaviour is native and move the patch to NATIVE_REQUIRED.", file=sys.stderr)
        return 1

    for path, text in texts.items():
        if text == original[path]:
            continue
        if audit:
            print(f"[patch_harbor] would write (audit): {path}")
            continue
        if path.suffix == ".py":
            compile(text, str(path), "exec")
        path.write_text(text, encoding="utf-8")
        print(f"[patch_harbor] written: {path}")
    print(f"[patch_harbor] harbor {found}: all patches accounted for.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
