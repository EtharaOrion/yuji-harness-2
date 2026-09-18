"""Stage the files the agent produced into Harbor's artifact publish dir.

Run as a ``[[verifier.collect]]`` hook in the main service:

    command = "python3 /harness/scoring/collect_artifacts.py"

Why a collect hook and not something in ``tests/test.sh``: Harbor downloads
artifacts *after* the agent phase and *before* the verifier
(``harbor/trial/single_step.py``), so by the time test.sh runs the collection
window has already closed. Collect hooks run in the main container immediately
before the download (``harbor/trial/trial.py``), which is the last moment the
agent's files still exist and can still be shipped.

What counts as agent output: everything under /workspace except /workspace/data.
The workspace volume is created empty per trial and only the agent (through the
``filesystem`` MCP server or its own shell) writes into it, while
/workspace/data is a read-only bind of the task's own input assets -- authored
ahead of the run and identical across every trial, so re-shipping it per run
would be duplication, not evidence.

Best-effort by contract: Harbor already treats collect-hook failures as
non-fatal, and a reporting step must never be the reason a graded trial fails.
Every error is caught, summarised on stdout, and exit is always 0.

Env:
    ARTIFACT_SRC        source root (default /workspace)
    ARTIFACT_DEST       publish dir (default /logs/artifacts)
    ARTIFACT_EXCLUDE    colon-separated src-relative dirs to skip (default "data")
    ARTIFACT_MAX_FILE   per-file byte cap  (default 100 MiB)
    ARTIFACT_MAX_TOTAL  total byte cap     (default 500 MiB)
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

DEFAULT_SRC = "/workspace"
DEFAULT_DEST = "/logs/artifacts"
DEFAULT_EXCLUDE = "data"
DEFAULT_MAX_FILE = 100 * 1024 * 1024
DEFAULT_MAX_TOTAL = 500 * 1024 * 1024

_SKIP_DIRS = frozenset({"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "node_modules", ".git"})
_SKIP_EXTS = frozenset({".pyc", ".pyo"})


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"[artifacts] {name}={raw!r} is not an integer; using default {default}")
        return default


def collect(src: Path, dest: Path, excluded: set[str],
            max_file: int, max_total: int) -> tuple[int, int, list[str]]:
    """Copy regular files from *src* into *dest*, preserving relative paths.

    Returns (files copied, bytes copied, skip reasons).
    """
    skipped: list[str] = []
    copied = 0
    total = 0

    if not src.is_dir():
        return 0, 0, [f"source {src} does not exist"]

    for path in sorted(src.rglob("*")):
        try:
            rel = path.relative_to(src)
        except ValueError:  # pragma: no cover - rglob always yields descendants
            continue
        if rel.parts and rel.parts[0] in excluded:
            continue
        if any(part in _SKIP_DIRS for part in rel.parts):
            continue
        if path.suffix in _SKIP_EXTS:
            continue
        # Symlinks, sockets and FIFOs are not evidence and can point outside
        # the workspace; is_file() follows links, so test for the link first.
        if path.is_symlink():
            skipped.append(f"{rel}: symlink")
            continue
        if not path.is_file():
            continue
        try:
            size = path.stat().st_size
        except OSError as exc:
            skipped.append(f"{rel}: {exc}")
            continue
        if size > max_file:
            skipped.append(f"{rel}: {size} bytes over per-file cap {max_file}")
            continue
        if total + size > max_total:
            skipped.append(f"{rel}: would exceed total cap {max_total}")
            continue
        target = dest / rel
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        except OSError as exc:
            skipped.append(f"{rel}: {exc}")
            continue
        copied += 1
        total += size

    return copied, total, skipped


def main() -> int:
    for suffix in ("SRC", "DEST", "EXCLUDE", "MAX_FILE", "MAX_TOTAL"):
        if f"DEFAULT_{suffix}" in os.environ and f"ARTIFACT_{suffix}" not in os.environ:
            print(f"[artifacts] DEFAULT_{suffix} is set in the environment but nothing "
                  f"reads it; ARTIFACT_{suffix} is the key that configures this")

    src = Path(os.environ.get("ARTIFACT_SRC", DEFAULT_SRC))
    dest = Path(os.environ.get("ARTIFACT_DEST", DEFAULT_DEST))
    excluded = {p for p in os.environ.get("ARTIFACT_EXCLUDE", DEFAULT_EXCLUDE).split(":") if p}
    max_file = _int_env("ARTIFACT_MAX_FILE", DEFAULT_MAX_FILE)
    max_total = _int_env("ARTIFACT_MAX_TOTAL", DEFAULT_MAX_TOTAL)

    try:
        dest.mkdir(parents=True, exist_ok=True)
        copied, total, skipped = collect(src, dest, excluded, max_file, max_total)
    except Exception as exc:  # noqa: BLE001 - must never fail the trial
        print(f"[artifacts] collection failed: {exc}")
        return 0

    print(f"[artifacts] {copied} file(s), {total} bytes from {src} -> {dest} "
          f"(excluded: {', '.join(sorted(excluded)) or 'none'})")
    for reason in skipped:
        print(f"[artifacts] skipped {reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
