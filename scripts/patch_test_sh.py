#!/usr/bin/env python3
"""Patch a task's tests/test.sh in place to the harness verifier, assets/verifier/test.sh.

The reference's canary line holds a ``{guid}`` placeholder, filled with the GUID from
the ``# harbor-canary GUID <uuid>`` line of the task's current test.sh. A task whose
test.sh has no such line cannot be patched and is refused.

Usage::

    python3 scripts/patch_test_sh.py <task_dir> ...

Exits 1 when any task is refused.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REFERENCE = Path(__file__).resolve().parent.parent / "assets" / "verifier" / "test.sh"
CANARY_RE = re.compile(rb"^# harbor-canary GUID ([0-9a-f-]+)\r?$", re.M)


def patch(task: Path, reference: bytes) -> bool:
    """Rewrite the task's test.sh from the reference; True when it changed.
    Raises ValueError when the task's test.sh is missing or carries no GUID."""
    script = task / "tests" / "test.sh"
    try:
        current = script.read_bytes()
    except OSError:
        raise ValueError(f"{script} is missing") from None
    m = CANARY_RE.search(current)
    if m is None:
        raise ValueError(f"{script} has no '# harbor-canary GUID <uuid>' line")
    wanted = reference.replace(b"{guid}", m.group(1))
    if current == wanted:
        return False
    script.write_bytes(wanted)
    return True


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__, file=sys.stderr)
        return 2
    reference = REFERENCE.read_bytes()
    rc = 0
    for task in map(Path, argv):
        try:
            if patch(task, reference):
                print(f"WARNING: patched {task.name}/tests/test.sh to assets/verifier/test.sh", file=sys.stderr)
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
