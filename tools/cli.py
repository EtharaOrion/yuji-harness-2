"""The harness's two registered entry points: inference and evaluation.

    harness-infer [--stage STAGE] <task-dir>       run the agent on one bundle
    harness-eval <trial-dir> <task-dir> [-- ...]   score a recorded rollout

Inference is scripts/run_task.sh, which runs the agent under test through
harbor and leaves a trial directory. Evaluation scores a recorded trial with the
bundle's OWN verifier: `harbor trial regrade` seeds a new trial with the
source's agent logs and artifacts and runs the task's tests/ in a separate
verifier environment, leaving the source untouched. Neither carries grading
logic; the reward is whatever the bundle's tests write.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUN_TASK = REPO / "scripts" / "run_task.sh"


def infer(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if not args or args[0] in ("-h", "--help"):
        print(__doc__.strip())
        return 0 if args else 2
    os.execv(str(RUN_TASK), [str(RUN_TASK), *args])
    return 0


def evaluate(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) < 2 or args[0] in ("-h", "--help"):
        print(__doc__.strip())
        return 0 if args[:1] in (["-h"], ["--help"]) else 2
    trial, task, rest = Path(args[0]), Path(args[1]), args[2:]
    if rest[:1] == ["--"]:
        rest = rest[1:]
    if not (task / "task.toml").is_file():
        print(f"harness-eval: not a task dir (no task.toml): {task}", file=sys.stderr)
        return 2
    if not trial.is_dir():
        print(f"harness-eval: no trial directory: {trial}", file=sys.stderr)
        return 2
    harbor = shutil.which("harbor")
    if harbor is None:
        print("harness-eval: harbor not found; make install-harbor", file=sys.stderr)
        return 2
    cmd = [harbor, "trial", "regrade", str(trial), "--task-path", str(task), *rest]
    os.execv(harbor, cmd)
    return 0
