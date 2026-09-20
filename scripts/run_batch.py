#!/usr/bin/env python3
"""Run many Harbor tasks as one resumable batch.

Where ``scripts/run_task.sh`` runs one task once, this drives a whole batch --
every task x every attempt -- through the four stages run_task.sh exposes,
recording each transition in a checkpoint (``scripts/checkpoint.py``). Kill it
at any point, restart the same command, and it picks up at the first step that
had not finished, for every unit, without re-running the agent phase it already
paid for.

    scripts/run_batch.py --all --model claude-opus-5 --n 3
    scripts/run_batch.py --all --model claude-opus-5 --n 3      # ... resume
    scripts/run_batch.py --all --model claude-opus-5 --n 3 --dry-run
    scripts/run_batch.py --batch-id nightly-2026-08-31 --status

A *unit* is one trajectory: ``<task-slug>::<model>::run-<i>``, landing in
``output/<slug>/trajectory/Run_<base+i>``. ``base`` is the number of Run_* dirs
that existed when the batch was first planned, so a batch appends to a task's
history instead of overwriting it, and the mapping is frozen in the checkpoint
so a resume always targets the same directory.

Steps, and what resume does with each:

    preflight   auth/docker/image      re-run freely, it is idempotent
    harbor_run  the agent phase        never re-run once it has any evidence on
                                       disk; this is the minutes-and-money step
    reshape     harbor_to_output.py    re-run freely, it rebuilds Run_N in place
    finance     Odoo usage report      guarded by finance_receipt.json so a
                                       re-drive cannot double-post

Units of the same task always run sequentially -- they share a job directory
and a Run_N counter. ``--concurrency`` only fans out across different tasks.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "tools" / "delivery"))
import checkpoint as ckpt  # noqa: E402
from harbor_to_output import norm_reward  # noqa: E402

RUN_TASK = REPO / "scripts" / "run_task.sh"

# The checkpoint's step list. Order matters: a unit runs them top to bottom.
# Changing this list invalidates in-flight batches (checkpoint.open refuses the
# mismatch), so add steps at a batch boundary, not mid-run.
STEPS = ["preflight", "harbor_run", "reshape", "finance"]

# checkpoint step name -> run_task.sh --stage name
STAGE_OF = {"preflight": "preflight", "harbor_run": "harbor",
            "reshape": "reshape", "finance": "finance"}


class StageError(RuntimeError):
    """A run_task.sh stage exited non-zero."""


def _log(msg: str) -> None:
    print(f"[run_batch] {msg}", flush=True)


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def _existing_runs(traj_dir: Path) -> int:
    """How many Run_N dirs a task already has, i.e. where this batch appends."""
    if not traj_dir.is_dir():
        return 0
    nums = [int(p.name[4:]) for p in traj_dir.iterdir()
            if p.is_dir() and p.name.startswith("run_") and p.name[4:].isdigit()]
    return max(nums) if nums else 0


def discover_tasks(args: argparse.Namespace) -> list[Path]:
    if args.all:
        root = Path(args.tasks_dir)
        found = sorted(p for p in root.iterdir()
                       if p.is_dir() and (p / "task.toml").is_file())
        if not found:
            raise SystemExit(f"no task dirs (with task.toml) under {root}")
        return found
    out = []
    for t in args.task:
        p = Path(t)
        if not (p / "task.toml").is_file():
            raise SystemExit(f"not a task dir (no task.toml): {p}")
        out.append(p)
    if not out:
        raise SystemExit("nothing to run: pass --task <dir> (repeatable) or --all")
    return out


# ---------------------------------------------------------------------------
# planning
# ---------------------------------------------------------------------------

def build_units(cp: ckpt.Checkpoint, tasks: Iterable[Path], args: argparse.Namespace
                ) -> list[dict[str, Any]]:
    """One spec per (task, attempt), reusing offsets this batch already fixed.

    The Run_N a unit owns is decided once, at first plan, and then read back out
    of the checkpoint forever after. Recomputing it from disk on every resume
    would slide the whole batch forward by however many runs it had completed.
    """
    prior_base: dict[str, int] = {}
    for u in cp.doc["units"]:
        slug, base = u.get("task_slug"), u.get("base_offset")
        if slug is not None and base is not None:
            prior_base[slug] = base

    output_dir = Path(args.output_dir)
    specs: list[dict[str, Any]] = []
    for task in tasks:
        slug = task.name
        if slug not in prior_base:
            prior_base[slug] = _existing_runs(output_dir / slug / "trajectory")
        base = prior_base[slug]
        for i in range(1, args.n + 1):
            run_no = base + i
            specs.append({
                "unit_id": f"{slug}::{args.model}::run-{i}",
                "task": str(task),
                "task_slug": slug,
                "agent": args.agent,
                "model": args.model,
                "run_index": i,
                "base_offset": base,
                # run_task.sh takes the offset, i.e. the run number minus one.
                "run_offset": run_no - 1,
                "run_dir": str(output_dir / slug / "trajectory" / f"run_{run_no}"),
                "job_dir": str(output_dir / slug),
            })
    return specs


# ---------------------------------------------------------------------------
# execution
# ---------------------------------------------------------------------------

def run_stage(stage: str, unit: dict[str, Any], args: argparse.Namespace,
              log_path: Path) -> None:
    """Invoke one run_task.sh stage, streaming its output to console and log."""
    env = os.environ.copy()
    env.update({
        "STAGE": stage,
        "AGENT": unit["agent"],
        "MODEL": unit["model"],
        "N": "1",                       # one attempt per unit; the batch owns the count
        "JOB": unit["task_slug"],
        "OUTPUT_DIR": str(Path(args.output_dir).resolve()),
        "AT": str(args.at),
        "BUILD_MULT": str(args.build_mult),
        "RUN_OFFSET": str(unit["run_offset"]),
    })
    if args.copy_to:
        env["COPY_TO"] = args.copy_to

    log_path.parent.mkdir(parents=True, exist_ok=True)
    prefix = f"[{unit['unit_id']}:{stage}]"
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"\n=== {stage} at {datetime.now(timezone.utc).isoformat()} ===\n")
        log.flush()
        proc = subprocess.Popen(
            [str(RUN_TASK), "--stage", stage, unit["task"]],
            cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            log.write(line)
            log.flush()
            print(f"{prefix} {line.rstrip()}", flush=True)
        code = proc.wait()
    if code != 0:
        raise StageError(f"run_task.sh --stage {stage} exited {code} (log: {log_path})")


def drive_unit(cp: ckpt.Checkpoint, unit_id: str, args: argparse.Namespace,
               batch_dir: Path, stop: threading.Event) -> None:
    """Advance one unit through every step it still needs."""
    for step in STEPS:
        if stop.is_set():
            return
        if not cp.needs(unit_id, step):
            continue
        unit = cp.unit(unit_id)
        log_path = batch_dir / "logs" / f"{unit_id.replace('::', '__')}.log"
        with cp.step(unit_id, step, on_error="record") as st:
            run_stage(STAGE_OF[step], unit, args, log_path)
            st.record(log=str(log_path))
            if step == "harbor_run":
                # Recorded as plain detail, not as `artifact`: reconcile must
                # never demote the agent phase on a path check alone.
                st.record(job_dir=unit["job_dir"])
                # run_task.sh deliberately swallows harbor's exit code so it can
                # reshape a partial run, which means a total failure (bad auth,
                # missing image, aborted build) still exits 0. Checking for the
                # artifacts instead is the difference between a batch that
                # reports 50 green units and one that produced anything.
                if not _harbor_evidence(unit, st.detail):
                    raise StageError(
                        f"harbor left no trial for this unit under {unit['job_dir']} "
                        f"— see {log_path}")
            elif step == "reshape":
                if not Path(unit["run_dir"]).is_dir():
                    raise StageError(_reshape_failure_hint(unit, log_path))
                st.record(artifact=unit["run_dir"])
                st.output(run_dir=unit["run_dir"], **_grade(Path(unit["run_dir"])))
            elif step == "finance":
                receipt = Path(unit["run_dir"]) / "finance_receipt.json"
                if receipt.is_file():
                    st.record(artifact=str(receipt))
                else:
                    st.skip("no finance receipt (ODOO_URL unset or reporting skipped)")
        if cp.unit(unit_id)["status"] == ckpt.FAILED:
            err = (cp.unit(unit_id).get("last_error") or {}).get("message", "")
            _log(f"{unit_id}: FAILED at {step}: {err}")
            if args.stop_on_failure:
                stop.set()
            return
    _log(f"{unit_id}: {cp.unit(unit_id)['status']}")


def _grade(run_dir: Path) -> dict[str, Any]:
    """Best-effort reward/pass readout, so the checkpoint doubles as a scoreboard."""
    out: dict[str, Any] = {}
    try:
        reward = json.loads((run_dir / "verifier" / "reward.json").read_text())
        out["reward"] = norm_reward(reward.get("reward", reward.get("value")))
    except (OSError, json.JSONDecodeError, AttributeError):
        pass
    try:
        report = json.loads((run_dir / "report.json").read_text())
        if isinstance(report, dict) and "passed" in report:
            out["passed"] = report["passed"]
    except (OSError, json.JSONDecodeError):
        pass
    return out


# ---------------------------------------------------------------------------
# reconcile support
# ---------------------------------------------------------------------------

def _raw_trials_present(job_dir: Path, slug: str) -> bool:
    """True while Harbor's unconsumed trial dirs are still in the job dir.

    reshape MOVES these into .raw/, so their presence is what distinguishes
    "reshape has not run yet" from "reshape already ran and ate its input".
    """
    if not job_dir.is_dir():
        return False
    return any(p.is_dir() and p.name.startswith(f"{slug}__") for p in job_dir.iterdir())


def _harbor_evidence(unit: dict[str, Any], step: dict[str, Any]) -> bool:
    """True while any on-disk trace of the agent phase survives.

    Deliberately generous. Re-running the agent because a heuristic could not
    find its output costs real money, so this only reports "gone" when the
    reshaped run, the raw trial dirs, and the .raw tree are all absent.
    """
    if Path(unit.get("run_dir") or "").exists():
        return True
    job_dir = Path(step.get("job_dir") or unit.get("job_dir") or "")
    if not job_dir.is_dir():
        return False
    if (job_dir / ".raw").exists():
        return True
    return _raw_trials_present(job_dir, unit.get("task_slug") or "")


def _reshape_failure_hint(unit: dict[str, Any], log_path: Path) -> str:
    """Explain a reshape that produced nothing, and name the only way out.

    reshape is re-runnable only while its input is still there. Once an earlier
    pass moved the trial into .raw/, a deleted Run_N cannot be rebuilt from
    disk, and the only remedy costs a fresh agent run -- which is the operator's
    call to make, never this driver's.
    """
    base = f"reshape did not produce {unit['run_dir']} — see {log_path}"
    if _raw_trials_present(Path(unit["job_dir"]), unit.get("task_slug") or ""):
        return base
    return (
        f"{base}\n"
        f"  The raw Harbor trial was already consumed into {unit['job_dir']}/.raw by an\n"
        f"  earlier reshape, so this run cannot be rebuilt from what is on disk.\n"
        f"  Re-run the agent phase (costs a fresh agent run):\n"
        f"    --reset-step harbor_run --retry-failed"
    )


VERIFIERS = {"harbor_run": _harbor_evidence}


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def print_plan(cp: ckpt.Checkpoint) -> None:
    print(cp.render_status())
    print("\nresume plan:")
    any_work = False
    for u in cp.doc["units"]:
        if not u.get("in_plan", True):
            continue
        todo = [s for s in STEPS if cp.needs(u["unit_id"], s)]
        done = [s for s in STEPS if not cp.needs(u["unit_id"], s)]
        if not todo:
            print(f"  {u['unit_id']:<52} nothing to do ({u['status']})")
            continue
        any_work = True
        held = " [held back: failed, pass --retry-failed]" if u["status"] == ckpt.FAILED else ""
        print(f"  {u['unit_id']:<52} skip={','.join(done) or '-'} "
              f"run={','.join(todo)}{held}")
    if not any_work:
        print("  (batch is complete)")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Re-running the same command resumes; nothing else is needed.",
    )
    ap.add_argument("--task", action="append", default=[], metavar="DIR",
                    help="task dir to run (repeatable)")
    ap.add_argument("--all", action="store_true", help="run every task under --tasks-dir")
    ap.add_argument("--tasks-dir", default="tasks", help="where --all looks (default: tasks)")
    ap.add_argument("--model", default="claude-opus-5")
    ap.add_argument("--agent", default="claude-code")
    ap.add_argument("--n", type=int, default=1, help="attempts per task (default: 1)")
    ap.add_argument("--at", default="auto",
                    help="pass@k ks handed to the reshaper ('auto' = every k from 1..runs)")
    ap.add_argument("--build-mult", default="3", help="environment build timeout multiplier")
    ap.add_argument("--output-dir", default=str(REPO / "output"))
    ap.add_argument("--copy-to", default="", help="extra mirror for the reshaped bundle")
    ap.add_argument("--batch-id", default=None,
                    help="batch name (default: <model>-<UTC timestamp>). Reuse it to resume.")
    ap.add_argument("--batch-dir", default=None,
                    help="checkpoint dir (default: <output-dir>/_batches/<batch-id>)")
    ap.add_argument("--concurrency", type=int, default=1,
                    help="tasks in flight at once (default: 1). Attempts of one task "
                         "always run sequentially.")
    ap.add_argument("--status", action="store_true", help="print progress and exit")
    ap.add_argument("--dry-run", action="store_true", help="print the resume plan and exit")
    ap.add_argument("--retry-failed", action="store_true",
                    help="reset failed steps so this run retries them")
    ap.add_argument("--reset-step", metavar="STEP", default=None,
                    help=f"reset one step across the batch before running ({'|'.join(STEPS)})")
    ap.add_argument("--no-reconcile", action="store_true",
                    help="skip the on-disk check that demotes steps whose output vanished")
    ap.add_argument("--stop-on-failure", action="store_true",
                    help="stop launching new work after the first failed unit")
    ap.add_argument("--force", action="store_true",
                    help="resume despite a changed config, and take the lock over")
    args = ap.parse_args(argv)
    if args.reset_step and args.reset_step not in STEPS:
        ap.error(f"--reset-step must be one of {STEPS}")
    if args.n < 1:
        ap.error("--n must be >= 1")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # --status must work without knowing the task list, so it reads and exits
    # before any planning happens.
    if args.status:
        batch_dir = _batch_dir(args, require_existing=True)
        cp = ckpt.Checkpoint.open(batch_dir, read_only=True)
        print(cp.render_status())
        return 0

    tasks = discover_tasks(args)
    batch_dir = _batch_dir(args)
    fingerprint = ckpt.fingerprint_of({
        "agent": args.agent, "model": args.model, "n": args.n, "at": args.at,
        "build_mult": args.build_mult, "output_dir": str(Path(args.output_dir).resolve()),
        "tasks": sorted(t.name for t in tasks),
        "harness_commit": _git_commit(),
    })

    try:
        cp = ckpt.Checkpoint.open(
            batch_dir,
            batch_id=batch_dir.name,
            steps=STEPS,
            invocation={"argv": sys.argv, "cwd": os.getcwd(),
                        "git_commit": _git_commit()},
            fingerprint=fingerprint,
            force=args.force,
            # A preview opens read-only: every write below (planning, reconcile,
            # resets, even the crash-recovery demotion open() would normally do)
            # then lands in memory only. The plan stays exact, the batch dir is
            # untouched, and no lock is taken -- so this is also safe to run
            # against a batch that is live in another terminal.
            read_only=args.dry_run,
        )
    except ckpt.CheckpointError as exc:
        print(f"run_batch: {exc}", file=sys.stderr)
        return 2

    exit_code = 0
    try:
        cp.plan_units(build_units(cp, tasks, args))

        verb = "would " if args.dry_run else ""
        if not args.no_reconcile:
            for pair in cp.reconcile(root=REPO, verifiers=VERIFIERS):
                _log(f"reconcile: {pair} -> pending (its output is no longer on disk)")

        if args.reset_step:
            for pair in cp.reset(step=args.reset_step):
                _log(f"{verb}reset {pair}")
        if args.retry_failed:
            for pair in cp.reset(failed_only=True):
                _log(f"{verb}retry {pair}")

        if args.dry_run:
            print_plan(cp)
            return 0

        pending = [u["unit_id"] for u in cp.iter_pending()]
        if not pending:
            # Nothing *runnable* is not the same as nothing wrong: units held
            # back because they failed still make this batch a failure, and a
            # caller looping on the exit code must not see green.
            status = cp.finish()
            _log("nothing to do — no unit has a runnable step left "
                 "(use --retry-failed or --reset-step to redo work)")
            print(cp.render_status())
            return 0 if status == ckpt.COMPLETED else 1

        _log(f"batch {cp.batch_id}: {len(pending)} unit(s) to advance "
             f"-> {cp.path.parent}")

        stop = threading.Event()
        previous = _install_signal_handlers(stop)
        try:
            _drive(cp, pending, args, batch_dir, stop)
        finally:
            _restore_signal_handlers(previous)

        status = cp.finish(interrupted=stop.is_set())
        print(cp.render_status())
        exit_code = 0 if status == ckpt.COMPLETED else 1
    except KeyboardInterrupt:
        cp.finish(interrupted=True)
        _log("interrupted — rerun the same command to resume")
        exit_code = 130
    finally:
        cp.release_lock()
    return exit_code


def _drive(cp: ckpt.Checkpoint, pending: list[str], args: argparse.Namespace,
           batch_dir: Path, stop: threading.Event) -> None:
    """Run the pending units, fanning out across tasks but never within one."""
    chains: dict[str, list[str]] = {}
    for unit_id in pending:
        chains.setdefault(cp.unit(unit_id)["task_slug"], []).append(unit_id)

    def run_chain(unit_ids: list[str]) -> None:
        for unit_id in unit_ids:
            if stop.is_set():
                return
            drive_unit(cp, unit_id, args, batch_dir, stop)

    if args.concurrency <= 1 or len(chains) == 1:
        for unit_ids in chains.values():
            run_chain(unit_ids)
        return
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        list(pool.map(run_chain, chains.values()))


def _install_signal_handlers(stop: threading.Event) -> dict[int, Any]:
    """First signal stops launching new work; the second is the usual hard exit.

    Returns the handlers it displaced so the caller can put them back -- this is
    a library-callable entry point, not only a CLI.
    """
    def handler(signum, _frame):
        if stop.is_set():
            raise KeyboardInterrupt
        stop.set()
        _log(f"signal {signum} — finishing the current step, then stopping "
             "(signal again to abort now)")
    previous: dict[int, Any] = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            previous[sig] = signal.signal(sig, handler)
        except ValueError:
            pass  # not on the main thread; the caller owns signals
    return previous


def _restore_signal_handlers(previous: dict[int, Any]) -> None:
    for sig, handler in previous.items():
        try:
            signal.signal(sig, handler)
        except ValueError:
            pass


def _batch_dir(args: argparse.Namespace, *, require_existing: bool = False) -> Path:
    if args.batch_dir:
        return Path(args.batch_dir)
    base = Path(args.output_dir) / "_batches"
    if args.batch_id:
        return base / args.batch_id
    if require_existing:
        # --status with no id: the most recently touched batch is the one meant.
        candidates = sorted((p for p in base.glob("*/checkpoint.json")),
                            key=lambda p: p.stat().st_mtime, reverse=True) if base.is_dir() else []
        if not candidates:
            raise SystemExit(f"no batches under {base} — pass --batch-id")
        return candidates[0].parent
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return base / f"{args.model}-{stamp}"


if __name__ == "__main__":
    raise SystemExit(main())
