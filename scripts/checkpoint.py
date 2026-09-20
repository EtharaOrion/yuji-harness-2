#!/usr/bin/env python3
"""Batch checkpoint: crash-safe, resumable progress for multi-step task runs.

A *batch* is one invocation of a driver over many *units*; a unit is one
trajectory -- concretely ``(task, agent, model, run_index)``, which lands in
``output/<slug>/trajectory/Run_N/``. Each unit advances through an ordered list
of *steps* (``preflight``, ``harbor_run``, ``reshape``, ``finance``, ...). This
module records that progress in one JSON file so an interrupted batch resumes
at the last step that actually completed instead of from scratch.

Nothing here knows about Harbor, mcp-atlas, or any particular step: the caller
declares the step names and the unit list, so the same format serves any task
from any run or batch.

Layout, next to the artifacts the batch produces::

    output/_batches/<batch_id>/
    ├── checkpoint.json       authoritative state (atomically replaced)
    ├── checkpoint.json.bak   previous good copy, read if the primary is torn
    ├── checkpoint.lock       {pid, host, started_at} -- one writer per batch
    └── events.jsonl          append-only transition log, for debugging

Four invariants make resume safe:

1. Every transition is flushed with write-tmp, fsync, rename, so a kill at any
   instant leaves either the old file or the new one, never half of one.
2. A step found ``running`` at load time is demoted to ``pending``. The process
   that owned it died mid-step, so its work is presumed incomplete -- trusting
   it optimistically is exactly how a resume silently skips real work.
3. The filesystem outranks the checkpoint. ``reconcile()`` re-checks the
   artifact each completed step recorded and demotes any whose artifact has
   since vanished (wiped ``output/``, restarted instance, partial sync).
4. Resuming a batch whose config fingerprint changed is refused by default. A
   trajectory set graded half under one model and half under another is worse
   than no resume at all.

Because demoted steps re-execute, a step body must either be safe to run twice
or guard itself. Ask ``needs()`` before entering ``step()``.

CLI, for shell callers that only need to move one step (see run_task.sh)::

    python3 scripts/checkpoint.py status  output/_batches/<id>
    python3 scripts/checkpoint.py units   output/_batches/<id> --pending
    python3 scripts/checkpoint.py mark    output/_batches/<id> \\
        --unit <uid> --step harbor_run --status completed --detail exit_code=0
    python3 scripts/checkpoint.py reset   output/_batches/<id> --failed
"""

from __future__ import annotations

import argparse
import contextlib
import errno
import hashlib
import json
import os
import socket
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

SCHEMA_VERSION = "harness-checkpoint-v1"

# Step and unit statuses. PENDING/RUNNING are the only non-terminal ones;
# SKIPPED is terminal and deliberate (a step that had nothing to do), FAILED is
# terminal but retryable via reset().
PENDING = "pending"
RUNNING = "running"
COMPLETED = "completed"
FAILED = "failed"
SKIPPED = "skipped"
ORPHANED = "orphaned"

DONE_STATUSES = (COMPLETED, SKIPPED)

# A lock whose owning pid is gone is stale immediately. This bound only matters
# for a lock written by a *different* host (shared checkpoint dir), where pid
# liveness cannot be tested: 6h is longer than any single batch we run.
STALE_LOCK_SECONDS = 6 * 60 * 60

# Fields on a unit that plan_units() must never overwrite from a plan spec:
# they are execution state, not description.
_STATE_KEYS = frozenset({
    "unit_id", "status", "steps", "outputs", "last_error",
    "created_at", "updated_at", "interruptions", "in_plan",
})


class CheckpointError(RuntimeError):
    """Base class for every failure this module raises deliberately."""


class LockedError(CheckpointError):
    """Another live process owns this batch's checkpoint."""


class FingerprintMismatch(CheckpointError):
    """The batch was created under a different config than the one resuming it."""


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Alive, owned by another user.
        return True
    except OSError:
        return False
    return True


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _atomic_write_json(path: Path, obj: Any, *, keep_backup: bool = True) -> None:
    """Replace ``path`` with ``obj`` such that no reader ever sees a partial file.

    The temp file is fsynced before the rename, and the directory afterwards, so
    the rename itself is durable: after a power loss the old file is intact even
    if the new one never landed. The prior copy is rotated to ``.bak`` first,
    which is what ``_load_doc`` falls back to if the primary is ever unreadable.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.tmp"
    data = json.dumps(obj, indent=2, ensure_ascii=False) + "\n"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, data.encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)
    if keep_backup and path.exists():
        os.replace(path, path.parent / f"{path.name}.bak")
    os.replace(tmp, path)
    dirfd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(dirfd)
    except OSError:
        pass  # not all filesystems allow directory fsync
    finally:
        os.close(dirfd)


def _load_doc(path: Path) -> dict[str, Any] | None:
    """Load the checkpoint, falling back to the backup copy.

    The two-rename write in ``_atomic_write_json`` has one instant where the
    primary name does not exist but ``.bak`` holds the last good state.
    """
    doc = _read_json(path)
    if isinstance(doc, dict):
        return doc
    doc = _read_json(path.parent / f"{path.name}.bak")
    return doc if isinstance(doc, dict) else None


def _resolve(target: str | os.PathLike[str]) -> Path:
    """Accept either the checkpoint file or the batch directory holding it."""
    p = Path(target)
    return p if p.suffix == ".json" else p / "checkpoint.json"


def fingerprint_of(config: Any) -> str:
    """Stable fingerprint of a batch config, for resume-safety checks.

    Canonical JSON (sorted keys) so key order in the caller's dict cannot make
    two identical configs look different.
    """
    blob = json.dumps(config, sort_keys=True, ensure_ascii=False, default=str)
    return "sha256:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# step handle
# ---------------------------------------------------------------------------

class StepHandle:
    """Handed to the body of ``Checkpoint.step()`` to annotate what happened."""

    def __init__(self) -> None:
        self.detail: dict[str, Any] = {}
        self.unit_outputs: dict[str, Any] = {}
        self.skipped: bool = False

    def record(self, **detail: Any) -> None:
        """Merge fields into this step's record (e.g. ``artifact=``, ``exit_code=``).

        ``artifact`` is special: ``reconcile()`` treats it as the on-disk proof
        that this step really produced something, so record it whenever a step
        writes a file or directory.
        """
        self.detail.update(detail)

    def output(self, **values: Any) -> None:
        """Merge fields into the *unit's* outputs (reward, run_dir, passed, ...)."""
        self.unit_outputs.update(values)

    def skip(self, reason: str) -> None:
        """Mark this step deliberately not-applicable; it will not be retried."""
        self.skipped = True
        self.detail["reason"] = reason


# ---------------------------------------------------------------------------
# checkpoint
# ---------------------------------------------------------------------------

class Checkpoint:
    """One batch's progress file. Not a database -- a JSON doc you can read."""

    def __init__(self, path: Path, doc: dict[str, Any], *, read_only: bool) -> None:
        self.path = path
        self._doc = doc
        self._read_only = read_only
        self._lock_path = path.parent / "checkpoint.lock"
        self._events_path = path.parent / "events.jsonl"
        self._lock_held = False
        # One writer per process too: a driver running units concurrently funnels
        # every mutation through this lock, so flushes never interleave.
        self._mutex = threading.RLock()

    # -- construction --------------------------------------------------------

    @classmethod
    def open(
        cls,
        target: str | os.PathLike[str],
        *,
        batch_id: str | None = None,
        steps: Sequence[str] | None = None,
        invocation: dict[str, Any] | None = None,
        fingerprint: str | None = None,
        force: bool = False,
        read_only: bool = False,
        stale_lock_after: float = STALE_LOCK_SECONDS,
    ) -> "Checkpoint":
        """Create or resume a batch checkpoint.

        ``steps`` is required when creating. On resume it is verified against
        the stored list: a driver whose step list changed cannot reason about
        state written by the old one, so the mismatch is refused unless
        ``force``. Same for ``fingerprint`` (see FingerprintMismatch).
        """
        path = _resolve(target)
        if not read_only:
            path.parent.mkdir(parents=True, exist_ok=True)
        doc = _load_doc(path)
        created = doc is None
        fp_override: dict[str, Any] | None = None

        if created:
            if not steps:
                raise CheckpointError(
                    f"no checkpoint at {path} and no steps= given to create one"
                )
            doc = {
                "schema_version": SCHEMA_VERSION,
                "batch_id": batch_id or path.parent.name,
                "status": RUNNING,
                "created_at": _utc(),
                "updated_at": _utc(),
                "steps": list(steps),
                "invocation": invocation or {},
                "config_fingerprint": fingerprint,
                "totals": {},
                "units": [],
            }
        else:
            got = doc.get("schema_version")
            if got != SCHEMA_VERSION:
                raise CheckpointError(
                    f"{path}: schema_version {got!r}, this build writes {SCHEMA_VERSION!r}"
                )
            if steps is not None and list(steps) != list(doc.get("steps") or []):
                if not force:
                    raise CheckpointError(
                        f"{path}: step list changed\n"
                        f"  stored: {doc.get('steps')}\n"
                        f"  now:    {list(steps)}\n"
                        "resume with force=True (--force) to adopt the new list"
                    )
                doc["steps"] = list(steps)
            stored_fp = doc.get("config_fingerprint")
            if fingerprint and stored_fp and stored_fp != fingerprint:
                if not force:
                    raise FingerprintMismatch(
                        f"{path}: batch was created under a different config\n"
                        f"  stored: {stored_fp}\n"
                        f"  now:    {fingerprint}\n"
                        "resume with force=True (--force), or start a new batch id"
                    )
                # --force means the operator accepted a config change mid-batch.
                # Record it loudly: the earlier units ran against the old config
                # and the later ones will not, and both land in the same
                # summary. An override that leaves no trace is indistinguishable
                # from a batch that never changed.
                fp_override = {"stored": stored_fp, "now": fingerprint}
                doc.setdefault("fingerprint_overrides", []).append(
                    {"at": _utc(), **fp_override}
                )
                doc["config_fingerprint"] = fingerprint
            elif fingerprint and not stored_fp:
                doc["config_fingerprint"] = fingerprint
            if batch_id:
                doc["batch_id"] = batch_id
            if invocation:
                # Keep the original, but record what the resuming process looked
                # like -- "which invocation touched this batch" is a real
                # debugging question and the answer is otherwise lost.
                doc.setdefault("resumed_by", []).append(
                    {"at": _utc(), **invocation}
                )

        cp = cls(path, doc, read_only=read_only)
        if read_only:
            return cp

        cp._acquire_lock(force=force, stale_after=stale_lock_after)
        try:
            demoted = cp._demote_running()
            doc["status"] = RUNNING
            cp._recount()
            cp._flush()
        except BaseException:
            cp.release_lock()
            raise
        cp._event(None, None, event="batch_created" if created else "batch_resumed",
                  batch_id=doc["batch_id"], demoted=demoted,
                  fingerprint_override=fp_override)
        if fp_override:
            cp._event(None, None, event="fingerprint_override_forced",
                      batch_id=doc["batch_id"], **fp_override)
        return cp

    # -- context manager -----------------------------------------------------

    def __enter__(self) -> "Checkpoint":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        # An exception escaping the driver means the batch stopped early; leave
        # it interrupted rather than claiming a verdict it did not earn.
        if not self._read_only:
            self.finish(interrupted=exc_type is not None)
        self.release_lock()
        return False

    # -- locking -------------------------------------------------------------

    def _lock_payload(self) -> dict[str, Any]:
        return {
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "started_at": _utc(),
            "epoch": time.time(),
        }

    def _acquire_lock(self, *, force: bool, stale_after: float) -> None:
        payload = json.dumps(self._lock_payload(), indent=2) + "\n"
        try:
            fd = os.open(self._lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileNotFoundError:
            raise
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
            holder = _read_json(self._lock_path) or {}
            if not force and not self._lock_is_stale(holder, stale_after):
                raise LockedError(
                    f"{self.path}: batch is already being written by "
                    f"pid {holder.get('pid')} on {holder.get('host')} "
                    f"(since {holder.get('started_at')}).\n"
                    "Stop it, or pass force=True (--force) to take the lock over."
                )
            # Stale or stolen: replace it atomically so two reclaimers cannot
            # both believe they won.
            _atomic_write_json(self._lock_path, self._lock_payload(), keep_backup=False)
            self._lock_held = True
            return
        try:
            os.write(fd, payload.encode("utf-8"))
            os.fsync(fd)
        finally:
            os.close(fd)
        self._lock_held = True

    @staticmethod
    def _lock_is_stale(holder: dict[str, Any], stale_after: float) -> bool:
        if not holder:
            return True
        pid, host = holder.get("pid"), holder.get("host")
        age = time.time() - float(holder.get("epoch") or 0)
        if host == socket.gethostname() and isinstance(pid, int):
            # Same machine: pid liveness is authoritative, no timeout needed.
            return not _pid_alive(pid)
        return age > stale_after

    def release_lock(self) -> None:
        if self._lock_held:
            with contextlib.suppress(OSError):
                self._lock_path.unlink()
            self._lock_held = False

    # -- persistence ---------------------------------------------------------

    def _flush(self) -> None:
        if self._read_only:
            return
        self._doc["updated_at"] = _utc()
        _atomic_write_json(self.path, self._doc)

    def _event(self, unit_id: str | None, step: str | None, **fields: Any) -> None:
        """Append one line to events.jsonl.

        The checkpoint says where a batch *is*; this says how it got there. It
        is never read back by this module -- it exists so a confusing resume can
        be reconstructed after the fact.
        """
        if self._read_only:
            return
        rec = {"ts": _utc(), "unit_id": unit_id, "step": step, **fields}
        try:
            with open(self._events_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        except OSError:
            pass  # the event log is diagnostic; losing a line must not fail a run

    # -- state -------------------------------------------------------------

    @property
    def doc(self) -> dict[str, Any]:
        return self._doc

    @property
    def batch_id(self) -> str:
        return self._doc["batch_id"]

    @property
    def step_names(self) -> list[str]:
        return list(self._doc["steps"])

    def unit(self, unit_id: str) -> dict[str, Any]:
        for u in self._doc["units"]:
            if u["unit_id"] == unit_id:
                return u
        raise KeyError(f"no unit {unit_id!r} in {self.path}")

    def _demote_running(self) -> list[str]:
        """Rule 2: nothing left ``running`` by a dead process may be trusted."""
        demoted: list[str] = []
        for u in self._doc["units"]:
            hit = False
            for name, st in u.get("steps", {}).items():
                if st.get("status") == RUNNING:
                    st["status"] = PENDING
                    st["demoted_reason"] = "process exited while this step was running"
                    st["demoted_at"] = _utc()
                    demoted.append(f"{u['unit_id']}:{name}")
                    self._event(u["unit_id"], name, event="demoted",
                                **{"from": RUNNING, "to": PENDING})
                    hit = True
            if u.get("status") == RUNNING:
                u["status"] = PENDING
                hit = True
            if hit:
                u["interruptions"] = int(u.get("interruptions") or 0) + 1
        return demoted

    def plan_units(self, specs: Iterable[dict[str, Any]]) -> None:
        """Declare the units this batch covers. Safe to call on every resume.

        New units are appended pending; units already present keep their state
        untouched; units no longer in the plan are marked ``in_plan: false`` and,
        if they had not finished, ``orphaned`` -- kept in the file because the
        record of what a batch once intended to run is evidence, not clutter.
        """
        with self._mutex:
            by_id = {u["unit_id"]: u for u in self._doc["units"]}
            planned: set[str] = set()
            for spec in specs:
                uid = spec.get("unit_id")
                if not uid:
                    raise CheckpointError(f"unit spec without unit_id: {spec!r}")
                planned.add(uid)
                u = by_id.get(uid)
                if u is None:
                    u = {
                        "unit_id": uid,
                        "status": PENDING,
                        "created_at": _utc(),
                        "updated_at": _utc(),
                        "interruptions": 0,
                        "in_plan": True,
                        "steps": {},
                        "outputs": {},
                        "last_error": None,
                    }
                    self._doc["units"].append(u)
                    by_id[uid] = u
                    self._event(uid, None, event="unit_added")
                for key, value in spec.items():
                    if key not in _STATE_KEYS:
                        u[key] = value
                # A step added to the driver between batches shows up pending
                # rather than missing.
                for name in self._doc["steps"]:
                    u["steps"].setdefault(name, {"status": PENDING})
                u["in_plan"] = True
                if u["status"] == ORPHANED:
                    u["status"] = self._unit_status(u)
            for uid, u in by_id.items():
                if uid in planned:
                    continue
                u["in_plan"] = False
                if u["status"] in (PENDING, RUNNING, FAILED):
                    u["status"] = ORPHANED
                    self._event(uid, None, event="unit_orphaned")
            self._recount()
            self._flush()

    # -- queries -------------------------------------------------------------

    def needs(self, unit_id: str, step: str) -> bool:
        """True when this step still has work to do (not completed, not skipped)."""
        u = self.unit(unit_id)
        if not u.get("in_plan", True) or u["status"] == ORPHANED:
            return False
        return u["steps"].get(step, {}).get("status") not in DONE_STATUSES

    def unit_pending(self, unit_id: str, *, include_failed: bool = False) -> bool:
        u = self.unit(unit_id)
        if not u.get("in_plan", True) or u["status"] == ORPHANED:
            return False
        if u["status"] == FAILED and not include_failed:
            return False
        return any(
            u["steps"].get(name, {}).get("status") not in DONE_STATUSES
            for name in self._doc["steps"]
        )

    def iter_pending(self, *, include_failed: bool = False) -> Iterator[dict[str, Any]]:
        """Units with at least one unfinished step, in plan order.

        Failed units are held back by default: re-running them automatically is
        how a broken task burns a whole batch's budget on repeat. ``reset()``
        or ``include_failed`` opts back in.
        """
        for u in list(self._doc["units"]):
            if self.unit_pending(u["unit_id"], include_failed=include_failed):
                yield u

    # -- mutation ------------------------------------------------------------

    @contextlib.contextmanager
    def step(
        self,
        unit_id: str,
        name: str,
        *,
        on_error: str = "raise",
    ) -> Iterator[StepHandle]:
        """Run one step, recording running -> completed/failed/skipped around it.

        The body gets a :class:`StepHandle` to record artifacts and outputs. An
        exception marks the step failed (with the traceback's last frames) and,
        unless ``on_error="record"``, propagates -- a driver that wants to keep
        going with other units passes ``"record"``.
        """
        with self._mutex:
            u = self.unit(unit_id)
            if name not in self._doc["steps"]:
                raise CheckpointError(
                    f"step {name!r} is not one of this batch's steps: {self._doc['steps']}"
                )
            prev = u["steps"].get(name, {})
            prev_status = prev.get("status", PENDING)
            reruns = int(prev.get("reruns") or 0)
            if prev_status in (COMPLETED, FAILED, SKIPPED):
                reruns += 1
            # A fresh record, so a stale error or artifact from the previous
            # attempt cannot be mistaken for this one's.
            st: dict[str, Any] = {"status": RUNNING, "started_at": _utc()}
            if reruns:
                st["reruns"] = reruns
            u["steps"][name] = st
            u["status"] = RUNNING
            u["updated_at"] = _utc()
            self._recount()
            self._flush()
        self._event(unit_id, name, event="step_start", **{"from": prev_status, "to": RUNNING})

        handle = StepHandle()
        t0 = time.monotonic()
        try:
            yield handle
        except (KeyboardInterrupt, SystemExit) as exc:
            # Ctrl-C or a terminating signal is an interruption, not a defect in
            # the step. Recording it as FAILED would hold the unit back from the
            # next resume's pending sweep, which is the opposite of what someone
            # who just interrupted a batch wants.
            with self._mutex:
                u["steps"][name] = {"status": PENDING,
                                    "interrupted_at": _utc(),
                                    "interrupted_by": type(exc).__name__,
                                    **({"reruns": reruns} if reruns else {})}
                u["interruptions"] = int(u.get("interruptions") or 0) + 1
                u["status"] = self._unit_status(u)
                u["updated_at"] = _utc()
                self._recount()
                self._flush()
            self._event(unit_id, name, event="step_interrupted",
                        **{"from": RUNNING, "to": PENDING})
            raise
        except BaseException as exc:
            with self._mutex:
                st["status"] = FAILED
                st["finished_at"] = _utc()
                st["duration_s"] = round(time.monotonic() - t0, 3)
                st.update(handle.detail)
                st["error"] = {
                    "type": type(exc).__name__,
                    "message": str(exc)[:2000],
                    "traceback": "".join(traceback.format_exception(exc))[-2000:],
                }
                u["outputs"].update(handle.unit_outputs)
                u["last_error"] = {"step": name, "at": _utc(),
                                   "type": type(exc).__name__, "message": str(exc)[:2000]}
                u["status"] = FAILED
                u["updated_at"] = _utc()
                self._recount()
                self._flush()
            self._event(unit_id, name, event="step_failed",
                        **{"from": RUNNING, "to": FAILED, "error": type(exc).__name__})
            if on_error == "raise":
                raise
            return
        with self._mutex:
            st["status"] = SKIPPED if handle.skipped else COMPLETED
            st["finished_at"] = _utc()
            st["duration_s"] = round(time.monotonic() - t0, 3)
            st.update(handle.detail)
            u["outputs"].update(handle.unit_outputs)
            u["last_error"] = None
            u["status"] = self._unit_status(u)
            u["updated_at"] = _utc()
            self._recount()
            self._flush()
        self._event(unit_id, name, event="step_done",
                    **{"from": RUNNING, "to": st["status"]})

    def mark(self, unit_id: str, step: str, status: str, **detail: Any) -> None:
        """Set one step's status directly. For shell callers (see the CLI).

        ``step()`` is the API for in-process drivers; this exists because
        run_task.sh crosses stage boundaries in bash, where a context manager
        is not available.
        """
        if status not in (PENDING, RUNNING, COMPLETED, FAILED, SKIPPED):
            raise CheckpointError(f"unknown status {status!r}")
        with self._mutex:
            u = self.unit(unit_id)
            if step not in self._doc["steps"]:
                raise CheckpointError(
                    f"step {step!r} is not one of this batch's steps: {self._doc['steps']}"
                )
            st = u["steps"].setdefault(step, {"status": PENDING})
            prev = st.get("status", PENDING)
            st["status"] = status
            st["updated_at"] = _utc()
            st.update(detail)
            if status == FAILED:
                u["last_error"] = {"step": step, "at": _utc(),
                                   "message": str(detail.get("message", ""))[:2000]}
            u["status"] = RUNNING if status == RUNNING else self._unit_status(u)
            u["updated_at"] = _utc()
            self._recount()
            self._flush()
        self._event(unit_id, step, event="step_marked", **{"from": prev, "to": status})

    def set_outputs(self, unit_id: str, **values: Any) -> None:
        with self._mutex:
            self.unit(unit_id)["outputs"].update(values)
            self._flush()

    def reset(
        self,
        *,
        unit_id: str | None = None,
        step: str | None = None,
        failed_only: bool = False,
    ) -> list[str]:
        """Return steps to ``pending`` so the next resume re-runs them.

        With no arguments this resets everything unfinished-or-failed in the
        batch. Returns the ``unit:step`` pairs it touched. On a read-only
        checkpoint this changes the in-memory doc only, which is how callers
        preview a resume.
        """
        touched: list[str] = []
        with self._mutex:
            for u in self._doc["units"]:
                if unit_id and u["unit_id"] != unit_id:
                    continue
                for name in self._doc["steps"]:
                    if step and name != step:
                        continue
                    st = u["steps"].setdefault(name, {"status": PENDING})
                    if failed_only and st.get("status") != FAILED:
                        continue
                    if st.get("status") == PENDING:
                        continue
                    prev = st.get("status")
                    u["steps"][name] = {"status": PENDING,
                                        "reset_from": prev, "reset_at": _utc()}
                    touched.append(f"{u['unit_id']}:{name}")
                    self._event(u["unit_id"], name, event="reset",
                                **{"from": prev, "to": PENDING})
                u["last_error"] = None
                if u["status"] in (FAILED, ORPHANED) and u.get("in_plan", True):
                    u["status"] = self._unit_status(u)
                elif u.get("in_plan", True):
                    u["status"] = self._unit_status(u)
            self._recount()
            self._flush()
        return touched

    def reconcile(
        self,
        *,
        root: str | os.PathLike[str] | None = None,
        verifiers: dict[str, Callable[[dict[str, Any], dict[str, Any]], bool | None]] | None = None,
    ) -> list[str]:
        """Rule 3: demote completed steps whose evidence is gone from disk.

        For each completed step, a per-step ``verifier(unit, step)`` decides --
        returning ``None`` to abstain. With no verifier, a step that recorded an
        ``artifact`` path is checked for that path's existence (resolved against
        ``root``); a step that recorded nothing is left alone, since there is
        nothing to check it against.

        Returns the ``unit:step`` pairs it demoted.
        """
        base = Path(root) if root else Path.cwd()
        verifiers = verifiers or {}
        demoted: list[str] = []
        with self._mutex:
            for u in self._doc["units"]:
                for name in self._doc["steps"]:
                    st = u["steps"].get(name)
                    if not st or st.get("status") != COMPLETED:
                        continue
                    verifier = verifiers.get(name)
                    if verifier is not None:
                        ok = verifier(u, st)
                        reason = f"verifier for {name!r} rejected the recorded result"
                    elif st.get("artifact"):
                        p = Path(st["artifact"])
                        p = p if p.is_absolute() else base / p
                        ok = p.exists()
                        reason = f"artifact missing: {st['artifact']}"
                    else:
                        ok = None
                        reason = ""
                    if ok is False:
                        demoted.append(f"{u['unit_id']}:{name}")
                        u["steps"][name] = {"status": PENDING,
                                            "demoted_reason": reason,
                                            "demoted_at": _utc(),
                                            "demoted_from": COMPLETED}
                        self._event(u["unit_id"], name, event="reconcile_demoted",
                                    **{"from": COMPLETED, "to": PENDING, "reason": reason})
                if u.get("in_plan", True) and u["status"] != ORPHANED:
                    u["status"] = self._unit_status(u)
            self._recount()
            self._flush()
        return demoted

    def finish(self, *, interrupted: bool = False) -> str:
        """Close the batch out, returning its final status."""
        with self._mutex:
            for u in self._doc["units"]:
                if u.get("in_plan", True) and u["status"] not in (ORPHANED, FAILED):
                    u["status"] = self._unit_status(u)
            live = [u for u in self._doc["units"] if u.get("in_plan", True)]
            if interrupted:
                status = "interrupted"
            elif any(u["status"] == FAILED for u in live):
                status = FAILED
            elif all(u["status"] in DONE_STATUSES for u in live):
                status = COMPLETED
            else:
                status = "interrupted"
            self._doc["status"] = status
            self._doc["finished_at"] = _utc()
            self._recount()
            self._flush()
        self._event(None, None, event="batch_finished", status=status)
        return status

    # -- rollups -------------------------------------------------------------

    def _unit_status(self, u: dict[str, Any]) -> str:
        states = [u["steps"].get(n, {}).get("status", PENDING) for n in self._doc["steps"]]
        if any(s == FAILED for s in states):
            return FAILED
        if all(s in DONE_STATUSES for s in states):
            return COMPLETED
        if any(s == RUNNING for s in states):
            return RUNNING
        return PENDING

    def _recount(self) -> None:
        live = [u for u in self._doc["units"] if u.get("in_plan", True)]
        counts = {PENDING: 0, RUNNING: 0, COMPLETED: 0, FAILED: 0}
        for u in live:
            counts[u["status"]] = counts.get(u["status"], 0) + 1
        self._doc["totals"] = {
            "units": len(live),
            "orphaned": len(self._doc["units"]) - len(live),
            **counts,
        }

    def summary(self) -> dict[str, Any]:
        """Counts by unit status and, per step, counts by step status."""
        live = [u for u in self._doc["units"] if u.get("in_plan", True)]
        per_step: dict[str, dict[str, int]] = {}
        for name in self._doc["steps"]:
            bucket: dict[str, int] = {}
            for u in live:
                s = u["steps"].get(name, {}).get("status", PENDING)
                bucket[s] = bucket.get(s, 0) + 1
            per_step[name] = bucket
        return {
            "batch_id": self._doc["batch_id"],
            "status": self._doc["status"],
            "totals": dict(self._doc.get("totals") or {}),
            "steps": per_step,
        }

    def render_status(self) -> str:
        s = self.summary()
        t = s["totals"]
        lines = [
            f"batch {s['batch_id']}  [{s['status']}]  {self.path}",
            f"  units: {t.get('units', 0)} "
            f"({t.get('completed', 0)} completed, {t.get('failed', 0)} failed, "
            f"{t.get('pending', 0)} pending, {t.get('orphaned', 0)} orphaned)",
            "  steps:",
        ]
        for name, bucket in s["steps"].items():
            detail = ", ".join(f"{k}={v}" for k, v in sorted(bucket.items()))
            lines.append(f"    {name:<14} {detail}")
        failed = [u for u in self._doc["units"] if u["status"] == FAILED]
        if failed:
            lines.append("  failed units:")
            for u in failed:
                err = u.get("last_error") or {}
                lines.append(
                    f"    {u['unit_id']}  at {err.get('step', '?')}: "
                    f"{str(err.get('message', '')).splitlines()[0][:100] if err.get('message') else ''}"
                )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _kv(pairs: Sequence[str]) -> dict[str, Any]:
    """Parse ``k=v`` CLI pairs, JSON-decoding values that look like JSON."""
    out: dict[str, Any] = {}
    for pair in pairs or ():
        if "=" not in pair:
            raise SystemExit(f"expected key=value, got {pair!r}")
        k, v = pair.split("=", 1)
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("target", help="batch dir (output/_batches/<id>) or checkpoint.json")
        p.add_argument("--force", action="store_true",
                       help="take the lock over even if another process holds it")
        return p

    p_status = common(sub.add_parser("status", help="print a progress summary"))
    p_status.add_argument("--json", action="store_true", help="emit summary as JSON")

    p_units = common(sub.add_parser("units", help="list unit ids"))
    p_units.add_argument("--pending", action="store_true", help="only unfinished units")
    p_units.add_argument("--include-failed", action="store_true")

    p_mark = common(sub.add_parser("mark", help="set one step's status"))
    p_mark.add_argument("--unit", required=True)
    p_mark.add_argument("--step", required=True)
    p_mark.add_argument("--status", required=True,
                        choices=[PENDING, RUNNING, COMPLETED, FAILED, SKIPPED])
    p_mark.add_argument("--detail", action="append", default=[], metavar="K=V",
                        help="extra fields on the step record (repeatable)")

    p_reset = common(sub.add_parser("reset", help="return steps to pending"))
    p_reset.add_argument("--unit")
    p_reset.add_argument("--step")
    p_reset.add_argument("--failed", action="store_true", help="only failed steps")

    p_rec = common(sub.add_parser("reconcile", help="demote steps whose artifacts vanished"))
    p_rec.add_argument("--root", default=".", help="base for relative artifact paths")

    a = ap.parse_args(argv)

    # status/units never write, so they never contend for the lock.
    read_only = a.cmd in ("status", "units")
    try:
        cp = Checkpoint.open(a.target, force=a.force, read_only=read_only)
    except CheckpointError as exc:
        print(f"checkpoint: {exc}", file=sys.stderr)
        return 2

    try:
        if a.cmd == "status":
            print(json.dumps(cp.summary(), indent=2) if a.json else cp.render_status())
        elif a.cmd == "units":
            units = (cp.iter_pending(include_failed=a.include_failed) if a.pending
                     else (u for u in cp.doc["units"] if u.get("in_plan", True)))
            for u in units:
                print(u["unit_id"])
        elif a.cmd == "mark":
            cp.mark(a.unit, a.step, a.status, **_kv(a.detail))
        elif a.cmd == "reset":
            for pair in cp.reset(unit_id=a.unit, step=a.step, failed_only=a.failed):
                print(f"reset {pair}")
        elif a.cmd == "reconcile":
            for pair in cp.reconcile(root=a.root):
                print(f"demoted {pair}")
    except (CheckpointError, KeyError) as exc:
        print(f"checkpoint: {exc}", file=sys.stderr)
        return 2
    finally:
        cp.release_lock()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
