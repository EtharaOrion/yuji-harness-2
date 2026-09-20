"""Unit tests for scripts/checkpoint.py.

The interesting cases are the ones that only matter after a crash: a step left
running, an artifact that vanished, a second driver on the same batch, a config
that changed under a resume. Those get the most coverage here.
"""
import json
import os
import socket
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import checkpoint as cp  # noqa: E402

STEPS = ["preflight", "harbor_run", "reshape", "finance"]
UNITS = [
    {"unit_id": "alpha::run-1", "task": "tasks/alpha", "run_index": 1},
    {"unit_id": "beta::run-1", "task": "tasks/beta", "run_index": 1},
]


def _open(tmp_path, **kw):
    kw.setdefault("steps", STEPS)
    kw.setdefault("batch_id", "test-batch")
    return cp.Checkpoint.open(tmp_path / "batch", **kw)


def _dead_pid() -> int:
    """A pid that is not currently running, for stale-lock tests."""
    for candidate in range(32000, 4000, -7):
        if not cp._pid_alive(candidate):
            return candidate
    pytest.skip("no dead pid available on this machine")


# ---------------------------------------------------------------------------
# creation, planning, persistence
# ---------------------------------------------------------------------------

def test_create_writes_file_and_plans_units(tmp_path):
    c = _open(tmp_path)
    c.plan_units(UNITS)
    c.release_lock()

    doc = json.loads((tmp_path / "batch" / "checkpoint.json").read_text())
    assert doc["schema_version"] == cp.SCHEMA_VERSION
    assert doc["steps"] == STEPS
    assert [u["unit_id"] for u in doc["units"]] == ["alpha::run-1", "beta::run-1"]
    assert doc["units"][0]["task"] == "tasks/alpha"
    assert all(s["status"] == "pending" for s in doc["units"][0]["steps"].values())
    assert doc["totals"] == {"units": 2, "orphaned": 0, "pending": 2,
                             "running": 0, "completed": 0, "failed": 0}


def test_create_requires_steps(tmp_path):
    with pytest.raises(cp.CheckpointError, match="no steps"):
        cp.Checkpoint.open(tmp_path / "batch")


def test_state_survives_replan_and_new_units_are_appended(tmp_path):
    c = _open(tmp_path)
    c.plan_units(UNITS)
    with c.step("alpha::run-1", "preflight"):
        pass
    c.release_lock()

    c2 = _open(tmp_path)
    c2.plan_units(UNITS + [{"unit_id": "gamma::run-1", "task": "tasks/gamma"}])
    assert c2.unit("alpha::run-1")["steps"]["preflight"]["status"] == "completed"
    assert c2.needs("gamma::run-1", "preflight")
    c2.release_lock()


def test_units_dropped_from_the_plan_are_orphaned_not_deleted(tmp_path):
    c = _open(tmp_path)
    c.plan_units(UNITS)
    c.plan_units(UNITS[:1])
    beta = c.unit("beta::run-1")
    assert beta["status"] == "orphaned" and beta["in_plan"] is False
    assert [u["unit_id"] for u in c.iter_pending()] == ["alpha::run-1"]
    # Re-declaring it brings it back to life.
    c.plan_units(UNITS)
    assert c.unit("beta::run-1")["status"] == "pending"
    c.release_lock()


def test_step_list_change_is_refused_without_force(tmp_path):
    _open(tmp_path).release_lock()
    with pytest.raises(cp.CheckpointError, match="step list changed"):
        _open(tmp_path, steps=STEPS + ["delivery"])
    c = _open(tmp_path, steps=STEPS + ["delivery"], force=True)
    assert c.step_names[-1] == "delivery"
    c.release_lock()


# ---------------------------------------------------------------------------
# step execution
# ---------------------------------------------------------------------------

def test_step_is_flushed_while_still_running(tmp_path):
    """Rule 1: progress is on disk before the step ends, not after."""
    c = _open(tmp_path)
    c.plan_units(UNITS)
    path = tmp_path / "batch" / "checkpoint.json"
    with c.step("alpha::run-1", "harbor_run") as st:
        mid = json.loads(path.read_text())
        assert mid["units"][0]["steps"]["harbor_run"]["status"] == "running"
        st.record(artifact="output/alpha/.raw", exit_code=0)
        st.output(run_dir="output/alpha/trajectory/run_1")
    after = json.loads(path.read_text())["units"][0]
    assert after["steps"]["harbor_run"]["status"] == "completed"
    assert after["steps"]["harbor_run"]["artifact"] == "output/alpha/.raw"
    assert after["steps"]["harbor_run"]["exit_code"] == 0
    assert after["outputs"]["run_dir"] == "output/alpha/trajectory/run_1"
    assert "duration_s" in after["steps"]["harbor_run"]
    c.release_lock()


def test_failed_step_records_error_and_fails_the_unit(tmp_path):
    c = _open(tmp_path)
    c.plan_units(UNITS)
    with pytest.raises(ValueError):
        with c.step("alpha::run-1", "harbor_run"):
            raise ValueError("harbor exited 3")
    u = c.unit("alpha::run-1")
    assert u["status"] == "failed"
    assert u["steps"]["harbor_run"]["status"] == "failed"
    assert u["steps"]["harbor_run"]["error"]["type"] == "ValueError"
    assert "harbor exited 3" in u["steps"]["harbor_run"]["error"]["message"]
    assert u["last_error"]["step"] == "harbor_run"
    # Failed units are held back from the default pending sweep.
    assert [x["unit_id"] for x in c.iter_pending()] == ["beta::run-1"]
    assert "alpha::run-1" in [x["unit_id"] for x in c.iter_pending(include_failed=True)]
    c.release_lock()


def test_on_error_record_swallows_the_exception(tmp_path):
    c = _open(tmp_path)
    c.plan_units(UNITS)
    with c.step("alpha::run-1", "harbor_run", on_error="record"):
        raise RuntimeError("boom")
    assert c.unit("alpha::run-1")["steps"]["harbor_run"]["status"] == "failed"
    c.release_lock()


def test_retry_clears_the_previous_attempts_error(tmp_path):
    c = _open(tmp_path)
    c.plan_units(UNITS)
    with c.step("alpha::run-1", "reshape", on_error="record"):
        raise RuntimeError("boom")
    with c.step("alpha::run-1", "reshape"):
        pass
    st = c.unit("alpha::run-1")["steps"]["reshape"]
    assert st["status"] == "completed"
    assert "error" not in st
    assert st["reruns"] == 1
    c.release_lock()


def test_skipped_step_is_terminal(tmp_path):
    c = _open(tmp_path)
    c.plan_units(UNITS)
    with c.step("alpha::run-1", "finance") as st:
        st.skip("ODOO_URL unset")
    st = c.unit("alpha::run-1")["steps"]["finance"]
    assert st["status"] == "skipped" and st["reason"] == "ODOO_URL unset"
    assert not c.needs("alpha::run-1", "finance")
    c.release_lock()


def test_unknown_step_name_is_rejected(tmp_path):
    c = _open(tmp_path)
    c.plan_units(UNITS)
    with pytest.raises(cp.CheckpointError, match="not one of this batch's steps"):
        with c.step("alpha::run-1", "nope"):
            pass
    c.release_lock()


def test_unit_completes_when_every_step_is_done(tmp_path):
    c = _open(tmp_path)
    c.plan_units(UNITS)
    for name in STEPS:
        with c.step("alpha::run-1", name):
            pass
    assert c.unit("alpha::run-1")["status"] == "completed"
    assert not c.unit_pending("alpha::run-1")
    c.release_lock()


# ---------------------------------------------------------------------------
# resume
# ---------------------------------------------------------------------------

def test_resume_reruns_only_the_unfinished_steps(tmp_path):
    """The whole point: a crash at step k costs you step k, not steps 0..k."""
    calls = []

    def drive(c, *, fail_at=None):
        for u in list(c.iter_pending(include_failed=True)):
            for name in c.step_names:
                if not c.needs(u["unit_id"], name):
                    continue
                with c.step(u["unit_id"], name, on_error="record") as st:
                    calls.append((u["unit_id"], name))
                    st.record(artifact=str(tmp_path / "art" / name))
                    (tmp_path / "art").mkdir(exist_ok=True)
                    (tmp_path / "art" / name).write_text("x")
                    if (u["unit_id"], name) == fail_at:
                        raise RuntimeError("crash")
                if c.unit(u["unit_id"])["status"] == "failed":
                    break

    c = _open(tmp_path)
    c.plan_units(UNITS)
    drive(c, fail_at=("alpha::run-1", "reshape"))
    c.release_lock()
    first_pass = list(calls)
    assert ("alpha::run-1", "preflight") in first_pass
    assert ("alpha::run-1", "finance") not in first_pass

    calls.clear()
    c2 = _open(tmp_path)
    c2.plan_units(UNITS)
    c2.reset(failed_only=True)
    drive(c2)
    c2.release_lock()

    # Nothing already completed was touched again.
    assert ("alpha::run-1", "preflight") not in calls
    assert ("alpha::run-1", "harbor_run") not in calls
    assert ("alpha::run-1", "reshape") in calls
    assert ("alpha::run-1", "finance") in calls
    assert c2.unit("alpha::run-1")["status"] == "completed"


def test_running_step_is_demoted_on_reopen(tmp_path):
    """Rule 2: work owned by a dead process is presumed unfinished."""
    c = _open(tmp_path)
    c.plan_units(UNITS)
    c.mark("alpha::run-1", "harbor_run", "running")
    c.release_lock()  # simulate a hard exit: no finish(), status left running

    c2 = _open(tmp_path)
    st = c2.unit("alpha::run-1")["steps"]["harbor_run"]
    assert st["status"] == "pending"
    assert "process exited" in st["demoted_reason"]
    assert c2.unit("alpha::run-1")["interruptions"] == 1
    assert c2.needs("alpha::run-1", "harbor_run")
    c2.release_lock()


def test_torn_primary_falls_back_to_the_backup(tmp_path):
    c = _open(tmp_path)
    c.plan_units(UNITS)
    with c.step("alpha::run-1", "preflight"):
        pass
    c.release_lock()

    path = tmp_path / "batch" / "checkpoint.json"
    assert (tmp_path / "batch" / "checkpoint.json.bak").exists()
    path.write_text('{"schema_ve')  # a torn write

    c2 = _open(tmp_path)
    assert c2.batch_id == "test-batch"
    assert [u["unit_id"] for u in c2.doc["units"]] == ["alpha::run-1", "beta::run-1"]
    c2.release_lock()


def test_fingerprint_mismatch_refuses_resume(tmp_path):
    fp1 = cp.fingerprint_of({"model": "claude-opus-4-8", "n": 1})
    fp2 = cp.fingerprint_of({"model": "claude-sonnet-4-6", "n": 1})
    _open(tmp_path, fingerprint=fp1).release_lock()

    with pytest.raises(cp.FingerprintMismatch, match="different config"):
        _open(tmp_path, fingerprint=fp2)
    c = _open(tmp_path, fingerprint=fp2, force=True)
    assert c.doc["config_fingerprint"] == fp2
    c.release_lock()


def test_fingerprint_is_order_insensitive(tmp_path):
    assert cp.fingerprint_of({"a": 1, "b": 2}) == cp.fingerprint_of({"b": 2, "a": 1})


# ---------------------------------------------------------------------------
# reconcile
# ---------------------------------------------------------------------------

def test_reconcile_demotes_steps_whose_artifact_vanished(tmp_path):
    """Rule 3: the filesystem outranks the checkpoint."""
    art = tmp_path / "output" / "alpha" / "trajectory" / "run_1"
    art.mkdir(parents=True)
    c = _open(tmp_path)
    c.plan_units(UNITS)
    with c.step("alpha::run-1", "preflight"):
        pass  # no artifact recorded -> nothing to check -> left alone
    with c.step("alpha::run-1", "reshape") as st:
        st.record(artifact="output/alpha/trajectory/run_1")

    assert c.reconcile(root=tmp_path) == []

    import shutil
    shutil.rmtree(art)
    assert c.reconcile(root=tmp_path) == ["alpha::run-1:reshape"]
    st = c.unit("alpha::run-1")["steps"]["reshape"]
    assert st["status"] == "pending" and "artifact missing" in st["demoted_reason"]
    assert c.unit("alpha::run-1")["steps"]["preflight"]["status"] == "completed"
    c.release_lock()


def test_reconcile_honours_a_custom_verifier(tmp_path):
    c = _open(tmp_path)
    c.plan_units(UNITS)
    with c.step("alpha::run-1", "finance") as st:
        st.record(receipt="output/alpha/finance_receipt.json")
    demoted = c.reconcile(verifiers={"finance": lambda unit, step: False})
    assert demoted == ["alpha::run-1:finance"]
    assert c.needs("alpha::run-1", "finance")
    c.release_lock()


# ---------------------------------------------------------------------------
# locking
# ---------------------------------------------------------------------------

def test_second_writer_is_locked_out(tmp_path):
    c = _open(tmp_path)
    with pytest.raises(cp.LockedError, match="already being written"):
        _open(tmp_path)
    c.release_lock()
    _open(tmp_path).release_lock()  # lock released, next writer gets in


def test_read_only_open_ignores_the_lock(tmp_path):
    c = _open(tmp_path)
    c.plan_units(UNITS)
    ro = cp.Checkpoint.open(tmp_path / "batch", read_only=True)
    assert [u["unit_id"] for u in ro.doc["units"]] == ["alpha::run-1", "beta::run-1"]
    ro.mark("alpha::run-1", "preflight", "completed")  # in-memory only
    assert json.loads((tmp_path / "batch" / "checkpoint.json").read_text(
    ))["units"][0]["steps"]["preflight"]["status"] == "pending"
    c.release_lock()


def test_stale_lock_from_a_dead_pid_is_reclaimed(tmp_path):
    _open(tmp_path).release_lock()
    lock = tmp_path / "batch" / "checkpoint.lock"
    lock.write_text(json.dumps(
        {"pid": _dead_pid(), "host": socket.gethostname(),
         "started_at": "2020-01-01T00:00:00Z", "epoch": 1577836800.0}))
    c = _open(tmp_path)  # must not raise
    assert json.loads(lock.read_text())["pid"] == os.getpid()
    c.release_lock()


def test_old_lock_from_another_host_is_reclaimed_on_age(tmp_path):
    _open(tmp_path).release_lock()
    lock = tmp_path / "batch" / "checkpoint.lock"
    lock.write_text(json.dumps(
        {"pid": 1, "host": "some-other-host", "started_at": "2020-01-01T00:00:00Z",
         "epoch": 1577836800.0}))
    _open(tmp_path).release_lock()  # older than STALE_LOCK_SECONDS


def test_force_takes_a_live_lock_over(tmp_path):
    c = _open(tmp_path)
    c2 = _open(tmp_path, force=True)
    c2.release_lock()
    c.release_lock()


# ---------------------------------------------------------------------------
# reset, finish, summary
# ---------------------------------------------------------------------------

def test_reset_failed_only_touches_failed_steps(tmp_path):
    c = _open(tmp_path)
    c.plan_units(UNITS)
    with c.step("alpha::run-1", "preflight"):
        pass
    with c.step("alpha::run-1", "harbor_run", on_error="record"):
        raise RuntimeError("boom")
    touched = c.reset(failed_only=True)
    assert touched == ["alpha::run-1:harbor_run"]
    u = c.unit("alpha::run-1")
    assert u["steps"]["preflight"]["status"] == "completed"
    assert u["steps"]["harbor_run"]["status"] == "pending"
    assert u["steps"]["harbor_run"]["reset_from"] == "failed"
    assert u["status"] == "pending" and u["last_error"] is None
    c.release_lock()


def test_finish_reports_the_batch_verdict(tmp_path):
    c = _open(tmp_path)
    c.plan_units(UNITS[:1])
    for name in STEPS:
        with c.step("alpha::run-1", name):
            pass
    assert c.finish() == "completed"
    c.release_lock()

    c2 = _open(tmp_path)
    c2.plan_units(UNITS)  # beta is untouched
    assert c2.finish() == "interrupted"
    c2.release_lock()


def test_context_manager_marks_an_escaping_exception_as_interrupted(tmp_path):
    with pytest.raises(RuntimeError):
        with _open(tmp_path) as c:
            c.plan_units(UNITS)
            raise RuntimeError("driver blew up")
    doc = json.loads((tmp_path / "batch" / "checkpoint.json").read_text())
    assert doc["status"] == "interrupted"
    assert not (tmp_path / "batch" / "checkpoint.lock").exists()


def test_summary_and_status_render(tmp_path):
    c = _open(tmp_path)
    c.plan_units(UNITS)
    with c.step("alpha::run-1", "preflight"):
        pass
    s = c.summary()
    assert s["steps"]["preflight"] == {"completed": 1, "pending": 1}
    assert s["totals"]["units"] == 2
    text = c.render_status()
    assert "test-batch" in text and "preflight" in text
    c.release_lock()


def test_events_log_records_every_transition(tmp_path):
    c = _open(tmp_path)
    c.plan_units(UNITS)
    with c.step("alpha::run-1", "preflight"):
        pass
    c.release_lock()
    events = [json.loads(l) for l in
              (tmp_path / "batch" / "events.jsonl").read_text().splitlines()]
    kinds = [e["event"] for e in events]
    assert kinds[0] == "batch_created"
    assert "unit_added" in kinds and "step_start" in kinds and "step_done" in kinds


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_cli_mark_status_and_reset_roundtrip(tmp_path, capsys):
    c = _open(tmp_path)
    c.plan_units(UNITS)
    c.release_lock()
    target = str(tmp_path / "batch")

    assert cp.main(["mark", target, "--unit", "alpha::run-1", "--step", "harbor_run",
                    "--status", "completed", "--detail", "exit_code=0",
                    "--detail", "artifact=output/alpha"]) == 0
    doc = json.loads((tmp_path / "batch" / "checkpoint.json").read_text())
    st = doc["units"][0]["steps"]["harbor_run"]
    assert st["status"] == "completed" and st["exit_code"] == 0

    assert cp.main(["status", target]) == 0
    assert "test-batch" in capsys.readouterr().out

    assert cp.main(["units", target, "--pending"]) == 0
    assert capsys.readouterr().out.split() == ["alpha::run-1", "beta::run-1"]

    assert cp.main(["reset", target, "--unit", "alpha::run-1"]) == 0
    assert "reset alpha::run-1:harbor_run" in capsys.readouterr().out


def test_cli_status_json(tmp_path, capsys):
    c = _open(tmp_path)
    c.plan_units(UNITS)
    c.release_lock()
    assert cp.main(["status", str(tmp_path / "batch"), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["batch_id"] == "test-batch"


def test_cli_reconcile_demotes(tmp_path, capsys):
    c = _open(tmp_path)
    c.plan_units(UNITS)
    with c.step("alpha::run-1", "reshape") as st:
        st.record(artifact="gone/run_1")
    c.release_lock()
    assert cp.main(["reconcile", str(tmp_path / "batch"), "--root", str(tmp_path)]) == 0
    assert "demoted alpha::run-1:reshape" in capsys.readouterr().out


def test_cli_missing_checkpoint_is_an_error_not_a_traceback(tmp_path, capsys):
    assert cp.main(["status", str(tmp_path / "nope")]) == 2
    assert "checkpoint:" in capsys.readouterr().err


def test_keyboard_interrupt_leaves_the_step_pending_not_failed(tmp_path):
    """Ctrl-C means "stopped", not "broken" -- the next resume must retry it."""
    c = _open(tmp_path)
    c.plan_units(UNITS)
    with pytest.raises(KeyboardInterrupt):
        with c.step("alpha::run-1", "harbor_run"):
            raise KeyboardInterrupt
    u = c.unit("alpha::run-1")
    assert u["steps"]["harbor_run"]["status"] == "pending"
    assert u["steps"]["harbor_run"]["interrupted_by"] == "KeyboardInterrupt"
    assert u["status"] == "pending" and u["interruptions"] == 1
    # No --retry-failed needed: it comes back in the default sweep.
    assert [x["unit_id"] for x in c.iter_pending()] == ["alpha::run-1", "beta::run-1"]
    c.release_lock()


def test_read_only_open_creates_nothing_on_disk(tmp_path):
    """A preview of a batch that does not exist must not bring it into being."""
    c = cp.Checkpoint.open(tmp_path / "batch", batch_id="preview",
                           steps=STEPS, read_only=True)
    c.plan_units(UNITS)
    with c.step("alpha::run-1", "preflight"):
        pass
    assert c.unit("alpha::run-1")["steps"]["preflight"]["status"] == "completed"
    assert not (tmp_path / "batch").exists()
