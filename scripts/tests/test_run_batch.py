"""Unit tests for scripts/run_batch.py.

run_task.sh is stubbed throughout: these tests are about what the driver decides
to run and what it refuses to run twice, not about Harbor or Docker.
"""
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import checkpoint as ckpt  # noqa: E402
import run_batch as rb  # noqa: E402

# A stand-in for run_task.sh: logs every stage it is asked to run, fakes the
# artifacts each stage would leave behind, and fails on demand.
FAKE_RUN_TASK = """#!/usr/bin/env bash
set -uo pipefail
STAGE_ARG=""
TASK=""
while [ $# -gt 0 ]; do
  case "$1" in
    --stage) STAGE_ARG="$2"; shift 2;;
    *) TASK="$1"; shift;;
  esac
done
echo "$JOB $STAGE_ARG" >> "$FAKE_LOG"
RUN_DIR="$OUTPUT_DIR/$JOB/trajectory/run_$((RUN_OFFSET+1))"
if [ -n "${FAIL_STAGE:-}" ] && [ "$STAGE_ARG" = "$FAIL_STAGE" ]; then
  echo "stage $STAGE_ARG blew up"
  exit 1
fi
case "$STAGE_ARG" in
  harbor)  [ -z "${NO_ARTIFACTS:-}" ] && mkdir -p "$OUTPUT_DIR/$JOB/.raw" ;;
  reshape) [ -z "${NO_ARTIFACTS:-}" ] && mkdir -p "$RUN_DIR" ;;
  finance) [ -n "${FAKE_RECEIPT:-}" ] && echo '{"ok": true}' > "$RUN_DIR/finance_receipt.json" ;;
esac
echo "ran $STAGE_ARG for $TASK"
exit 0
"""


@pytest.fixture
def env(tmp_path, monkeypatch):
    """A stubbed harness: two fake tasks, a fake run_task.sh, an empty output/."""
    fake = tmp_path / "fake_run_task.sh"
    fake.write_text(FAKE_RUN_TASK)
    fake.chmod(0o755)
    monkeypatch.setattr(rb, "RUN_TASK", fake)

    tasks_dir = tmp_path / "tasks"
    for slug in ("alpha", "beta"):
        (tasks_dir / slug).mkdir(parents=True)
        (tasks_dir / slug / "task.toml").write_text('name = "acme/%s"\n' % slug)

    log = tmp_path / "stages.log"
    log.write_text("")
    monkeypatch.setenv("FAKE_LOG", str(log))
    monkeypatch.delenv("FAIL_STAGE", raising=False)
    monkeypatch.delenv("FAKE_RECEIPT", raising=False)
    monkeypatch.delenv("NO_ARTIFACTS", raising=False)

    class Env:
        root = tmp_path
        tasks = tasks_dir
        output = tmp_path / "output"
        stages_log = log

        def argv(self, *extra):
            return ["--all", "--tasks-dir", str(tasks_dir),
                    "--output-dir", str(self.output),
                    "--batch-id", "test-batch", *extra]

        def stages(self, job=None):
            lines = [l.split() for l in self.stages_log.read_text().split("\n") if l.strip()]
            return [s for j, s in lines if job is None or j == job]

        def checkpoint(self):
            return json.loads(
                (self.output / "_batches" / "test-batch" / "checkpoint.json").read_text())

        def unit(self, unit_id):
            return next(u for u in self.checkpoint()["units"] if u["unit_id"] == unit_id)

    return Env()


# ---------------------------------------------------------------------------
# planning
# ---------------------------------------------------------------------------

def test_units_append_after_existing_runs(env, tmp_path):
    """A batch adds to a task's history; it never reuses a run_N."""
    (env.output / "alpha" / "trajectory" / "run_1").mkdir(parents=True)
    (env.output / "alpha" / "trajectory" / "run_2").mkdir(parents=True)
    args = rb.parse_args(env.argv("--n", "2", "--model", "m1"))
    cp = ckpt.Checkpoint.open(tmp_path / "cp", batch_id="b", steps=rb.STEPS)
    specs = rb.build_units(cp, rb.discover_tasks(args), args)
    cp.release_lock()

    alpha = [s for s in specs if s["task_slug"] == "alpha"]
    assert [Path(s["run_dir"]).name for s in alpha] == ["run_3", "run_4"]
    assert [s["run_offset"] for s in alpha] == [2, 3]
    beta = [s for s in specs if s["task_slug"] == "beta"]
    assert [Path(s["run_dir"]).name for s in beta] == ["run_1", "run_2"]


def test_run_numbering_is_frozen_at_first_plan(env, tmp_path):
    """Resume must not slide the batch forward by the runs it already produced."""
    args = rb.parse_args(env.argv("--n", "1", "--model", "m1"))
    tasks = rb.discover_tasks(args)
    cp = ckpt.Checkpoint.open(tmp_path / "cp", batch_id="b", steps=rb.STEPS)
    cp.plan_units(rb.build_units(cp, tasks, args))

    # The batch runs, so run_1 now exists on disk...
    (env.output / "alpha" / "trajectory" / "run_1").mkdir(parents=True)
    again = rb.build_units(cp, tasks, args)
    cp.plan_units(again)
    cp.release_lock()

    alpha = next(s for s in again if s["task_slug"] == "alpha")
    assert Path(alpha["run_dir"]).name == "run_1"  # not run_2
    assert cp.unit("alpha::m1::run-1")["run_offset"] == 0


def test_discover_tasks_rejects_a_non_task_dir(env, tmp_path):
    (tmp_path / "nope").mkdir()
    with pytest.raises(SystemExit, match="no task.toml"):
        rb.discover_tasks(rb.parse_args(["--task", str(tmp_path / "nope")]))


def test_nothing_to_run_is_an_error(env):
    with pytest.raises(SystemExit, match="nothing to run"):
        rb.discover_tasks(rb.parse_args([]))


# ---------------------------------------------------------------------------
# happy path
# ---------------------------------------------------------------------------

def test_full_batch_runs_every_stage_once(env):
    assert rb.main(env.argv("--n", "1", "--model", "m1")) == 0
    assert env.stages("alpha") == ["preflight", "harbor", "reshape", "finance"]
    assert env.stages("beta") == ["preflight", "harbor", "reshape", "finance"]

    doc = env.checkpoint()
    assert doc["status"] == "completed"
    assert doc["totals"]["completed"] == 2
    u = env.unit("alpha::m1::run-1")
    assert u["status"] == "completed"
    assert u["outputs"]["run_dir"].endswith("alpha/trajectory/run_1")
    assert u["steps"]["reshape"]["artifact"].endswith("run_1")
    # No receipt was written, so finance is skipped rather than claimed.
    assert u["steps"]["finance"]["status"] == "skipped"


def test_finance_completes_when_a_receipt_lands(env, monkeypatch):
    monkeypatch.setenv("FAKE_RECEIPT", "1")
    assert rb.main(env.argv("--n", "1", "--model", "m1")) == 0
    st = env.unit("alpha::m1::run-1")["steps"]["finance"]
    assert st["status"] == "completed"
    assert st["artifact"].endswith("finance_receipt.json")


def test_rerunning_a_complete_batch_does_no_work(env):
    assert rb.main(env.argv("--n", "1", "--model", "m1")) == 0
    before = env.stages()
    assert rb.main(env.argv("--n", "1", "--model", "m1")) == 0
    assert env.stages() == before  # not one extra stage invocation


def test_attempts_of_one_task_each_get_their_own_run_dir(env):
    assert rb.main(env.argv("--n", "3", "--model", "m1", "--task", str(env.tasks / "alpha"))) == 0
    runs = sorted(p.name for p in (env.output / "alpha" / "trajectory").iterdir())
    assert runs == ["run_1", "run_2", "run_3"]


# ---------------------------------------------------------------------------
# resume
# ---------------------------------------------------------------------------

def test_resume_does_not_rerun_the_agent_phase(env, monkeypatch):
    """The point of the whole feature: a crash after harbor costs no agent time."""
    monkeypatch.setenv("FAIL_STAGE", "reshape")
    assert rb.main(env.argv("--n", "1", "--model", "m1")) == 1
    assert env.stages("alpha") == ["preflight", "harbor", "reshape"]
    assert env.unit("alpha::m1::run-1")["status"] == "failed"
    assert env.unit("alpha::m1::run-1")["steps"]["harbor_run"]["status"] == "completed"

    env.stages_log.write_text("")
    monkeypatch.delenv("FAIL_STAGE")
    assert rb.main(env.argv("--n", "1", "--model", "m1", "--retry-failed")) == 0
    # preflight and harbor are already done; only the tail re-runs.
    assert env.stages("alpha") == ["reshape", "finance"]
    assert env.checkpoint()["status"] == "completed"


def test_failed_units_are_held_back_until_retry_is_asked_for(env, monkeypatch):
    monkeypatch.setenv("FAIL_STAGE", "harbor")
    assert rb.main(env.argv("--n", "1", "--model", "m1")) == 1
    env.stages_log.write_text("")
    monkeypatch.delenv("FAIL_STAGE")

    assert rb.main(env.argv("--n", "1", "--model", "m1")) == 1
    assert env.stages() == []  # nothing retried without being asked

    assert rb.main(env.argv("--n", "1", "--model", "m1", "--retry-failed")) == 0
    assert env.stages("alpha") == ["harbor", "reshape", "finance"]


def test_stop_on_failure_leaves_the_rest_pending(env, monkeypatch):
    monkeypatch.setenv("FAIL_STAGE", "harbor")
    assert rb.main(env.argv("--n", "1", "--model", "m1", "--stop-on-failure")) == 1
    doc = env.checkpoint()
    assert doc["status"] == "interrupted"
    assert env.unit("beta::m1::run-1")["status"] == "pending"
    assert env.stages("beta") == []


def test_reset_step_redoes_one_stage_across_the_batch(env):
    assert rb.main(env.argv("--n", "1", "--model", "m1")) == 0
    env.stages_log.write_text("")
    assert rb.main(env.argv("--n", "1", "--model", "m1", "--reset-step", "reshape")) == 0
    # finance already reached a terminal state; only the reset step re-runs.
    assert env.stages("alpha") == ["reshape"]


def test_changing_the_model_refuses_to_resume_the_batch(env, capsys):
    assert rb.main(env.argv("--n", "1", "--model", "m1")) == 0
    assert rb.main(env.argv("--n", "1", "--model", "m2")) == 2
    assert "different config" in capsys.readouterr().err
    assert rb.main(env.argv("--n", "1", "--model", "m2", "--force")) == 0


# ---------------------------------------------------------------------------
# reconcile
# ---------------------------------------------------------------------------

def test_deleted_run_dir_redoes_reshape_but_not_the_agent(env):
    import shutil
    assert rb.main(env.argv("--n", "1", "--model", "m1")) == 0
    shutil.rmtree(env.output / "alpha" / "trajectory" / "run_1")
    env.stages_log.write_text("")

    assert rb.main(env.argv("--n", "1", "--model", "m1")) == 0
    assert env.stages("alpha") == ["reshape"]
    assert env.stages("beta") == []


def test_harbor_evidence_survives_reshape_consuming_the_raw_trials(tmp_path):
    unit = {"run_dir": str(tmp_path / "out/alpha/trajectory/run_1"),
            "job_dir": str(tmp_path / "out/alpha"), "task_slug": "alpha"}
    step = {"job_dir": unit["job_dir"]}
    assert rb._harbor_evidence(unit, step) is False          # nothing on disk yet

    (tmp_path / "out/alpha/alpha__abc123").mkdir(parents=True)  # raw trial dir
    assert rb._harbor_evidence(unit, step) is True

    import shutil
    shutil.rmtree(tmp_path / "out/alpha/alpha__abc123")
    (tmp_path / "out/alpha/.raw").mkdir()                     # reshaped away
    assert rb._harbor_evidence(unit, step) is True

    shutil.rmtree(tmp_path / "out/alpha/.raw")
    Path(unit["run_dir"]).mkdir(parents=True)                 # only the run survives
    assert rb._harbor_evidence(unit, step) is True


def test_no_reconcile_skips_the_disk_check(env):
    import shutil
    assert rb.main(env.argv("--n", "1", "--model", "m1")) == 0
    shutil.rmtree(env.output / "alpha" / "trajectory" / "run_1")
    env.stages_log.write_text("")
    assert rb.main(env.argv("--n", "1", "--model", "m1", "--no-reconcile")) == 0
    assert env.stages() == []


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def test_dry_run_reports_the_plan_and_runs_nothing(env, monkeypatch, capsys):
    monkeypatch.setenv("FAIL_STAGE", "reshape")
    rb.main(env.argv("--n", "1", "--model", "m1"))
    env.stages_log.write_text("")
    monkeypatch.delenv("FAIL_STAGE")

    assert rb.main(env.argv("--n", "1", "--model", "m1", "--dry-run")) == 0
    out = capsys.readouterr().out
    assert "resume plan:" in out
    assert "skip=preflight,harbor_run run=reshape,finance" in out
    assert "held back: failed" in out
    assert env.stages() == []


def test_status_reads_without_taking_the_lock(env, capsys):
    assert rb.main(env.argv("--n", "1", "--model", "m1")) == 0
    assert rb.main(["--status", "--output-dir", str(env.output), "--batch-id", "test-batch"]) == 0
    out = capsys.readouterr().out
    assert "test-batch" in out and "2 completed" in out
    assert not (env.output / "_batches" / "test-batch" / "checkpoint.lock").exists()


def test_status_without_an_id_picks_the_latest_batch(env):
    assert rb.main(env.argv("--n", "1", "--model", "m1")) == 0
    assert rb.main(["--status", "--output-dir", str(env.output)]) == 0


def test_status_with_no_batches_at_all_is_a_clean_error(env):
    with pytest.raises(SystemExit, match="no batches"):
        rb.main(["--status", "--output-dir", str(env.output)])


def test_per_unit_logs_are_written(env):
    assert rb.main(env.argv("--n", "1", "--model", "m1")) == 0
    log = env.output / "_batches" / "test-batch" / "logs" / "alpha__m1__run-1.log"
    assert log.is_file()
    assert "ran preflight" in log.read_text()


def test_a_batch_with_held_back_failures_does_not_exit_green(env, monkeypatch):
    """A loop driving this must not read "nothing runnable" as "all good"."""
    monkeypatch.setenv("FAIL_STAGE", "harbor")
    assert rb.main(env.argv("--n", "1", "--model", "m1")) == 1
    monkeypatch.delenv("FAIL_STAGE")
    assert rb.main(env.argv("--n", "1", "--model", "m1")) == 1
    assert env.checkpoint()["status"] == "failed"


def test_a_stage_that_exits_zero_but_produced_nothing_still_fails(env, monkeypatch):
    """run_task.sh swallows harbor's exit code, so the driver checks artifacts.

    This is the shape of a real failure: bad auth or a missing image aborts
    harbor, run_task.sh reshapes "whatever landed" and exits 0, and without this
    check the whole batch reports green having produced no trajectories.
    """
    monkeypatch.setenv("NO_ARTIFACTS", "1")
    assert rb.main(env.argv("--n", "1", "--model", "m1")) == 1
    u = env.unit("alpha::m1::run-1")
    assert u["status"] == "failed"
    assert u["steps"]["harbor_run"]["status"] == "failed"
    assert "left no trial" in u["steps"]["harbor_run"]["error"]["message"]
    # The batch stops that unit there rather than reshaping nothing.
    assert env.stages("alpha") == ["preflight", "harbor"]


def test_reshape_that_leaves_no_run_dir_fails(env, monkeypatch):
    import shutil
    assert rb.main(env.argv("--n", "1", "--model", "m1")) == 0
    # Drop the run dir so reconcile demotes reshape, then make reshape a no-op:
    # it must not be able to claim success over a directory that is not there.
    shutil.rmtree(env.output / "alpha" / "trajectory" / "run_1")
    monkeypatch.setenv("NO_ARTIFACTS", "1")
    assert rb.main(env.argv("--n", "1", "--model", "m1")) == 1
    st = env.unit("alpha::m1::run-1")["steps"]["reshape"]
    assert st["status"] == "failed" and "did not produce" in st["error"]["message"]
    # The raw trial was consumed by the first reshape, so the message must name
    # the only real remedy instead of leaving the operator to guess.
    assert "already consumed" in st["error"]["message"]
    assert "--reset-step harbor_run --retry-failed" in st["error"]["message"]


def test_reshape_hint_stays_plain_while_the_raw_trial_is_still_there(env, tmp_path):
    """Only the unrecoverable case gets the expensive-remedy advice."""
    job = env.output / "alpha"
    (job / "alpha__abc123").mkdir(parents=True)
    unit = {"run_dir": str(job / "trajectory" / "run_1"),
            "job_dir": str(job), "task_slug": "alpha"}
    msg = rb._reshape_failure_hint(unit, tmp_path / "x.log")
    assert "did not produce" in msg and "already consumed" not in msg

    import shutil
    shutil.rmtree(job / "alpha__abc123")
    msg = rb._reshape_failure_hint(unit, tmp_path / "x.log")
    assert "already consumed" in msg


def test_dry_run_never_mutates_the_checkpoint(env, monkeypatch):
    """--dry-run previews resets and reconcile instead of performing them."""
    import shutil
    assert rb.main(env.argv("--n", "1", "--model", "m1")) == 0
    # beta's output disappears, so reconcile has something it would demote
    shutil.rmtree(env.output / "beta" / "trajectory" / "run_1")
    env.stages_log.write_text("")
    before = env.checkpoint()

    assert rb.main(env.argv("--n", "1", "--model", "m1", "--dry-run",
                            "--retry-failed", "--reset-step", "preflight")) == 0
    after = env.checkpoint()
    assert {u["unit_id"]: u["steps"] for u in before["units"]} == \
           {u["unit_id"]: u["steps"] for u in after["units"]}
    assert env.stages() == []


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

def test_force_resumes_and_records_the_override(env):
    """--force is allowed, but it may not be silent: the override lands in
    events.jsonl naming both fingerprints."""
    assert rb.main(env.argv("--n", "1", "--model", "m1")) == 0
    assert rb.main(env.argv("--n", "1", "--model", "m2", "--force")) == 0

    events = (env.output / "_batches" / "test-batch" / "events.jsonl").read_text()
    forced = [json.loads(l) for l in events.splitlines()
              if l.strip() and json.loads(l).get("event") == "fingerprint_override_forced"]
    assert len(forced) == 1, events
    assert forced[0]["stored"] != forced[0]["now"]

    cp = env.checkpoint()
    assert len(cp["fingerprint_overrides"]) == 1

