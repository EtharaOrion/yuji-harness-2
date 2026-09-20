"""Stage plumbing in scripts/run_task.sh, and the harbor patch it runs first.

Harbor itself is stubbed on PATH, so these cover what the script decides -- the
run_N a stage owns, the state it hands to the next stage, the runs it protects
from being wiped -- without building a container. The same PATH stub lets the
last section patch a mirrored harbor rather than the installed one.
"""
import importlib.util
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from conftest import mirror_harbor_package, requires_docker, requires_harbor

REPO = Path(__file__).resolve().parent.parent.parent
RUN_TASK = REPO / "scripts" / "run_task.sh"

HARBOR_STUB = """#!/usr/bin/env bash
printf '%s\\n' "$@" >> "$HARBOR_ARGS"
mkdir -p "$JOB_DIR"
touch "$JOB_DIR/result.json"
# STUB_TRIAL makes the stub behave like a harbor run that actually created a
# trial directory, which is what the run-ownership tests below need to see.
if [ -n "${STUB_TRIAL:-}" ]; then
  for t in ${STUB_TRIAL//,/ }; do
    mkdir -p "$JOB_DIR/$t"
    echo '{}' > "$JOB_DIR/$t/config.json"
  done
fi
# A trial harbor created but never started: the directory exists, config.json
# does not. run_task.sh must refuse to reshape rather than lose the run.
if [ -n "${STUB_TRIAL_EMPTY:-}" ]; then
  mkdir -p "$JOB_DIR/$STUB_TRIAL_EMPTY"
fi
exit 0
"""


@pytest.fixture
def env(tmp_path):
    task = tmp_path / "tasks" / "alpha"
    task.mkdir(parents=True)
    (task / "task.toml").write_text('name = "acme/alpha"\nimage = "example/img:1"\n')

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = bin_dir / "harbor"
    stub.write_text(HARBOR_STUB)
    stub.chmod(0o755)

    # Shadowing `harbor` on PATH also redirects patch_harbor.py, which run_task.sh
    # runs before dispatch and which resolves the harbor package relative to that
    # binary. Without a mirrored package it raises, run_task.sh aborts, and every
    # assertion below fails on an empty argv -- looking like a dispatch bug rather
    # than a missing fixture. See scripts/tests/conftest.py.
    if not mirror_harbor_package(tmp_path):
        pytest.skip("harbor is not installed; cannot mirror its package")

    class Env:
        root = tmp_path
        task_dir = task
        output = tmp_path / "output"
        harbor_args = tmp_path / "harbor_args.txt"

        def run(self, stage, **overrides):
            e = dict(os.environ)
            e.update({
                "PATH": f"{bin_dir}:{e['PATH']}",
                "OUTPUT_DIR": str(self.output),
                "JOB": "alpha",
                "JOB_DIR": str(self.output / "alpha"),
                "HARBOR_ARGS": str(self.harbor_args),
            })
            e.update({k: str(v) for k, v in overrides.items()})
            return subprocess.run(
                [str(RUN_TASK), "--stage", stage, str(task)],
                capture_output=True, text=True, env=e, cwd=str(REPO), timeout=120)

        def state(self):
            return json.loads((self.output / "alpha" / ".run_state.json").read_text())

        def harbor_argv(self):
            return self.harbor_args.read_text().split("\n") if self.harbor_args.exists() else []

    return Env()


def test_unknown_stage_is_rejected(env):
    r = env.run("nonsense")
    assert r.returncode == 2
    assert "unknown stage" in r.stderr


def test_missing_task_dir_is_rejected(tmp_path):
    r = subprocess.run([str(RUN_TASK), "--stage", "harbor", str(tmp_path / "nope")],
                       capture_output=True, text=True, cwd=str(REPO), timeout=60)
    assert r.returncode == 2
    assert "not a task dir" in r.stderr


def test_help_lists_the_stages():
    r = subprocess.run([str(RUN_TASK), "--help"], capture_output=True, text=True,
                       cwd=str(REPO), timeout=60)
    assert r.returncode == 0
    for stage in ("preflight", "harbor", "reshape", "finance"):
        assert stage in r.stdout


@requires_docker
def test_harbor_stage_records_state_for_the_next_stage(env):
    r = env.run("harbor", RUN_OFFSET=2, MODEL="m1", AGENT="claude-code", N=1)
    assert r.returncode == 0, r.stderr
    state = env.state()
    assert state["run_offset"] == 2
    assert state["slug"] == "alpha" and state["job"] == "alpha"
    assert state["harbor_done"] == 1


def test_explicit_run_offset_beats_what_is_on_disk(env):
    (env.output / "alpha" / "trajectory" / "run_1").mkdir(parents=True)
    (env.output / "alpha" / "trajectory" / "run_2").mkdir(parents=True)
    env.run("harbor", RUN_OFFSET=0)
    assert env.state()["run_offset"] == 0


def test_offset_falls_back_to_counting_existing_runs(env):
    """A bare run_task.sh, with no driver deciding for it, still appends."""
    for n in (1, 2, 3):
        (env.output / "alpha" / "trajectory" / f"run_{n}").mkdir(parents=True)
    env.run("harbor")
    assert env.state()["run_offset"] == 3


def test_earlier_runs_are_stashed_outside_the_job_dir(env):
    """Harbor may clear output/<job>/ wholesale, stash included, if it lives there."""
    (env.output / "alpha" / "trajectory" / "run_1").mkdir(parents=True)
    (env.output / "alpha" / "trajectory" / "run_1" / "marker").write_text("keep me")
    env.run("harbor", RUN_OFFSET=1)
    stash = Path(env.state()["stash_dir"])
    assert stash == env.output / ".stash" / "alpha"
    assert (stash / "run_1" / "marker").read_text() == "keep me"
    assert env.output / "alpha" not in stash.parents


@requires_docker
def test_harbor_argv_carries_the_job_and_attempt_count(env):
    env.run("harbor", RUN_OFFSET=0, MODEL="m1", N=1)
    argv = env.harbor_argv()
    assert "--job-name" in argv and "alpha" in argv
    assert "--n-attempts" in argv
    assert "--model" in argv and "m1" in argv


def test_oracle_agent_gets_no_model_flag(env):
    env.run("harbor", RUN_OFFSET=0, AGENT="oracle", MODEL="m1")
    assert "--model" not in env.harbor_argv()


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

DOCKER_STUB = """#!/usr/bin/env bash
printf '%s\\n' "$*" >> "$DOCKER_CALLS"
exit 0
"""

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

BUNDLES = sorted(p for p in (REPO / "tasks").glob("*/environment/docker-compose.yaml")) \
    if (REPO / "tasks").is_dir() else []



def _stale_trial(env, name="alpha__stale"):
    """A trial dir harbor left behind when an earlier invocation died."""
    d = env.output / "alpha" / name
    d.mkdir(parents=True)
    (d / "config.json").write_text("{}")
    return d


@requires_docker
def test_reshape_is_handed_only_this_invocations_trial(env):
    """The extra-runs bug: convert_job globs every *__* dir with a config.json,
    so one N=1 invocation emitted one run per stale trial dir as well."""
    _stale_trial(env)
    r = env.run("harbor", RUN_OFFSET=0, N=1, STUB_TRIAL="alpha__fresh")
    assert r.returncode == 0, r.stderr
    assert env.state()["trials"] == "alpha__fresh"


@requires_docker
def test_host_rubric_does_not_grade_a_stale_trial(env):
    """stage_harbor names the stale dirs "NOT part of this run" and stage_reshape
    honours that, but the rubric pass globbed every *__* dir and graded one
    anyway -- `find | sort` is alphabetical, so which one won was arbitrary. A
    real run spent a full judge pass (60 criteria, ~1M chars) on a trial from an
    earlier invocation and published it as this run's."""
    _stale_trial(env)
    assert env.run("harbor", RUN_OFFSET=0, N=1, STUB_TRIAL="alpha__fresh").returncode == 0
    # stage_host_rubric is not independently selectable; stage_reshape calls it
    # first, so reshape is the driver.
    r = env.run("reshape", RUN_OFFSET=0)
    blob = r.stdout + r.stderr
    # Guard against a vacuous pass: if the stage never ran, absence proves nothing.
    assert "unknown stage" not in blob, blob[-800:]
    assert "alpha__stale" not in blob, (
        f"the rubric pass reached a trial from an earlier invocation: {blob[-1500:]}")


def _select_trials(tmp_path, state, names=("task__AAA", "task__BBB", "task__STALE")):
    """Drive stage_host_rubric's trial selector directly.

    The loop names a trial only on its "already graded" / "failed" branches, so
    asserting on stage output cannot distinguish "selected and skipped" from
    "never selected". This calls the selector itself.
    `state` is None for an absent `trials` key, else the stored comma string.
    """
    job = tmp_path / "job"
    for n in names:
        (job / n).mkdir(parents=True)
    has = "return 1" if state is None else "return 0"
    get = "" if state is None else state
    script = f"""
        set -uo pipefail
        OUTPUT_DIR={tmp_path}; JOB=job
        state_has() {{ {has}; }}
        state_get() {{ printf '%s' '{get}'; }}
        {_selector_source()}
        this_invocations_trials
    """
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    return [Path(p).name for p in out.stdout.split() if p.strip()]


def _selector_source():
    body, keep = [], False
    for line in RUN_TASK.read_text().splitlines():
        if line.startswith("this_invocations_trials() {"):
            keep = True
        if keep:
            body.append(line)
            if line == "}":
                break
    assert body, "this_invocations_trials() not found in run_task.sh"
    return "\n".join(body)


def test_selector_absent_key_grades_every_trial(tmp_path):
    """Hand-driven stage over a job dir with no state: grade what is there."""
    assert sorted(_select_trials(tmp_path, None)) == ["task__AAA", "task__BBB", "task__STALE"]


def test_selector_empty_value_grades_nothing(tmp_path):
    """"harbor made nothing" must not collapse into "grade everything" -- that
    is the same distinction stage_reshape draws, and the bug this pass had."""
    assert _select_trials(tmp_path, "") == []


def test_selector_keeps_every_trial_of_a_multi_attempt_run(tmp_path):
    """Regression: an early version read the comma list with `printf '%s' | read`,
    which dropped the last field because it was unterminated -- silently
    narrowing an N=2 run to one trial."""
    assert sorted(_select_trials(tmp_path, "task__AAA,task__BBB")) == ["task__AAA", "task__BBB"]


def test_selector_skips_a_name_whose_dir_was_removed(tmp_path):
    """A dir deleted by hand must not fail the whole stage."""
    assert _select_trials(tmp_path, "task__AAA,task__GONE") == ["task__AAA"]


def test_selector_never_returns_a_stale_dir(tmp_path):
    """The bug: `find | sort` returned task__STALE and the judge graded it."""
    assert "task__STALE" not in _select_trials(tmp_path, "task__AAA")


@requires_docker
def test_host_rubric_says_so_when_this_invocation_made_no_trial(env):
    """"harbor made nothing" must not read as "the job dir is empty" -- a
    state-tracking bug that grades nothing would otherwise look benign."""
    _stale_trial(env)
    assert env.run("harbor", RUN_OFFSET=0, N=1, STUB_TRIAL="").returncode == 0
    r = env.run("reshape", RUN_OFFSET=0)
    blob = r.stdout + r.stderr
    assert "unknown stage" not in blob, blob[-800:]
    assert "no trial dir of its own" in blob, blob[-2000:]


@requires_docker
def test_stale_trial_dirs_are_named_not_silently_dropped(env):
    _stale_trial(env)
    r = env.run("harbor", RUN_OFFSET=0, N=1, STUB_TRIAL="alpha__fresh")
    assert "trial dir(s) from earlier invocations" in r.stderr
    assert "alpha__stale" in r.stderr


@requires_docker
def test_every_trial_of_a_multi_attempt_run_is_kept(env):
    """N>1 legitimately makes several trials; only the pre-existing ones drop out."""
    _stale_trial(env)
    r = env.run("harbor", RUN_OFFSET=0, N=2, STUB_TRIAL="alpha__c,alpha__b")
    assert r.returncode == 0, r.stderr
    assert env.state()["trials"] == "alpha__b,alpha__c"


@requires_docker
def test_no_trial_at_all_records_an_empty_list(env):
    """Nothing ran: reshape must convert nothing, not adopt the leftovers."""
    _stale_trial(env)
    r = env.run("harbor", RUN_OFFSET=0, N=1)
    assert r.returncode == 0, r.stderr
    assert env.state()["trials"] == ""


@requires_docker
def test_a_new_trial_that_never_started_is_still_caught(env):
    """A stale dir has a config.json, so probing the newest dir by mtime let an
    aborted trial pass the guard silently -- the case the guard exists for."""
    _stale_trial(env)
    r = env.run("harbor", RUN_OFFSET=0, N=1, STUB_TRIAL_EMPTY="alpha__aborted")
    assert r.returncode != 0
    assert "AGENT PHASE DID NOT RUN" in r.stderr


@requires_docker
def test_one_empty_trial_among_several_is_caught(env):
    r = env.run("harbor", RUN_OFFSET=0, N=2,
                STUB_TRIAL="alpha__good", STUB_TRIAL_EMPTY="alpha__aborted")
    assert r.returncode != 0
    assert "AGENT PHASE DID NOT RUN" in r.stderr


REAL_TRIAL_CFG = '{"task": {"path": "tasks/alpha", "name": "acme/alpha"}, "trial_name": "%s"}'


def _trial(env, name, *, started=True):
    d = env.output / "alpha" / name
    d.mkdir(parents=True, exist_ok=True)
    if started:
        (d / "config.json").write_text(REAL_TRIAL_CFG % name)
        (d / "result.json").write_text('{"reward": 0.0}')
    return d


@requires_docker
def test_one_invocation_writes_exactly_one_run(env, tmp_path):
    """End to end over both stages: a job dir carrying two dead trials from
    earlier invocations must still yield exactly ONE new trajectory/run_N."""
    (env.output / "alpha").mkdir(parents=True)
    _trial(env, "alpha__old1")
    _trial(env, "alpha__old2")
    (env.output / "alpha" / "config.json").write_text(
        '{"agents": [{"name": "claude-code", "model_name": "m1"}]}')
    (env.output / "alpha" / "result.json").write_text('{"id": "job-1"}')

    assert env.run("harbor", RUN_OFFSET=0, N=1, STUB_TRIAL="alpha__fresh").returncode == 0
    _trial(env, "alpha__fresh")          # the stub cannot write harbor's real config
    r = env.run("reshape", RUN_OFFSET=0)
    assert r.returncode == 0, r.stderr + r.stdout

    runs = [p.name for d in env.output.glob("*/trajectory") for p in d.glob("run_*")]
    assert runs == ["run_1"], runs


def test_host_rubric_pass_looks_only_at_this_invocations_trials(env):
    """The host pass globbed every *__* dir in the job, so a leftover trial from
    an earlier invocation was graded again -- judge quota spent on a run this
    invocation never made. It now takes the trial list stage_harbor recorded."""
    (env.output / "alpha").mkdir(parents=True)
    old = _trial(env, "alpha__old1")
    (env.output / "alpha" / "config.json").write_text(
        '{"agents": [{"name": "claude-code", "model_name": "m1"}]}')
    (env.output / "alpha" / "result.json").write_text('{"id": "job-1"}')
    assert env.run("harbor", RUN_OFFSET=0, N=1, STUB_TRIAL="alpha__fresh").returncode == 0
    fresh = _trial(env, "alpha__fresh")
    for d in (old, fresh):
        (d / "verifier").mkdir(exist_ok=True)
        (d / "verifier" / "reward.json").write_text('{"reward": 0.5}')
        (d / "verifier" / "reward_producer.json").write_text('{"producer": "judge_container"}')
    r = env.run("reshape", RUN_OFFSET=0)
    out = r.stdout + r.stderr
    assert "judge container for alpha__fresh" in out, out[-3000:]
    graded = [l for l in out.splitlines() if "judge container for" in l]
    assert not any("alpha__old1" in l for l in graded), graded
    # harbor validates the container's reward.json as numbers only, so the label
    # is added on the host -- to this invocation's trial alone.
    assert "producer" not in json.loads((old / "verifier" / "reward.json").read_text())
    published = json.loads(
        (env.output / "alpha" / "trajectory" / "run_1" / "verifier" / "reward.json").read_text())
    assert published.get("producer") == "judge_container", published


def test_a_colon_in_the_path_is_refused_before_anything_runs(env, tmp_path):
    """Docker bind mounts are colon-delimited, so a checkout or jobs dir with a
    ":" in its name cannot be mounted -- compose says "too many colons" and the
    long syntax fails identically. Measured on a clone under
    ~/Downloads/feat:new_eval_container: the trial died in "starting
    environment..." and read like a harness bug."""
    colon_out = tmp_path / "feat:new_eval" / "output"
    colon_out.mkdir(parents=True)
    r = env.run("harbor", RUN_OFFSET=0, N=1, OUTPUT_DIR=str(colon_out))
    assert r.returncode != 0, r.stdout
    assert "colon" in r.stderr, r.stderr[-1500:]
    assert not env.harbor_argv() or env.harbor_argv() == [""], "harbor was invoked anyway"
