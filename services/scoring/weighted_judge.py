"""
services/scoring/weighted_judge.py

Channel A of the weighted grader: runs a task's tests/test_outputs.py pytest
suite against one trajectory and scores it against tests/test_weights.json.

This module is Phase 1 of the "pytest + rubric" weighted-grading rollout.
Channel B (the rubric judge) and the top-level judge_weighted() ledger
combiner that merges Channel A + Channel B land in a later phase; today this
module only produces the `traj_tests` component value, so it's usable
standalone as soon as a task ships a tests/test_outputs.py + test_weights.json.

Guard-test polarity (read this before writing a test_weights.json):
    Every test in test_outputs.py is phrased positively — "X happened" — and
    whether X happening is good or bad is decided entirely by the *sign* of
    that test's weight here:
      * positive weight  -> a "goal" test. Passing earns credit.
      * negative weight  -> a "guard" test. Passing (the forbidden thing
        happened) earns a penalty; *not* triggering earns nothing (not a
        bonus) — a run that never goes near the guard shouldn't score higher
        than one that was never at risk of it.
    This is why nothing in traj_asserts.py is ever wrapped in `assert not
    ...` — the polarity belongs here, in the weights file, not in the test
    body. See traj_asserts.py's module docstring for the assertion API.

Scoring formula (mirrors complex-mcp's weighted grader):
    pos_total = sum(w for w in weights.values() if w > 0)
    earned    = sum(w for name, w in weights.items() if w > 0 and results[name])
    penalty   = sum(abs(w) for name, w in weights.items() if w < 0 and results[name])
    value     = max(0, (earned - penalty) / pos_total)   if pos_total else None

    `value is None` (no goal tests defined — a guard-only suite) means this
    component has no positive stake to normalize against, so it's dropped
    out of the top-level reward ledger entirely rather than counted as 0.

A task opts into Channel A by giving `components.traj_tests.weight` a
non-zero value in test_weights.json — a task with no test_weights.json (or a
flat legacy shape, see load_weights()) contributes nothing to the ledger,
so adding this module changes no existing task's score by default.
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_SCORING_DIR = Path(__file__).resolve().parent
if str(_SCORING_DIR) not in sys.path:
    sys.path.insert(0, str(_SCORING_DIR))

import traj_asserts  # noqa: E402

_ENV_VAR = traj_asserts._ENV_VAR  # "MCPATLAS_TRAJECTORY" — single source of truth


@dataclass
class ComponentWeight:
    weight: float = 0.0
    tests: dict[str, float] = field(default_factory=dict)


@dataclass
class Weights:
    threshold: float = 1.0
    traj_tests: ComponentWeight = field(default_factory=ComponentWeight)
    rubric: ComponentWeight = field(default_factory=ComponentWeight)
    state_completion: ComponentWeight = field(default_factory=ComponentWeight)
    state_misbehave: ComponentWeight = field(default_factory=ComponentWeight)


def load_weights(path: str | Path | None) -> Weights:
    """Load tests/test_weights.json. Two accepted shapes:

    Flat (legacy/simple) — a bare mapping of test name -> signed weight.
    This populates traj_tests.tests but leaves traj_tests.weight at 0, so a
    task doesn't silently start contributing to the ledger just by having a
    test_weights.json file — a task author must opt in explicitly with the
    component shape below.

        {"test_used_search_tool": 1, "test_called_forbidden_tool": -2}

    Component (full) shape:

        {
          "threshold": 1.0,
          "components": {
            "traj_tests": {"weight": 3, "tests": {"test_x": 1, "test_y": -2}},
            "rubric":     {"weight": 5}
          }
        }

    A missing file returns all-zero weights (traj_tests inert, nothing
    changes vs. today's grading).
    """
    if path is None:
        return Weights()
    p = Path(path)
    if not p.exists():
        return Weights()
    data = json.loads(p.read_text(encoding="utf-8"))

    if "components" not in data and "threshold" not in data:
        # flat shape: bare {test_name: weight}
        return Weights(traj_tests=ComponentWeight(weight=0.0, tests=dict(data)))

    components = data.get("components", {})
    traj_raw = components.get("traj_tests", {})
    rubric_raw = components.get("rubric", {})
    state_raw = components.get("state_completion", {})
    misbehave_raw = components.get("state_misbehave", {})
    return Weights(
        threshold=float(data.get("threshold", 1.0)),
        traj_tests=ComponentWeight(
            weight=float(traj_raw.get("weight", 0.0)),
            tests=dict(traj_raw.get("tests", {})),
        ),
        rubric=ComponentWeight(weight=float(rubric_raw.get("weight", 0.0))),
        state_completion=ComponentWeight(weight=float(state_raw.get("weight", 0.0))),
        state_misbehave=ComponentWeight(weight=float(misbehave_raw.get("weight", 0.0))),
    )


# Source of the pytest plugin that runs inside the grading subprocess.
#
# Embedded as text rather than shipped as a file because this module is copied
# verbatim into every task bundle as tests/weighted_judge.py, so it has to stay
# self-contained.
#
# A test's verdict is its RETURN VALUE when it returns one, and otherwise its
# call-phase outcome. Both styles exist in the bundles and they must both grade
# honestly:
#
#   assert style   def test_x(): assert cond      -> returns None, verdict is
#                                                    the pytest outcome
#   return style   def test_x(): return cond      -> verdict is bool(cond)
#
# Reading only the pytest outcome silently mis-scores every return-style test,
# because a function that returns instead of asserting ALWAYS passes in pytest
# -- pytest's only failure signals are a raised exception or a failed assert.
# The bundles are overwhelmingly return-style (88 of 90 tests in one, 59 of 61
# in another) and they run pytest with
# `-W ignore::pytest.PytestReturnNotNoneWarning`, so nothing warned either. The
# effect was not a missing score but a fabricated one: every goal recorded as
# met and every guard as tripped, whatever the agent actually did.
#
# pytest_pyfunc_call runs the function itself and returns True to claim the
# call, so the body executes exactly once -- calling it here and letting pytest
# call it again would double every side effect the test performs.
_COLLECTOR_PLUGIN = '''\
import json
import os

import pytest

_OUT = os.environ["MCPATLAS_GRADER_RESULTS"]
_results = {}
_returns = {}


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem):
    """Run the test body once and keep whatever it returned."""
    argnames = getattr(pyfuncitem, "_fixtureinfo", None)
    argnames = getattr(argnames, "argnames", ()) or ()
    kwargs = {name: pyfuncitem.funcargs[name] for name in argnames}
    _returns[pyfuncitem.name] = pyfuncitem.obj(**kwargs)
    return True


def pytest_runtest_logreport(report):
    name = report.nodeid.rsplit("::", 1)[-1]
    if report.when == "call":
        # A test that raised never reached its return statement, so the pytest
        # outcome is the only truth available for it.
        verdict = _returns.get(name)
        if report.outcome == "passed" and verdict is not None:
            _results[name] = bool(verdict)
        else:
            _results[name] = report.outcome == "passed"
    elif report.when in ("setup", "teardown") and report.outcome == "failed":
        _results[name] = False


def pytest_sessionfinish(session, exitstatus):
    with open(_OUT, "w", encoding="utf-8") as fh:
        json.dump(_results, fh)
'''

_RESULTS_VAR = "MCPATLAS_GRADER_RESULTS"
_TIMEOUT_VAR = "MCPATLAS_GRADER_TIMEOUT"
_SANDBOX_VAR = "MCPATLAS_SANDBOX"
_DEFAULT_TIMEOUT = 300.0


def _grader_timeout() -> float:
    """Wall-clock ceiling for one graded suite.

    A bundle-authored test that never returns would otherwise hang the
    grader indefinitely. Timing out scores the task 0 rather than crediting
    it, so there is nothing to gain by hanging: a suite that never reports
    earns no goal and triggers no guard, and score_traj_tests floors at 0.
    """
    raw = os.environ.get(_TIMEOUT_VAR, "").strip()
    if not raw:
        return _DEFAULT_TIMEOUT
    try:
        value = float(raw)
    except ValueError:
        return _DEFAULT_TIMEOUT
    return value if value > 0 else _DEFAULT_TIMEOUT


def _sandbox_prefix() -> list[str]:
    """Optional argv prefix that confines the grading subprocess.

    The subprocess boundary below stops a bundle test from reaching the
    grader's own objects, which is the channel that let a planted test
    rebind the scoring functions that were about to score it. It does not
    stop that test from reaching the filesystem the grading user can reach.

    Closing that needs a real sandbox, and which one is correct depends on
    where the grader runs -- in Harbor the suite is already inside a task
    container, on a developer box it is not. So the choice is bound by the
    operator here rather than picked in this module. Set MCPATLAS_SANDBOX to
    the command that should wrap the interpreter, for example:

        MCPATLAS_SANDBOX="bwrap --ro-bind /usr /usr --dev /dev --unshare-all"
    """
    raw = os.environ.get(_SANDBOX_VAR, "").strip()
    return shlex.split(raw) if raw else []


def _run_traj_pytest(trajectory: dict | list, test_file: str | Path) -> dict[str, bool]:
    """Run `test_file` under pytest against `trajectory`, return
    {test_function_name: passed}.

    The suite runs in a separate interpreter, and that is the point rather
    than an implementation detail. `test_file` is supplied by the task
    bundle and pytest executes it at collection time, so running it in this
    process put task-authored code inside the process that computes and
    writes the reward that code is being scored by -- close enough to rebind
    score_traj_tests, edit the ledger, or write reward.json outright. A
    subprocess makes the grader's own state unreachable from the graded
    code.

    Two limits worth stating rather than implying. The boundary is a process
    boundary, not a filesystem one: a test can still read and write whatever
    the grading user can, and MCPATLAS_SANDBOX exists to close that (see
    _sandbox_prefix). And a run that dies or times out returns whatever the
    plugin recorded, usually nothing, which scores 0 rather than crediting
    anything.

    `trajectory` is staged to a temp file that MCPATLAS_TRAJECTORY points at
    for the life of the subprocess only; this process's own environment is
    never mutated, so concurrent graders no longer race each other over it.
    """
    test_file = Path(test_file)
    if not test_file.exists():
        return {}

    traj_asserts.reset_cache()

    with tempfile.TemporaryDirectory(prefix="mcpatlas-grade-") as workdir:
        work = Path(workdir)
        traj_path = work / "trajectory.json"
        traj_path.write_text(json.dumps(trajectory), encoding="utf-8")
        results_path = work / "results.json"
        (work / "mcpatlas_grader_collect.py").write_text(_COLLECTOR_PLUGIN, encoding="utf-8")

        env = dict(os.environ)
        env[_ENV_VAR] = str(traj_path)
        env[_RESULTS_VAR] = str(results_path)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONPATH"] = os.pathsep.join(
            part for part in (str(work), str(_SCORING_DIR), os.environ.get("PYTHONPATH", ""))
            if part
        )

        argv = [
            *_sandbox_prefix(),
            sys.executable, "-m", "pytest", str(test_file),
            "-q", "-p", "no:cacheprovider", "--import-mode=importlib",
            "-p", "mcpatlas_grader_collect",
        ]
        try:
            subprocess.run(
                argv, env=env, capture_output=True, timeout=_grader_timeout(), check=False
            )
        except (subprocess.TimeoutExpired, OSError):
            return {}

        if not results_path.is_file():
            return {}
        try:
            loaded = json.loads(results_path.read_text(encoding="utf-8"))
        except ValueError:
            return {}

    if not isinstance(loaded, dict):
        return {}
    return {str(name): bool(passed) for name, passed in loaded.items()}


def traj_test_rows(results: dict[str, bool], weights: dict[str, float]) -> list[dict[str, Any]]:
    """One row per weighted test, for downstream reporting (ctrf.json /
    detail.json). `outcome` is the *polarity-applied* verdict — "credited"
    for a passing goal or a non-triggering guard, "penalized" for a
    triggering guard, "missed" for a failing goal, "not_run" if the test
    wasn't collected (e.g. it errored at collection time)."""
    rows: list[dict[str, Any]] = []
    for name, weight in weights.items():
        raw = results.get(name)
        if raw is None:
            outcome = "not_run"
        elif weight >= 0:
            outcome = "credited" if raw else "missed"
        else:
            outcome = "penalized" if raw else "credited"
        rows.append({
            "name": name,
            "weight": weight,
            "raw_passed": raw,
            "outcome": outcome,
        })
    return rows


def score_traj_tests(results: dict[str, bool], weights: dict[str, float]) -> float | None:
    """Combine per-test pass/fail with signed weights into one value in
    [0, 1], or None if there are no goal (positive-weight) tests to
    normalize against — see the module docstring."""
    pos_total = sum(w for w in weights.values() if w > 0)
    if pos_total <= 0:
        return None
    earned = sum(w for name, w in weights.items() if w > 0 and results.get(name))
    penalty = sum(abs(w) for name, w in weights.items() if w < 0 and results.get(name))
    return max(0.0, (earned - penalty) / pos_total)


def score_task(
    trajectory: dict | list,
    test_file: str | Path,
    weights_file: str | Path | None,
) -> dict[str, Any]:
    """Convenience end-to-end entry point for one task: load weights, run
    the pytest suite, score it. Returns a dict with `value` (float|None),
    `weight` (this component's ledger weight — 0 unless the task's
    test_weights.json opts in), and `rows` (per-test detail for reporting)."""
    weights = load_weights(weights_file)
    results = _run_traj_pytest(trajectory, test_file)
    value = score_traj_tests(results, weights.traj_tests.tests)
    return {
        "value": value,
        "weight": weights.traj_tests.weight,
        "rows": traj_test_rows(results, weights.traj_tests.tests),
    }


# ============================================================================
# Phase 3: reward ledger combiner + verifier artifact writer
# ============================================================================
#
# The ledger carries up to four components: traj_tests (Channel A), rubric
# (Channel B), and the state channel pair state_completion (Rc, positive
# weight) / state_misbehave (Rb, negative weight), captured by the bundle's
# tests/state_dump.py and emitted to state_channel.json. graph_plan is retired
# (no graph-plan grader runs in tests/test.sh). Positive components fold their
# own internal goal/guard polarity into a single "goodness" value in [0, 1]
# (see score_traj_tests() and rubric_weighted.evaluate_rubric()); a
# negative-weight component is a penalty whose value in [0, 1] measures how
# much of the forbidden thing happened. The combination follows the documented
# formula: reward = max(0, (sum of weight*value over all components) / basis),
# where basis is the sum of the positive weights.


def combine_ledger(ledger: dict[str, dict[str, Any]]) -> float | None:
    """Combine component values into one reward in [0, 1].

    `ledger` maps component name -> {"weight": float, "value": float|None}.
    A component only counts if its weight is nonzero and its value isn't None
    (None means "no goal tests/criteria defined" — see score_traj_tests()/
    evaluate_rubric() — so there's nothing to weigh in for that component).
    Negative-weight components (state_misbehave) subtract from the numerator
    but never join the basis, and the result floors at 0.

    Returns None if no positive-weight component is active — the caller should
    fall back to whatever the pipeline's default (unweighted) scoring already
    does rather than treating an all-inert ledger as a reward of 0.
    """
    active = {
        name: c for name, c in ledger.items()
        if c.get("weight", 0) != 0 and c.get("value") is not None
    }
    positive = {name: c for name, c in active.items() if c["weight"] > 0}
    if not positive:
        return None
    basis = sum(c["weight"] for c in positive.values())
    raw = sum(c["weight"] * c["value"] for c in active.values()) / basis
    return max(0.0, raw)


def judge_weighted(
    *,
    trajectory: dict | list | None = None,
    test_file: str | Path | None = None,
    test_weights_file: str | Path | None = None,
    traj_results: dict[str, bool] | None = None,
    rubric_value: float | None = None,
    rubric_weight: float = 0.0,
    rubric_rows: list[dict[str, Any]] | None = None,
    state_channel: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Top-level combiner for one task's grading run.

    Channel A is computed here directly (given trajectory + test_file +
    test_weights_file), since it's synchronous and self-contained. Channel B
    is async (it makes LLM-judge calls via rubric_weighted.evaluate_rubric)
    so its already-computed `value`/`weight`/`rows` are passed in instead of
    computed here — callers `await evaluate_rubric(...)` first, then pass the
    result through. Pass `traj_results` directly (skipping the pytest run)
    if you've already run it, e.g. from a caller that also wants the raw
    per-test dict for other purposes.

    Returns {"reward": float|None, "ledger": {...}, "traj_test_rows": [...],
    "rubric_rows": [...]}. `reward` is None only if neither channel is
    active for this task (see combine_ledger()) — callers should treat that
    as "this task isn't opted into weighted grading" and fall back to
    whatever default scoring the pipeline already does.
    """
    weights = load_weights(test_weights_file) if test_weights_file else Weights()
    ledger: dict[str, dict[str, Any]] = {}
    rows_traj: list[dict[str, Any]] = []

    if weights.traj_tests.weight and (traj_results is not None or (trajectory is not None and test_file)):
        results = traj_results if traj_results is not None else _run_traj_pytest(trajectory, test_file)
        value = score_traj_tests(results, weights.traj_tests.tests)
        rows_traj = traj_test_rows(results, weights.traj_tests.tests)
        if value is not None:
            ledger["traj_tests"] = {"weight": weights.traj_tests.weight, "value": value}

    effective_rubric_weight = rubric_weight or weights.rubric.weight
    if effective_rubric_weight and rubric_value is not None:
        ledger["rubric"] = {"weight": effective_rubric_weight, "value": rubric_value}

    # State channel (Rc/Rb): the parsed contents of state_channel.json, written
    # by the bundle's tests. A dump that could not run reports available: False
    # and stays out of the ledger — an unmeasured world is not a scored zero.
    if state_channel and state_channel.get("available"):
        if weights.state_completion.weight and state_channel.get("completion") is not None:
            ledger["state_completion"] = {
                "weight": weights.state_completion.weight,
                "value": float(state_channel["completion"]),
            }
        if weights.state_misbehave.weight and state_channel.get("misbehave") is not None:
            ledger["state_misbehave"] = {
                "weight": weights.state_misbehave.weight,
                "value": float(state_channel["misbehave"]),
            }

    reward = combine_ledger(ledger)
    return {
        "reward": reward,
        "ledger": ledger,
        "traj_test_rows": rows_traj,
        "rubric_rows": rubric_rows or [],
    }


def _ctrf_report(traj_test_rows: list[dict[str, Any]], tool_name: str = "mcp-atlas-weighted-judge") -> dict[str, Any]:
    """Render Channel A's per-test rows as a CTRF (Common Test Report Format)
    document. A row's `outcome` (already polarity-applied — "credited" /
    "penalized" / "missed" / "not_run") maps to CTRF status: "credited" ->
    passed, everything else -> failed/skipped, so CTRF readers see pass/fail
    from the *grading* perspective, not raw pytest pass/fail."""
    now_ms = int(time.time() * 1000)
    tests = []
    passed = failed = skipped = 0
    for row in traj_test_rows:
        if row["outcome"] == "not_run":
            status = "skipped"
            skipped += 1
        elif row["outcome"] == "credited":
            status = "passed"
            passed += 1
        else:
            status = "failed"
            failed += 1
        tests.append({
            "name": row["name"],
            "status": status,
            "duration": 0,
            "extra": {"weight": row["weight"], "raw_passed": row["raw_passed"], "outcome": row["outcome"]},
        })
    return {
        "results": {
            "tool": {"name": tool_name},
            "summary": {
                "tests": len(tests),
                "passed": passed,
                "failed": failed,
                "pending": 0,
                "skipped": skipped,
                "other": 0,
                "start": now_ms,
                "stop": now_ms,
            },
            "tests": tests,
        }
    }


def write_verifier_artifacts(out_dir: str | Path, result: dict[str, Any]) -> None:
    """Write Harbor's expected verifier/ artifact set from a judge_weighted()
    result: reward.json (Harbor's VerifierResult shape, {"reward": float}),
    reward.txt (the same value as plain text), ctrf.json (Channel A's
    per-test results in CTRF format), and detail.json (the full breakdown —
    ledger + both channels' per-item rows — for debugging, not read by
    Harbor). Mirrors tests/agent_judge.py's existing reward.json/reward.txt
    convention exactly, so a task's verifier output is drop-in compatible
    whether it comes from agent_judge.py or this weighted judge.

    A None reward (no component active) is written as 0.0 — a task that
    opted into weighted grading but has nothing scoreable shouldn't emit an
    invalid/missing reward file.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reward = result.get("reward")
    reward_value = reward if reward is not None else 0.0
    (out_dir / "reward.txt").write_text(str(reward_value))
    (out_dir / "reward.json").write_text(json.dumps({"reward": reward_value}, indent=2))
    (out_dir / "ctrf.json").write_text(json.dumps(_ctrf_report(result.get("traj_test_rows", [])), indent=2))
    (out_dir / "detail.json").write_text(json.dumps(result, indent=2))
