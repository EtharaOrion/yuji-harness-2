"""Both-halves tests for the principle 5a judge-reliability discipline.

These assert behaviour, not the presence of keywords. The audit instrument
that raised J-JUDGE-* matches source text, so a comment mentioning
"conformal" would silence it; that would be gaming the check rather than
satisfying it. Each test below drives the real code path with a stub judge.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

rj = pytest.importorskip("services.scoring.rubric_judge")

CRIT = {"id": "c1", "description": "did the thing", "weight": 1.0}
TRAJ = {"messages": [{"role": "user", "content": "hi"}]}


def _run(monkeypatch, scores):
    """Drive the disciplined scorer with a scripted sequence of judge replies."""
    seen = []

    def fake_call(model, messages):
        seen.append(messages[1]["content"])
        return {"score": scores[min(len(seen) - 1, len(scores) - 1)], "reason": "r"}

    monkeypatch.setattr(rj, "_call_judge_once", fake_call)
    out = rj._score_criterion_with_discipline(
        CRIT, "prompt", TRAJ, "m", n_trials=11
    )
    return out, seen


def test_trials_actually_run_eleven_times(monkeypatch):
    (score, _, err, rel), seen = _run(monkeypatch, [1.0])
    assert len(seen) == 11, "the discipline must issue one call per trial"
    assert rel["trials_requested"] == 11
    assert rel["trials_succeeded"] == 11
    assert rel["discipline_satisfied"] is True
    assert err is None


def test_floor_is_at_least_eleven():
    assert rj._JUDGE_TRIALS_FLOOR >= 11


def test_position_is_actually_randomized(monkeypatch):
    (_, _, _, rel), seen = _run(monkeypatch, [1.0])
    assert rel["position_randomized"] is True
    # Section order must genuinely vary across trials, not merely be declared.
    assert len(set(seen)) > 1, "prompts were identical; no position variation occurred"
    orders = [t["section_order"] for t in rel["trials"]]
    assert orders[0] is None, "trial 0 must be the canonical order"
    assert any(o is not None and o != [0, 1, 2] for o in orders[1:])


def test_permutations_are_deterministic():
    a = rj._permutations_for("c1", 11)
    b = rj._permutations_for("c1", 11)
    assert a == b, "same criterion must reproduce the same orderings"
    assert rj._permutations_for("c2", 11) != a, "different criteria must differ"


def test_canonical_score_is_trial_zero_unchanged(monkeypatch):
    # Trial 0 returns 1.0, later trials drift. The reported score must stay 1.0
    # so downstream `score == 1.0` comparisons are unaffected.
    (score, _, _, rel), _ = _run(monkeypatch, [1.0, 0.0])
    assert score == 1.0
    assert rel["stability"]["unstable"] is True


def test_conformal_set_widens_on_disagreement(monkeypatch):
    (_, _, _, unstable), _ = _run(monkeypatch, [1.0, 0.0])
    (_, _, _, stable), _ = _run(monkeypatch, [1.0])
    assert unstable["conformal"]["available"] is True
    assert stable["conformal"]["available"] is True
    assert unstable["conformal"]["width"] > stable["conformal"]["width"], (
        "a judge that disagrees with itself must produce a wider prediction set"
    )
    assert stable["conformal"]["width"] == 0.0
    assert stable["conformal"]["low_confidence"] is False
    assert unstable["conformal"]["low_confidence"] is True


def test_stable_judge_is_reported_stable(monkeypatch):
    (_, _, _, rel), _ = _run(monkeypatch, [0.5])
    assert rel["stability"]["spread"] == 0.0
    assert rel["stability"]["unstable"] is False
    assert rel["stability"]["stdev"] == 0.0


def test_failed_trials_are_dropped_not_defaulted(monkeypatch):
    calls = {"n": 0}

    def flaky(model, messages):
        calls["n"] += 1
        if calls["n"] % 2 == 0:
            return {"nonsense": True}  # forces the retry path then a failure
        return {"score": 1.0, "reason": "r"}

    monkeypatch.setattr(rj, "_call_judge_once", flaky)
    _, _, _, rel = rj._score_criterion_with_discipline(
        CRIT, "p", TRAJ, "m", n_trials=11
    )
    # No trial may contribute a fabricated score to the sample.
    assert all(s is not None for s in rel["trial_scores"])
    assert rel["trials_succeeded"] + rel["trials_failed"] == rel["trials_requested"]


def test_re_grade_returns_reliability_only(monkeypatch):
    monkeypatch.setattr(
        rj, "_call_judge_once", lambda *a, **k: {"score": 0.25, "reason": "r"}
    )
    rel = rj.re_grade(CRIT, "p", TRAJ, "m", n_trials=11)
    assert rel["trials_succeeded"] == 11
    assert "conformal" in rel and "stability" in rel


def test_single_trial_refuses_to_fake_a_prediction_set():
    c = rj._conformal_interval([0.7], rj._CONFORMAL_ALPHA)
    assert c["available"] is False, "one trial cannot calibrate a prediction set"
    empty = rj._conformal_interval([], rj._CONFORMAL_ALPHA)
    assert empty["available"] is False
