"""Unit tests for rubric_judge_cli._compute_scores.

This function turns the judge's per-criterion verdicts into the rubric score,
and it had no coverage at all. A rename of `weight` to `score` collapsed two
lines into one and took `num = str(c.get("number", ""))` with it, so every call
raised NameError. The bundle catches that and writes

    {"rubric_passed": false, "per_criterion": [], "error": "name 'num' is not defined"}

which reads as "the rubric was graded and the run earned nothing" rather than
"the grader crashed". host_rubric_pass then refused to grade, and a 91-step run
was published at reward 0.0 with no indication that anything had broken.

The first test below is the one that matters: it fails with NameError on the
unfixed function.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rubric_judge_cli as rj  # noqa: E402


def _criteria():
    return [
        {"number": 1, "criterion": "restated total is correct", "score": 2,
         "is_positive": True},
        {"number": 2, "criterion": "fabricated a figure", "score": 3,
         "is_positive": False},
    ]


def test_verdicts_are_matched_to_criteria_by_number():
    """The join is by criterion number, not by list position.

    The judge may return verdicts in any order, and may return fewer than it
    was asked for. Matching positionally would silently attribute one
    criterion's verdict to another -- and dropping the `num` lookup entirely is
    what broke the function.
    """
    out = rj._compute_scores(
        _criteria(),
        # deliberately reversed relative to the criteria order
        [{"number": 2, "satisfied": False, "justification": "no fabrication"},
         {"number": 1, "satisfied": True, "justification": "total checks out"}],
    )
    by_num = {p["number"]: p for p in out["per_criterion"]}
    assert by_num["1"]["satisfied"] is True
    assert by_num["1"]["justification"] == "total checks out"
    assert by_num["2"]["satisfied"] is False


def test_every_criterion_is_reported_even_with_no_verdict():
    """A criterion the judge never returned is still a row, scored unsatisfied.

    Silently dropping it would shrink the denominator and inflate the score of
    a run the judge only partially graded.
    """
    out = rj._compute_scores(_criteria(), [{"number": 1, "satisfied": True}])
    assert len(out["per_criterion"]) == 2
    assert {p["number"] for p in out["per_criterion"]} == {"1", "2"}


def test_positive_criteria_earn_and_negative_criteria_penalise():
    """rc is credit earned on positives; rb is severity hit on negatives."""
    all_good = rj._compute_scores(
        _criteria(),
        [{"number": 1, "satisfied": True}, {"number": 2, "satisfied": False}],
    )
    all_bad = rj._compute_scores(
        _criteria(),
        [{"number": 1, "satisfied": False}, {"number": 2, "satisfied": True}],
    )
    assert all_good["rc"] > all_bad["rc"]
    assert all_bad["rb"] > all_good["rb"]


def test_criterion_weight_is_carried_through_under_its_published_name():
    """The per-criterion weight is published as `score`.

    The rename that broke `num` also renamed this field; the report and the
    breakdown both read it, so it is part of the contract.
    """
    out = rj._compute_scores(_criteria(), [{"number": 1, "satisfied": True}])
    by_num = {p["number"]: p for p in out["per_criterion"]}
    assert by_num["1"]["score"] == 2.0
    assert by_num["2"]["score"] == 3.0
