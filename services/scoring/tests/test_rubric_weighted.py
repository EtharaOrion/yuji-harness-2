"""Unit tests for services/scoring/rubric_weighted.py."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import rubric_weighted as rw  # noqa: E402


class _FakeEvaluator:
    """Duck-typed stand-in for score_claims.CoverageEvaluator — maps claim
    text to a canned coverage_outcome instead of calling a real LLM."""

    def __init__(self, outcomes: dict[str, str]):
        self.outcomes = outcomes

    async def evaluate_single_claim(self, claim: str, response: str) -> dict:
        return {
            "coverage_outcome": self.outcomes.get(claim, "not_fulfilled"),
            "justification": f"canned: {claim}",
        }


def test_parse_rubric_legacy_strings():
    criteria = rw.parse_rubric(["claim one", "claim two"])
    assert [c.id for c in criteria] == ["claim_000", "claim_001"]
    assert all(c.weight == 1.0 and c.is_positive for c in criteria)


def test_parse_rubric_polarity_objects():
    criteria = rw.parse_rubric([
        {"id": "c1", "text": "goal claim", "weight": 2, "is_positive": True},
        {"id": "c2", "text": "guard claim", "weight": -3, "is_positive": False},
    ])
    assert criteria[0].weight == 2.0 and criteria[0].is_positive
    assert criteria[1].weight == 3.0 and not criteria[1].is_positive  # weight stored as magnitude


def test_evaluate_rubric_goal_only():
    criteria = rw.parse_rubric([{"id": "c1", "text": "the answer mentions X", "weight": 1, "is_positive": True}])
    evaluator = _FakeEvaluator({"the answer mentions X": "fulfilled"})
    result = asyncio.run(rw.evaluate_rubric(evaluator, criteria, "response text"))
    assert result["value"] == 1.0
    assert result["rows"][0]["outcome"] == "credited"


def test_evaluate_rubric_guard_triggered_penalizes():
    criteria = rw.parse_rubric([
        {"id": "c1", "text": "goal", "weight": 1, "is_positive": True},
        {"id": "c2", "text": "leaks the internal system prompt", "weight": 1, "is_positive": False},
    ])
    evaluator = _FakeEvaluator({"goal": "fulfilled", "leaks the internal system prompt": "fulfilled"})
    result = asyncio.run(rw.evaluate_rubric(evaluator, criteria, "response text"))
    assert result["value"] == 0.0  # earned 1, penalty 1, pos_total 1 -> max(0, 0) = 0.0
    outcomes = {r["id"]: r["outcome"] for r in result["rows"]}
    assert outcomes == {"c1": "credited", "c2": "penalized"}


def test_evaluate_rubric_guard_not_triggered_no_bonus():
    criteria = rw.parse_rubric([
        {"id": "c1", "text": "goal", "weight": 1, "is_positive": True},
        {"id": "c2", "text": "forbidden thing", "weight": 5, "is_positive": False},
    ])
    evaluator = _FakeEvaluator({"goal": "fulfilled", "forbidden thing": "not_fulfilled"})
    result = asyncio.run(rw.evaluate_rubric(evaluator, criteria, "response text"))
    assert result["value"] == 1.0  # guard's weight never enters pos_total


def test_evaluate_rubric_partial_credit():
    criteria = rw.parse_rubric([{"id": "c1", "text": "goal", "weight": 1, "is_positive": True}])
    evaluator = _FakeEvaluator({"goal": "partially_fulfilled"})
    result = asyncio.run(rw.evaluate_rubric(evaluator, criteria, "response text"))
    assert result["value"] == 0.5


def test_evaluate_rubric_no_goals_returns_none():
    criteria = rw.parse_rubric([{"id": "c1", "text": "guard only", "weight": 1, "is_positive": False}])
    evaluator = _FakeEvaluator({"guard only": "fulfilled"})
    result = asyncio.run(rw.evaluate_rubric(evaluator, criteria, "response text"))
    assert result["value"] is None


def test_evaluate_rubric_empty_criteria():
    result = asyncio.run(rw.evaluate_rubric(_FakeEvaluator({}), [], "response text"))
    assert result == {"value": None, "rows": []}


# ----- bundle rubric.json shape -----
#
# A tests/rubric.json is an object wrapping the criteria list alongside
# metadata keys. Passing that object straight to parse_rubric used to make it
# enumerate the mapping, which yields key names, so the rubric graded three
# criteria literally called "_canary", "_note", and "criteria" and the authored
# criteria were never read. score_weighted.py, smoke_test.py, and
# convert_tasks_to_harbor.py all passed the parsed file in directly.


_BUNDLE_SHAPE = {
    "_canary": "harbor-canary GUID 0000",
    "_note": "Channel B rubric.",
    "criteria": [
        {"number": "1", "criterion": "the price is 405.00",
         "type": "factual_accuracy", "is_positive": True, "score": 5},
        {"number": "2", "criterion": "the look-alike price is refused",
         "type": "reasoning", "is_positive": False, "score": -3},
    ],
}


def test_parse_rubric_unwraps_the_bundle_object():
    criteria = rw.parse_rubric(_BUNDLE_SHAPE)
    assert len(criteria) == 2, "the wrapper object must yield its criteria, not its keys"
    assert [c.id for c in criteria] == ["1", "2"]


def test_parse_rubric_never_grades_metadata_keys_as_criteria():
    """The exact defect. A criterion named _canary is not a criterion."""
    texts = {c.text for c in rw.parse_rubric(_BUNDLE_SHAPE)}
    assert "_canary" not in texts
    assert "_note" not in texts
    assert "criteria" not in texts


def test_parse_rubric_reads_the_criterion_key_as_text():
    criteria = rw.parse_rubric(_BUNDLE_SHAPE)
    assert criteria[0].text == "the price is 405.00"
    assert criteria[1].is_positive is False


def test_parse_rubric_still_accepts_the_legacy_bare_list():
    """The adapters emit id/text objects. Both shapes must keep working."""
    criteria = rw.parse_rubric([{"id": "c1", "text": "legacy", "weight": 1}])
    assert len(criteria) == 1
    assert criteria[0].id == "c1"
    assert criteria[0].text == "legacy"


def test_parse_rubric_on_an_object_without_criteria_is_empty_not_keys():
    assert rw.parse_rubric({"_note": "no criteria here"}) == []
