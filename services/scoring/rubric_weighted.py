"""
services/scoring/rubric_weighted.py

Channel B of the weighted grader: judges a set of polarity-tagged rubric
criteria against a final response, then combines the per-criterion verdicts
with signed weights — mirroring weighted_judge.py's Channel A formula.

Deliberately does NOT import services/scoring/score_claims.py. It only needs
one capability from it — "judge one claim against one response, return a
coverage_outcome" — so that's expressed here as a duck-typed interface
instead of a hard import, keeping this module free of score_claims.py's
heavy import chain (matplotlib, aiohttp, dotenv) for callers that only want
to *combine* already-computed verdicts, or that supply a fake evaluator in
tests. In production, pass a real score_claims.CoverageEvaluator instance —
it already satisfies the interface:

    evaluator.evaluate_single_claim(claim: str, response: str) -> Awaitable[
        {"coverage_outcome": "fulfilled" | "partially_fulfilled" | "not_fulfilled", ...}
    ]

Rubric criterion shape (tests/rubric.json):

    [
      {"id": "claim_000", "text": "...", "score": 1, "is_positive": true},
      {"id": "claim_001", "text": "...", "score": -2, "is_positive": false}
    ]

`score` states the magnitude, and authored rubrics sign it by polarity
(a guard reads -5). Only the magnitude is read here; polarity is carried by
the explicit `is_positive` field. That's a deliberate difference from
Channel A's test_weights.json (where polarity *is* the sign): a rubric
criterion's LLM-judged verdict is inherently graded, not boolean, so "how
much of a violation occurred" needs its own magnitude separate from "is this
good or bad" — folding both into one number would conflate them.

Older rubrics also carry a `weight` twin holding the same magnitude
unsigned, and it is still accepted when `score` is absent. `score` is read
first because it is the field the graders and the published breakdown rows
agree on, and because four of this tree's rubrics state only that one.

Legacy compatibility: a bare list of strings (score_claims.py's
extract_claims() output — today's GTFA_CLAIMS shape) is accepted too — each
string becomes a weight=1, is_positive=True goal criterion. This is what
lets a task with no rubric.json at all keep working unchanged.

Scoring formula, symmetric with weighted_judge.score_traj_tests():
    scale     = {"fulfilled": 1.0, "partially_fulfilled": 0.5, "not_fulfilled": 0.0}[outcome]
    pos_total = sum(|w| for goal criteria)
    earned    = sum(|w| * scale for goal criteria)
    penalty   = sum(|w| * scale for guard criteria)   # "fulfilled" == violation occurred
    value     = max(0, (earned - penalty) / pos_total)   if pos_total else None
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Protocol

_SCALE = {"fulfilled": 1.0, "partially_fulfilled": 0.5, "not_fulfilled": 0.0}


class ClaimEvaluator(Protocol):
    def evaluate_single_claim(self, claim: str, response: str) -> Awaitable[dict[str, Any]]: ...


@dataclass
class RubricCriterion:
    id: str
    text: str
    weight: float = 1.0
    is_positive: bool = True


def parse_rubric(raw: Any) -> list[RubricCriterion]:
    """Accepts either the legacy bare-string-list shape (score_claims.py's
    extract_claims() output) or the polarity-tagged object shape described
    in the module docstring."""
    if not raw:
        return []
    # A bundle rubric.json is an object wrapping the criteria list alongside
    # metadata keys such as _canary and _note. Enumerating that mapping yields
    # its key names, so every caller that passed the parsed file straight in
    # was building criteria literally called "_canary", "_note", and
    # "criteria" while the authored criteria were never read. Unwrapping here
    # rather than at each call site fixes score_weighted, smoke_test, and
    # convert_tasks_to_harbor together, and matches what
    # rubric_judge_cli._load_criteria already does.
    if isinstance(raw, dict):
        raw = raw.get("criteria") or []
    out: list[RubricCriterion] = []
    for i, item in enumerate(raw):
        if isinstance(item, str):
            out.append(RubricCriterion(id=f"claim_{i:03d}", text=item))
        elif isinstance(item, dict):
            # `criterion` and `number` are the bundle shape; `text` and `id`
            # are the legacy shape adapters emit. Both are accepted so neither
            # producer has to change.
            text = (
                item.get("text") or item.get("criterion") or item.get("description")
                or item.get("title") or ""
            )
            identifier = item.get("id") or item.get("number") or f"claim_{i:03d}"
            magnitude = item.get("score", item.get("weight", 1.0))
            out.append(RubricCriterion(
                id=str(identifier),
                text=text,
                weight=abs(float(magnitude)),
                is_positive=bool(item.get("is_positive", True)),
            ))
    return out


async def evaluate_rubric(
    evaluator: ClaimEvaluator,
    criteria: list[RubricCriterion],
    response: str,
) -> dict[str, Any]:
    """Judge every criterion (one evaluator call each, run concurrently),
    then combine into one polarity-applied value in [0, 1] (or None if there
    are no goal criteria to normalize against)."""
    if not criteria:
        return {"value": None, "rows": []}

    results = await asyncio.gather(*[
        evaluator.evaluate_single_claim(c.text, response) for c in criteria
    ])
    return score_verdicts(criteria, results)


def score_verdicts(
    criteria: list[RubricCriterion],
    verdicts: list[dict[str, Any]],
) -> dict[str, Any]:
    """Combine already-computed per-criterion verdicts into one value.

    Split out of evaluate_rubric so a caller that already has verdicts can
    apply the same polarity/weight formula without an evaluator. That is what
    a Harbor bundle needs: tests/agent_judge.py has already judged every
    criterion and written its per-criterion results to rubric_breakdown.json,
    so re-judging them in-container would mean a second round of LLM calls
    for verdicts we already hold. One formula, one place, both callers.

    Each verdict is a dict carrying `coverage_outcome`; a plain
    binary `score` (1.0/0.0), which is what agent_judge.py emits, is accepted
    and mapped onto fulfilled/not_fulfilled.
    """
    if not criteria:
        return {"value": None, "rows": []}

    rows: list[dict[str, Any]] = []
    pos_total = 0.0
    earned = 0.0
    penalty = 0.0
    for c, result in zip(criteria, verdicts):
        outcome = result.get("coverage_outcome")
        if outcome is None and "score" in result:
            outcome = "fulfilled" if float(result.get("score") or 0.0) >= 1.0 else "not_fulfilled"
        outcome = outcome or "not_fulfilled"
        if outcome not in _SCALE:
            raise ValueError(
                f"judge returned coverage_outcome {outcome!r}, outside the declared "
                f"vocabulary {sorted(_SCALE)}; refusing rather than silently scoring it "
                "as not_fulfilled"
            )
        scale = _SCALE[outcome]
        contribution = c.weight * scale
        if c.is_positive:
            pos_total += c.weight
            earned += contribution
            outcome_label = "credited" if scale >= 1.0 else ("partial_credit" if scale > 0 else "missed")
        else:
            penalty += contribution
            outcome_label = "penalized" if scale > 0 else "credited"
        rows.append({
            "id": c.id,
            "text": c.text,
            "score": c.weight if c.is_positive else -c.weight,
            "is_positive": c.is_positive,
            "coverage_outcome": outcome,
            "justification": result.get("justification", ""),
            "outcome": outcome_label,
        })

    value = max(0.0, (earned - penalty) / pos_total) if pos_total > 0 else None
    return {"value": value, "rows": rows}
