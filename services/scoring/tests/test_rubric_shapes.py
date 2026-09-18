"""Every rubric shape this project ships, loaded or refused with a reason.

Three shapes exist: a bare list and {"rubric": [...]} are rubric_judge's own,
{"criteria": [...]} is what the dataset bundles carry. Only the first two
loaded, so score_rubric could not read a shipped rubric at all.

The interesting half is what must still be refused. The bundle shape carries
`is_positive`, and rubric_judge_cli grades a false one against a separate
negative pool. score_rubric scalarizes one weighted mean and has no such pool,
so admitting a penalty criterion would award reward for satisfying a criterion
that describes a fault. Both shipped bundles contain such criteria, so the
refusal is the live path rather than a hypothetical.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

rj = pytest.importorskip("services.scoring.rubric_judge")
cli = pytest.importorskip("services.scoring.rubric_judge_cli")


def _write(tmp_path: Path, doc) -> Path:
    p = tmp_path / "rubric.json"
    p.write_text(json.dumps(doc))
    return p


_POSITIVE = [
    {"number": "1", "criterion": "a", "score": 5, "is_positive": True},
    {"number": "2", "criterion": "b", "score": 3, "is_positive": True},
]


# ----- shapes that load -----


def test_bare_list_loads(tmp_path):
    assert len(rj._load_rubric(_write(tmp_path, [{"id": "c1", "criterion": "a"}]))) == 1


def test_rubric_key_loads(tmp_path):
    assert len(rj._load_rubric(_write(tmp_path, {"rubric": [{"id": "c1"}]}))) == 1


def test_bundle_criteria_shape_loads(tmp_path):
    """The shape every dataset bundle ships, which used to raise."""
    out = rj._load_rubric(_write(tmp_path, {"criteria": _POSITIVE}))
    assert len(out) == 2


def test_bundle_number_becomes_the_permutation_seed(tmp_path):
    """Bundle criteria carry `number`, not `id`. Without the mapping every
    criterion seeds _permutations_for from "" and the position discipline
    degenerates to one shared ordering that looks like it ran."""
    out = rj._load_rubric(_write(tmp_path, {"criteria": _POSITIVE}))
    assert [c["id"] for c in out] == ["1", "2"]
    assert rj._permutations_for("1", 4) != rj._permutations_for("2", 4)


def test_bundle_score_becomes_weight_magnitude(tmp_path):
    """`score` is a point value on {-5,-3,-1,1,3,5}; its sign is carried by
    is_positive, so the weight is the magnitude."""
    out = rj._load_rubric(_write(tmp_path, {"criteria": _POSITIVE}))
    assert [c["weight"] for c in out] == [5.0, 3.0]
    assert rj._normalize_weights(out) == [0.625, 0.375]


def test_an_explicit_weight_is_not_overwritten(tmp_path):
    doc = {"criteria": [{"number": "1", "score": 5, "weight": 2.0, "is_positive": True}]}
    assert rj._load_rubric(_write(tmp_path, doc))[0]["weight"] == 2.0


# ----- shapes that must be refused -----


def test_penalty_criteria_are_refused(tmp_path):
    doc = {"criteria": _POSITIVE + [
        {"number": "3", "criterion": "c", "score": -5, "is_positive": False},
    ]}
    with pytest.raises(ValueError) as e:
        rj._load_rubric(_write(tmp_path, doc))
    assert "is_positive=false" in str(e.value)
    assert "rubric_judge_cli" in str(e.value)


@pytest.mark.parametrize("bundle", [
    "draft-side-table-lot-price", "bull-street-lot-expense-claim",
])
def test_shipped_rubrics_are_refused_for_the_right_reason(bundle):
    """Not a shape error any more: an accurate statement that this grader
    cannot express what the rubric declares."""
    path = _ROOT.parent / "dataset" / bundle / "tests" / "rubric.json"
    if not path.is_file():
        pytest.skip(f"{bundle} not checked out")
    with pytest.raises(ValueError) as e:
        rj._load_rubric(path)
    assert "penalty criteria" in str(e.value)


def test_an_unrecognised_shape_still_raises(tmp_path):
    with pytest.raises(ValueError):
        rj._load_rubric(_write(tmp_path, {"something": "else"}))


# ----- the CLI loader -----


def test_cli_loads_every_shape(tmp_path):
    assert len(cli._load_criteria(_write(tmp_path, {"criteria": _POSITIVE}))) == 2
    assert len(cli._load_criteria(_write(tmp_path, {"rubric": _POSITIVE}))) == 2
    assert len(cli._load_criteria(_write(tmp_path, _POSITIVE))) == 2


def test_cli_refuses_instead_of_grading_nothing(tmp_path):
    """It used to return []. Zero criteria grade cleanly and the run reports a
    finished rubric channel that judged nothing, which is the worst outcome
    available: a silent wrong answer that looks like a pass."""
    with pytest.raises(ValueError):
        cli._load_criteria(_write(tmp_path, {"something": "else"}))
