"""Unit tests for weighted_judge.py's Phase 3 additions: combine_ledger(),
judge_weighted(), and write_verifier_artifacts()."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import weighted_judge as wj  # noqa: E402


def test_combine_ledger_weighted_average():
    ledger = {
        "traj_tests": {"weight": 3, "value": 1.0},
        "rubric": {"weight": 1, "value": 0.0},
    }
    assert wj.combine_ledger(ledger) == 0.75  # (3*1 + 1*0) / 4


def test_combine_ledger_ignores_zero_weight_and_none_value():
    ledger = {
        "traj_tests": {"weight": 0, "value": 1.0},   # inert — excluded
        "rubric": {"weight": 5, "value": None},       # no criteria — excluded
    }
    assert wj.combine_ledger(ledger) is None


def test_combine_ledger_empty():
    assert wj.combine_ledger({}) is None


def test_judge_weighted_traj_only(tmp_path):
    test_file = tmp_path / "test_outputs.py"
    test_file.write_text(
        'from traj_asserts import called_with\n'
        'def test_goal():\n'
        '    assert called_with("search")\n'
    )
    weights_file = tmp_path / "test_weights.json"
    weights_file.write_text(json.dumps({
        "components": {"traj_tests": {"weight": 4, "tests": {"test_goal": 1}}}
    }))
    trajectory = [
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "1", "type": "function", "function": {"name": "search", "arguments": "{}"}}
        ]},
    ]
    result = wj.judge_weighted(trajectory=trajectory, test_file=test_file, test_weights_file=weights_file)
    assert result["reward"] == 1.0
    assert result["ledger"] == {"traj_tests": {"weight": 4.0, "value": 1.0}}
    assert len(result["traj_test_rows"]) == 1


def test_judge_weighted_combines_both_channels(tmp_path):
    test_file = tmp_path / "test_outputs.py"
    test_file.write_text(
        'from traj_asserts import called_with\n'
        'def test_goal():\n'
        '    assert called_with("search")\n'
    )
    weights_file = tmp_path / "test_weights.json"
    weights_file.write_text(json.dumps({
        "components": {
            "traj_tests": {"weight": 1, "tests": {"test_goal": 1}},
            "rubric": {"weight": 1},
        }
    }))
    trajectory = [
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "1", "type": "function", "function": {"name": "search", "arguments": "{}"}}
        ]},
    ]
    result = wj.judge_weighted(
        trajectory=trajectory, test_file=test_file, test_weights_file=weights_file,
        rubric_value=0.5, rubric_weight=1.0, rubric_rows=[{"id": "c1"}],
    )
    assert result["reward"] == 0.75  # (1*1.0 + 1*0.5) / 2
    assert result["ledger"]["rubric"] == {"weight": 1.0, "value": 0.5}
    assert result["rubric_rows"] == [{"id": "c1"}]


def test_judge_weighted_no_active_components_returns_none_reward(tmp_path):
    result = wj.judge_weighted()
    assert result["reward"] is None
    assert result["ledger"] == {}


def test_write_verifier_artifacts(tmp_path):
    result = {
        "reward": 0.75,
        "ledger": {"traj_tests": {"weight": 1, "value": 0.75}},
        "traj_test_rows": [
            {"name": "test_goal", "weight": 1, "raw_passed": True, "outcome": "credited"},
            {"name": "test_guard", "weight": -1, "raw_passed": True, "outcome": "penalized"},
        ],
        "rubric_rows": [],
    }
    out_dir = tmp_path / "verifier"
    wj.write_verifier_artifacts(out_dir, result)

    assert (out_dir / "reward.txt").read_text() == "0.75"
    reward_json = json.loads((out_dir / "reward.json").read_text())
    assert reward_json == {"reward": 0.75}

    ctrf = json.loads((out_dir / "ctrf.json").read_text())
    assert ctrf["results"]["summary"]["tests"] == 2
    assert ctrf["results"]["summary"]["passed"] == 1
    assert ctrf["results"]["summary"]["failed"] == 1
    statuses = {t["name"]: t["status"] for t in ctrf["results"]["tests"]}
    assert statuses == {"test_goal": "passed", "test_guard": "failed"}

    detail = json.loads((out_dir / "detail.json").read_text())
    assert detail == result


def test_write_verifier_artifacts_none_reward_writes_zero(tmp_path):
    out_dir = tmp_path / "verifier"
    wj.write_verifier_artifacts(out_dir, {"reward": None, "ledger": {}, "traj_test_rows": [], "rubric_rows": []})
    assert (out_dir / "reward.txt").read_text() == "0.0"
    assert json.loads((out_dir / "reward.json").read_text()) == {"reward": 0.0}
