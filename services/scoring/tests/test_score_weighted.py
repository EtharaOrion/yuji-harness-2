"""Unit tests for services/scoring/score_weighted.py."""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import score_weighted as sw  # noqa: E402

TRAJECTORY = [
    {"role": "assistant", "content": None, "tool_calls": [
        {"id": "1", "type": "function", "function": {"name": "search", "arguments": "{}"}}
    ]},
    {"role": "assistant", "content": "final answer", "tool_calls": None},
]


def _make_task_bundle(tasks_dir: Path, task_id: str, weights: dict) -> None:
    task_tests = tasks_dir / task_id / "tests"
    task_tests.mkdir(parents=True)
    (task_tests / "test_weights.json").write_text(json.dumps(weights))
    (task_tests / "test_outputs.py").write_text(
        'from traj_asserts import called_with\n'
        'def test_goal():\n'
        '    assert called_with("search")\n'
    )



async def _score_row(row, **kwargs):
    """Call sw.score_row with a semaphore built inside the running loop.

    asyncio.Semaphore() binds to the current event loop at construction time
    on Python <= 3.9, so constructing one at call-argument position (outside
    asyncio.run) raises "There is no current event loop".
    """
    return await sw.score_row(row, semaphore=asyncio.Semaphore(4), **kwargs)


def test_score_row_task_never_opted_in(tmp_path):
    tasks_dir = tmp_path / "tasks"
    tasks_dir.mkdir()
    row = {"task_id": "t1", "run_trajectory_json": json.dumps(TRAJECTORY), "response": "final answer"}
    result = asyncio.run(_score_row(
        row, tasks_dir=tasks_dir, model_name="m", rubric_evaluator=None,
    ))
    assert result["weighted_reward"] == ""


def test_score_row_no_tasks_dir(tmp_path):
    row = {"task_id": "t1", "run_trajectory_json": json.dumps(TRAJECTORY), "response": "final answer"}
    result = asyncio.run(_score_row(
        row, tasks_dir=None, model_name="m", rubric_evaluator=None,
    ))
    assert result["weighted_reward"] == ""


def test_score_row_traj_only(tmp_path):
    tasks_dir = tmp_path / "tasks"
    _make_task_bundle(tasks_dir, "t1", {
        "components": {"traj_tests": {"weight": 1, "tests": {"test_goal": 1}}}
    })
    row = {"task_id": "t1", "run_trajectory_json": json.dumps(TRAJECTORY), "response": "final answer"}
    result = asyncio.run(_score_row(
        row, tasks_dir=tasks_dir, model_name="m", rubric_evaluator=None,
    ))
    assert result["weighted_reward"] == 1.0
    assert result["traj_tests_value"] == 1.0
    assert result["rubric_value"] == ""
    ledger = json.loads(result["weighted_ledger_json"])
    assert ledger == {"traj_tests": {"weight": 1.0, "value": 1.0}}


def test_score_row_falls_back_to_raw_conversation_history(tmp_path):
    tasks_dir = tmp_path / "tasks"
    _make_task_bundle(tasks_dir, "t1", {
        "components": {"traj_tests": {"weight": 1, "tests": {"test_goal": 1}}}
    })
    row = {"task_id": "t1", "run_trajectory_json": "", "raw_conversation_history": json.dumps(TRAJECTORY), "response": "final answer"}
    result = asyncio.run(_score_row(
        row, tasks_dir=tasks_dir, model_name="m", rubric_evaluator=None,
    ))
    assert result["weighted_reward"] == 1.0


def test_score_row_rubric_weight_without_evaluator_skips_channel_b(tmp_path):
    tasks_dir = tmp_path / "tasks"
    _make_task_bundle(tasks_dir, "t1", {
        "components": {
            "traj_tests": {"weight": 1, "tests": {"test_goal": 1}},
            "rubric": {"weight": 1},
        }
    })
    row = {"task_id": "t1", "run_trajectory_json": json.dumps(TRAJECTORY), "response": "final answer"}
    result = asyncio.run(_score_row(
        row, tasks_dir=tasks_dir, model_name="m", rubric_evaluator=None,
    ))
    # rubric_evaluator=None -> Channel B skipped, only traj_tests counts
    assert result["weighted_reward"] == 1.0
    assert result["rubric_value"] == ""


def test_response_column_prefers_model_specific():
    row = {"gpt-4o_response": "specific", "response": "generic"}
    assert sw._response_column(row, "gpt-4o") == "specific"
    assert sw._response_column({"response": "generic"}, "gpt-4o") == "generic"
