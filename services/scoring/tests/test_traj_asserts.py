"""Unit tests for services/scoring/traj_asserts.py."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import traj_asserts as ta  # noqa: E402

RUN_TRAJECTORY = {
    "schema_version": "mcp-atlas-trajectory-v1",
    "session_id": "s1",
    "task_id": "t1",
    "agent": {"name": "litellm", "model_name": "test-model"},
    "final_metrics": {
        "total_prompt_tokens": 10,
        "total_completion_tokens": 10,
        "total_cost_usd": None,
        "total_steps": 2,
    },
    "steps": [
        {
            "step_id": 1,
            "timestamp": "2026-01-01T00:00:00Z",
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "arxiv_search_papers",
                            "arguments": json.dumps({"query": "locality"}),
                        },
                    }
                ],
            },
            "tool_results": [
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": [{"type": "text", "text": "found 3 papers"}],
                }
            ],
        },
        {
            "step_id": 2,
            "timestamp": "2026-01-01T00:00:05Z",
            "message": {
                "role": "assistant",
                "content": "Here is the answer.",
                "tool_calls": None,
            },
        },
    ],
}

FLAT_MESSAGES = [
    {"role": "system", "content": "you are an agent"},
    {"role": "user", "content": "find the paper"},
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "arxiv_search_papers", "arguments": '{"query": "locality"}'},
            }
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": [{"type": "text", "text": "Error: rate limited"}],
    },
    {"role": "assistant", "content": "Sorry, I couldn't find it.", "tool_calls": None},
]


def _stage(monkeypatch, tmp_path, data, name="traj.json"):
    p = tmp_path / name
    p.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setenv(ta._ENV_VAR, str(p))
    ta.reset_cache()
    return p


def test_calls_and_called_with_run_trajectory(monkeypatch, tmp_path):
    _stage(monkeypatch, tmp_path, RUN_TRAJECTORY)
    assert ta.called_with("arxiv_search_papers")
    assert ta.called_with("arxiv_search_papers", query="locality")
    assert not ta.called_with("arxiv_search_papers", query="something else")
    assert ta.call_count("arxiv_search_papers") == 1
    assert ta.call_count() == 1
    assert ta.distinct_tools_called() == {"arxiv_search_papers"}


def test_never_called_and_called_any(monkeypatch, tmp_path):
    _stage(monkeypatch, tmp_path, RUN_TRAJECTORY)
    assert ta.never_called("slack_channels_list")
    assert not ta.called_any("slack_channels_list")
    assert ta.called_any("arxiv_search_papers", "slack_channels_list")


def test_final_message(monkeypatch, tmp_path):
    _stage(monkeypatch, tmp_path, RUN_TRAJECTORY)
    assert ta.final() == "Here is the answer."


def test_predicate_matcher(monkeypatch, tmp_path):
    _stage(monkeypatch, tmp_path, RUN_TRAJECTORY)
    assert ta.called_with("arxiv_search_papers", query=lambda q: "local" in q)
    assert not ta.called_with("arxiv_search_papers", query=lambda q: "slack" in q)


def test_flat_message_shape(monkeypatch, tmp_path):
    _stage(monkeypatch, tmp_path, FLAT_MESSAGES)
    assert ta.called_with("arxiv_search_papers", query="locality")
    assert ta.tool_errored("arxiv_search_papers")
    assert ta.tool_errored()
    assert ta.final() == "Sorry, I couldn't find it."


def test_cache_reset_picks_up_new_file(monkeypatch, tmp_path):
    _stage(monkeypatch, tmp_path, RUN_TRAJECTORY, name="a.json")
    assert ta.distinct_tools_called() == {"arxiv_search_papers"}
    _stage(monkeypatch, tmp_path, FLAT_MESSAGES, name="b.json")
    assert ta.tool_errored()


def test_missing_env_var_raises(monkeypatch):
    monkeypatch.delenv(ta._ENV_VAR, raising=False)
    ta.reset_cache()
    try:
        ta.calls()
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError when MCPATLAS_TRAJECTORY is unset")
