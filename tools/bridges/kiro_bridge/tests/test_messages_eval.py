"""The Anthropic Messages API emulation and the rubric evaluator, offline.

A fake bridge whose `complete()` returns a canned text stands in for a real
`kiro-cli` run, so the prompt rendering, tool-call parsing and verdict scoring
are all exercised without the CLI or the network.
"""

from __future__ import annotations

import json

import pytest

from kiro_bridge import KiroBridgeError, evaluate, parse_verdicts, score_verdicts
from kiro_bridge.evaluate import _criterion, complete_with_reasoning
from kiro_bridge.messages_api import (
    ESTIMATED_USAGE_SOURCE,
    answer,
    estimate_request_tokens,
    estimate_text_tokens,
    kiro_model,
    render_prompt,
    sse_events,
)


class FakeBridge:
    def __init__(self, reply: str, thinking: str = ""):
        self._reply = reply
        self._thinking = thinking
        self.prompts: list[str] = []
        self.efforts: list[str | None] = []

        class _C:
            effective_judge_model = "claude-sonnet-5"

        self.config = _C()

    def complete(self, system, user, *, model=None, timeout_sec=None, effort=None):
        self.prompts.append(user)
        self.efforts.append(effort)
        return self._reply, {"input_tokens": 3, "output_tokens": 4}

    def complete_with_reasoning(self, system, user, *, model=None, timeout_sec=None, effort=None):
        self.prompts.append(user)
        self.efforts.append(effort)
        return self._reply, {"input_tokens": 3, "output_tokens": 4}, self._thinking


class SequenceBridge(FakeBridge):
    """A fake Kiro model that returns one canned reply per repair attempt."""

    def __init__(self, replies: list[str], thinkings: list[str] | None = None):
        super().__init__("")
        self._replies = replies
        self._thinkings = thinkings or [""] * len(replies)
        self.calls = 0

    def complete_with_reasoning(self, system, user, *, model=None, timeout_sec=None, effort=None):
        self.prompts.append(user)
        self.efforts.append(effort)
        index = self.calls
        self.calls += 1
        return (
            self._replies[index],
            {"input_tokens": 3, "output_tokens": 4},
            self._thinkings[index],
        )


def _openhands_body(*, reasoning: bool = False) -> dict:
    body = {
        "model": "claude-opus-5",
        "messages": [{"role": "user", "content": "complete the task"}],
        "tools": [
            {"name": "shell", "input_schema": {"type": "object"}},
            {"name": "finish", "input_schema": {
                "type": "object",
                "properties": {"message": {"type": "string"}},
                "required": ["message"],
            }},
        ],
        "tool_choice": {"type": "auto"},
    }
    if reasoning:
        body["thinking"] = {"type": "enabled", "budget_tokens": 8000}
    return body


def test_kiro_model_strips_prefixes():
    assert kiro_model("kiro/claude-sonnet-5") == "claude-sonnet-5"
    assert kiro_model("anthropic/claude-opus-5") == "claude-opus-5"
    assert kiro_model("claude-opus-5", override="kiro/gpt-5.6-sol") == "gpt-5.6-sol"


def test_plain_text_reply_becomes_text_block():
    bridge = FakeBridge("BRIDGE_OK")
    body = {"model": "claude-sonnet-5", "messages": [{"role": "user", "content": "ping"}]}
    reply = answer(bridge, body, model="claude-sonnet-5")
    assert reply.stop_reason == "end_turn"
    assert reply.content == [{"type": "text", "text": "BRIDGE_OK"}]
    assert reply.usage["input_tokens"] == 3 and reply.usage["output_tokens"] == 4


def test_tool_call_block_is_parsed():
    bridge = FakeBridge('Working.\n<tool_call>\n{"name": "shell", "input": {"command": "ls"}}\n</tool_call>')
    body = {
        "model": "claude-sonnet-5",
        "messages": [{"role": "user", "content": "run ls"}],
        "tools": [{"name": "shell", "input_schema": {"type": "object"}}],
    }
    reply = answer(bridge, body, model="claude-sonnet-5")
    assert reply.stop_reason == "tool_use"
    tool_uses = [b for b in reply.content if b["type"] == "tool_use"]
    assert len(tool_uses) == 1
    assert tool_uses[0]["name"] == "shell" and tool_uses[0]["input"] == {"command": "ls"}


def test_reasoning_request_returns_anthropic_thinking_block_and_effort():
    bridge = FakeBridge("FINAL", "Compared the evidence and selected the supported answer.")
    body = {
        "model": "gpt-5.6-sol",
        "thinking": {"type": "enabled", "budget_tokens": 8000},
        "messages": [{"role": "user", "content": "decide"}],
    }
    reply = answer(bridge, body, model="gpt-5.6-sol")
    assert reply.content[0] == {
        "type": "thinking",
        "thinking": "Compared the evidence and selected the supported answer.",
        "signature": "kiro_bridge_reasoning_summary_v1",
    }
    assert reply.content[1] == {"type": "text", "text": "FINAL"}
    assert bridge.efforts == ["high"]
    assert "<reasoning_summary>" in bridge.prompts[0]


def test_thinking_block_streams_as_thinking_and_signature_deltas():
    bridge = FakeBridge("FINAL", "Short reasoning summary.")
    body = {
        "model": "gpt-5.6-sol",
        "reasoning_effort": "medium",
        "messages": [{"role": "user", "content": "decide"}],
    }
    reply = answer(bridge, body, model="gpt-5.6-sol")
    payloads = []
    for chunk in sse_events(reply):
        data_line = next(line for line in chunk.decode().splitlines() if line.startswith("data: "))
        payloads.append(json.loads(data_line.removeprefix("data: ")))
    delta_types = [p.get("delta", {}).get("type") for p in payloads]
    assert "thinking_delta" in delta_types
    assert "signature_delta" in delta_types
    assert bridge.efforts == ["medium"]


def test_disabled_reasoning_does_not_request_or_return_thinking():
    bridge = FakeBridge("FINAL", "should not be requested")
    body = {
        "model": "gpt-5.6-sol",
        "thinking": {"type": "disabled"},
        "messages": [{"role": "user", "content": "decide"}],
    }
    reply = answer(bridge, body, model="gpt-5.6-sol")
    assert bridge.efforts == ["none"]
    assert "<reasoning_summary>" not in bridge.prompts[0]
    assert [block["type"] for block in reply.content] == ["text"]


def test_forced_structured_tool_still_requests_reasoning_summary():
    body = {
        "model": "gpt-5.6-sol",
        "thinking": {"type": "enabled", "budget_tokens": 8000},
        "messages": [{"role": "user", "content": "choose"}],
        "tools": [{"name": "act", "input_schema": {
            "type": "object", "properties": {"path": {"type": "string"}},
            "required": ["path"],
        }}],
        "tool_choice": {"type": "tool", "name": "act"},
    }
    prompt = render_prompt(body)
    assert "Always include it when requested" in prompt
    assert "After the required <reasoning_summary> block" in prompt


def test_render_prompt_includes_system_and_tools():
    body = {
        "system": "be terse",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"name": "shell", "description": "run", "input_schema": {"type": "object"}}],
    }
    prompt = render_prompt(body)
    assert "be terse" in prompt and "shell" in prompt and "<tool_call>" in prompt


def test_parse_verdicts_matches_by_id_and_abstains_on_missing():
    criteria = [_criterion({"id": "A", "criterion": "x"}, 1),
                _criterion({"id": "B", "criterion": "y"}, 2)]
    text = '{"verdicts": [{"id": "A", "satisfied": true, "rationale": "ok"}]}'
    verdicts = parse_verdicts(text, criteria)
    by_id = {v.id: v for v in verdicts}
    assert by_id["A"].satisfied is True
    assert by_id["B"].satisfied is None  # missing -> abstains


def test_score_verdicts_positive_and_negative():
    criteria = [
        _criterion({"id": "A", "criterion": "x", "weight": 1}, 1),
        _criterion({"id": "B", "criterion": "y", "weight": 1}, 2),
        _criterion({"id": "N", "criterion": "bad", "is_positive": False, "weight": 1}, 3),
    ]
    text = ('{"verdicts": [{"id": "A", "satisfied": true}, {"id": "B", "satisfied": false}, '
            '{"id": "N", "satisfied": true}]}')
    verdicts = parse_verdicts(text, criteria)
    # earned 1 (A) of positive total 2, minus 1 triggered negative = 0.0
    assert score_verdicts(verdicts) == 0.0


def test_evaluate_end_to_end_with_fake_bridge(monkeypatch):
    # evaluate() runs a tool-less agent for the judge call; fake that run.
    from kiro_bridge import AgentResult, Usage

    def fake_run_agent(bridge, prompt, **kwargs):
        return AgentResult(
            status="finished",
            text='{"verdicts": [{"id": "C1", "satisfied": true, "rationale": "good"}]}',
            agent_id="s", run_id="s", model="claude-sonnet-5", runtime="local",
            duration_sec=0.1, usage=Usage(credits=0.01),
        )

    monkeypatch.setattr("kiro_bridge.agent.run_agent", fake_run_agent)
    bridge = FakeBridge("")
    result = evaluate(bridge, "do x", [{"criterion": "did x"}], {"transcript": "did x"})
    assert result.ok is True and result.score == 1.0
    assert result.verdicts[0].satisfied is True


def test_complete_with_reasoning_prefers_genuine_acp_thought(monkeypatch):
    from kiro_bridge import AgentResult, Usage

    def fake_run_agent(bridge, prompt, **kwargs):
        return AgentResult(
            status="finished",
            text=("<reasoning_summary>Fallback summary.</reasoning_summary>\n"
                  "Visible answer"),
            agent_id="s", run_id="s", model="gpt-5.6-sol", runtime="local",
            duration_sec=0.1, usage=Usage(),
            events=[{"type": "thinking", "text": "Genuine ACP thought summary."}],
        )

    monkeypatch.setattr("kiro_bridge.agent.run_agent", fake_run_agent)
    bridge = FakeBridge("")
    text, _usage, thinking = complete_with_reasoning(
        bridge, "", "prompt", model="gpt-5.6-sol", effort="high"
    )
    assert text == "Visible answer"
    assert thinking == "Genuine ACP thought summary."


def test_finish_tool_repairs_incomplete_prose_into_work_tool_call():
    bridge = SequenceBridge([
        "The output was clipped, so I need the full bodies.",
        '<tool_call>\n{"name": "shell", "input": {"command": "cat ev/full.json"}}\n</tool_call>',
    ])
    reply = answer(bridge, _openhands_body(), model="claude-opus-5")

    assert reply.stop_reason == "tool_use"
    call = next(block for block in reply.content if block["type"] == "tool_use")
    assert call["name"] == "shell"
    assert call["input"] == {"command": "cat ev/full.json"}
    assert bridge.calls == 2
    assert "every reply MUST call" in bridge.prompts[0]
    assert "Your previous reply could not be used" in bridge.prompts[1]
    assert "call a work tool to continue" in bridge.prompts[1]
    assert reply.usage["input_tokens"] == 6
    assert reply.usage["output_tokens"] == 8


def test_finish_tool_repairs_incomplete_prose_into_finish_call():
    bridge = SequenceBridge([
        "Everything is done.",
        '<tool_call>\n{"name": "finish", "input": {"message": "Completed."}}\n</tool_call>',
    ])
    reply = answer(bridge, _openhands_body(), model="claude-opus-5")

    assert reply.stop_reason == "tool_use"
    call = next(block for block in reply.content if block["type"] == "tool_use")
    assert call["name"] == "finish"
    assert call["input"] == {"message": "Completed."}
    assert bridge.calls == 2


def test_auto_tool_request_without_finish_still_allows_ordinary_prose():
    bridge = FakeBridge("No tool is needed for this answer.")
    body = {
        "model": "claude-opus-5",
        "messages": [{"role": "user", "content": "answer normally"}],
        "tools": [{"name": "shell", "input_schema": {"type": "object"}}],
        "tool_choice": {"type": "auto"},
    }
    reply = answer(bridge, body, model="claude-opus-5")

    assert reply.stop_reason == "end_turn"
    assert reply.content == [{"type": "text", "text": "No tool is needed for this answer."}]
    assert len(bridge.prompts) == 1


def test_finish_tool_repair_preserves_reasoning_from_successful_attempt():
    bridge = SequenceBridge(
        [
            "I need to inspect another file.",
            '<tool_call>\n{"name": "shell", "input": {"command": "cat next.json"}}\n</tool_call>',
        ],
        ["Premature attempt reasoning.", "The next file contains the required evidence."],
    )
    reply = answer(bridge, _openhands_body(reasoning=True), model="claude-opus-5")

    assert reply.stop_reason == "tool_use"
    assert reply.content[0] == {
        "type": "thinking",
        "thinking": "The next file contains the required evidence.",
        "signature": "kiro_bridge_reasoning_summary_v1",
    }
    assert sum(block["type"] == "thinking" for block in reply.content) == 1
    assert bridge.efforts == ["high", "high"]


@pytest.mark.parametrize(("repair_attempts", "expected_calls"), [(0, 1), (1, 2)])
def test_finish_tool_repair_exhaustion_raises_instead_of_ending_turn(
    repair_attempts, expected_calls
):
    bridge = SequenceBridge(["I still need more evidence."] * expected_calls)

    with pytest.raises(KiroBridgeError, match="model reply remained invalid") as exc:
        answer(
            bridge,
            _openhands_body(),
            model="claude-opus-5",
            repair_attempts=repair_attempts,
        )

    assert bridge.calls == expected_calls
    assert f"after {expected_calls} attempt" in str(exc.value)


def test_zero_kiro_usage_gets_labelled_estimates_and_preserves_credits():
    class CreditBridge(FakeBridge):
        def complete_with_reasoning(self, system, user, **kwargs):
            self.prompts.append(user)
            self.efforts.append(kwargs.get("effort"))
            return (
                "VISIBLE",
                {"credits": 0.1257, "request_count": 1},
                "Reasoning summary.",
            )

    bridge = CreditBridge("")
    body = {
        "model": "claude-opus-5",
        "reasoning_effort": "high",
        "messages": [{"role": "user", "content": "answer"}],
    }
    reply = answer(bridge, body, model="claude-opus-5")

    assert reply.usage["input_tokens"] == estimate_request_tokens(body)
    assert reply.usage["reasoning_tokens"] == estimate_text_tokens("Reasoning summary.")
    assert reply.usage["output_tokens"] == (
        estimate_text_tokens("VISIBLE") + estimate_text_tokens("Reasoning summary.")
    )
    assert reply.usage["cache_read_input_tokens"] == 0
    assert reply.usage["cache_creation_input_tokens"] == 0
    assert reply.usage["kiro_credits"] == 0.1257
    assert reply.usage["request_count"] == 1
    assert reply.usage["usage_estimated"] is True
    assert reply.usage["usage_source"] == ESTIMATED_USAGE_SOURCE
    assert set(reply.usage["estimated_fields"]) == {
        "input_tokens", "output_tokens", "reasoning_tokens"
    }


def test_positive_upstream_counts_are_never_replaced_by_estimates():
    bridge = FakeBridge("VISIBLE", "Reasoning summary.")
    body = {
        "model": "claude-opus-5",
        "reasoning_effort": "high",
        "messages": [{"role": "user", "content": "answer"}],
    }
    reply = answer(bridge, body, model="claude-opus-5")

    assert reply.usage["input_tokens"] == 3
    assert reply.usage["output_tokens"] == 4
    assert reply.usage["reasoning_tokens"] > 0
    assert reply.usage["estimated_fields"] == ["reasoning_tokens"]
