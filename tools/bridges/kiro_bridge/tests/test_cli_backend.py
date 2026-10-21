"""The Kiro ACP JSON-Lines stream → the bridge's normalised events.

The lines here are the shapes `kiro-cli chat --output-format stream-json`
actually emits (captured from a live run): runStarted, metadata with
meteringUsage, sessionUpdate carrying agent_message_chunk / tool_call /
tool_call_update, and runFinished.
"""

from __future__ import annotations

import json

from kiro_bridge.cli_backend import _StreamState


def _feed(lines: list[dict]) -> _StreamState:
    state = _StreamState()
    for line in lines:
        state.feed(json.dumps(line) + "\n")
    return state


def test_text_chunks_coalesce_into_one_assistant_event():
    state = _feed([
        {"type": "runStarted", "data": {"engine": "v2"}},
        {"type": "sessionUpdate", "data": {"sessionId": "s1", "update": {
            "sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": "HELLO"}}}},
        {"type": "sessionUpdate", "data": {"sessionId": "s1", "update": {
            "sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": "_KIRO"}}}},
        {"type": "runFinished", "data": {"sessionId": "s1", "status": "success",
                                         "stopReason": "end_turn", "finalText": "HELLO_KIRO"}},
    ])
    assert state.started is True
    assert state.session_id == "s1"
    assistants = [e for e in state.events if e["type"] == "assistant"]
    assert len(assistants) == 1 and assistants[0]["text"] == "HELLO_KIRO"
    assert state.result["status"] == "success"


def test_metering_usage_accumulates_credits():
    state = _feed([
        {"type": "runStarted", "data": {}},
        {"type": "metadata", "data": {"sessionId": "s", "meteringUsage": [
            {"value": 0.01, "unit": "credit"}, {"value": 0.02, "unit": "credit"}]}},
        {"type": "runFinished", "data": {"sessionId": "s", "status": "success", "finalText": ""}},
    ])
    assert round(state.usage().credits, 6) == 0.03


def test_tool_call_start_then_completed_update():
    state = _feed([
        {"type": "runStarted", "data": {}},
        {"type": "sessionUpdate", "data": {"sessionId": "s", "update": {
            "sessionUpdate": "tool_call", "toolCallId": "t1", "kind": "execute",
            "title": "Running: echo hi",
            "rawInput": {"command": "echo hi", "__tool_use_purpose": "noise"},
            "_meta": {"kiro": {"toolName": "shell"}}}}},
        {"type": "sessionUpdate", "data": {"sessionId": "s", "update": {
            "sessionUpdate": "tool_call_update", "toolCallId": "t1", "kind": "execute",
            "status": "completed",
            "rawOutput": {"items": [{"Json": {"exit_status": "exit status: 0", "stdout": ""}}]}}}},
        {"type": "runFinished", "data": {"sessionId": "s", "status": "success", "finalText": "ok"}},
    ])
    tool_events = [e for e in state.events if e["type"] == "tool_call"]
    # One running, one completed — both named `shell`, noise arg stripped.
    assert [e["status"] for e in tool_events] == ["running", "completed"]
    assert all(e["name"] == "shell" for e in tool_events)
    assert tool_events[0]["args"] == {"command": "echo hi"}
    assert "exit status: 0" in tool_events[1]["result"]


def test_failed_tool_update_is_error():
    state = _feed([
        {"type": "runStarted", "data": {}},
        {"type": "sessionUpdate", "data": {"sessionId": "s", "update": {
            "sessionUpdate": "tool_call_update", "toolCallId": "t2", "kind": "read",
            "status": "failed",
            "content": [{"type": "content", "content": {"type": "text", "text": "does not exist"}}],
            "_meta": {"kiro": {"toolName": "read"}}}}},
        {"type": "runFinished", "data": {"sessionId": "s", "status": "success", "finalText": ""}},
    ])
    tool_events = [e for e in state.events if e["type"] == "tool_call"]
    assert tool_events[-1]["status"] == "error"
    assert "does not exist" in str(tool_events[-1]["result"])


def test_run_error_envelope_records_error_result():
    state = _feed([
        {"type": "runStarted", "data": {"engine": "v2"}},
        {"type": "runError", "data": {"sessionId": "s", "stage": "init",
                                      "message": "Mode 'x' not found"}},
    ])
    assert state.started is True
    assert state.result["status"] == "error"
    assert "not found" in state.result["error"]


def test_thinking_chunks_become_thinking_event():
    state = _feed([
        {"type": "runStarted", "data": {}},
        {"type": "sessionUpdate", "data": {"sessionId": "s", "update": {
            "sessionUpdate": "agent_thought_chunk", "content": {"type": "text", "text": "hmm "}}}},
        {"type": "sessionUpdate", "data": {"sessionId": "s", "update": {
            "sessionUpdate": "agent_thought_chunk", "content": {"type": "text", "text": "ok"}}}},
        {"type": "sessionUpdate", "data": {"sessionId": "s", "update": {
            "sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": "done"}}}},
        {"type": "runFinished", "data": {"sessionId": "s", "status": "success", "finalText": "done"}},
    ])
    kinds = [e["type"] for e in state.events]
    assert "thinking" in kinds and "assistant" in kinds
    thinking = next(e for e in state.events if e["type"] == "thinking")
    assert thinking["text"] == "hmm ok"


def test_generated_agent_profile_is_v2_v3_universal_and_confined(tmp_path):
    """V3 rejects legacy allowedTools unless capability permissions also exist."""
    from kiro_bridge.cli_backend import AGENT_NAME, _write_agent_config

    name = _write_agent_config(
        tmp_path,
        None,
        {"sandbox": {"command": "printf", "args": ["ready"]}},
        confined=True,
    )
    assert name == AGENT_NAME
    profile = json.loads(
        (tmp_path / ".kiro" / "agents" / f"{AGENT_NAME}.json").read_text()
    )
    assert profile["tools"] == ["@sandbox"]
    assert profile["allowedTools"] == ["@sandbox"]  # CLI v2 compatibility
    assert profile["permissions"] == {
        "rules": [
            {"capability": "mcp", "match": ["sandbox/*"], "effect": "allow"}
        ]
    }
    assert {rule["capability"] for rule in profile["permissions"]["rules"]} == {"mcp"}
    assert profile["includeMcpJson"] is False
