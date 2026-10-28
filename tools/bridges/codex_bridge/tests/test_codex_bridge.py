"""Offline checks for the codex judge bridge: the `codex exec --json` event
parsing and argv in backend.py, and the Messages API emulation in
messages_api.py. No codex CLI, no network: subprocess.run is monkeypatched
and a fake bridge returns canned text.

The event shapes here are copied from a real `codex exec --json` run of
codex-cli 0.146.0 against gpt-5.6-sol (2026-10-27).
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from codex_bridge import CodexBridgeError, CodexConfig
from codex_bridge.backend import CodexBackend
from codex_bridge.messages_api import _requested_effort, _text_of, answer, render_prompt

EXEC_ARGV = ["exec", "--json", "--ephemeral", "--skip-git-repo-check", "--ignore-user-config",
             "--ignore-rules", "-s", "read-only", "-m", "gpt-5.6-sol"]


def _events(*evs: dict) -> str:
    return "\n".join(json.dumps(e) for e in evs) + "\n"


AGENT_OK = {"type": "item.completed", "item": {"type": "agent_message", "text": "hello"}}
TURN_OK = {"type": "turn.completed", "usage": {"input_tokens": 13879, "cached_input_tokens": 13184,
                                                "cache_write_input_tokens": 0, "output_tokens": 15,
                                                "reasoning_output_tokens": 0}}
TURN_FAILED = {"type": "turn.failed", "error": {"message": "{\"error\":{\"message\":\"Unsupported value: "
                                                            "'minimal' is not supported with the 'gpt-5.6-sol' model.\"}}"}}


class FakeRun:
    """Stands in for subprocess.run: records argv/kwargs, returns canned output."""

    def __init__(self, stdout: str, returncode: int = 0, stderr: str = ""):
        self.stdout, self.returncode, self.stderr = stdout, returncode, stderr
        self.calls: list[tuple[list[str], dict]] = []

    def __call__(self, cmd, **kw):
        self.calls.append((list(cmd), kw))
        return self


@pytest.fixture
def backend() -> CodexBackend:
    # Any existing file satisfies require_cli(); the subprocess is faked anyway.
    return CodexBackend(CodexConfig(cli_path=sys.executable))


def test_parses_agent_message_usage_and_argv(backend, monkeypatch):
    fake = FakeRun(_events({"type": "thread.started"}, {"type": "turn.started"}, AGENT_OK, TURN_OK))
    monkeypatch.setattr(subprocess, "run", fake)
    text, usage, thinking = backend.complete_with_reasoning("", "hi", model="gpt-5.6-sol")
    assert text == "hello" and thinking == ""
    # OpenAI counts cached tokens inside input_tokens; the Anthropic wire does not.
    assert usage["input_tokens"] == 13879 - 13184
    assert usage["cache_read_tokens"] == 13184 and usage["output_tokens"] == 15
    cmd, kw = fake.calls[0]
    assert cmd[1:] == EXEC_ARGV + ["-"]          # no effort flag when none requested
    assert kw["input"] == "hi" and kw["cwd"]      # prompt on stdin, throwaway cwd


def test_turn_failed_raises_with_codex_message(backend, monkeypatch):
    monkeypatch.setattr(subprocess, "run", FakeRun(_events({"type": "turn.started"}, TURN_FAILED), returncode=1))
    with pytest.raises(CodexBridgeError, match="Unsupported value"):
        backend.complete_with_reasoning("", "hi", model="gpt-5.6-sol")


def test_no_agent_message_raises(backend, monkeypatch):
    monkeypatch.setattr(subprocess, "run", FakeRun(_events({"type": "turn.started"}, TURN_OK)))
    with pytest.raises(CodexBridgeError, match="no agent_message"):
        backend.complete_with_reasoning("", "hi", model="gpt-5.6-sol")


@pytest.mark.parametrize("effort,flag", [
    ("none", "none"),        # disabled thinking: codex takes `none` natively (gpt-5.6-sol 400s on minimal)
    ("low", "low"), ("xhigh", "xhigh"), ("max", "max"), ("minimal", "minimal"),
    ("bogus", None), ("", None),
])
def test_effort_flag(backend, monkeypatch, effort, flag):
    fake = FakeRun(_events(AGENT_OK, {"type": "turn.completed", "usage": {}}))
    monkeypatch.setattr(subprocess, "run", fake)
    backend.complete_with_reasoning("", "hi", model="m", effort=effort)
    cmd = fake.calls[0][0]
    flags = [a for a in cmd if a.startswith("model_reasoning_effort=")]
    assert flags == ([f"model_reasoning_effort={flag}"] if flag else [])


def test_config_effort_is_the_fallback_only(monkeypatch):
    b = CodexBackend(CodexConfig(cli_path=sys.executable, reasoning_effort="high"))
    fake = FakeRun(_events(AGENT_OK, {"type": "turn.completed", "usage": {}}))
    monkeypatch.setattr(subprocess, "run", fake)
    b.complete_with_reasoning("", "hi", model="m")
    b.complete_with_reasoning("", "hi", model="m", effort="low")
    assert "model_reasoning_effort=high" in fake.calls[0][0]
    assert "model_reasoning_effort=low" in fake.calls[1][0]


def test_config_rejects_unknown_effort():
    with pytest.raises(CodexBridgeError, match="CODEX_REASONING_EFFORT"):
        CodexConfig.from_env({"CODEX_REASONING_EFFORT": "ultra"})
    assert CodexConfig.from_env({"CODEX_REASONING_EFFORT": "none"}).reasoning_effort == "none"


def test_requested_effort_mapping():
    assert _requested_effort({"thinking": {"type": "disabled"}}) == "none"
    assert _requested_effort({"thinking": {"type": "enabled", "budget_tokens": 2048}}) == "medium"
    assert _requested_effort({"thinking": {"type": "adaptive"}}) == "high"
    assert _requested_effort({}) == ""


# ----------------------------------------------------------------- messages_api

class FakeBridge:
    def __init__(self, *replies: str):
        self.replies, self.prompts, self.efforts = list(replies), [], []

    def complete_with_reasoning(self, system, user, *, model, timeout_sec=None, effort=""):
        self.prompts.append(user)
        self.efforts.append(effort)
        return self.replies.pop(0), {"input_tokens": 3, "output_tokens": 4}, ""


def _body(**extra):
    return {"model": "claude-opus-5", "max_tokens": 200,
            "messages": [{"role": "user", "content": "grade this"}], **extra}


def test_json_schema_output_format_strips_fence():
    b = FakeBridge('```json\n{"score": 0.8, "rationale": "fine"}\n```')
    reply = answer(b, _body(output_format={"type": "json_schema", "schema": {"type": "object"}}),
                   model="gpt-5.6-sol")
    assert reply.stop_reason == "end_turn" and reply.model == "gpt-5.6-sol"
    assert json.loads(reply.content[0]["text"]) == {"score": 0.8, "rationale": "fine"}
    assert reply.usage["input_tokens"] == 3 and not reply.usage.get("usage_estimated")


def test_forced_single_tool_becomes_tool_use():
    b = FakeBridge('{"verdict": "pass", "confidence": 0.9}')
    body = _body(tools=[{"name": "verdict", "input_schema": {"type": "object", "required": ["verdict"]}}],
                 tool_choice={"type": "tool", "name": "verdict"})
    reply = answer(b, body, model="gpt-5.6-sol")
    assert reply.stop_reason == "tool_use"
    assert reply.content[0]["type"] == "tool_use" and reply.content[0]["name"] == "verdict"
    assert reply.content[0]["input"] == {"verdict": "pass", "confidence": 0.9}


def test_invalid_reply_is_repaired_once_then_errors():
    b = FakeBridge("not json", "still not json")
    with pytest.raises(CodexBridgeError, match="remained invalid"):
        answer(b, _body(output_format={"type": "json_schema", "schema": {"type": "object"}}), model="m")
    assert len(b.prompts) == 2 and "<framework_note>" in b.prompts[1]


def test_disabled_thinking_reaches_backend_as_none():
    b = FakeBridge("ok")
    reply = answer(b, _body(thinking={"type": "disabled"}), model="m")
    assert b.efforts == ["none"]
    assert reply.content == [{"type": "text", "text": "ok"}]   # no thinking block exposed


def test_missing_usage_is_estimated_and_flagged():
    class NoUsage(FakeBridge):
        def complete_with_reasoning(self, system, user, **kw):
            return "ok", {}, ""
    reply = answer(NoUsage(), _body(), model="m")
    assert reply.usage["usage_estimated"] is True
    assert reply.usage["input_tokens"] > 0 and reply.usage["output_tokens"] > 0


def test_text_of_strips_nul_and_omits_images():
    assert _text_of("a\x00b") == "ab"
    assert _text_of([{"type": "text", "text": "x\x00"}, {"type": "image"}]) == "x\n[image omitted]"


def test_render_prompt_carries_system_tools_and_turns():
    p = render_prompt({"system": "SYS", "messages": [{"role": "user", "content": "hi"}],
                       "tools": [{"name": "t", "input_schema": {"type": "object"}}]})
    assert "<system_prompt>\nSYS\n</system_prompt>" in p
    assert '"name": "t"' in p and "<tool_call>" in p and '<turn role="user">\nhi\n</turn>' in p
