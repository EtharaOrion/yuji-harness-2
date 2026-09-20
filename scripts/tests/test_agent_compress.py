"""The agent-path compressor: what it is allowed to change, and what it is not.

Written against the failure that produced it. `headroom proxy` rewrote the whole
request and deleted Claude Code's deferred tool loading, so the next request
referenced a tool it no longer declared and the API answered

    400 Tool reference 'tool_search_tool_regex' not found in available tools

eleven turns into a paid run, twice. compress_proxy.py may shorten text and
nothing else; every test here is a way that promise could be broken quietly.

The compressor is faked throughout. What is under test is the guard around it,
not headroom -- and a test that needed the real library would pass or fail for
reasons that have nothing to do with this code.
"""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / "tools" / "headroom" / "compress_proxy.py"


@pytest.fixture
def proxy(monkeypatch):
    spec = importlib.util.spec_from_file_location("compress_proxy", SOURCE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "_disabled", False)
    monkeypatch.setattr(mod, "_failures", 0)
    return mod


TOOLS = [
    {"type": "tool_search_tool_regex_20251119", "name": "tool_search_tool_regex"},
    {"name": "mcp__LightGmail__search", "description": "d" * 400,
     "input_schema": {"type": "object", "properties": {}}},
]

BODY = {
    "model": "claude-opus-5",
    "system": [{"type": "text", "text": "S" * 500, "cache_control": {"type": "ephemeral"}}],
    "tools": TOOLS,
    "messages": [
        {"role": "user", "content": [{"type": "text", "text": "evidence " * 2000,
                                      "cache_control": {"type": "ephemeral"}}]},
        {"role": "assistant", "content": [
            {"type": "server_tool_use", "id": "srvtoolu_1", "name": "tool_search_tool_regex",
             "input": {"pattern": "mail"}},
            {"type": "tool_search_tool_result", "tool_use_id": "srvtoolu_1", "content": []}]},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "toolu_1", "name": "mcp__LightGmail__search",
             "input": {"q": "x"}}]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "toolu_1", "content": "rows " * 3000}]},
    ],
}


def _raw(body=None) -> bytes:
    return json.dumps(body or BODY).encode()


def _shrink_text(model, messages):
    """A well-behaved compressor: shortens text blocks, touches nothing else."""
    out = copy.deepcopy(messages)
    for message in out:
        content = message.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    block["text"] = block["text"][:20]
    return out


# --- what must never change --------------------------------------------------

def test_tools_are_never_touched(proxy, monkeypatch):
    """The whole failure in one assertion."""
    monkeypatch.setattr(proxy, "compress_messages", _shrink_text)
    out = json.loads(proxy.compress_request(_raw()))
    assert out["tools"] == TOOLS


def test_the_system_prompt_is_never_touched(proxy, monkeypatch):
    monkeypatch.setattr(proxy, "compress_messages", _shrink_text)
    out = json.loads(proxy.compress_request(_raw()))
    assert out["system"] == BODY["system"]


def test_a_dropped_tool_block_is_refused(proxy, monkeypatch):
    """What headroom actually did: delete blocks it did not recognise."""
    def _drop_server_blocks(model, messages):
        out = copy.deepcopy(messages)
        for message in out:
            content = message.get("content")
            if isinstance(content, list):
                message["content"] = [b for b in content
                                      if b.get("type") not in ("server_tool_use",
                                                               "tool_search_tool_result")]
        return out
    monkeypatch.setattr(proxy, "compress_messages", _drop_server_blocks)
    assert proxy.compress_request(_raw()) == _raw(), (
        "a compressor deleted a block type and the request still went upstream"
    )


def test_a_renamed_tool_result_is_refused(proxy, monkeypatch):
    """A tool_result whose id no longer matches its tool_use is a 400, not a saving."""
    def _rename(model, messages):
        out = copy.deepcopy(messages)
        for message in out:
            for block in message.get("content") or []:
                if isinstance(block, dict) and block.get("tool_use_id"):
                    block["tool_use_id"] = "toolu_other"
        return out
    monkeypatch.setattr(proxy, "compress_messages", _rename)
    assert proxy.compress_request(_raw()) == _raw()


def test_a_lost_message_is_refused(proxy, monkeypatch):
    monkeypatch.setattr(proxy, "compress_messages",
                        lambda model, messages: list(messages)[:-1])
    assert proxy.compress_request(_raw()) == _raw()


# --- fail-open ---------------------------------------------------------------

def test_a_raising_compressor_sends_the_original(proxy, monkeypatch):
    def _boom(model, messages):
        raise RuntimeError("router exploded")
    monkeypatch.setattr(proxy, "compress_messages", _boom)
    assert proxy.compress_request(_raw()) == _raw()


def test_a_body_that_is_not_json_is_relayed(proxy, monkeypatch):
    monkeypatch.setattr(proxy, "compress_messages", _shrink_text)
    assert proxy.compress_request(b"not json at all") == b"not json at all"


def test_a_body_without_messages_is_relayed(proxy, monkeypatch):
    monkeypatch.setattr(proxy, "compress_messages", _shrink_text)
    raw = json.dumps({"model": "m", "tools": TOOLS}).encode()
    assert proxy.compress_request(raw) == raw


def test_a_bigger_result_is_discarded(proxy, monkeypatch):
    def _inflate(model, messages):
        out = copy.deepcopy(messages)
        for message in out:
            for block in message.get("content") or []:
                if isinstance(block, dict) and block.get("type") == "text":
                    block["text"] = block["text"] * 2
        return out
    monkeypatch.setattr(proxy, "compress_messages", _inflate)
    assert proxy.compress_request(_raw()) == _raw()


def test_repeated_failures_stop_the_compressor(proxy, monkeypatch):
    """A compressor that keeps failing is not worth a per-request gamble."""
    def _boom(model, messages):
        raise RuntimeError("nope")
    monkeypatch.setattr(proxy, "compress_messages", _boom)
    monkeypatch.setattr(proxy, "DISABLE_AFTER", 2)
    for _ in range(2):
        proxy.compress_request(_raw())
    assert proxy._disabled, "compression stayed on after repeated failures"
    # And once off it does not even call the compressor again.
    def _tripwire(model, messages):
        raise AssertionError("compressor called after being switched off")
    monkeypatch.setattr(proxy, "compress_messages", _tripwire)
    assert proxy.compress_request(_raw()) == _raw()


def test_a_missing_library_is_a_plain_relay(proxy, monkeypatch):
    monkeypatch.setattr(proxy, "compress_messages", None)
    assert proxy.compress_request(_raw()) == _raw()


# --- caching -----------------------------------------------------------------

def test_cache_breakpoints_survive(proxy, monkeypatch):
    """Losing the marker forfeits an ~88% discount, silently.

    Compression strips cache_control (grader_compress stands down on cached
    traffic, which is right for the grader and wrong here), so the proxy carries
    the markers back afterwards.
    """
    monkeypatch.setattr(proxy, "compress_messages", _shrink_text)
    out = json.loads(proxy.compress_request(_raw()))
    first = out["messages"][0]["content"]
    assert any(b.get("cache_control") for b in first), first


def test_a_flattened_message_still_carries_its_breakpoint(proxy, monkeypatch):
    """Headroom returns bare strings; a marker needs a block to live on."""
    def _flatten(model, messages):
        out = copy.deepcopy(messages)
        out[0]["content"] = "short"
        return out
    monkeypatch.setattr(proxy, "compress_messages", _flatten)
    out = json.loads(proxy.compress_request(_raw()))
    first = out["messages"][0]["content"]
    assert isinstance(first, list) and first[0].get("cache_control"), first


def test_a_real_saving_is_kept(proxy, monkeypatch):
    monkeypatch.setattr(proxy, "compress_messages", _shrink_text)
    out = proxy.compress_request(_raw())
    assert len(out) < len(_raw())
    body = json.loads(out)
    assert len(body["messages"]) == len(BODY["messages"])
    assert body["messages"][3]["content"][0]["tool_use_id"] == "toolu_1"
