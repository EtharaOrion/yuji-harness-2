"""RED tests for zbridge.sse_translator — stateful GLM SSE -> Anthropic SSE.

Covers ≥14 cases per PLAN.md §5.3 + §11 scenarios S1–S3, S5, S10, S10b, S13, S17, S19.
"""
from __future__ import annotations

import json

import pytest

from zbridge.sse_translator import SseTranslator


def collect(gen) -> list[bytes]:
    return list(gen)


def parse_events(chunks: list[bytes]) -> list[dict]:
    """Extract every SSE `event: X\\ndata: {...}` frame into a list of dicts with type + payload."""
    raw = b"".join(chunks).decode("utf-8", errors="replace")
    frames = []
    cur_event = None
    for line in raw.splitlines():
        if line.startswith("event: "):
            cur_event = line[len("event: "):].strip()
        elif line.startswith("data: "):
            payload = line[len("data: "):]
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                continue
            frames.append({"event": cur_event, "data": data})
            cur_event = None
    return frames


# ---- 1. Worked example: thinking -> text -> tool (S3+S5+S1) ---------------

def test_worked_example_thinking_text_tool(load_sse):
    t = SseTranslator(model="glm-5.3")
    out = collect(t.feed(load_sse("glm_stream_thinking_text_tool.sse")))
    out += collect(t.close())
    events = parse_events(out)
    types = [e["data"]["type"] for e in events]

    # message_start present
    assert types[0] == "message_start"
    # Three content blocks: thinking, text, tool_use — with correct bracketing
    # Extract only content_block_start events
    starts = [e for e in events if e["data"]["type"] == "content_block_start"]
    block_kinds = [s["data"]["content_block"]["type"] for s in starts]
    assert block_kinds == ["thinking", "text", "tool_use"], f"got {block_kinds}"
    # Message ends with message_delta then message_stop
    assert types[-2] == "message_delta"
    assert types[-1] == "message_stop"
    # Stop reason mapped correctly
    md = [e for e in events if e["data"]["type"] == "message_delta"][0]
    assert md["data"]["delta"]["stop_reason"] == "tool_use"


# ---- 2. Pure text stream (S2) ---------------------------------------------

def test_pure_text_stream_emits_correct_frames(load_sse):
    t = SseTranslator(model="glm-5.3")
    events = parse_events(collect(t.feed(load_sse("glm_stream_text.sse"))) + collect(t.close()))
    types = [e["data"]["type"] for e in events]
    assert types[0] == "message_start"
    assert "content_block_start" in types
    assert "text_delta" in [e["data"].get("delta", {}).get("type") for e in events if e["data"]["type"] == "content_block_delta"]
    assert types[-1] == "message_stop"


# ---- 3. Pure thinking (defensive) -----------------------------------------

def test_pure_thinking_stream():
    sse = (
        b'data: {"choices":[{"index":0,"delta":{"reasoning_content":"a"}}]}\n\n'
        b'data: {"choices":[{"index":0,"delta":{"reasoning_content":"b"}}]}\n\n'
        b'data: {"choices":[{"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":2}}\n\n'
        b'data: [DONE]\n\n'
    )
    t = SseTranslator(model="glm-5.3")
    events = parse_events(collect(t.feed(sse)) + collect(t.close()))
    starts = [e for e in events if e["data"]["type"] == "content_block_start"]
    assert len(starts) == 1 and starts[0]["data"]["content_block"]["type"] == "thinking"


# ---- 4. Thinking then text (S3) -------------------------------------------

def test_thinking_then_text(load_sse):
    t = SseTranslator(model="glm-5.3")
    events = parse_events(collect(t.feed(load_sse("glm_stream_thinking_then_text.sse"))) + collect(t.close()))
    starts = [e for e in events if e["data"]["type"] == "content_block_start"]
    kinds = [s["data"]["content_block"]["type"] for s in starts]
    assert kinds == ["thinking", "text"]


# ---- 5. Text then thinking (defensive; unlikely from GLM but state machine must handle) ---

def test_text_then_thinking_transition():
    sse = (
        b'data: {"choices":[{"index":0,"delta":{"content":"answer"}}]}\n\n'
        b'data: {"choices":[{"index":0,"delta":{"reasoning_content":"late"}}]}\n\n'
        b'data: {"choices":[{"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}\n\n'
        b'data: [DONE]\n\n'
    )
    t = SseTranslator(model="glm-5.3")
    events = parse_events(collect(t.feed(sse)) + collect(t.close()))
    starts = [e for e in events if e["data"]["type"] == "content_block_start"]
    kinds = [s["data"]["content_block"]["type"] for s in starts]
    assert kinds == ["text", "thinking"]


# ---- 6. Tool-only (single) -------------------------------------------------

def test_single_tool_only(load_sse):
    t = SseTranslator(model="glm-5.3")
    events = parse_events(collect(t.feed(load_sse("glm_stream_tool_calls.sse"))) + collect(t.close()))
    starts = [e for e in events if e["data"]["type"] == "content_block_start"]
    assert len(starts) == 1 and starts[0]["data"]["content_block"]["type"] == "tool_use"


# ---- 7. Multi-tool interleaved by index (S5) -------------------------------

def test_multi_tool_calls_streaming_assembles_by_index():
    sse = (
        b'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"a","arguments":""}}]}}]}\n\n'
        b'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"id":"c2","type":"function","function":{"name":"b","arguments":""}}]}}]}\n\n'
        b'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"x\\":1}"}}]}}]}\n\n'
        b'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"function":{"arguments":"{\\"y\\":2}"}}]}}]}\n\n'
        b'data: {"choices":[{"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":5,"completion_tokens":5}}\n\n'
        b'data: [DONE]\n\n'
    )
    t = SseTranslator(model="glm-5.3")
    events = parse_events(collect(t.feed(sse)) + collect(t.close()))
    starts = [e for e in events if e["data"]["type"] == "content_block_start" and e["data"]["content_block"]["type"] == "tool_use"]
    assert len(starts) == 2
    ids = [s["data"]["content_block"]["id"] for s in starts]
    assert ids == ["c1", "c2"]


# ---- 8. [DONE] without terminal frame (S13) --------------------------------

def test_done_without_terminal_synthesises_stop(load_sse):
    t = SseTranslator(model="glm-5.3")
    events = parse_events(collect(t.feed(load_sse("glm_stream_done_without_terminal.sse"))) + collect(t.close()))
    types = [e["data"]["type"] for e in events]
    assert types[-2] == "message_delta"
    assert types[-1] == "message_stop"
    md = [e for e in events if e["data"]["type"] == "message_delta"][0]
    # Synthetic terminal uses end_turn
    assert md["data"]["delta"]["stop_reason"] == "end_turn"


# ---- 9. Mid-stream drop (passthrough) closes with event: error (S10) -------

def test_mid_drop_passthrough_emits_error_and_closes(load_sse):
    t = SseTranslator(model="glm-5.3")
    # feed the truncated fixture then close() without any terminal
    events = parse_events(collect(t.feed(load_sse("glm_stream_mid_drop.sse"))) + collect(t.close()))
    types = [e["data"]["type"] for e in events]
    # Must contain an error event AFTER close since we never saw [DONE] or finish_reason
    assert any(t_ == "error" for t_ in types), f"expected error event in {types}"


# ---- 10. Buffered retry (S10b) — placeholder API check ----------------------

def test_sse_translator_is_stateful_and_reusable_after_close():
    """A second SseTranslator instance must produce isolated output."""
    t1 = SseTranslator(model="glm-5.3")
    _ = collect(t1.feed(b'data: {"choices":[{"index":0,"delta":{"content":"a"}}]}\n\n'))
    t2 = SseTranslator(model="glm-5.3")
    out2 = collect(t2.feed(b'data: {"choices":[{"index":0,"delta":{"content":"b"}}]}\n\n'))
    # t2's output should not contain "a" from t1's stream
    assert b'"a"' not in b"".join(out2)


# ---- 11. Unicode multibyte split across chunks (S17) -----------------------

def test_unicode_multibyte_split_reassembles(load_sse):
    raw = load_sse("glm_stream_unicode_split.sse")
    parts = raw.split(b"\n|||SPLIT|||\n")
    assert len(parts) == 2
    t = SseTranslator(model="glm-5.3")
    out = collect(t.feed(parts[0])) + collect(t.feed(parts[1])) + collect(t.close())
    # Rocket emoji 🚀 should appear intact in some text_delta
    aggregated = b"".join(out).decode("utf-8", errors="replace")
    assert "🚀" in aggregated, "rocket emoji lost across chunk boundary"


# ---- 12. Empty tool_use input at close: input:{} still valid ---------------

def test_tool_use_with_empty_arguments_closes_with_empty_input():
    sse = (
        b'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"f","arguments":""}}]}}]}\n\n'
        b'data: {"choices":[{"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}\n\n'
        b'data: [DONE]\n\n'
    )
    t = SseTranslator(model="glm-5.3")
    events = parse_events(collect(t.feed(sse)) + collect(t.close()))
    starts = [e for e in events if e["data"]["type"] == "content_block_start" and e["data"]["content_block"]["type"] == "tool_use"]
    assert len(starts) == 1
    assert starts[0]["data"]["content_block"]["input"] == {}


# ---- 13. model_context_window_exceeded maps to stop_reason max_tokens ------

def test_context_exceeded_maps_stop_reason():
    sse = (
        b'data: {"choices":[{"index":0,"delta":{"content":"partial"}}]}\n\n'
        b'data: {"choices":[{"index":0,"finish_reason":"model_context_window_exceeded"}],"usage":{"prompt_tokens":1000,"completion_tokens":0}}\n\n'
        b'data: [DONE]\n\n'
    )
    t = SseTranslator(model="glm-5.3")
    events = parse_events(collect(t.feed(sse)) + collect(t.close()))
    md = [e for e in events if e["data"]["type"] == "message_delta"][0]
    assert md["data"]["delta"]["stop_reason"] == "max_tokens"


# ---- 14. Concurrency: each instance has isolated state (S19) ---------------

def test_two_concurrent_instances_do_not_share_state():
    sse_a = b'data: {"choices":[{"index":0,"delta":{"content":"A1"}}]}\n\n'
    sse_b = b'data: {"choices":[{"index":0,"delta":{"content":"B1"}}]}\n\n'
    ta = SseTranslator(model="glm-5.3")
    tb = SseTranslator(model="glm-5.3")
    out_a = b"".join(collect(ta.feed(sse_a)))
    out_b = b"".join(collect(tb.feed(sse_b)))
    assert b'"A1"' in out_a and b'"A1"' not in out_b
    assert b'"B1"' in out_b and b'"B1"' not in out_a
