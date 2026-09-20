"""RED tests for zbridge.translate — GLM non-stream response -> Anthropic Message.

Every row in PLAN.md §5.2 has at least one assertion.
"""
from __future__ import annotations

from zbridge.translate import glm_to_anthropic_response


def test_text_only_maps_to_text_block(load_json):
    out = glm_to_anthropic_response(load_json("glm_nonstream_text.json"))
    assert out["type"] == "message"
    assert out["role"] == "assistant"
    assert out["model"] == "glm-5.3"
    assert len(out["content"]) == 1
    assert out["content"][0] == {"type": "text", "text": "Hi"}


def test_message_id_gets_msg_prefix_when_missing(load_json):
    out = glm_to_anthropic_response(load_json("glm_nonstream_text.json"))
    assert out["id"].startswith("msg_"), f"expected msg_ prefix, got {out['id']!r}"


def test_message_id_prefix_not_double_added():
    # If GLM already returns "msg_..." somehow, don't double-prefix
    out = glm_to_anthropic_response({
        "id": "msg_already_prefixed",
        "model": "glm-5.3",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "x"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    })
    assert out["id"] == "msg_already_prefixed"


def test_thinking_block_emitted_before_text(load_json):
    out = glm_to_anthropic_response(load_json("glm_nonstream_thinking_text.json"))
    kinds = [b["type"] for b in out["content"]]
    assert kinds == ["thinking", "text"]
    assert out["content"][0]["thinking"].startswith("The user greeted me")
    assert out["content"][1]["text"] == "Hello!"


def test_thinking_signature_empty_by_default(load_json):
    out = glm_to_anthropic_response(load_json("glm_nonstream_thinking_text.json"))
    assert out["content"][0].get("signature", None) == ""


def test_thinking_signature_stable_hex_when_key_set(load_json):
    key = b"test-key-abc"
    out1 = glm_to_anthropic_response(load_json("glm_nonstream_thinking_text.json"), thinking_sig_key=key)
    out2 = glm_to_anthropic_response(load_json("glm_nonstream_thinking_text.json"), thinking_sig_key=key)
    sig1 = out1["content"][0]["signature"]
    sig2 = out2["content"][0]["signature"]
    assert sig1 == sig2 and len(sig1) == 24 and all(c in "0123456789abcdef" for c in sig1)


def test_tool_calls_map_to_tool_use_blocks(load_json):
    out = glm_to_anthropic_response(load_json("glm_nonstream_tool_calls.json"))
    tool_uses = [b for b in out["content"] if b["type"] == "tool_use"]
    assert len(tool_uses) == 1
    tu = tool_uses[0]
    assert tu["id"] == "call_8208132785081240070"
    assert tu["name"] == "get_weather"
    assert tu["input"] == {"city": "Beijing"}


def test_finish_reason_maps_stop_to_end_turn(load_json):
    out = glm_to_anthropic_response(load_json("glm_nonstream_text.json"))
    assert out["stop_reason"] == "end_turn"


def test_finish_reason_tool_calls_maps_to_tool_use(load_json):
    out = glm_to_anthropic_response(load_json("glm_nonstream_tool_calls.json"))
    assert out["stop_reason"] == "tool_use"


def test_finish_reason_length_maps_to_max_tokens():
    out = glm_to_anthropic_response({
        "id": "x", "model": "glm-5.3",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "long..."}, "finish_reason": "length"}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 100},
    })
    assert out["stop_reason"] == "max_tokens"


def test_finish_reason_context_window_maps_to_max_tokens():
    out = glm_to_anthropic_response({
        "id": "x", "model": "glm-5.3",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": ""}, "finish_reason": "model_context_window_exceeded"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 0},
    })
    assert out["stop_reason"] == "max_tokens"


def test_usage_fields_mapped(load_json):
    out = glm_to_anthropic_response(load_json("glm_nonstream_text.json"))
    assert out["usage"]["input_tokens"] == 5
    assert out["usage"]["output_tokens"] == 1


def test_cached_tokens_mapped_when_present(load_json):
    out = glm_to_anthropic_response(load_json("glm_nonstream_thinking_text.json"))
    assert out["usage"].get("cache_read_input_tokens") == 3


def test_tool_calls_with_malformed_json_falls_back_to_empty_input():
    out = glm_to_anthropic_response({
        "id": "x", "model": "glm-5.3",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant", "content": None,
                "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{not json"}}],
            },
            "finish_reason": "tool_calls",
        }],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    })
    tu = [b for b in out["content"] if b["type"] == "tool_use"][0]
    assert tu["input"] == {}
