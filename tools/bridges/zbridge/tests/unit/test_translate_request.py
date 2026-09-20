"""RED tests for zbridge.translate — Anthropic /v1/messages -> GLM chat.completions.

Every row in PLAN.md §5.1 has at least one assertion. Failures are expected until T10.
"""
from __future__ import annotations

import pytest

from zbridge.translate import (
    TranslationError,
    anthropic_to_glm_request,
    map_reasoning_effort,
)


# ----- Model alias ----------------------------------------------------------

def test_model_alias_claude_maps_to_glm_5_3():
    out = anthropic_to_glm_request({
        "model": "claude-3-5-sonnet-latest",
        "max_tokens": 8,
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert out["model"] == "glm-5.3"


def test_model_passthrough_for_exact_glm_ids():
    out = anthropic_to_glm_request({
        "model": "glm-4.6",
        "max_tokens": 8,
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert out["model"] == "glm-4.6"


def test_model_alias_custom_map_overrides():
    out = anthropic_to_glm_request(
        {"model": "myalias", "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]},
        model_alias={"myalias": "glm-4.5"},
    )
    assert out["model"] == "glm-4.5"


# ----- Messages: strings + block lists --------------------------------------

def test_user_string_content_passthrough():
    out = anthropic_to_glm_request({
        "model": "glm-5.3", "max_tokens": 8,
        "messages": [{"role": "user", "content": "hello"}],
    })
    assert out["messages"][-1] == {"role": "user", "content": "hello"}


def test_user_text_block_list_flattens_to_string():
    out = anthropic_to_glm_request({
        "model": "glm-5.3", "max_tokens": 8,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "part1"},
            {"type": "text", "text": "part2"},
        ]}],
    })
    assert out["messages"][-1]["role"] == "user"
    assert "part1" in out["messages"][-1]["content"]
    assert "part2" in out["messages"][-1]["content"]


def test_user_image_block_maps_to_glm_multimodal(load_json):
    out = anthropic_to_glm_request(load_json("anthropic_req_image.json"))
    msg = out["messages"][-1]
    assert msg["role"] == "user"
    assert isinstance(msg["content"], list)
    kinds = [b["type"] for b in msg["content"]]
    assert kinds.count("image_url") == 2
    assert kinds.count("text") == 1
    # Base64 image becomes data URL
    urls = [b["image_url"]["url"] for b in msg["content"] if b["type"] == "image_url"]
    assert any(u.startswith("data:image/png;base64,") for u in urls)
    assert any(u == "https://example.com/pic.jpg" for u in urls)


def test_assistant_text_blocks_concatenate():
    out = anthropic_to_glm_request({
        "model": "glm-5.3", "max_tokens": 8,
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [
                {"type": "text", "text": "A "},
                {"type": "text", "text": "B"},
            ]},
        ],
    })
    assistant = [m for m in out["messages"] if m["role"] == "assistant"][0]
    assert assistant["content"].strip() == "A  B" or assistant["content"] == "A B"


def test_assistant_thinking_blocks_dropped_by_default():
    out = anthropic_to_glm_request({
        "model": "glm-5.3", "max_tokens": 8,
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "secret reasoning", "signature": "sig"},
                {"type": "text", "text": "answer"},
            ]},
        ],
    })
    for m in out["messages"]:
        assert "secret reasoning" not in str(m.get("content", ""))


def test_assistant_tool_use_maps_to_tool_calls(load_json):
    out = anthropic_to_glm_request(load_json("anthropic_req_tool_roundtrip.json"))
    assistant = [m for m in out["messages"] if m["role"] == "assistant"][0]
    assert "tool_calls" in assistant
    tc = assistant["tool_calls"][0]
    assert tc["id"] == "toolu_call_a"
    assert tc["type"] == "function"
    assert tc["function"]["name"] == "lookup"
    import json as _json
    assert _json.loads(tc["function"]["arguments"]) == {"q": "x"}


def test_user_tool_result_becomes_role_tool(load_json):
    out = anthropic_to_glm_request(load_json("anthropic_req_tool_roundtrip.json"))
    tool_msgs = [m for m in out["messages"] if m["role"] == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0]["tool_call_id"] == "toolu_call_a"
    assert '"ok"' in tool_msgs[0]["content"]


def test_mixed_content_user_message_splits_tool_first_then_text(load_json):
    """PLAN.md S15: N tool_result blocks -> N role=tool msgs, then residual text -> one role=user msg AFTER."""
    out = anthropic_to_glm_request(load_json("anthropic_req_mixed_content.json"))
    roles = [m["role"] for m in out["messages"]]
    # Find last tool message and last user message; tool must come before user
    last_tool_idx = max(i for i, r in enumerate(roles) if r == "tool")
    last_user_idx = max(i for i, r in enumerate(roles) if r == "user")
    assert last_tool_idx < last_user_idx, f"tool must precede residual user text; got roles={roles}"
    residual_user = out["messages"][last_user_idx]
    assert "Additional context after the result" in residual_user["content"]


# ----- System ---------------------------------------------------------------

def test_system_string_prepended_as_system_message():
    out = anthropic_to_glm_request({
        "model": "glm-5.3", "max_tokens": 8,
        "system": "Be helpful.",
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert out["messages"][0] == {"role": "system", "content": "Be helpful."}


def test_system_list_concatenates_cache_control_dropped(load_json):
    out = anthropic_to_glm_request(load_json("anthropic_req_system_list.json"))
    sys_msg = out["messages"][0]
    assert sys_msg["role"] == "system"
    assert "helpful assistant" in sys_msg["content"]
    assert "concisely" in sys_msg["content"]
    # cache_control marker must not leak
    assert "cache_control" not in str(sys_msg)
    assert "ephemeral" not in sys_msg["content"]


# ----- Params ---------------------------------------------------------------

def test_stop_sequences_truncated_to_four():
    out = anthropic_to_glm_request({
        "model": "glm-5.3", "max_tokens": 8,
        "stop_sequences": ["a", "b", "c", "d", "e", "f"],
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert out["stop"] == ["a", "b", "c", "d"]


def test_top_p_clamped_to_valid_range():
    out = anthropic_to_glm_request({
        "model": "glm-5.3", "max_tokens": 8,
        "top_p": 1.5,
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert out["top_p"] == 1.0


def test_temperature_passthrough():
    out = anthropic_to_glm_request({
        "model": "glm-5.3", "max_tokens": 8,
        "temperature": 0.5,
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert out["temperature"] == 0.5


def test_max_tokens_forwarded():
    out = anthropic_to_glm_request({
        "model": "glm-5.3", "max_tokens": 256,
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert out["max_tokens"] == 256


def test_stream_forwarded():
    out = anthropic_to_glm_request({
        "model": "glm-5.3", "max_tokens": 8, "stream": True,
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert out["stream"] is True


def test_top_k_dropped_with_warning():
    out = anthropic_to_glm_request({
        "model": "glm-5.3", "max_tokens": 8, "top_k": 40,
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert "top_k" not in out


# ----- Tools ----------------------------------------------------------------

def test_tools_renamed_to_function_shape(load_json):
    out = anthropic_to_glm_request(load_json("anthropic_req_tool.json"))
    assert isinstance(out.get("tools"), list)
    t = out["tools"][0]
    assert t["type"] == "function"
    assert t["function"]["name"] == "lookup"
    assert "parameters" in t["function"]
    assert "input_schema" not in str(t)


def test_tool_choice_auto_passthrough():
    out = anthropic_to_glm_request({
        "model": "glm-5.3", "max_tokens": 8,
        "tools": [{"name": "x", "description": "y", "input_schema": {"type": "object"}}],
        "tool_choice": "auto",
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert out.get("tool_choice") == "auto"


def test_tool_choice_forced_type_tool_rejected(load_json):
    """PLAN.md §5.1 + S14: forced tool selection REJECTED with 400."""
    with pytest.raises(TranslationError) as exc:
        anthropic_to_glm_request(load_json("anthropic_req_forced_tool_choice.json"))
    assert "tool_choice" in str(exc.value).lower() or "forced" in str(exc.value).lower()


def test_tool_choice_any_rejected():
    with pytest.raises(TranslationError):
        anthropic_to_glm_request({
            "model": "glm-5.3", "max_tokens": 8,
            "tools": [{"name": "x", "description": "y", "input_schema": {"type": "object"}}],
            "tool_choice": "any",
            "messages": [{"role": "user", "content": "hi"}],
        })


# ----- Unsupported field rejections (S18) -----------------------------------

@pytest.mark.parametrize("field,value", [
    ("n", 2),
    ("logprobs", True),
    ("seed", 42),
    ("response_format", {"type": "json_schema", "json_schema": {"schema": {}}}),
    ("service_tier", "priority"),
])
def test_unsupported_fields_rejected(field, value):
    with pytest.raises(TranslationError) as exc:
        anthropic_to_glm_request({
            "model": "glm-5.3", "max_tokens": 8,
            field: value,
            "messages": [{"role": "user", "content": "hi"}],
        })
    assert field in str(exc.value).lower() or "unsupported" in str(exc.value).lower()


# ----- Thinking -------------------------------------------------------------

def test_thinking_enabled_maps_to_glm_thinking_and_effort():
    out = anthropic_to_glm_request({
        "model": "glm-5.3", "max_tokens": 8,
        "thinking": {"type": "enabled", "budget_tokens": 8192},
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert out["thinking"] == {"type": "enabled"}
    assert out["reasoning_effort"] == "max"


def test_reasoning_effort_gated_by_model_family_glm5_never_medium():
    # GLM-5.3 with budget 3000 should NOT map to "medium"
    assert map_reasoning_effort(3000, "glm-5.3") in ("low", "high", "max")
    assert map_reasoning_effort(3000, "glm-5.3") != "medium"


def test_reasoning_effort_glm4_family_can_be_medium():
    assert map_reasoning_effort(3000, "glm-4.6") == "medium"


# ----- Metadata / user_id ---------------------------------------------------

def test_metadata_user_id_forwarded_when_valid_length():
    out = anthropic_to_glm_request({
        "model": "glm-5.3", "max_tokens": 8,
        "metadata": {"user_id": "user-abc-123"},
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert out.get("user_id") == "user-abc-123"


def test_metadata_user_id_dropped_when_too_short():
    out = anthropic_to_glm_request({
        "model": "glm-5.3", "max_tokens": 8,
        "metadata": {"user_id": "x"},
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert "user_id" not in out
