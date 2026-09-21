"""Tests for openhands-adapter trajectory serialization to Anthropic JSONL.

Verifies that entrypoint.py's serialize_message_to_anthropic_jsonl() and
_block_to_dict() produce output matching what harness's test.sh
parse_anthropic() function expects to read from /logs/agent/claude-code.txt.

Run:
    pytest services/openhands-adapter/tests/test_trajectory_serialization.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from entrypoint import serialize_message_to_anthropic_jsonl, _block_to_dict


def test_text_block_serializes_to_anthropic_shape():
    block = SimpleNamespace(type="text", text="Hello world")
    result = _block_to_dict(block)
    assert result == {"type": "text", "text": "Hello world"}


def test_tool_use_block_serializes_with_id_name_input():
    block = SimpleNamespace(
        type="tool_use",
        id="toolu_01ABC",
        name="LightXero.list_invoices",
        input={"limit": 24},
    )
    result = _block_to_dict(block)
    assert result == {
        "type": "tool_use",
        "id": "toolu_01ABC",
        "name": "LightXero.list_invoices",
        "input": {"limit": 24},
    }


def test_tool_result_block_serializes_with_tool_use_id_content():
    block = SimpleNamespace(
        type="tool_result",
        tool_use_id="toolu_01ABC",
        content='[{"id": "RC-004", ...}]',
    )
    result = _block_to_dict(block)
    assert result == {
        "type": "tool_result",
        "tool_use_id": "toolu_01ABC",
        "content": '[{"id": "RC-004", ...}]',
    }


def test_assistant_message_serializes_with_role_and_content_array():
    message = SimpleNamespace(
        role="assistant",
        content=[
            SimpleNamespace(type="text", text="I will start by finding tools"),
            SimpleNamespace(type="tool_use", id="toolu_01", name="ListTools", input={}),
        ],
    )
    result = serialize_message_to_anthropic_jsonl(message)
    assert result["type"] == "assistant"
    assert len(result["content"]) == 2
    assert result["content"][0]["type"] == "text"
    assert result["content"][1]["type"] == "tool_use"


def test_string_content_wraps_in_text_block():
    message = SimpleNamespace(role="user", content="What is the settlement rule?")
    result = serialize_message_to_anthropic_jsonl(message)
    assert result == {
        "type": "user",
        "content": [{"type": "text", "text": "What is the settlement rule?"}],
    }


def test_dict_input_message_supported():
    message = {"role": "assistant", "content": [{"type": "text", "text": "Done"}]}
    result = serialize_message_to_anthropic_jsonl(message)
    assert result == {"type": "assistant", "content": [{"type": "text", "text": "Done"}]}


def test_jsonl_output_parses_via_harness_parse_anthropic_shape():
    """End-to-end: our output must be readable by test.sh's parse_anthropic() function.

    Reference logic from staging/ed8fbb42/tests/test.sh step 1:
        blocks = obj.get("content") or []
        for block in blocks:
            t = block.get("type")
            if t == "tool_use":
                steps.append({"kind": "tool_use", "name": block.get("name"), "input": block.get("input")})
    """
    message = SimpleNamespace(
        role="assistant",
        content=[
            SimpleNamespace(type="tool_use", id="tu_1", name="LightMonday.update_item", input={"item_id": "rc-001"}),
        ],
    )
    result = serialize_message_to_anthropic_jsonl(message)
    line = json.dumps(result)
    parsed = json.loads(line)

    assert parsed.get("type") in ("assistant", "user")
    blocks = parsed.get("content") or []
    assert len(blocks) == 1
    assert blocks[0].get("type") == "tool_use"
    assert blocks[0].get("name") == "LightMonday.update_item"
    assert blocks[0].get("input") == {"item_id": "rc-001"}
