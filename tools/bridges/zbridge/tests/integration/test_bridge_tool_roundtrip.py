"""Two-turn tool round-trip: assistant emits tool_use → client sends tool_result → next turn."""
from __future__ import annotations

import json

import pytest
import respx
from httpx import Response

from .conftest import UPSTREAM

pytestmark = pytest.mark.integration


@respx.mock
async def test_two_turn_roundtrip_preserves_ids(client, secret_headers, load_fx):
    """Send a request that already contains a prior turn's tool_use + client's tool_result.
    Assert the bridge translates it to GLM's role=tool with matching tool_call_id."""

    # Mock upstream to accept whatever GLM request we send
    route = respx.post(UPSTREAM).mock(return_value=Response(
        200, json={
            "id": "final", "model": "glm-5.3",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "result observed"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 20, "completion_tokens": 3},
        },
    ))

    r = await client.post(
        "/v1/messages",
        headers=secret_headers,
        content=load_fx("anthropic_req_tool_roundtrip.json"),
    )
    assert r.status_code == 200
    outbound = json.loads(route.calls[0].request.content)

    # Assistant turn in outbound should have tool_calls with matching id
    assistant_msgs = [m for m in outbound["messages"] if m.get("role") == "assistant"]
    assert assistant_msgs, "expected an assistant message in outbound"
    tcs = assistant_msgs[0].get("tool_calls") or []
    assert tcs and tcs[0]["id"] == "toolu_call_a"

    # Tool result becomes role=tool with matching tool_call_id
    tool_msgs = [m for m in outbound["messages"] if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0]["tool_call_id"] == "toolu_call_a"
    assert '"ok"' in tool_msgs[0]["content"]

    # Response has expected text
    body = r.json()
    assert body["content"][0]["text"] == "result observed"
