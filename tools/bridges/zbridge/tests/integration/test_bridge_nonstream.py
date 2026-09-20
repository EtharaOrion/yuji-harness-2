"""Non-stream /v1/messages: happy path, translation correctness, header forwarding."""
from __future__ import annotations

import pytest
import respx
from httpx import Response

from .conftest import UPSTREAM

pytestmark = pytest.mark.integration


@respx.mock
async def test_nonstream_text_returns_valid_anthropic(client, secret_headers, load_fx_json, load_fx):
    respx.post(UPSTREAM).mock(return_value=Response(
        200, content=load_fx("glm_nonstream_text.json"),
        headers={"content-type": "application/json", "x-request-id": "req-1"}))

    r = await client.post(
        "/v1/messages",
        headers=secret_headers,
        content=load_fx("anthropic_req_text.json"),
    )
    assert r.status_code == 200
    body = r.json()
    assert body["type"] == "message"
    assert body["role"] == "assistant"
    assert body["model"] == "glm-5.3"
    assert body["content"][0]["type"] == "text"
    assert body["content"][0]["text"] == "Hi"
    assert body["id"].startswith("msg_")
    assert body["stop_reason"] == "end_turn"
    assert body["usage"]["input_tokens"] > 0
    # Upstream headers forwarded
    assert r.headers.get("zbridge-upstream-x-request-id") == "req-1"


@respx.mock
async def test_nonstream_forwards_model_alias_upstream(client, secret_headers, load_fx):
    route = respx.post(UPSTREAM).mock(return_value=Response(
        200, json={
            "id": "abc", "model": "glm-5.3",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }))
    r = await client.post("/v1/messages", headers=secret_headers, content=load_fx("anthropic_req_text.json"))
    assert r.status_code == 200
    # The outbound body to z.ai must have model=glm-5.3 (aliased from claude-3-5-sonnet-latest)
    outbound = route.calls[0].request
    import json as _json
    payload = _json.loads(outbound.content)
    assert payload["model"] == "glm-5.3"


@respx.mock
async def test_nonstream_rejects_forced_tool_choice_with_400(client, secret_headers, load_fx):
    # No upstream call expected
    respx.post(UPSTREAM).mock(return_value=Response(500))
    r = await client.post("/v1/messages", headers=secret_headers, content=load_fx("anthropic_req_forced_tool_choice.json"))
    assert r.status_code == 400
    body = r.json()
    assert body["type"] == "error"
    assert body["error"]["type"] == "invalid_request_error"


@respx.mock
async def test_nonstream_bearer_auth_forwarded_to_upstream(client, secret_headers, load_fx):
    route = respx.post(UPSTREAM).mock(return_value=Response(
        200, json={
            "id": "abc", "model": "glm-5.3",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "x"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }))
    await client.post("/v1/messages", headers=secret_headers, content=load_fx("anthropic_req_text.json"))
    assert route.calls[0].request.headers.get("authorization") == "Bearer test-zai-key"


@respx.mock
async def test_healthz(client):
    r = await client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["model_default"] == "glm-5.3"
