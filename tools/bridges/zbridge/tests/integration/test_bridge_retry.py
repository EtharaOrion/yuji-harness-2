"""Retry semantics: 1302 retriable then succeeds; 1308 not-retriable surfaces immediately."""
from __future__ import annotations

import pytest
import respx
from httpx import Response

from .conftest import UPSTREAM

pytestmark = pytest.mark.integration


@respx.mock
async def test_1302_retries_then_succeeds(client, secret_headers, load_fx, load_fx_json, monkeypatch):
    monkeypatch.setenv("ZB_MAX_INLINE_WAIT_S", "0")  # snap retries as fast as possible

    # Two failures then a success
    route = respx.post(UPSTREAM).mock(side_effect=[
        Response(429, json=load_fx_json("glm_error_1302.json")),
        Response(429, json=load_fx_json("glm_error_1302.json")),
        Response(200, json={
            "id": "abc", "model": "glm-5.3",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "recovered"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }),
    ])
    r = await client.post("/v1/messages", headers=secret_headers, content=load_fx("anthropic_req_text.json"))
    assert r.status_code == 200
    assert r.json()["content"][0]["text"] == "recovered"
    assert route.call_count == 3


@respx.mock
async def test_1308_subscription_cap_surfaces_immediately(client, secret_headers, load_fx, load_fx_json):
    route = respx.post(UPSTREAM).mock(return_value=Response(429, json=load_fx_json("glm_error_1308.json")))
    r = await client.post("/v1/messages", headers=secret_headers, content=load_fx("anthropic_req_text.json"))
    assert r.status_code == 429
    body = r.json()
    assert body["error"]["type"] == "rate_limit_error"
    assert r.headers.get("zbridge-error-kind") == "subscription_cap"
    assert route.call_count == 1  # NOT retried


@respx.mock
async def test_1301_waf_surfaces_400_with_hint(client, secret_headers, load_fx, load_fx_json):
    respx.post(UPSTREAM).mock(return_value=Response(400, json=load_fx_json("glm_error_1301.json")))
    r = await client.post("/v1/messages", headers=secret_headers, content=load_fx("anthropic_req_text.json"))
    assert r.status_code == 400
    body = r.json()
    assert body["error"]["type"] == "invalid_request_error"
    assert "tool" in body["error"]["message"].lower() or "syntax" in body["error"]["message"].lower()


@respx.mock
async def test_upstream_500_retries_then_surfaces(client, secret_headers, load_fx, monkeypatch):
    monkeypatch.setenv("ZB_MAX_INLINE_WAIT_S", "0")
    route = respx.post(UPSTREAM).mock(return_value=Response(500, content=b"internal"))
    r = await client.post("/v1/messages", headers=secret_headers, content=load_fx("anthropic_req_text.json"))
    assert r.status_code >= 500
    assert route.call_count > 1  # retried
