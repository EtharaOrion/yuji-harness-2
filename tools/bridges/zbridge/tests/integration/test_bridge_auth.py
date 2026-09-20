"""Auth gate: ZB_BRIDGE_SECRET required. Multiple header formats accepted."""
from __future__ import annotations

import pytest
import respx
from httpx import Response

from .conftest import UPSTREAM

pytestmark = pytest.mark.integration


@respx.mock
async def test_missing_or_wrong_secret_is_401(client, load_fx):
    respx.post(UPSTREAM).mock(return_value=Response(500))  # should NEVER be hit
    # No secret
    r = await client.post(
        "/v1/messages",
        headers={"content-type": "application/json"},
        content=load_fx("anthropic_req_text.json"),
    )
    assert r.status_code == 401
    body = r.json()
    assert body["error"]["type"] == "authentication_error"

    # Wrong secret
    r = await client.post(
        "/v1/messages",
        headers={"content-type": "application/json", "x-zbridge-secret": "wrong"},
        content=load_fx("anthropic_req_text.json"),
    )
    assert r.status_code == 401


@respx.mock
async def test_accepts_x_api_key_header(client, load_fx):
    respx.post(UPSTREAM).mock(return_value=Response(
        200, json={"id": "1", "model": "glm-5.3",
                   "choices": [{"index": 0, "message": {"role": "assistant", "content": "x"}, "finish_reason": "stop"}],
                   "usage": {"prompt_tokens": 1, "completion_tokens": 1}}))
    r = await client.post(
        "/v1/messages",
        headers={"content-type": "application/json", "x-api-key": "test-bridge-secret"},
        content=load_fx("anthropic_req_text.json"),
    )
    assert r.status_code == 200


@respx.mock
async def test_accepts_authorization_bearer(client, load_fx):
    respx.post(UPSTREAM).mock(return_value=Response(
        200, json={"id": "1", "model": "glm-5.3",
                   "choices": [{"index": 0, "message": {"role": "assistant", "content": "x"}, "finish_reason": "stop"}],
                   "usage": {"prompt_tokens": 1, "completion_tokens": 1}}))
    r = await client.post(
        "/v1/messages",
        headers={"content-type": "application/json", "authorization": "Bearer test-bridge-secret"},
        content=load_fx("anthropic_req_text.json"),
    )
    assert r.status_code == 200
