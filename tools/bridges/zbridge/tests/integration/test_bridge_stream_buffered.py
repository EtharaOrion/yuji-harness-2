"""Stream buffered mode: full capture, atomic replay, transparent retry on mid-drop."""
from __future__ import annotations

import json

import pytest
import respx
from httpx import Response

from .conftest import UPSTREAM

pytestmark = pytest.mark.integration


def _parse_sse(raw: bytes) -> list[dict]:
    frames = []
    cur = None
    for line in raw.decode("utf-8", errors="replace").splitlines():
        if line.startswith("event: "):
            cur = line[len("event: "):].strip()
        elif line.startswith("data: "):
            try:
                d = json.loads(line[len("data: "):])
            except json.JSONDecodeError:
                continue
            frames.append({"event": cur, "data": d})
            cur = None
    return frames


@respx.mock
async def test_buffered_default_delivers_complete_stream(client, secret_headers, load_fx):
    # buffered default (ZB_BUFFER_AND_RETRY=1)
    respx.post(UPSTREAM).mock(return_value=Response(
        200, content=load_fx("glm_stream_thinking_text_tool.sse"),
        headers={"content-type": "text/event-stream"},
    ))
    req_body = json.loads(load_fx("anthropic_req_text.json"))
    req_body["stream"] = True

    async with client.stream(
        "POST", "/v1/messages",
        headers=secret_headers,
        content=json.dumps(req_body).encode(),
    ) as r:
        assert r.status_code == 200
        assert r.headers.get("zbridge-stream-mode") == "buffered"
        raw = b""
        async for chunk in r.aiter_bytes():
            raw += chunk

    frames = _parse_sse(raw)
    types = [f["data"]["type"] for f in frames]
    assert "message_start" in types
    assert types[-1] == "message_stop"
    # thinking + text + tool_use blocks should all appear
    starts = [f["data"]["content_block"]["type"] for f in frames
              if f["data"]["type"] == "content_block_start"]
    assert starts == ["thinking", "text", "tool_use"]


@respx.mock
async def test_buffered_transparent_retry_on_mid_drop(client, secret_headers, load_fx, load_fx_json, monkeypatch):
    """S10b: first attempt drops mid-stream, second succeeds, client only sees success."""
    monkeypatch.setenv("ZB_MAX_INLINE_WAIT_S", "0")
    monkeypatch.setenv("ZB_STREAM_BUFFER_RETRIES", "2")

    # Attempt 1: partial stream (no [DONE], no finish_reason) — mid-drop
    # Attempt 2: full success
    route = respx.post(UPSTREAM).mock(side_effect=[
        Response(200, content=load_fx("glm_stream_mid_drop.sse"),
                 headers={"content-type": "text/event-stream"}),
        Response(200, content=load_fx("glm_stream_text.sse"),
                 headers={"content-type": "text/event-stream"}),
    ])

    req_body = json.loads(load_fx("anthropic_req_text.json"))
    req_body["stream"] = True

    async with client.stream(
        "POST", "/v1/messages",
        headers=secret_headers,
        content=json.dumps(req_body).encode(),
    ) as r:
        assert r.status_code == 200
        raw = b""
        async for chunk in r.aiter_bytes():
            raw += chunk

    frames = _parse_sse(raw)
    types = [f["data"]["type"] for f in frames]
    # Client sees a COMPLETE stream — no error event
    assert "error" not in types
    assert "message_start" in types
    assert types[-1] == "message_stop"
    # Upstream was retried
    assert route.call_count == 2
