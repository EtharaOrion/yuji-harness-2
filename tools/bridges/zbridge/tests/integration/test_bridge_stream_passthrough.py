"""Stream passthrough mode: chunk-by-chunk translation."""
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
async def test_stream_text_passthrough_emits_full_anthropic_event_sequence(
    client, secret_headers, load_fx, monkeypatch,
):
    monkeypatch.setenv("ZB_BUFFER_AND_RETRY", "0")

    respx.post(UPSTREAM).mock(return_value=Response(
        200, content=load_fx("glm_stream_text.sse"),
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
        assert r.headers.get("content-type", "").startswith("text/event-stream")
        raw = b""
        async for chunk in r.aiter_bytes():
            raw += chunk

    frames = _parse_sse(raw)
    types = [f["data"]["type"] for f in frames]
    assert "message_start" in types
    assert "message_stop" in types
    assert types[-1] == "message_stop"
    # At least one text_delta
    text_deltas = [f for f in frames if f["data"]["type"] == "content_block_delta"
                   and f["data"]["delta"]["type"] == "text_delta"]
    assert len(text_deltas) >= 1
