"""T20a smoke — one live call per surface against real GLM-5.3.

Requires ZB_ZAI_API_KEY. Skipped if not set.

Run:
    ZB_ZAI_API_KEY=... .venv/bin/python -m pytest tests/smoke/smoke_real_zai.py -m smoke -q
"""
from __future__ import annotations

import json
import os

import httpx
import pytest

from zbridge.bridge import Config, build_app

pytestmark = [pytest.mark.smoke]


def _need_key():
    if not os.environ.get("ZB_ZAI_API_KEY"):
        pytest.skip("ZB_ZAI_API_KEY not set")


@pytest.fixture
def app():
    _need_key()
    os.environ.setdefault("ZB_BRIDGE_SECRET", "smoke-secret")
    cfg = Config()
    return build_app(config=cfg)


async def _post(app, body: dict) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://smoke") as c:
        return await c.post(
            "/v1/messages",
            headers={
                "content-type": "application/json",
                "x-zbridge-secret": os.environ["ZB_BRIDGE_SECRET"],
            },
            content=json.dumps(body).encode(),
            timeout=180,
        )


async def test_smoke_text(app):
    r = await _post(app, {
        "model": "claude-3-5-sonnet-latest",
        "max_tokens": 32,
        "messages": [{"role": "user", "content": "Say hi in one word."}],
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["type"] == "message"
    assert body["content"][0]["type"] == "text"
    assert isinstance(body["content"][0]["text"], str) and body["content"][0]["text"]
    assert body["stop_reason"] in ("end_turn", "max_tokens")
    assert body["usage"]["input_tokens"] > 0
    assert body["usage"]["output_tokens"] > 0


async def test_smoke_stream_text(app):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://smoke") as c:
        async with c.stream(
            "POST", "/v1/messages",
            headers={
                "content-type": "application/json",
                "x-zbridge-secret": os.environ["ZB_BRIDGE_SECRET"],
                "accept": "text/event-stream",
            },
            content=json.dumps({
                "model": "claude-3-5-sonnet-latest",
                "max_tokens": 64,
                "stream": True,
                "messages": [{"role": "user", "content": "Say hi in three words."}],
            }).encode(),
            timeout=300,
        ) as r:
            assert r.status_code == 200
            raw = b""
            async for chunk in r.aiter_bytes():
                raw += chunk
    text = raw.decode("utf-8", errors="replace")
    assert "event: message_start" in text
    assert "event: message_stop" in text
    assert "text_delta" in text


async def test_smoke_probe_no_gzip(app):
    """Verify z.ai does not gzip SSE streams (would break byte-level parsing)."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://smoke") as c:
        async with c.stream(
            "POST", "/v1/messages",
            headers={
                "content-type": "application/json",
                "x-zbridge-secret": os.environ["ZB_BRIDGE_SECRET"],
                "accept": "text/event-stream",
                "accept-encoding": "gzip",  # ADVERTISE gzip; upstream should still send plain
            },
            content=json.dumps({
                "model": "claude-3-5-sonnet-latest",
                "max_tokens": 8, "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            }).encode(),
            timeout=180,
        ) as r:
            # We STRIP content-encoding upstream, so client should never see gzip
            assert r.headers.get("content-encoding") not in ("gzip", "deflate", "br")
