"""Integration test fixtures — spin up the FastAPI app with a mocked upstream."""
from __future__ import annotations

import os
from pathlib import Path

import httpx
import pytest
import pytest_asyncio
import respx

from zbridge.bridge import Config, build_app

UPSTREAM = "https://api.z.ai/api/coding/paas/v4/chat/completions"

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "synth"


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("ZB_ZAI_API_KEY", "test-zai-key")
    monkeypatch.setenv("ZB_BRIDGE_SECRET", "test-bridge-secret")
    monkeypatch.setenv("ZB_MAX_INLINE_RETRIES", "3")
    monkeypatch.setenv("ZB_MAX_INLINE_WAIT_S", "5")
    monkeypatch.setenv("ZB_PING_INTERVAL_S", "1")
    monkeypatch.setenv("ZB_STREAM_BUFFER_RETRIES", "2")
    # Clean anything a prior test set
    monkeypatch.delenv("ZB_STREAM_LOG_PATH", raising=False)


@pytest_asyncio.fixture
async def client():
    """AsyncClient bound to the FastAPI app via ASGITransport."""
    cfg = Config()
    upstream_client = httpx.AsyncClient()
    app = build_app(config=cfg, http_client=upstream_client)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    await upstream_client.aclose()


@pytest.fixture
def secret_headers():
    return {
        "content-type": "application/json",
        "x-zbridge-secret": "test-bridge-secret",
        "anthropic-version": "2023-06-01",
    }


@pytest.fixture
def load_fx():
    def _load(name: str) -> bytes:
        return (FIXTURES_DIR / name).read_bytes()
    return _load


@pytest.fixture
def load_fx_json(load_fx):
    import json
    def _load(name: str) -> dict:
        return json.loads(load_fx(name))
    return _load
