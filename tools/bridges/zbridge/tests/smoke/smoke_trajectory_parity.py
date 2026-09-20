"""T20b smoke — trajectory schema parity with ccbridge + latency check.

Requires:
  - ZB_ZAI_API_KEY
  - ZB_BRIDGE_SECRET
  - ZB_STREAM_LOG_PATH (will be populated by the bridge)
  - Optional: CCBRIDGE_SAMPLE_JSONL — path to a ccbridge-produced JSONL for schema diff

Run:
    ZB_ZAI_API_KEY=... ZB_BRIDGE_SECRET=... ZB_STREAM_LOG_PATH=/tmp/zbridge-smoke.jsonl \
    CCBRIDGE_SAMPLE_JSONL=/path/to/ccbridge-sample.jsonl \
    .venv/bin/python -m pytest tests/smoke/smoke_trajectory_parity.py -m smoke -q
"""
from __future__ import annotations

import json
import os
import time

import httpx
import pytest

from zbridge.bridge import Config, build_app

pytestmark = [pytest.mark.smoke]

CCBRIDGE_SCHEMA_KEYS = {"ts", "seq", "source", "request_id", "model", "kind", "event", "delta"}


def _need():
    for k in ("ZB_ZAI_API_KEY", "ZB_BRIDGE_SECRET", "ZB_STREAM_LOG_PATH"):
        if not os.environ.get(k):
            pytest.skip(f"{k} not set")


async def test_trajectory_jsonl_schema_parity_and_latency():
    _need()
    log_path = os.environ["ZB_STREAM_LOG_PATH"]
    # Truncate any prior run
    open(log_path, "w").close()

    cfg = Config()
    app = build_app(config=cfg)
    transport = httpx.ASGITransport(app=app)

    t0 = time.perf_counter()
    ttft = None
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
                "max_tokens": 256, "stream": True,
                "messages": [{"role": "user", "content": "Say hi in five words."}],
            }).encode(),
            timeout=300,
        ) as r:
            assert r.status_code == 200
            async for chunk in r.aiter_bytes():
                if ttft is None and b"content_block_delta" in chunk:
                    ttft = time.perf_counter() - t0
    total = time.perf_counter() - t0

    # Schema parity
    rows = [json.loads(line) for line in open(log_path).read().splitlines() if line.strip()]
    assert rows, "stream tee produced no rows"
    row_keys = set(rows[0].keys())
    assert row_keys == CCBRIDGE_SCHEMA_KEYS, (
        f"schema mismatch: extra={row_keys - CCBRIDGE_SCHEMA_KEYS} "
        f"missing={CCBRIDGE_SCHEMA_KEYS - row_keys}"
    )

    # Optional ccbridge sample diff
    ccb_path = os.environ.get("CCBRIDGE_SAMPLE_JSONL")
    if ccb_path and os.path.exists(ccb_path):
        ccb_row = json.loads(open(ccb_path).readline())
        assert set(ccb_row.keys()) == row_keys, "ccbridge sample schema differs from zbridge"

    # Latency: TTFT should be < 60s, total < 300s for a trivial 5-word prompt.
    # Absolute bound — z.ai TTFT is typically 1-5s.
    assert ttft is not None and ttft < 60.0, f"TTFT={ttft}s exceeds 60s"
    assert total < 300.0, f"total={total}s exceeds 300s"

    # Emit a latency report artifact
    open("tests/smoke/latency_report.json", "w").write(json.dumps({
        "ttft_seconds": round(ttft, 3),
        "total_seconds": round(total, 3),
    }, indent=2))
