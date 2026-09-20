"""S19 — four parallel streaming requests on the same bridge process must not cross state."""
from __future__ import annotations

import asyncio
import json

import pytest
import respx
from httpx import Response

from .conftest import UPSTREAM

pytestmark = pytest.mark.integration


def _make_sse(marker: str) -> bytes:
    """Build a small unique SSE payload; the marker appears in the streamed content."""
    frames = [
        f'data: {{"choices":[{{"index":0,"delta":{{"role":"assistant","content":""}},"finish_reason":null}}]}}',
        f'data: {{"choices":[{{"index":0,"delta":{{"content":"{marker}"}},"finish_reason":null}}]}}',
        f'data: {{"choices":[{{"index":0,"delta":{{}},"finish_reason":"stop"}}],"usage":{{"prompt_tokens":1,"completion_tokens":1}}}}',
        f'data: [DONE]',
    ]
    return ("\n\n".join(frames) + "\n\n").encode("utf-8")


@respx.mock
async def test_four_parallel_streams_do_not_share_state(client, secret_headers, load_fx):
    """
    Fire 4 concurrent streaming requests. The mock returns a different-marked payload
    per call, and each request must receive back only its own marker.
    """
    markers = ["MARK_A", "MARK_B", "MARK_C", "MARK_D"]

    # Use side_effect list so each request in the batch gets a unique payload
    respx.post(UPSTREAM).mock(side_effect=[
        Response(200, content=_make_sse(m), headers={"content-type": "text/event-stream"})
        for m in markers
    ])

    req_body = json.loads(load_fx("anthropic_req_text.json"))
    req_body["stream"] = True
    body_bytes = json.dumps(req_body).encode()

    async def do_one() -> bytes:
        async with client.stream(
            "POST", "/v1/messages", headers=secret_headers, content=body_bytes,
        ) as r:
            assert r.status_code == 200
            raw = b""
            async for chunk in r.aiter_bytes():
                raw += chunk
            return raw

    outputs = await asyncio.gather(*(do_one() for _ in markers))

    # Each output contains EXACTLY ONE of the markers (no crossing)
    hits_per_output = []
    for out in outputs:
        hit = [m for m in markers if m.encode() in out]
        hits_per_output.append(hit)
    # Every output has exactly one marker
    assert all(len(h) == 1 for h in hits_per_output), f"marker collisions: {hits_per_output}"
    # Each marker used exactly once (order may differ due to concurrency, but multiset equal)
    assert sorted(h[0] for h in hits_per_output) == sorted(markers)
