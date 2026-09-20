#!/usr/bin/env python3
"""T0 — Live probe of z.ai usage semantics. Read-only: touches no zbridge source.

Answers three questions that the usage-mapping fix depends on:

  Q1  Does z.ai's `usage.prompt_tokens` INCLUDE `prompt_tokens_details.cached_tokens`
      (OpenAI convention) or EXCLUDE it (Anthropic convention)?
  Q2  Does z.ai report any cache-WRITE / cache-creation field at all?
  Q3  Given the answer to Q1, does the current bridge translation double-count
      cached tokens once LiteLLM reconstructs `prompt_tokens`?

Method for Q1: send the SAME large prompt N times. Call 1 is a cache miss, later
calls are cache hits. If `prompt_tokens` stays flat while `cached_tokens` climbs,
prompt_tokens INCLUDES cached. If `prompt_tokens` drops by roughly the cached
amount, it EXCLUDES cached.

Usage:
    python scripts/probe_usage.py            # both phases
    python scripts/probe_usage.py --upstream-only
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import secrets
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

UPSTREAM = "https://api.z.ai/api/coding/paas/v4/chat/completions"
N_CALLS = 3
MAX_TOKENS = 16
CACHE_KEY_HINTS = ("cach", "creat", "write", "miss", "hit")


# ---------------------------------------------------------------------------
# .env loading (no python-dotenv dependency)
# ---------------------------------------------------------------------------

def load_dotenv(path: Path) -> list[str]:
    loaded = []
    if not path.exists():
        return loaded
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v
            loaded.append(k)
    return loaded


# ---------------------------------------------------------------------------
# Deterministic ~3k-token cacheable prompt
# ---------------------------------------------------------------------------

_PARA = (
    "The bridge translates Anthropic Messages API payloads into the OpenAI-compatible "
    "chat completions schema exposed by the GLM Coding Plan endpoint. Each request is "
    "rewritten field by field: system prompts are hoisted into a leading system message, "
    "tool definitions are rewrapped as function declarations, tool results are emitted as "
    "discrete tool-role messages, and thinking budgets are collapsed onto a reasoning "
    "effort ladder. Responses travel the same path in reverse. "
)


def build_prompt() -> tuple[str, str]:
    """Return (system, user). System is the big stable cacheable prefix.

    A per-run nonce is prepended so call 1 is a GUARANTEED cache miss even if an
    earlier run of this probe already warmed z.ai's cache. It must go at the very
    front: a shared prefix would let the cache match everything before the nonce.
    """
    nonce = secrets.token_hex(16)
    system = f"[probe-run {nonce}] " + "".join(
        f"[block {i:03d}] {_PARA}" for i in range(60)
    )
    user = "Reply with exactly one word: ok"
    return system, user


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_cache_keys(obj: Any, path: str = "usage") -> list[tuple[str, Any]]:
    """Recursively collect every key whose name hints at cache accounting."""
    hits = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}.{k}"
            if any(h in k.lower() for h in CACHE_KEY_HINTS):
                hits.append((p, v))
            hits.extend(find_cache_keys(v, p))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            hits.extend(find_cache_keys(v, f"{path}[{i}]"))
    return hits


def cached_of(usage: dict[str, Any]) -> int:
    ptd = usage.get("prompt_tokens_details") or {}
    if isinstance(ptd, dict) and ptd.get("cached_tokens") is not None:
        return int(ptd["cached_tokens"])
    if usage.get("cached_tokens") is not None:
        return int(usage["cached_tokens"])
    return 0


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# ---------------------------------------------------------------------------
# Phase A — raw upstream
# ---------------------------------------------------------------------------

async def phase_a(client, api_key: str) -> list[dict[str, Any]]:
    import httpx

    system, user = build_prompt()
    payload = {
        "model": os.environ.get("ZB_DEFAULT_MODEL", "glm-5.3"),
        "max_tokens": MAX_TOKENS,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }

    rule("PHASE A — raw z.ai upstream usage objects")
    print(f"upstream : {UPSTREAM}")
    print(f"model    : {payload['model']}")
    print(f"system   : {len(system)} chars (stable prefix, identical every call)")

    usages: list[dict[str, Any]] = []
    for i in range(1, N_CALLS + 1):
        resp = await client.post(
            UPSTREAM,
            json=payload,
            headers={
                "authorization": f"Bearer {api_key}",
                "content-type": "application/json",
                "accept": "application/json",
                "accept-language": "en-US,en",
            },
            timeout=httpx.Timeout(120.0, connect=30.0),
        )
        if resp.status_code != 200:
            print(f"\ncall {i}: HTTP {resp.status_code}\n{resp.text[:800]}")
            return usages
        body = resp.json()
        usage = body.get("usage") or {}
        usages.append(usage)
        print(f"\n--- call {i} (raw usage verbatim) ---")
        print(json.dumps(usage, indent=2, sort_keys=True, ensure_ascii=False))
        if i == 1:
            print(f"    response.model = {body.get('model')!r}")
        await asyncio.sleep(2)
    return usages


# ---------------------------------------------------------------------------
# Phase B — through the bridge (in-process ASGI, no port needed)
# ---------------------------------------------------------------------------

async def phase_b(upstream_client) -> list[dict[str, Any]]:
    import httpx

    from zbridge.bridge import build_app

    system, user = build_prompt()
    app = build_app(http_client=upstream_client)
    secret = os.environ.get("ZB_BRIDGE_SECRET", "").strip()

    rule("PHASE B — bridge-translated usage")

    payload = {
        "model": "claude-3-5-sonnet-latest",
        "max_tokens": MAX_TOKENS,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }

    out: list[dict[str, Any]] = []
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://probe") as bc:
        for i in range(1, N_CALLS + 1):
            headers = {"content-type": "application/json", "anthropic-version": "2023-06-01"}
            if secret:
                headers["x-zbridge-secret"] = secret
            r = await bc.post("/v1/messages", json=payload, headers=headers, timeout=120.0)
            if r.status_code != 200:
                print(f"\ncall {i}: HTTP {r.status_code}\n{r.text[:800]}")
                return out
            body = r.json()
            usage = body.get("usage") or {}
            out.append(usage)
            print(f"\n--- call {i} ---")
            print(f"  response.model     = {body.get('model')!r}")
            print(f"  bridge usage       = {json.dumps(usage, sort_keys=True)}")
            # LiteLLM's anthropic transform: prompt_tokens = input + cache_read + cache_creation
            recon = (
                int(usage.get("input_tokens", 0))
                + int(usage.get("cache_read_input_tokens", 0))
                + int(usage.get("cache_creation_input_tokens", 0))
            )
            print(f"  LiteLLM prompt_tokens (input+read+creation) = {recon}")
            await asyncio.sleep(2)
    return out


# ---------------------------------------------------------------------------
# Phase C — streaming vs non-streaming parity, raw upstream
# ---------------------------------------------------------------------------

def _sse_usage_frames(raw: bytes) -> list[dict[str, Any]]:
    """Extract every non-empty `usage` object from a raw GLM SSE byte stream."""
    found = []
    for block in raw.split(b"\n\n"):
        for line in block.split(b"\n"):
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            payload = line[len(b"data:"):].strip()
            if payload == b"[DONE]" or not payload:
                continue
            try:
                d = json.loads(payload)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if isinstance(d, dict) and d.get("usage"):
                found.append(d["usage"])
    return found


async def _upstream_call(client, api_key: str, payload: dict[str, Any]) -> tuple[int, bytes]:
    import httpx

    stream = bool(payload.get("stream"))
    r = await client.post(
        UPSTREAM,
        json=payload,
        headers={
            "authorization": f"Bearer {api_key}",
            "content-type": "application/json",
            "accept": "text/event-stream" if stream else "application/json",
            "accept-language": "en-US,en",
        },
        timeout=httpx.Timeout(120.0, connect=30.0),
    )
    return r.status_code, r.content


async def phase_c(client, api_key: str) -> dict[str, Any]:
    """Does z.ai report the SAME usage over SSE as it does non-streaming?

    Sequence on one fresh nonce prompt:
      1. non-stream  -> forced cache MISS, establishes true context size
      2. stream      -> cache hit, usage read from the SSE terminal frame
      3. non-stream  -> cache hit, the number stream must match
    Also retries the stream with `stream_options.include_usage` if the default
    stream carries no usage at all.
    """
    system, user = build_prompt()
    model = os.environ.get("ZB_DEFAULT_MODEL", "glm-5.3")
    base = {
        "model": model,
        "max_tokens": MAX_TOKENS,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }

    rule("PHASE C — upstream stream vs non-stream usage parity")
    out: dict[str, Any] = {}

    st, body = await _upstream_call(client, api_key, dict(base))
    if st != 200:
        print(f"non-stream miss call failed: HTTP {st}\n{body[:400]!r}")
        return out
    out["nonstream_miss"] = (json.loads(body).get("usage") or {})
    print("1. non-stream (forced MISS):")
    print("   " + json.dumps(out["nonstream_miss"], sort_keys=True))
    await asyncio.sleep(2)

    st, raw = await _upstream_call(client, api_key, {**base, "stream": True})
    if st != 200:
        print(f"stream call failed: HTTP {st}\n{raw[:400]!r}")
        return out
    usages = _sse_usage_frames(raw)
    print(f"\n2. stream (no stream_options): {len(usages)} frame(s) carried a usage object")
    if usages:
        out["stream_hit"] = usages[-1]
        print("   terminal usage: " + json.dumps(usages[-1], sort_keys=True))
    else:
        print("   NONE — retrying with stream_options.include_usage=true")
        await asyncio.sleep(2)
        st, raw2 = await _upstream_call(
            client, api_key,
            {**base, "stream": True, "stream_options": {"include_usage": True}},
        )
        u2 = _sse_usage_frames(raw2) if st == 200 else []
        print(f"   with stream_options: HTTP {st}, {len(u2)} usage frame(s)")
        if u2:
            out["stream_hit"] = u2[-1]
            out["needs_stream_options"] = True
            print("   terminal usage: " + json.dumps(u2[-1], sort_keys=True))
    await asyncio.sleep(2)

    st, body = await _upstream_call(client, api_key, dict(base))
    if st == 200:
        out["nonstream_hit"] = (json.loads(body).get("usage") or {})
        print("\n3. non-stream (cache HIT):")
        print("   " + json.dumps(out["nonstream_hit"], sort_keys=True))
    return out


# ---------------------------------------------------------------------------
# Phase D — what the bridge currently emits over SSE
# ---------------------------------------------------------------------------

async def phase_d(upstream_client) -> dict[str, Any]:
    import httpx

    from zbridge.bridge import build_app

    system, user = build_prompt()
    app = build_app(http_client=upstream_client)
    secret = os.environ.get("ZB_BRIDGE_SECRET", "").strip()

    rule("PHASE D — bridge SSE output: is usage preserved?")

    payload = {
        "model": "claude-3-5-sonnet-latest",
        "max_tokens": MAX_TOKENS,
        "stream": True,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
    headers = {"content-type": "application/json", "anthropic-version": "2023-06-01",
               "accept": "text/event-stream"}
    if secret:
        headers["x-zbridge-secret"] = secret

    found: dict[str, Any] = {}
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://probe") as bc:
        r = await bc.post("/v1/messages", json=payload, headers=headers, timeout=180.0)
        if r.status_code != 200:
            print(f"HTTP {r.status_code}\n{r.text[:400]}")
            return found
        for block in r.content.split(b"\n\n"):
            for line in block.split(b"\n"):
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                try:
                    d = json.loads(line[len(b"data:"):].strip())
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                if not isinstance(d, dict):
                    continue
                t = d.get("type")
                if t == "message_start":
                    found["message_start"] = (d.get("message") or {}).get("usage")
                elif t == "message_delta":
                    found["message_delta"] = d.get("usage")
                elif t == "error":
                    found["error"] = d.get("error")

    for k in ("message_start", "message_delta", "error"):
        if k in found:
            print(f"  {k:<14} usage = {json.dumps(found[k], sort_keys=True)}")
    return found


# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------

def stream_verdict(c: dict[str, Any], d: dict[str, Any]) -> int:
    rule("STREAMING VERDICT")
    rc = 0
    if not c:
        print("no upstream streaming data collected.")
        return 1

    miss = c.get("nonstream_miss") or {}
    shit = c.get("stream_hit")
    nhit = c.get("nonstream_hit") or {}

    print("Q4  does z.ai report usage at all over SSE?")
    if shit is None:
        print("      NO — no usage object in the SSE stream, with or without")
        print("      stream_options.include_usage. Streaming cannot reach parity;")
        print("      token counts would have to be estimated.")
        return 1
    print("      YES" + ("  (requires stream_options.include_usage)"
                         if c.get("needs_stream_options") else "  (by default)"))
    if c.get("needs_stream_options"):
        print("      => the bridge must inject stream_options.include_usage upstream.")

    print("\nQ5  is streamed usage identical to non-streamed for the same prompt?")
    print("      field                 non-stream(miss)  stream(hit)  non-stream(hit)")
    for f in ("prompt_tokens", "completion_tokens"):
        print(f"      {f:<21} {miss.get(f)!s:<17} {shit.get(f)!s:<12} {nhit.get(f)!s}")
    print(f"      {'cached_tokens':<21} {cached_of(miss)!s:<17} {cached_of(shit)!s:<12} "
          f"{cached_of(nhit)!s}")

    same_prompt = shit.get("prompt_tokens") == nhit.get("prompt_tokens") == miss.get("prompt_tokens")
    same_cached = cached_of(shit) == cached_of(nhit)
    if same_prompt and same_cached:
        print("\n      PASS: prompt_tokens and cached_tokens match across stream/non-stream.")
        print("            Parity is achievable — the numbers exist upstream.")
    else:
        print("\n      MISMATCH: streamed usage differs from non-streamed.")
        if not same_prompt:
            print("            prompt_tokens differs.")
        if not same_cached:
            print("            cached_tokens differs (may be cache timing, re-run to confirm).")
        rc = 1

    print("\nQ6  does the BRIDGE preserve it today?")
    ms = d.get("message_start") or {}
    md = d.get("message_delta") or {}
    got_in = int(ms.get("input_tokens", 0) or 0)
    got_rd = int(ms.get("cache_read_input_tokens", 0) or 0)
    got_cr = int(ms.get("cache_creation_input_tokens", 0) or 0)
    print(f"      message_start input_tokens          = {got_in}")
    print(f"      message_start cache_read_input      = {got_rd}")
    print(f"      message_start cache_creation_input  = {got_cr}")
    print(f"      message_delta output_tokens         = {md.get('output_tokens')}")
    if got_in == 0 and got_rd == 0 and got_cr == 0:
        print("\n      FAIL: all input-side usage is 0 over SSE. Enabling stream=true today")
        print("            would report input=0, cache_read=0, cache_write=0 and a cost")
        print("            covering output tokens only. Non-stream and stream do NOT agree.")
        rc = 1
    else:
        print("\n      PASS: input-side usage present on the SSE path.")
    return rc

def verdict(a_usages: list[dict[str, Any]], b_usages: list[dict[str, Any]]) -> int:
    rule("VERDICT")
    if not a_usages:
        print("Q1/Q2 UNRESOLVED — no successful upstream call.")
        return 1

    print("Q2  cache-related fields present in z.ai usage:")
    all_hits: dict[str, Any] = {}
    for u in a_usages:
        for p, v in find_cache_keys(u):
            all_hits[p] = v
    if all_hits:
        for p, v in sorted(all_hits.items()):
            print(f"      {p} = {v}")
    else:
        print("      (none)")
    write_like = [p for p in all_hits if any(h in p.lower() for h in ("creat", "write", "miss"))]
    print(f"    cache-WRITE field: {'FOUND -> ' + str(write_like) if write_like else 'ABSENT'}")

    print("\nQ1  prompt_tokens vs cached_tokens across identical calls:")
    print("      call  prompt_tokens  cached_tokens  prompt-cached")
    for i, u in enumerate(a_usages, 1):
        p, c = int(u.get("prompt_tokens", 0)), cached_of(u)
        print(f"      {i:<5} {p:<14} {c:<14} {p - c}")

    p1, c1 = int(a_usages[0].get("prompt_tokens", 0)), cached_of(a_usages[0])
    hit = next((u for u in a_usages[1:] if cached_of(u) > 0), None)

    rc = 0
    if c1 != 0:
        print(f"\n    INCONCLUSIVE: call 1 reported cached_tokens={c1}, expected 0.")
        print("    The per-run nonce should have forced a cache miss. Without a clean")
        print("    miss baseline the true context size is unknown and Q1 cannot be decided.")
        rc = 1
    elif hit is None:
        print("\n    INCONCLUSIVE: cached_tokens never became non-zero (no cache hit).")
        print("    Prompt may be under z.ai's minimum cacheable length, or caching is off")
        print("    for this key. Re-run with a longer prompt before concluding.")
        rc = 1
    else:
        ph, ch = int(hit.get("prompt_tokens", 0)), cached_of(hit)
        # Call 1 is a guaranteed cache miss (nonce), so its prompt_tokens IS the true
        # context size, uncontaminated by cache accounting. The prompt is byte-identical
        # on every call, so that size is fixed for the whole run.
        total = p1
        incl = abs(ph - total)          # hit prompt flat at total  => INCLUDES cached
        excl = abs(ph - (total - ch))   # hit prompt fell by cached => EXCLUDES cached
        print(f"\n    call 1 is a forced cache MISS (cached=0), so true context size = {total}.")
        print(f"    prompt is byte-identical on all {len(a_usages)} calls, so that size is fixed.")
        print(f"    hit call: prompt_tokens = {ph}, cached_tokens = {ch}")
        print(f"      predicted if prompt_tokens INCLUDES cached: {total}   -> |error| = {incl}")
        print(f"      predicted if prompt_tokens EXCLUDES cached: {total - ch}   -> |error| = {excl}")
        if incl <= excl:
            print("\n    => prompt_tokens INCLUDES cached_tokens (OpenAI convention).")
            print("       Therefore the bridge MUST report")
            print("           input_tokens = prompt_tokens - cached_tokens")
            print("       (split further into input + cache_creation, see map_usage), or any")
            print("       consumer reconstructing prompt_tokens = input + cache_read +")
            print("       cache_creation — LiteLLM's Anthropic transform does — will count")
            print("       every cached token twice. Q3 below checks whether it does.")
        else:
            print("\n    => prompt_tokens EXCLUDES cached_tokens (Anthropic convention).")
            print("       The current mapping is CORRECT; the double-count diagnosis is wrong.")
            print("       Do NOT apply the T1 subtraction. Re-investigate the 282k>200k anomaly.")

    if b_usages:
        print("\nQ3  bridge output — does it survive LiteLLM's reconstruction?")
        print("    The bridge prompt is byte-identical on all calls, so the reconstructed")
        print("    prompt_tokens must be CONSTANT. Call 1 is a forced miss, so its value is")
        print("    the ground truth; any later inflation is the cached tokens being double-counted.")
        print("      call  input  cache_read  cache_creation  recon")
        recons = []
        for i, bu in enumerate(b_usages, 1):
            inp = int(bu.get("input_tokens", 0))
            rd = int(bu.get("cache_read_input_tokens", 0))
            cr = int(bu.get("cache_creation_input_tokens", 0))
            recons.append(inp + rd + cr)
            print(f"      {i:<5} {inp:<6} {rd:<11} {cr:<15} {inp + rd + cr}")

        truth = recons[0]
        drift = [r for r in recons[1:] if r != truth]
        zero_input = [i for i, bu in enumerate(b_usages, 1)
                      if int(bu.get("input_tokens", 0)) == 0]
        if drift:
            print(f"\n    FAIL: recon drifted from {truth} to {drift} — cached tokens are")
            print("          still being counted twice.")
            rc = 1
        else:
            print(f"\n    PASS: recon constant at {truth} across miss and hit calls;")
            print("          no double-count.")
        if zero_input:
            print(f"    FAIL: input_tokens is 0 on call(s) {zero_input}; the cache-write split")
            print("          must never zero out input_tokens.")
            rc = 1
        else:
            print("    PASS: input_tokens non-zero on every call.")
        wrote = [int(bu.get("cache_creation_input_tokens", 0)) for bu in b_usages]
        print(f"    cache_creation_input_tokens per call: {wrote}"
              f"  ({'non-zero reported' if any(wrote) else 'ALL ZERO'})")
    return rc


# ---------------------------------------------------------------------------

async def main_async(args) -> int:
    import httpx

    loaded = load_dotenv(REPO / ".env")
    if loaded:
        print(f"loaded from .env: {', '.join(loaded)}")

    api_key = os.environ.get("ZB_ZAI_API_KEY", "").strip()
    if not api_key:
        print("ZB_ZAI_API_KEY is not set (checked env and .env).", file=sys.stderr)
        return 2

    rc = 0
    async with httpx.AsyncClient() as client:
        if not args.stream_only:
            a = await phase_a(client, api_key)
            b = [] if args.upstream_only or not a else await phase_b(client)
            rc |= verdict(a, b)
        if args.stream or args.stream_only:
            c = await phase_c(client, api_key)
            d = {} if args.upstream_only else await phase_d(client)
            rc |= stream_verdict(c, d)
    return rc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream-only", action="store_true", help="skip the bridge phases")
    ap.add_argument("--stream", action="store_true", help="also run the streaming phases (C/D)")
    ap.add_argument("--stream-only", action="store_true", help="run only the streaming phases")
    args = ap.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
