# zbridge — Anthropic-shaped bridge to z.ai GLM Coding Plan

> **Revision R1** — incorporates all 7 critical + 11 important findings from oracle review 2026-09-03.

## 1. Goal & Success Criteria

Build a local FastAPI proxy that exposes an Anthropic-shaped `/v1/messages` endpoint on port 8766 and translates every request/response/SSE event to and from z.ai's OpenAI-compat GLM Coding Plan endpoint (`https://api.z.ai/api/coding/paas/v4/chat/completions`), so the existing trajectory harness — which today points at ccbridge — can point at zbridge with only a base-URL change and produce faithful, thinking-preserving, tool-use-round-tripped trajectories on GLM-5.3.

Binary success checklist (all must be observable):

- [ ] `curl http://127.0.0.1:8766/v1/messages` returns a valid Anthropic Messages response for a text-only prompt, backed by a real GLM-5.3 completion.
- [ ] SSE stream to the same endpoint emits `message_start → content_block_start(thinking) → content_block_delta(thinking) → content_block_stop → content_block_start(text) → content_block_delta(text) → content_block_stop → message_delta → message_stop` in order, driven by real GLM stream chunks.
- [ ] A tool-use round trip (assistant `tool_use` → client `tool_result` → assistant final text) completes across two calls without any ID renaming or JSON re-encoding drift.
- [ ] `pytest -q` runs zero-network unit tests green (translation + SSE state machine + error mapping).
- [ ] `pytest -q -m integration` runs green using `respx`-mocked GLM upstream.
- [ ] The trajectory harness completes one real end-to-end task pointed at `http://127.0.0.1:8766` and emits a stream-tee JSONL file whose schema is byte-identical to ccbridge's.
- [ ] Buffered-stream mode survives at least one injected mid-stream drop (respx close-mid-frame) and re-issues upstream transparently; the client never observes the drop.
- [ ] `ZB_BRIDGE_SECRET` is required — requests without it get 401.
- [ ] `GET /healthz` returns `{"ok":true, "upstream":"api.z.ai", "model_default":"glm-5.3"}` in <100ms.
- [ ] No secrets appear in `git status` output or in any file tracked by git.
- [ ] `docker build -f Dockerfile.zbridge .` produces an image that starts and serves `/healthz` on port 8766.

## 2. Scope

**In scope (MVP / v1):**
- Anthropic `/v1/messages` non-stream + stream endpoint.
- `/healthz` liveness endpoint.
- Single-key auth (`ZB_ZAI_API_KEY`).
- Full Anthropic ↔ GLM translation for: text, system prompt (list-of-blocks), thinking, tool_use / tool_result (including mixed-content user messages), images (base64 and URL), stop_sequences, temperature, top_p, max_tokens.
- Stateful SSE translator (thinking → text → tool_calls interleaving) with `[DONE]`-without-terminal recovery.
- **Buffer-and-retry stream mode** — full upstream buffer, replay on mid-stream drop, SSE ping keepalive to client. Default ON (matches ccbridge). Gate via `ZB_BUFFER_AND_RETRY` env.
- Error taxonomy port (GLM business codes → ErrorKind → Anthropic-shape error), including upstream request-id + rate-limit header forwarding.
- Inline retry with exponential backoff for retriable errors.
- Reject unsupported `tool_choice` variants (`any`, `tool`, `{name:...}`) with 400 — GLM Coding Plan has no forced-tool knob today.
- Stream tee JSONL compatible with ccbridge schema.
- Bridge secret auth, bind 127.0.0.1 default.
- Dockerfile, pyproject.toml, README, .gitignore.
- pytest unit + respx integration + one documented real-endpoint smoke.

**Out of scope (deferred):**
- Multi-key pool + failover (v1.1) — but `credentials.py` interface is factored to slot it in.
- `recovery.py` pause-and-resume on `SUBSCRIPTION_CAP` (v1.1).
- `/quota` polling endpoint (v1.1).
- Metrics endpoint, cost surface, prompt-caching cache_control passthrough (v1.2).
- OAuth flow, Keychain reader, billing-attribution injection, tool-name prefix rename, `You are Claude Code` system prefix — all dropped (see Appendix).
- z.ai native Anthropic passthrough endpoint (rejected in decision).
- GLM vision models beyond the text-image case in `glm-5.3-flash` schema (models kept as configurable aliases).

## 3. Architecture

### 3.0 Alternatives Considered (and why rejected)

Three viable upstream architectures were evaluated before settling on Option A. Recording them here so future maintainers can audit the decision without re-deriving it.

| Option | Upstream endpoint | Bridge does | Rejected because |
|---|---|---|---|
| **A (chosen)** | `https://api.z.ai/api/coding/paas/v4/chat/completions` (OpenAI-compat, Coding Plan) | Full Anthropic↔OpenAI translation in-process | Requires building `translate.py` + `sse_translator.py` — non-trivial. But this is the ONLY path that gets GLM-5.3 (Coding Plan gate) and gives us full control over reasoning_content mapping, error taxonomy, header forwarding, stream buffering, and tee schema fidelity. |
| **B** | Same as A, but the trajectory harness talks OpenAI directly | Thin proxy: auth injection + logging + retry | Would force rewriting every Anthropic-SDK call site in the harness. Zero pipeline reuse from ccbridge. High blast radius on the client side. |
| **C** | `https://api.z.ai/api/anthropic/v1/messages` (z.ai's native Anthropic passthrough) | Trivial reverse-proxy — no translation code | **Blocking**: (a) this endpoint does NOT front the Coding Plan, so no GLM-5.3 access (verified against z.ai docs Q3 2026); (b) inherits z.ai's known translation bugs (`reasoning_content` drop on multi-turn per zai-org/GLM-4.5#100; WAF false-positives on function-call syntax in text); (c) no hook for our SSE tee (bytes never traverse our process in a form we can parse); (d) no hook for buffer-and-retry — mid-stream drops surface as truncated turns; (e) z.ai owns the SSE event shapes, so any upstream change silently mutates our trajectory bytes — unacceptable for a reproducible eval pipeline. |

Option C is retained as a **v1.1 opt-in fast path** (`ZB_UPSTREAM_MODE=anthropic_passthrough`) for callers on non-Coding-Plan models (GLM-4.6 and below) who want zero-translation cost and accept the tradeoffs.

### 3.1 Data-flow diagram

```
        harness / litellm / aider  (Anthropic SDK client)
                       │
                       │  POST /v1/messages   (Anthropic JSON, optional SSE)
                       ▼
     ┌───────────────────────────────────────────────────────────┐
     │                     zbridge :8766                         │
     │                                                           │
     │  bridge.py (FastAPI)                                      │
     │    ├─ auth: ZB_BRIDGE_SECRET check                        │
     │    ├─ translate.py: Anthropic req → GLM req               │
     │    ├─ httpx.AsyncClient → api.z.ai coding/paas/v4         │
     │    ├─ non-stream:      translate.py: GLM resp → Anthropic │
     │    ├─ stream path A:   sse_translator.py chunk-passthrough │
     │    ├─ stream path B:   sse_translator.py buffer-and-retry │
     │    │                   (default; ping keepalive)          │
     │    ├─ errors.py: classify + retry inline + hdr forward    │
     │    └─ stream_tee.py: JSONL tap (schema == ccbridge)       │
     │                                                           │
     │  credentials.py: KeyProvider (single-key v1, pool-ready)  │
     │  __main__.py: uvicorn CLI                                 │
     └───────────────────────────────────────────────────────────┘
                       │
                       ▼
       https://api.z.ai/api/coding/paas/v4/chat/completions
                (GLM-5.3, Bearer ZB_ZAI_API_KEY)
```

Module map vs ccbridge:

| zbridge module | Source lineage | Rationale |
|---|---|---|
| `bridge.py` | ported skeleton from ccbridge `bridge.py` | Same FastAPI shape, streaming vs non-streaming split, timeouts, secret gate, upstream header forwarding. Drops OAuth, billing-attribution, tool-prefix rename, `You are Claude Code` prefix. |
| `translate.py` | NEW | Pure functions Anthropic⇄GLM (request-out, response-in). No I/O — trivially unit-testable. |
| `sse_translator.py` | NEW | Stateful class that consumes GLM SSE frames and yields Anthropic SSE events. Two modes: chunk-passthrough and buffer-and-retry (parallel to ccbridge's two modes). |
| `errors.py` | ported enum shape from ccbridge `errors.py` | Same `ErrorKind` categories, rewritten classifier for GLM business codes; keeps `is_retryable` / `is_account_problem` semantics; upstream request-id and rate-limit header forwarding helper. |
| `credentials.py` | reduced port of ccbridge `credentials.py` | `KeyProvider` protocol only, single `EnvKeyProvider` impl. Structured so `MultiKeyPoolProvider` can be added in v1.1 with no call-site change. Drops Keychain, OAuth refresh, fcntl lock. |
| `stream_tee.py` | ported verbatim schema from ccbridge `stream_tee.py` | Same JSONL schema so downstream trajectory consumers work unchanged; implementation trimmed to what zbridge emits. |
| `__main__.py` | ported from ccbridge `__main__.py` | Same CLI shape (`--host --port --check`), port default 8766. |
| `recovery.py` | DEFERRED to v1.1 | Not needed for single-key MVP. |

## 4. Repo Layout

```
/Users/apple/Desktop/zbridge/
├── .gitignore
├── .python-version              # 3.12
├── pyproject.toml               # fastapi==0.115.5 uvicorn==0.32.1 httpx==0.27.2 pydantic==2.10.3 pytest respx
├── README.md
├── PLAN.md
├── Dockerfile.zbridge
├── docs/
│   ├── ARCHITECTURE.md          # links back to PLAN.md
│   ├── TRANSLATION.md           # frozen translation contract tables; BOLD "Silent deviations" section
│   ├── ERRORS.md                # GLM code → ErrorKind table
│   └── RUNBOOK.md               # start/stop, env vars, smoke test, data-residency callout
├── zbridge/
│   ├── __init__.py
│   ├── __main__.py
│   ├── bridge.py
│   ├── translate.py
│   ├── sse_translator.py
│   ├── errors.py
│   ├── credentials.py
│   └── stream_tee.py
├── tests/
│   ├── conftest.py
│   ├── fixtures/
│   │   ├── synth/                                # T03a: hand-authored, no key required
│   │   │   ├── glm_stream_text.sse
│   │   │   ├── glm_stream_thinking_then_text.sse
│   │   │   ├── glm_stream_tool_calls.sse
│   │   │   ├── glm_stream_thinking_text_tool.sse
│   │   │   ├── glm_stream_mid_drop.sse
│   │   │   ├── glm_stream_done_without_terminal.sse
│   │   │   ├── glm_stream_unicode_split.sse
│   │   │   ├── glm_nonstream_text.json
│   │   │   ├── glm_nonstream_tool_calls.json
│   │   │   ├── glm_error_1301.json
│   │   │   ├── glm_error_1302.json
│   │   │   ├── glm_error_1308.json
│   │   │   └── anthropic_req_*.json
│   │   └── real/                                 # T03b: recorded from real z.ai, optional
│   │       └── (populated when ZB_ZAI_API_KEY available)
│   ├── unit/
│   │   ├── test_translate_request.py
│   │   ├── test_translate_response.py
│   │   ├── test_sse_translator.py
│   │   ├── test_errors.py
│   │   ├── test_credentials.py
│   │   └── test_stream_tee.py
│   ├── integration/
│   │   ├── test_bridge_nonstream.py              # respx-mocked upstream
│   │   ├── test_bridge_stream_passthrough.py
│   │   ├── test_bridge_stream_buffered.py
│   │   ├── test_bridge_auth.py
│   │   ├── test_bridge_retry.py
│   │   ├── test_bridge_tool_roundtrip.py
│   │   └── test_bridge_concurrency.py
│   └── smoke/
│       ├── smoke_real_zai.py                     # opt-in; needs ZB_ZAI_API_KEY
│       └── smoke_trajectory_parity.py            # v1 harness parity + latency
└── scripts/
    ├── run_dev.sh                                 # exports env, launches uvicorn --reload
    └── curl_smoke.sh                              # curl example against local bridge
```

`.gitignore` at minimum: `.venv/`, `__pycache__/`, `*.pyc`, `.env`, `.bridge_secret`, `bridge.log`, `stream_tee.jsonl`, `.pytest_cache/`, `dist/`, `build/`, `tests/fixtures/real/`.

## 5. Translation Contract

### 5.1 Request: Anthropic `/v1/messages` → GLM `/chat/completions`

| Anthropic field | GLM field | Notes / edge cases |
|---|---|---|
| `model` | `model` | Alias map: flat dict `{"claude-3-5-sonnet-latest":"glm-5.3", ...}` (no globs). Passthrough for exact GLM ids. Configurable via `ZB_MODEL_ALIAS_JSON`. Default alias: every alias key resolves to `glm-5.3`. |
| `messages[].role="user"` with string content | `messages[].role="user"`, `content=str` | Direct. |
| `messages[].role="user"` with block list (text-only) | `messages[].role="user"`, `content=str` (joined) | Concatenate `text` blocks with newline. |
| `messages[].role="user"` with block list (mixed image + text) | `messages[].role="user"`, `content=[{type:"text",text:…},{type:"image_url",image_url:{url:…}}]` | GLM multimodal array. For `type:"image"` with `source.type:"base64"` → data URL `data:{media_type};base64,{data}`. For `type:"image"` with `source.type:"url"` → passthrough URL. |
| `messages[].role="user"` with mixed `text` + `tool_result` blocks | Emit N `role="tool"` messages first (in original order, one per `tool_result`), then any residual `text` blocks flatten into a single trailing `role="user"` message | Preserves order-of-observation semantics; scenario S15. |
| `messages[].role="assistant"` with `text` blocks | `messages[].role="assistant"`, `content=str` | Concatenate text blocks. |
| `messages[].role="assistant"` with `thinking` blocks | Drop from outgoing messages by default. If `ZB_PRESERVE_THINKING_IN_CONTEXT=1`, emit as `reasoning_content` on that turn AND set `thinking:{type:"enabled"}` + `clear_thinking:false`. | v1 default: drop, since GLM only accepts thinking on the current turn's response. |
| `messages[].role="assistant"` with `tool_use` blocks | Same message becomes `role="assistant"`, `content=""` (or accumulated text) + `tool_calls=[{id, type:"function", function:{name, arguments:JSON.stringify(input)}}]` | `tool_use.id` maps 1:1 to `tool_calls[].id`. Store map in translator scope to reverse. |
| `messages[].role="user"` with single `tool_result` block | `messages[].role="tool"`, `tool_call_id=<original id>`, `content=<result string or JSON-encoded>` | Content blocks flattened to string. |
| `system` (string) | Prepend to `messages` as `{role:"system", content:str}` | If already-empty, skip. |
| `system` (list of `text` blocks) | Concatenate texts, join with `\n\n`, then as above | **`cache_control` markers silently dropped in v1** — documented deviation. |
| `stop_sequences` (list) | `stop` (list, ≤4) | Truncate to 4, warn in log. |
| `temperature` (0..1) | `temperature` (0..1) | Passthrough. |
| `top_p` | `top_p` (clamp to [0.01, 1]) | Clamp with debug log if outside. |
| `max_tokens` (required in Anthropic) | `max_tokens` | Direct. |
| `stream` | `stream` | Direct. |
| `tools[].name/description/input_schema` | `tools[].function.{name, description, parameters}` wrapped with `type:"function"` | Direct rename. Reject `tools` that lack `input_schema`. |
| `tool_choice="auto"` / omitted | `tool_choice="auto"` | Direct. |
| `tool_choice="any"` / `"tool"` / `{type:"tool",name:...}` | **REJECT: HTTP 400 `invalid_request_error`, message: "GLM Coding Plan supports only tool_choice=auto; forced tool selection unavailable in v1."** | Silent coercion would break agents relying on forced tool use. Scenario S14. |
| `metadata.user_id` | `user_id` (if 6..128 chars, else drop) | Length gate. |
| `top_k` | Not supported by GLM → drop with warn. | |
| Anthropic thinking config `thinking:{type:"enabled", budget_tokens:N}` in request | GLM `thinking:{type:"enabled"}` (+ `reasoning_effort` mapped from budget_tokens, gated by model family) | **GLM-5.x** (only `low`/`high`/`max` accepted): <4k→low, <8k→high, ≥8k→max. **GLM-4.x** (full ladder): <2k→low, <4k→medium, <8k→high, ≥8k→max. Never emit `medium` to GLM-5.3 — upstream rejects it. |
| `anthropic-version` header | Ignored (we own the shape). | |
| `x-api-key` / `authorization` from client | Reject if `ZB_BRIDGE_SECRET` mismatch. | Bridge secret gate. |

Fields with NO clean mapping — **REJECT** upfront with HTTP 400 `invalid_request_error`: `n>1`, `logprobs`, `seed`, `response_format=json_schema`, `service_tier`.

### 5.2 Response: GLM `/chat/completions` (non-stream) → Anthropic Message

| GLM field | Anthropic field | Notes |
|---|---|---|
| `id` | `id` (with `msg_` prefix if raw id lacks it) | Preserve for correlation. |
| `model` | `model` | Passthrough. |
| `choices[0].message.reasoning_content` | `content[].type="thinking"`, `thinking=<str>`, `signature=<see §5.2.1>` | Emit BEFORE text block. |
| `choices[0].message.content` (string) | `content[].type="text"`, `text=<str>` | Emit AFTER thinking block. |
| `choices[0].message.tool_calls[]` | `content[].type="tool_use"`, `id=<tool_calls[].id>`, `name=<function.name>`, `input=JSON.parse(function.arguments)` | If JSON parse fails, emit `input={}` and log a `warning`. |
| `choices[0].finish_reason` | `stop_reason`: `stop`→`end_turn`, `tool_calls`→`tool_use`, `length`→`max_tokens`, `sensitive`→`stop_sequence` (with warning), `model_context_window_exceeded`→`max_tokens`, `network_error`→propagate as 502 (non-stream) / `event: error` (stream) | |
| `usage.prompt_tokens` | `usage.input_tokens` | Direct. |
| `usage.completion_tokens` | `usage.output_tokens` | Direct. |
| `usage.prompt_tokens_details.cached_tokens` | `usage.cache_read_input_tokens` | If present. |
| `role` | Always emit `role:"assistant"`, `type:"message"`. | |

#### 5.2.1 Thinking block `signature`

If `ZB_THINKING_SIG_KEY` is set (non-empty): `signature = hmac_sha256(ZB_THINKING_SIG_KEY, thinking_text)[:24]` (hex). Deterministic across runs with the same key.

If `ZB_THINKING_SIG_KEY` is empty (default): `signature = ""`. On startup, if thinking is enabled anywhere in the config AND the key is empty, log a WARNING: `thinking signatures will be empty; set ZB_THINKING_SIG_KEY for reproducible trajectories`.

**Rationale**: an ephemeral per-process key breaks trajectory replay determinism and content-hash-keyed harness caches. Explicit opt-in avoids silent drift.

### 5.2a Response headers (both non-stream and stream, success and error paths)

Forward the following upstream headers to the client, prefixed with `zbridge-upstream-`:
- `x-request-id` (or whichever request-id header z.ai emits — confirmed from a captured fixture in T03a/T03b)
- Any `x-ratelimit-*` header
- `retry-after` (if present)

Strip: `content-encoding`, `transfer-encoding`, `content-length`, `connection` (chunking artifacts).

Emit `zbridge-error-kind: <ErrorKind.value>` on every non-2xx response (mirrors ccbridge's `X-WCB-Bridge-Error`).

### 5.3 SSE Translation: GLM stream → Anthropic events

State machine tracks: `current_block_kind ∈ {none, thinking, text, tool_use[i]}`, `text_index`, `tool_by_index: dict[int→(anthropic_index, id, name, args_buffer)]`, `emitted_message_start`, `terminal_emitted`, `usage_accumulator`.

Rules:

1. On first frame: emit `event: message_start` with placeholder `message.id`, `model`, `usage={input_tokens:0,output_tokens:0}`.
2. Frame with `delta.reasoning_content`:
   - If `current != thinking`: close any open block (`content_block_stop`), then `event: content_block_start` with `content_block:{type:"thinking", thinking:"", signature:""}` at next index.
   - Emit `event: content_block_delta` with `delta:{type:"thinking_delta", thinking:<chunk>}`.
3. Frame with `delta.content`:
   - If `current != text`: close open block, emit `content_block_start` with `{type:"text", text:""}` at next index.
   - Emit `content_block_delta` with `{type:"text_delta", text:<chunk>}`.
4. Frame with `delta.tool_calls[]`:
   - For each entry keyed by `index`:
     - If unseen index and has `id`+`function.name`: close open block, allocate next Anthropic index, emit `content_block_start` with `{type:"tool_use", id, name, input:{}}`, remember buffer="".
     - Append `function.arguments` to buffer, emit `content_block_delta` with `{type:"input_json_delta", partial_json:<chunk>}`.
   - **A single delta with BOTH `reasoning_content`/`content` AND `tool_calls`** (defensive; not observed but permitted by schema): process fields in fixed order thinking → text → tool_calls; each subsequent transition triggers close-open-block + start-new-block per rules 2-4.
5. On terminal frame (has `finish_reason` and/or `usage`):
   - Close any open block.
   - Validate any tool-call `args_buffer` parses as JSON — on failure emit `event: error` and abort.
   - Emit `event: message_delta` with `delta:{stop_reason:<mapped>, stop_sequence:null}` + `usage:{output_tokens:<from usage.completion_tokens>}`.
   - Emit `event: message_stop`.
   - Set `terminal_emitted=true`.
5.5. **On `[DONE]` sentinel when `terminal_emitted=false`** (defensive — protects against upstream infra closing the stream without a `finish_reason` frame):
   - Close any open block.
   - Emit `event: message_delta` with `delta:{stop_reason:"end_turn", stop_sequence:null}` + `usage:{output_tokens: len(seen_text_bytes)/4}` (rough estimate) OR the last known accumulator value if any partial `usage` was seen.
   - Emit `event: message_stop`.
   - Log WARNING `sse: DONE received without prior finish_reason frame; synthesised terminal events`.
6. `[DONE]` sentinel (post-terminal): close TCP; no additional Anthropic event.
7. Every ≥`ZB_PING_INTERVAL_S` (default 10s) of upstream silence: emit `event: ping`.

**Concurrency**: `sse_translator.SseTranslator` instances are per-request, stateful, and NOT reentrant. `bridge.py` allocates one per incoming `/v1/messages` stream request. No shared state across requests. Verified by T17b/S19 concurrency test.

**Buffer-and-retry mode** (default, `ZB_BUFFER_AND_RETRY=1`):
1. Open upstream stream, buffer all bytes into a `bytearray`.
2. Track `tail = last 256 bytes` for terminal-marker detection (`data: [DONE]`, `finish_reason`).
3. On successful capture (terminal seen), replay through `SseTranslator` and stream the resulting Anthropic events to the client atomically.
4. On mid-stream drop (upstream closes without terminal): discard buffer, wait `min(2**attempt, ZB_MAX_INLINE_WAIT_S)`, re-issue upstream. Cap at `ZB_STREAM_BUFFER_RETRIES=3`.
5. During capture: emit `event: ping` to client every `ZB_PING_INTERVAL_S` to keep connection alive.
6. If all retries exhausted: emit `event: error` with `type:"api_error", message:"wcb-bridge: upstream stream incomplete after retries"`, close.

**Passthrough mode** (`ZB_BUFFER_AND_RETRY=0`): stream translated frames directly through `SseTranslator` as they arrive from upstream; on mid-stream drop, close client stream with `event: error`.

#### 5.3.1 Worked example — thinking → text → tool_use → terminal

GLM SSE (abbreviated):
```
data: {"choices":[{"index":0,"delta":{"reasoning_content":"Let me "}}]}
data: {"choices":[{"index":0,"delta":{"reasoning_content":"think."}}]}
data: {"choices":[{"index":0,"delta":{"content":"OK, "}}]}
data: {"choices":[{"index":0,"delta":{"content":"here."}}]}
data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"lookup","arguments":""}}]}}]}
data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"q\":"}}]}}]}
data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"x\"}"}}}]}}]}
data: {"choices":[{"index":0,"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":10,"completion_tokens":42}}
data: [DONE]
```

Anthropic emit:
```
event: message_start
data: {"type":"message_start","message":{"id":"msg_...","type":"message","role":"assistant","model":"glm-5.3","content":[],"stop_reason":null,"usage":{"input_tokens":10,"output_tokens":0}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}
event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Let me "}}
event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"think."}}
event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: content_block_start
data: {"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}
event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"OK, "}}
event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"here."}}
event: content_block_stop
data: {"type":"content_block_stop","index":1}

event: content_block_start
data: {"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"call_1","name":"lookup","input":{}}}
event: content_block_delta
data: {"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{\"q\":"}}
event: content_block_delta
data: {"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"\"x\"}"}}
event: content_block_stop
data: {"type":"content_block_stop","index":2}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"output_tokens":42}}
event: message_stop
data: {"type":"message_stop"}
```

## 6. Error Mapping

| GLM biz code | HTTP | ErrorKind | Retriable | Anthropic-shape response | Action |
|---|---|---|---|---|---|
| 1000/1001/1003 | 401 | `OAUTH_TOKEN_INVALID` | No | `authentication_error`, 401 | Surface immediately. v1.1 multi-key: mark slot bad, failover. |
| 1301 | 400 | `WAF_BLOCKED` | No | `invalid_request_error`, 400, message includes hint about tool-syntax-in-text | Surface. Log a `warning` in stream tee. |
| 1302 | 429 | `TRANSIENT_THROTTLE` | Yes | `rate_limit_error`, 429 (only if retries exhausted) | Retry inline: 1s, 2s, 4s (cap `ZB_MAX_INLINE_RETRIES=3`, matching ccbridge). |
| 1305 | 429 | `OVERLOADED` | Yes | `overloaded_error`, 529 (only if retries exhausted) | Same backoff. 1305 storms observed in prior-art bridges — if telemetry shows exhaustion, raise retries to 4 in v1.1. |
| 1308-1321 | 429 | `SUBSCRIPTION_CAP` | No | `rate_limit_error`, 429 with header `zbridge-error-kind: subscription_cap` | Surface. v1.1 hook: pause harness (recovery.py). |
| 1113 | 429 | `BILLING_ERROR` | No | `permission_error`, 403 | Surface. |
| 1211 | 400 | `UNKNOWN_MODEL` | No | `not_found_error`, 404 | Surface. |
| 500 / 1234 | 500 | `UPSTREAM_5XX` | Yes | `api_error`, 500 (only if retries exhausted) | Retry with jitter. |
| Network/timeout | — | `UPSTREAM_5XX` | Yes | 502 | Retry. |
| `finish_reason: network_error` mid-stream | — | `UPSTREAM_5XX` | Yes (buffered mode re-issues) / No (passthrough closes) | Buffered: re-issue upstream. Passthrough: close with `event: error`. | |
| `finish_reason: model_context_window_exceeded` | — | `CONTEXT_TOO_LARGE` | No | `stop_reason:"max_tokens"` on stream / `invalid_request_error` 400 on non-stream | Surface. |
| Any unrecognised business code | Preserve HTTP | `UNKNOWN` | Follows HTTP class (5xx retriable, 4xx not) | Anthropic-shape `api_error` w/ passthrough message | Log at WARN with full body for triage. |

No `Retry-After` header from z.ai → we compute backoff ourselves. Base 1s, factor 2, jitter ±20%, cap 60s, cap total wait `ZB_MAX_INLINE_WAIT_S=30s` (matching ccbridge).

**Header forwarding on error path**: `zbridge-upstream-request-id`, `zbridge-upstream-x-ratelimit-*`, `retry-after` (if present), `zbridge-error-kind: <ErrorKind.value>` — same envelope pattern as ccbridge's `_build_error_response` (bridge.py:587-617).

## 7. Configuration

| Env var | Default | Purpose | ccbridge counterpart |
|---|---|---|---|
| `ZB_ZAI_API_KEY` | (required) | z.ai bearer token | (via OAuth in ccbridge) |
| `ZB_BRIDGE_SECRET` | (required; generated to `.bridge_secret` on first `--check`) | Local caller auth | `WCB_CC_BRIDGE_SECRET` |
| `ZB_HOST` | `127.0.0.1` | Bind host | `WCB_CC_HOST` |
| `ZB_PORT` | `8766` | Bind port | `WCB_CC_PORT` (8765) |
| `ZB_UPSTREAM_URL` | `https://api.z.ai/api/coding/paas/v4/chat/completions` | GLM endpoint | (Anthropic hard-coded) |
| `ZB_DEFAULT_MODEL` | `glm-5.3` | Fallback if model missing | — |
| `ZB_MODEL_ALIAS_JSON` | `{"claude-3-5-sonnet-latest":"glm-5.3","claude-3-opus-latest":"glm-5.3","claude-sonnet-4-5":"glm-5.3","claude-opus-4-8":"glm-5.3"}` | Flat alias map (no globs) | — |
| `ZB_STREAM_LOG_PATH` | unset | Enable JSONL tee if set | `WCB_CC_STREAM_LOG_PATH` |
| `ZB_BUFFER_AND_RETRY` | `1` | Buffer full stream + retry on drop | `WCB_CC_BUFFER_AND_RETRY` (also default 1) |
| `ZB_STREAM_BUFFER_RETRIES` | `3` | Buffer mode: mid-drop re-issue count | `WCB_CC_STREAM_BUFFER_RETRIES` |
| `ZB_READ_TIMEOUT_NONSTREAM_S` | `180` | httpx read timeout | (matches ccbridge default) |
| `ZB_READ_TIMEOUT_STREAM_S` | `600` | httpx stream read timeout | (matches ccbridge default) |
| `ZB_MAX_INLINE_RETRIES` | `3` | Retry count for retriable non-stream | `WCB_CC_MAX_INLINE_RETRIES` |
| `ZB_MAX_INLINE_WAIT_S` | `30` | Total wait ceiling | `WCB_CC_MAX_INLINE_WAIT` |
| `ZB_THINKING_SIG_KEY` | empty (see §5.2.1) | HMAC key for deterministic thinking signatures | — |
| `ZB_PRESERVE_THINKING_IN_CONTEXT` | `0` | Send prior thinking back to GLM | — |
| `ZB_LOG_LEVEL` | `INFO` | Python logging | `WCB_CC_LOG_LEVEL` |
| `ZB_PING_INTERVAL_S` | `10` | SSE keepalive | — |

## 8. Task Dependency Graph (revised — 25 tasks)

| Task | Depends On | Reason |
|---|---|---|
| T01 repo scaffolding | None | Foundation. |
| T02 pyproject + venv | T01 | Needs repo. |
| T03a synth fixtures (no key) | T02 | Unblocks all RED tests. |
| T03b real fixtures (needs key) | T02 | Blocks only T20. |
| T04 translate.py RED (req) | T02, T03a | TDD RED before impl. |
| T05 translate.py RED (resp) | T02, T03a | Parallel with T04. |
| T06 errors.py RED | T02, T03a | Parallel with T04/T05. |
| T07 sse_translator.py RED | T02, T03a | Parallel — includes ≥14 tests (see T07 spec). |
| T08 stream_tee.py RED | T02 | Parallel; schema only. |
| T09 credentials.py RED | T02 | Parallel; interface only. |
| T10 translate.py GREEN | T04, T05 | |
| T11 errors.py GREEN (+ header forwarder) | T06 | |
| T12a sse_translator.py GREEN (passthrough + state machine) | T07, T10, T11 | Core state machine. |
| T12b sse_translator.py GREEN (buffer-and-retry mode) | T12a | Adds capture + replay + ping. |
| T13 stream_tee.py GREEN | T08 | Independent. |
| T14 credentials.py GREEN | T09 | Independent. |
| T15a bridge.py FastAPI wiring (auth + non-stream + retry + hdr fwd) | T10, T11, T13, T14 | |
| T15b bridge.py streaming paths (passthrough + buffered) + tee integration | T12a, T12b, T15a | |
| T15c `/healthz` endpoint | T15a | Trivial add; separate for atomicity. |
| T16 __main__.py CLI | T15a | |
| T17a integration tests: nonstream + auth + retry | T15a | respx-backed. |
| T17b integration tests: stream (passthrough + buffered) + tool_roundtrip + concurrency | T15b | respx-backed. |
| T18 Dockerfile + build test | T15a, T15b, T15c | |
| T19 README + docs/RUNBOOK.md + docs/TRANSLATION.md + docs/ERRORS.md | T15b | Needs final interface. |
| T20a real-endpoint smoke (curl) | T17a, T17b, T18, T03b | Live GLM-5.3 call. |
| T20b harness parity + latency check | T20a | Trajectory diff vs ccbridge; TTFT + total-latency within 3× ccbridge baseline. |

## 9. Parallel Execution Graph (revised)

```
Wave 1:
  T01  repo scaffolding + .gitignore

Wave 2:
  T02  pyproject + venv + pinned deps

Wave 3 (fire in parallel — T03a unblocks RED tests):
  T03a synth fixtures (no key)
  T03b real fixtures (needs key; optional; blocks only T20a)

Wave 4 (fire ALL in parallel — RED tests):
  T04  translate.py request tests
  T05  translate.py response tests
  T06  errors.py classification tests
  T07  sse_translator.py state-machine tests (≥14)
  T08  stream_tee.py schema tests
  T09  credentials.py interface tests

Wave 5 (fire in parallel — GREEN impls except SSE):
  T10  translate.py
  T11  errors.py (+ header forwarder)
  T13  stream_tee.py
  T14  credentials.py

Wave 6:
  T12a sse_translator.py — passthrough + state machine

Wave 7:
  T12b sse_translator.py — buffer-and-retry mode

Wave 8:
  T15a bridge.py — FastAPI wiring (auth + non-stream + retry + hdr fwd)

Wave 9 (fire in parallel):
  T15b bridge.py — streaming paths + tee
  T15c /healthz
  T16  __main__.py CLI
  T17a integration: nonstream + auth + retry
  T19  README + docs (draft; finalise after T15b)

Wave 10 (fire in parallel):
  T17b integration: stream + tool_roundtrip + concurrency
  T18  Dockerfile + build test

Wave 11 (needs real key):
  T20a real-endpoint smoke

Wave 12:
  T20b harness parity + latency
```

Critical path: T01 → T02 → T03a → T07 → T12a → T12b → T15a → T15b → T17b → T20a → T20b.

## 10. Tasks (atomic, TDD-first, binary-verifiable)

| ID | Wave | Title | File(s) | Category | Skills | Verification | Depends |
|---|---|---|---|---|---|---|---|
| T01 | 1 | Repo scaffolding | dirs + `.gitignore` + empty `__init__.py`s | quick | (none) | `tree -a -L 2 /Users/apple/Desktop/zbridge` matches §4; `.gitignore` present | — |
| T02 | 2 | pyproject + venv + pinned deps | `pyproject.toml`, `.python-version` | quick | (none) | `uv sync` succeeds; `python -c "import fastapi,httpx,pydantic,respx"` exits 0 | T01 |
| T03a | 3 | Synth fixtures (no key) | `tests/fixtures/synth/*` | unspecified-high | (none) | ≥13 fixture files valid JSON/SSE (`jq . *.json`; `grep -c '^data: ' *.sse`); includes `done_without_terminal.sse`, `unicode_split.sse` | T02 |
| T03b | 3 | Real fixtures (needs key) | `tests/fixtures/real/*` (gitignored) | unspecified-high | (none) | ≥5 real fixtures recorded via one-off helper (text, thinking, tool, multi-tool, oversize); helper deleted after | T02 |
| T04 | 4 | RED: request translator | `tests/unit/test_translate_request.py` | unspecified-high | (none) | `pytest ...` shows ≥14 failing tests, one per §5.1 row including mixed-content and `tool_choice` rejection | T02, T03a |
| T05 | 4 | RED: response translator | `tests/unit/test_translate_response.py` | unspecified-high | (none) | ≥9 failing tests covering §5.2 including `signature=""` default and `msg_` prefix injection | T02, T03a |
| T06 | 4 | RED: error classification + header forwarder | `tests/unit/test_errors.py` | unspecified-high | (none) | ≥12 failing tests, one per §6 row + `zbridge-upstream-request-id` forwarding + `zbridge-error-kind` header | T02, T03a |
| T07 | 4 | RED: SSE state machine (≥14) | `tests/unit/test_sse_translator.py` | ultrabrain | (none) | Failing tests for: (1) worked example §5.3.1, (2) pure text, (3) pure thinking, (4) thinking-then-text, (5) text-then-thinking (defensive), (6) tool-only, (7) multi-tool interleaved, (8) `[DONE]` without terminal, (9) mid-stream drop → passthrough emits `event: error`, (10) mid-stream drop → buffered retries, (11) unicode multibyte split across chunks, (12) empty tool_use `input` at close, (13) `model_context_window_exceeded` maps stop_reason, (14) concurrent instance state isolation | T02, T03a |
| T08 | 4 | RED: stream tee schema | `tests/unit/test_stream_tee.py` | quick | (none) | Assert JSONL keys `{ts,seq,source,request_id,model,kind,event,delta}` | T02 |
| T09 | 4 | RED: credentials interface | `tests/unit/test_credentials.py` | quick | (none) | Assert `KeyProvider` protocol + `EnvKeyProvider.get()` reads `ZB_ZAI_API_KEY` | T02 |
| T10 | 5 | GREEN: translate.py | `zbridge/translate.py` | deep | (none) | T04+T05 all green | T04, T05 |
| T11 | 5 | GREEN: errors.py (+ header forwarder) | `zbridge/errors.py` | unspecified-high | (none) | T06 green | T06 |
| T13 | 5 | GREEN: stream_tee.py | `zbridge/stream_tee.py` | quick | (none) | T08 green; sample JSONL parseable by `jq` | T08 |
| T14 | 5 | GREEN: credentials.py | `zbridge/credentials.py` | quick | (none) | T09 green | T09 |
| T12a | 6 | GREEN: sse_translator.py — passthrough + state machine | `zbridge/sse_translator.py` | ultrabrain | debugging | T07 tests 1-9, 11-14 all green | T07, T10, T11 |
| T12b | 7 | GREEN: sse_translator.py — buffer-and-retry mode | `zbridge/sse_translator.py` (extend) | ultrabrain | debugging | T07 test 10 (buffered retry) green; injected respx mid-drop is invisible to client | T12a |
| T15a | 8 | GREEN: bridge.py — FastAPI + auth + non-stream + retry + hdr fwd | `zbridge/bridge.py` | deep | debugging | Hand-run `curl /v1/messages` (respx-mocked) returns valid Anthropic JSON with `zbridge-upstream-request-id`; missing secret returns 401 | T10, T11, T13, T14 |
| T15b | 9 | GREEN: bridge.py — streaming (passthrough + buffered) + tee | `zbridge/bridge.py` (extend) | deep | debugging | Both stream modes serve full worked example; tee JSONL emitted when `ZB_STREAM_LOG_PATH` set | T12a, T12b, T15a |
| T15c | 9 | `/healthz` endpoint | `zbridge/bridge.py` (extend) | quick | (none) | `curl /healthz` returns 200 with `{ok, upstream, model_default}` in <100ms | T15a |
| T16 | 9 | __main__.py CLI | `zbridge/__main__.py` | quick | (none) | `python -m zbridge --host 127.0.0.1 --port 8766` starts uvicorn; `--check` prints config summary and exits 0 | T15a |
| T17a | 9 | Integration: nonstream + auth + retry | `tests/integration/test_bridge_{nonstream,auth,retry}.py` | deep | (none) | `pytest -m integration -q -k "nonstream or auth or retry"` green | T15a |
| T17b | 10 | Integration: stream + tool_roundtrip + concurrency | `tests/integration/test_bridge_{stream_passthrough,stream_buffered,tool_roundtrip,concurrency}.py` | deep | (none) | `pytest -m integration -q -k "stream or tool_roundtrip or concurrency"` green; concurrency test proves 4 parallel requests don't cross state | T15b |
| T18 | 10 | Dockerfile + build | `Dockerfile.zbridge` | unspecified-high | (none) | `docker build -f Dockerfile.zbridge -t zbridge:dev .` succeeds; `docker run --rm -p 8766:8766 -e ZB_ZAI_API_KEY=x -e ZB_BRIDGE_SECRET=y zbridge:dev` serves `/healthz` in <5s | T15a, T15b, T15c |
| T19 | 9→10 | README + docs | `README.md`, `docs/*.md` | writing | (none) | README has full runnable curl (with headers + body + first SSE line); RUNBOOK has start/stop/health/tee-inspect/data-residency callout; TRANSLATION.md has bold "Silent deviations" (cache_control drop, tool_choice rejection, thinking signature key requirement) | T15b |
| T20a | 11 | Real-endpoint smoke (curl) | `tests/smoke/smoke_real_zai.py`, `scripts/curl_smoke.sh` | deep | debugging | With real `ZB_ZAI_API_KEY` + `ZB_BRIDGE_SECRET`: `bash scripts/curl_smoke.sh text` returns Anthropic-shape JSON; `stream-text`, `thinking`, `tool` variants exit 0; probe verifies z.ai does not gzip SSE | T17a, T17b, T18, T03b |
| T20b | 12 | Harness parity + latency | `tests/smoke/smoke_trajectory_parity.py` | deep | debugging | Harness pointed at `http://127.0.0.1:8766` completes one 5-10-tool-cycle task; emitted JSONL tee schema matches ccbridge (`jq -r 'keys' | sort -u` equal); content diff shows only expected model differences (not framing bugs); TTFT + total-latency within 3× ccbridge baseline | T20a |

### Delegation & skills breakdown

**T01/T02/T08/T09/T13/T14/T15c/T16** — Category `quick`. Skills: []. (No specialised skill needed; scoped, contained.)

**T03a/T03b/T04/T05/T06/T11/T18** — Category `unspecified-high`. Skills: [].

**T07/T12a/T12b** — Category `ultrabrain`. Skills: [`debugging`] on GREEN tasks only. (State machine complexity + edge-case interleavings + buffer/retry.)

**T10/T15a/T15b/T17a/T17b/T20a/T20b** — Category `deep`. Skills: [`debugging`] on GREEN + smoke tasks.

**T19** — Category `writing`. Skills: [].

Skills universally omitted with justification: `frontend-ui-ux`, `visual-qa`, `playwright` (no UI); `init-deep` (fresh single-package repo); `remove-ai-slops`, `review-work`, `security-research` (post-v1 pass).

## 11. Scenarios (the TDD contract — revised, 19 scenarios)

### S1 — Happy-path text turn (non-stream)
Input: user "say hi in one word", `max_tokens=32`, `stream=false`. Expected: 200 with `content[0].type=="text"`, `stop_reason=="end_turn"`, `usage.input_tokens>0`. Unit: `test_translate_response.py::test_text_only_maps_to_text_block`. Real: `bash scripts/curl_smoke.sh text | jq .content[0].type` → `"text"`.

### S2 — Happy-path streaming text turn
Input: same as S1 with `stream=true`. Expected event order: `message_start, content_block_start(text), content_block_delta(text_delta)+, content_block_stop, message_delta, message_stop`. Unit: `test_sse_translator.py::test_pure_text_stream`. Real: `bash scripts/curl_smoke.sh stream-text | grep -c '^event: '` returns ≥6.

### S3 — Thinking + text turn (stream)
Input: `stream=true`, `thinking={type:"enabled", budget_tokens:2048}`. Expected: index 0 thinking, index 1 text. Unit: `test_sse_translator.py::test_thinking_then_text`. Real: `bash scripts/curl_smoke.sh thinking | awk '/content_block_start/{print}'` → exactly two matches, thinking then text.

### S4 — Single tool_use turn (non-stream)
Input: user asks to call `lookup(q:string)`, `tools=[lookup]`, `tool_choice="auto"`. Expected: `stop_reason=="tool_use"`, one `tool_use` block with valid `input` JSON. Unit: `test_translate_response.py::test_single_tool_call`. Real: `bash scripts/curl_smoke.sh tool | jq '.content[]|select(.type=="tool_use")'` returns one object.

### S5 — Multi tool_use turn (stream)
Input: prompt for two parallel tool calls. Expected: two `tool_use` blocks distinct `id`s, indices 1 and 2 (thinking possibly at 0), assembled `input` JSON valid for both. Unit: `test_sse_translator.py::test_multi_tool_calls_assembles_by_index`. Real: `bash scripts/curl_smoke.sh multitool | jq '[.[]|select(.type=="content_block_start" and .content_block.type=="tool_use")]|length'` → 2.

### S6 — tool_result round-trip (multi-turn)
Input: turn 1 → assistant emits `tool_use id=call_a`; turn 2 → user sends `{type:"tool_result", tool_use_id:"call_a", content:"{\"ok\":true}"}`. Expected: outbound GLM request has `messages` ending in `{role:"tool", tool_call_id:"call_a", content:"..."}`. Unit: `test_translate_request.py::test_tool_result_becomes_role_tool`. Integration: `test_bridge_tool_roundtrip.py::test_two_turn_roundtrip`.

### S7 — Rate-limit 1302 inline retry (mock-only — intentional; real-surface would waste real quota to reproduce)
Input: respx returns `{"error":{"code":"1302"}}` twice then success. Expected: single 200 to client. Unit: `test_errors.py::test_1302_retryable`; integration `test_bridge_retry.py::test_1302_succeeds_after_retry`.

### S8 — Quota-exhausted 1308 surface (mock-only — intentional)
Input: respx returns `{"error":{"code":"1308"}}`. Expected: 429 body with `type:"rate_limit_error"`, header `zbridge-error-kind: subscription_cap`. Unit + integration.

### S9 — WAF 1301 surface (mock-only — intentional)
Input: respx returns `{"error":{"code":"1301","message":"content policy"}}`. Expected: 400 with `invalid_request_error`, hint about tool-syntax-in-text. Unit test.

### S10 — Mid-stream drop → passthrough emits `event: error`
Input: respx SSE emits partial thinking + partial text then closes mid-frame. `ZB_BUFFER_AND_RETRY=0`. Expected: `event: error` with `type:"api_error"`, prior blocks properly `content_block_stop`-ped. Unit: `test_sse_translator.py::test_mid_drop_passthrough_error`.

### S10b — Mid-stream drop → buffered retries and succeeds
Input: respx SSE mid-drops on attempt 1, delivers full stream on attempt 2. `ZB_BUFFER_AND_RETRY=1`. Expected: client receives ONE complete stream with the second attempt's content only; sees only `event: ping` frames during retry gap. Unit + integration `test_bridge_stream_buffered.py::test_transparent_retry`.

### S11 — Oversize context maps stop_reason
Input: respx SSE terminates with `finish_reason:"model_context_window_exceeded"`. Expected: `message_delta.stop_reason=="max_tokens"`, no `event: error`. Unit test.

### S12 — Auth gate
Input: no `x-zbridge-secret` or `x-api-key` header, or wrong value. Expected: 401 with Anthropic-shape `authentication_error`. Integration `test_bridge_auth.py::test_missing_or_wrong_secret_401`. Real: `curl -i http://127.0.0.1:8766/v1/messages` → `HTTP/1.1 401`.

### S13 — `[DONE]` without prior terminal frame (NEW)
Input: respx SSE emits partial text then `data: [DONE]` without a `finish_reason` frame. Expected: bridge emits synthetic `message_delta(stop_reason:"end_turn")` + `message_stop`; log line `sse: DONE received without prior finish_reason frame`. Unit: `test_sse_translator.py::test_done_without_terminal_synthesises_stop`.

### S14 — Forced-tool `tool_choice` rejection (NEW)
Input: request with `tool_choice={"type":"tool","name":"lookup"}` or `tool_choice="any"`. Expected: HTTP 400 `invalid_request_error`, message names GLM Coding Plan limitation. Unit: `test_translate_request.py::test_tool_choice_forced_rejected`; integration `test_bridge_nonstream.py::test_tool_choice_forced_400`.

### S15 — Mixed-content user message (text + tool_result) (NEW)
Input: user message with `content = [{type:"text",text:"context"},{type:"tool_result",tool_use_id:"call_a",content:"..."}]`. Expected: outbound messages ends with `[role:"tool" tool_call_id:"call_a", role:"user" content:"context"]` (tool first, then residual text as trailing user message). Unit: `test_translate_request.py::test_mixed_content_user_message_split_ordering`.

### S16 — Image input (base64 + URL) (NEW)
Input: user message with two image blocks (one base64 png, one https URL) + text. Expected: outbound GLM message has `content=[{type:"text",text:...},{type:"image_url",image_url:{url:"data:image/png;base64,..."}},{type:"image_url",image_url:{url:"https://..."}}]`. Unit: `test_translate_request.py::test_image_blocks_map_to_glm_multimodal`.

### S17 — Unicode multibyte split across two SSE chunks (NEW)
Input: respx SSE emits an emoji "🚀" split across two `data:` chunks such that the first ends with byte 0xF0 and the second starts with 0x9F 0x9A 0x80. Expected: emitted `text_delta` in the aggregated content is `"🚀"` (correctly reassembled). Unit: `test_sse_translator.py::test_unicode_multibyte_split`.

### S18 — Rejected unsupported fields (NEW)
Input: request with `n:2` OR `logprobs:true` OR `response_format:{type:"json_schema",json_schema:{...}}` OR `service_tier:"priority"`. Expected: HTTP 400 `invalid_request_error` per field with the field name in the message. Unit: `test_translate_request.py::test_unsupported_fields_rejected` (parametrised).

### S19 — Concurrency: 4 parallel requests on same bridge process (NEW)
Input: 4 async clients POST 4 different streaming requests. Expected: each response is complete and contains only its own request's content; no interleaving; no shared state leaks. Integration: `test_bridge_concurrency.py::test_four_parallel_streams_isolated`. Verifies `SseTranslator` instances are per-request-scoped.

## 12. Risks & Open Questions

- **Multi-turn `reasoning_content` drop (zai-org/GLM-4.5#100)**: bridge cannot fix upstream. Emit an empty `thinking` block only when GLM sends `reasoning_content`; do not synthesise. Documented in RUNBOOK.
- **WAF 1301 false-positives on tool-syntax-in-text**: mitigation via explicit hint in error message; scenario S9. Consider v1.1 `ZB_WAF_TEXT_SANITISER`.
- **GLM-5.3 Coding-Plan availability**: `ZB_MODEL_ALIAS_JSON` is authoritative; no hard-coded model id in source.
- **SSE state-machine complexity**: T07 requires ≥14 unit tests before T12a starts.
- **Cost tracking**: GLM Coding Plan does not per-request bill; `usage` reports tokens only. `/usage` aggregation deferred to v1.2.
- **Prompt caching**: pass `cached_tokens` through as `cache_read_input_tokens`; **`cache_control` markers silently dropped in v1** — bold callout in RUNBOOK + docs/TRANSLATION.md.
- **Data residency**: all traffic hits Chinese infrastructure. RUNBOOK includes a bold "do not send confidential data" callout.
- **Harness cache key collision on `message.id`**: if the trajectory harness dedupes turns by response ID, our `msg_`-prefixed passthrough of GLM's opaque `id` could collide across replays. **Open**: audit harness cache key derivation before shipping T20b; if it hashes on `id`, add a bridge-side salt (`ZB_MESSAGE_ID_SALT`).
- **`Accept-Encoding: gzip` + SSE**: httpx decodes gzip transparently, but T20a includes an explicit probe of whether z.ai gzips SSE (typically no) — if yes, verify streaming still works chunk-by-chunk.
- **`stream_options.include_usage` — DECIDED**: not forwarded. GLM terminal chunk always carries `usage`.
- **z.ai upstream endpoint rename / model deprecation**: `ZB_UPSTREAM_URL` + `ZB_MODEL_ALIAS_JSON` env-var-driven; no code change required to retarget.
- **`role="tool"` message ordering — DECIDED**: split N `tool_result` blocks into N `role="tool"` messages; mixed-content case per S15.
- **Trajectory replay determinism** (thinking signature): `ZB_THINKING_SIG_KEY` default empty (`signature=""`); set to a stable value for reproducible bytes. Warning logged on startup if key empty AND thinking enabled.
- **Does harness tolerate `event: error` mid-stream truncation?**: buffered-mode default (`ZB_BUFFER_AND_RETRY=1`) removes the risk of this being an issue by transparently retrying upstream; the client only sees complete streams. Passthrough mode is opt-in for callers that prefer incremental delivery over completeness.

## 13. Verification Plan

1. **Unit** — `pytest -q` runs all `tests/unit/*` offline; every §5 mapping row and §6 error row has ≥1 test.
2. **Integration** — `pytest -q -m integration` uses `respx` to mock `api.z.ai`. Covers §11 scenarios that don't need a live key (S1-S19 except the "real:" surface).
3. **Health** — `curl -s http://127.0.0.1:8766/healthz` returns `{"ok":true, "upstream":"api.z.ai", "model_default":"glm-5.3"}`.
4. **Real smoke** — `bash scripts/curl_smoke.sh {text|stream-text|thinking|tool|multitool}` all exit 0 against real z.ai. Requires `ZB_ZAI_API_KEY` + `ZB_BRIDGE_SECRET`. Includes a `--probe-encoding` mode that verifies z.ai does not gzip SSE.
5. **Trajectory schema parity** — harness pointed at `ANTHROPIC_API_BASE=http://127.0.0.1:8766` runs one 5-10-tool-cycle task; JSONL tee's `jq -r 'keys' | sort -u` matches a ccbridge-produced sample byte-for-byte.
6. **Trajectory content parity** — spot-diff the same trajectory's message shapes (`content_block` sequence, `stop_reason`, `usage` presence) against a ccbridge sample; only expected differences (model output text) should remain.
7. **Latency parity** — TTFT (time to first `content_block_delta`) and total-latency for a streaming turn are within **3× ccbridge's baseline** on the same prompt. Recorded in `tests/smoke/latency_report.json`.
8. **Docker** — `docker build && docker run` starts and passes health check.
9. **Secret hygiene** — `grep -R "sk-" .` and `git status --porcelain` after `git init && git add .` both show nothing sensitive tracked.

## 14. Rollout Plan

- **v1.0 (MVP)** — single API key, full translation (text/thinking/tools/images/mixed-content), streaming state machine, `[DONE]`-without-terminal recovery, **buffer-and-retry stream mode (default ON)**, inline retry, header forwarding, stream tee, `/healthz`, Docker.
- **v1.1** — `MultiKeyPoolProvider` in `credentials.py` with failover, `recovery.py` (pause-and-resume on `SUBSCRIPTION_CAP`), `/quota` polling endpoint if z.ai exposes one, `ZB_WAF_TEXT_SANITISER`.
- **v1.2** — `/metrics` (Prometheus), `/usage` (aggregated JSONL summary), prompt-cache `cache_control` passthrough (when GLM supports it).

## 15. Timeline (revised)

| Wave | Tasks | Est hours (single senior eng) |
|---|---|---|
| 1 | T01 | 0.5 |
| 2 | T02 | 1 |
| 3 | T03a, T03b | 3 |
| 4 | T04–T09 (parallel) | 10 |
| 5 | T10, T11, T13, T14 (parallel) | 8 |
| 6 | T12a | 8 |
| 7 | T12b | 4 |
| 8 | T15a | 4 |
| 9 | T15b, T15c, T16, T17a, T19 (parallel) | 8 |
| 10 | T17b, T18 (parallel) | 4 |
| 11 | T20a | 2 |
| 12 | T20b | 3 |
| **Total** | **25 tasks** | **~55h (~7 working days)** |

Bumped from 40h → 55h to reflect: (a) buffer-and-retry now in v1, (b) T12a estimate raised to 8h given the actual SSE edge cases, (c) added scenarios S13-S19, (d) task-splits for atomicity.

## 16. Commit Strategy (atomic)

- Init: `git init && git commit -m "chore: initial scaffolding" -m "T01"` — includes only §4 layout + `.gitignore`, no code.
- One commit per task, subject `<type>(scope): <what>`, body includes `Task: T##` and `Verification: <exact command that turned green>`. Types: `test` (RED), `feat` (GREEN), `docs`, `build`, `chore`.
- Rules: no commit contains both RED and GREEN for the same module. No commit mixes source and unrelated docs. No commit larger than the task it maps to. Pre-commit hook (`ruff` if configured) runs; no `--no-verify`.

## 17. Appendix — Deviations from ccbridge

Deliberately dropped from the port:

| Dropped | Why |
|---|---|
| OAuth token flow, refresh, `console.anthropic.com/v1/oauth/token` | z.ai uses static bearer tokens. |
| macOS Keychain reader, `~/.claude/.credentials.json` fallback | Not applicable. Static `ZB_ZAI_API_KEY`. |
| `fcntl.flock` cross-process refresh serialization | No refresh → no shared-state contention. |
| `MultiAccountCredentialProvider` slot exhaustion | Deferred to v1.1 (same interface). |
| Billing-attribution block injection | Anthropic-specific hack. |
| Tool-name prefix rename | z.ai accepts plain names. |
| `You are Claude Code` system prefix | Anthropic-specific quota gate. |
| `anthropic-beta: oauth-2025-04-20` header | Anthropic-specific. |
| 3rd-party fingerprint header stripping + `claude-cli/1.0.60` reinjection | Anthropic-specific. |
| `recovery.py` pause-and-resume | Deferred v1.1. |
| `thinking:{type:"adaptive"}` normalization | GLM only has enabled/disabled. |

Retained verbatim from ccbridge:
- Stream tee JSONL schema `{ts,seq,source,request_id,model,kind,event,delta}`.
- Env-var-driven config style (`ZB_*` mirrors `WCB_CC_*`).
- 127.0.0.1 default bind + secret gate.
- Timeout defaults (180s non-stream, 600s stream).
- CLI shape (`--host --port --check`).
- Pinned dependency versions in Dockerfile.
- **Buffer-and-retry stream mode as default** (ccbridge parity — load-bearing for trajectory correctness).
- Upstream request-id + rate-limit header forwarding pattern.
- Inline retry defaults (3 retries, 30s cap — matching ccbridge; not doubled).

### Silent v1 deviations (must be documented BOLD in docs/TRANSLATION.md and README)

1. **`cache_control` markers on system blocks / message content: silently dropped.** No cache tokens observed on the Anthropic side; cost tracking unaffected but harness authors should not expect `cache_creation_input_tokens` in `usage`.
2. **`tool_choice != "auto"`: REJECTED with 400.** Forced tool selection unavailable on GLM Coding Plan. If the harness relies on this, deferral or a different upstream is required.
3. **Thinking block `signature`: empty by default.** Set `ZB_THINKING_SIG_KEY` to a stable value for reproducible trajectory bytes.
4. **Unsupported request fields (`n>1`, `logprobs`, `seed`, `response_format=json_schema`, `service_tier`): REJECTED with 400.**

---

## Execution TODO (wave-by-wave)

Fire per Wave. Sub-agent invocation template:
`task(category="<per-table>", load_skills=<per-table>, run_in_background=<true if parallel>, prompt="Task T##: <what>. Verification: <exact QA>. Deps satisfied. Files: <paths>. Reference: /Users/apple/Desktop/ccbridge/claude_oauth/<file>.py for pattern.")`

### Wave 1
- **T01. Repo scaffolding + `.gitignore`** — Category `quick`. Skills: [].

### Wave 2
- **T02. pyproject + pinned deps + venv** — Category `quick`. Skills: [].

### Wave 3 (parallel)
- **T03a. Synth fixtures (no key)** — Category `unspecified-high`. Skills: [].
- **T03b. Real fixtures (needs key; optional)** — Category `unspecified-high`. Skills: [].

### Wave 4 (parallel — RED tests)
- **T04. RED: request translator** — Category `unspecified-high`. Skills: [].
- **T05. RED: response translator** — Category `unspecified-high`. Skills: [].
- **T06. RED: error classification + header forwarder** — Category `unspecified-high`. Skills: [].
- **T07. RED: SSE state machine (≥14)** — Category `ultrabrain`. Skills: [].
- **T08. RED: stream tee schema** — Category `quick`. Skills: [].
- **T09. RED: credentials interface** — Category `quick`. Skills: [].

### Wave 5 (parallel — GREEN except SSE)
- **T10. GREEN translate.py** — Category `deep`. Skills: [].
- **T11. GREEN errors.py + header forwarder** — Category `unspecified-high`. Skills: [].
- **T13. GREEN stream_tee.py** — Category `quick`. Skills: [].
- **T14. GREEN credentials.py** — Category `quick`. Skills: [].

### Wave 6
- **T12a. GREEN sse_translator — passthrough + state machine** — Category `ultrabrain`. Skills: [`debugging`].

### Wave 7
- **T12b. GREEN sse_translator — buffer-and-retry mode** — Category `ultrabrain`. Skills: [`debugging`].

### Wave 8
- **T15a. GREEN bridge.py — FastAPI wiring** — Category `deep`. Skills: [`debugging`].

### Wave 9 (parallel)
- **T15b. GREEN bridge.py — streaming paths + tee** — Category `deep`. Skills: [`debugging`].
- **T15c. `/healthz`** — Category `quick`. Skills: [].
- **T16. `__main__.py` CLI** — Category `quick`. Skills: [].
- **T17a. Integration: nonstream + auth + retry** — Category `deep`. Skills: [].
- **T19. README + docs** — Category `writing`. Skills: [].

### Wave 10 (parallel)
- **T17b. Integration: stream + tool_roundtrip + concurrency** — Category `deep`. Skills: [].
- **T18. Dockerfile + build** — Category `unspecified-high`. Skills: [].

### Wave 11
- **T20a. Real-endpoint smoke** — Category `deep`. Skills: [`debugging`].

### Wave 12
- **T20b. Harness parity + latency** — Category `deep`. Skills: [`debugging`].

### Execution Instructions
1. Never fire GREEN for a module before its RED task is complete AND red.
2. Every task ends with a git commit whose body cites the exact verification command.
3. If a wave's parallel task fails, do not proceed to the next wave — fix, re-verify, then advance.
4. Wave 11 (T20a) requires `ZB_ZAI_API_KEY`; if unavailable, stop after Wave 10 and stage T20a/T20b for a follow-up window.
