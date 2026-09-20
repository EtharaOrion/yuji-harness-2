# Architecture

See [PLAN.md §3](../PLAN.md#3-architecture) for the definitive design + [PLAN.md §3.0](../PLAN.md#30-alternatives-considered-and-why-rejected) for rejected alternatives.

## Module map

| Module | Role | Reference |
|---|---|---|
| `zbridge/bridge.py` | FastAPI app: auth gate, `/v1/messages`, `/healthz`, retry loop, streaming modes | ccbridge/bridge.py |
| `zbridge/translate.py` | Pure Anthropic↔GLM translation (request-out, response-in) | NEW |
| `zbridge/sse_translator.py` | Stateful GLM SSE → Anthropic SSE state machine | NEW |
| `zbridge/errors.py` | GLM error classification + Anthropic envelope + header forwarding | ccbridge/errors.py |
| `zbridge/credentials.py` | KeyProvider protocol + EnvKeyProvider | ccbridge/credentials.py (single-key subset) |
| `zbridge/stream_tee.py` | JSONL trajectory tee (schema-parity with ccbridge) | ccbridge/stream_tee.py |
| `zbridge/__main__.py` | CLI: `--host --port --check` | ccbridge/__main__.py |

## Data flow

```
harness / litellm / aider  (Anthropic SDK client)
                     │
                     │ POST /v1/messages   (Anthropic JSON, optional SSE)
                     ▼
     ┌───────────────────────────────────────────────────────────┐
     │  zbridge :8766                                            │
     │                                                           │
     │  bridge.py                                                │
     │    ├─ ZB_BRIDGE_SECRET gate                               │
     │    ├─ translate.anthropic_to_glm_request                  │
     │    ├─ httpx.AsyncClient → z.ai coding plan                │
     │    ├─ non-stream: translate.glm_to_anthropic_response     │
     │    ├─ stream (passthrough): SseTranslator chunk-by-chunk  │
     │    ├─ stream (buffered): capture+retry, replay atomically │
     │    ├─ errors.classify_glm_error + retry inline            │
     │    └─ stream_tee.StreamTee (JSONL tap)                    │
     │                                                           │
     │  credentials.EnvKeyProvider (v1)  |  MultiKeyPool (v1.1) │
     └───────────────────────────────────────────────────────────┘
                     │
                     ▼
       https://api.z.ai/api/coding/paas/v4/chat/completions
                (GLM-5.3, Bearer ZB_ZAI_API_KEY)
```

## Threading / concurrency

- FastAPI runs on uvicorn's async event loop; every `/v1/messages` handler is `async`.
- `SseTranslator` is **stateful and per-request** — one instance per streaming call, never reused across requests. Verified by `tests/integration/test_bridge_concurrency.py`.
- `StreamTee` has an internal `threading.Lock` for file writes; safe to instantiate per-request.
- `httpx.AsyncClient` is shared across the process — thread-safe for concurrent use.

## Extension points (v1.1+)

- **`credentials.MultiKeyPoolProvider`**: drop-in replacement for `EnvKeyProvider`; pool spec via `ZB_ACCOUNT_POOL` env var.
- **`recovery.py`**: pause-and-resume on `SUBSCRIPTION_CAP` (mirrors ccbridge/recovery.py).
- **`/quota`**: expose z.ai's quota status if the upstream publishes it.
- **`/metrics`** (Prometheus): request rate, retry count, latency histograms.
