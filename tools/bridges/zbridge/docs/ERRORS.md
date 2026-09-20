# Error Taxonomy

Source: [PLAN.md §6](../PLAN.md#6-error-mapping). Implementation: [zbridge/errors.py](../zbridge/errors.py).

## ErrorKind enum

```python
class ErrorKind(str, Enum):
    OAUTH_TOKEN_INVALID   # 401
    WAF_BLOCKED           # 400 (safety filter — 1301)
    TRANSIENT_THROTTLE    # 429 retriable (1302)
    OVERLOADED            # 429 retriable (1305)
    SUBSCRIPTION_CAP      # 429 non-retriable (1308-1321)
    BILLING_ERROR         # 403 (1113)
    UNKNOWN_MODEL         # 404 (1211)
    UPSTREAM_5XX          # 500 retriable
    CONTEXT_TOO_LARGE     # (stream-only, mapped via stop_reason)
    UNKNOWN               # fallback
```

Properties:
- `is_retryable` — `TRANSIENT_THROTTLE`, `OVERLOADED`, `UPSTREAM_5XX`
- `is_account_problem` — `OAUTH_TOKEN_INVALID`, `SUBSCRIPTION_CAP`, `BILLING_ERROR` (feeds v1.1 multi-key failover)

## GLM business code → ErrorKind

| Code | HTTP (from z.ai) | ErrorKind | Retriable | Anthropic type |
|---|---|---|---|---|
| 1000 / 1001 / 1003 | 401 | OAUTH_TOKEN_INVALID | ✗ | authentication_error |
| 1113 | 429 | BILLING_ERROR | ✗ | permission_error |
| 1211 | 400 | UNKNOWN_MODEL | ✗ | not_found_error |
| 1301 | 400 | WAF_BLOCKED | ✗ | invalid_request_error |
| 1302 | 429 | TRANSIENT_THROTTLE | ✓ | rate_limit_error |
| 1305 | 429 | OVERLOADED | ✓ | overloaded_error |
| 1308–1321 | 429 | SUBSCRIPTION_CAP | ✗ | rate_limit_error |
| 500 / 1234 | 500 | UPSTREAM_5XX | ✓ | api_error |
| (any other) | passthrough | UNKNOWN | HTTP-class default | api_error |

## Client-facing envelope

```json
{
  "type": "error",
  "error": {
    "type": "<anthropic_error_type>",
    "message": "<human-readable message; WAF adds hint about tool-syntax-in-text>"
  }
}
```

Response headers:
- `zbridge-error-kind: <ErrorKind.value>` — always on non-2xx
- `zbridge-upstream-x-request-id: <z.ai id>` — when upstream provided one
- `zbridge-upstream-x-ratelimit-*` — forwarded when present
- `retry-after: <N>` — when classified error carries a hint

## Retry semantics

Inline retries follow `min(cap, base * 2^attempt) ± 20% jitter`, capped at:
- `ZB_MAX_INLINE_RETRIES=3` attempts
- `ZB_MAX_INLINE_WAIT_S=30` total wait

Buffered streaming retries (mid-stream drops) follow:
- `ZB_STREAM_BUFFER_RETRIES=3` attempts

z.ai does not emit `Retry-After` — we compute our own backoff.
