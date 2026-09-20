# zbridge — Runbook

## Start / stop

**Local dev:**
```bash
export ZB_ZAI_API_KEY=your-zai-key
export ZB_BRIDGE_SECRET=your-local-secret
export ZB_THINKING_SIG_KEY=your-stable-hmac-key   # for reproducible trajectory bytes
python -m zbridge --host 127.0.0.1 --port 8766
```

**Docker:**
```bash
docker build -f Dockerfile.zbridge -t zbridge:dev .
docker run --rm -p 8766:8766 \
    -e ZB_ZAI_API_KEY=$ZB_ZAI_API_KEY \
    -e ZB_BRIDGE_SECRET=$ZB_BRIDGE_SECRET \
    -e ZB_THINKING_SIG_KEY=$ZB_THINKING_SIG_KEY \
    zbridge:dev
```

Stop: `Ctrl+C` locally; `docker stop <container>` or `docker kill` for the container.

## Health

```bash
curl -sS http://127.0.0.1:8766/healthz | jq
# {"ok": true, "upstream": "api.z.ai/api/coding/paas/v4/chat/completions",
#  "model_default": "glm-5.3", "stream_mode": "buffered"}
```

## Environment variables

See [PLAN.md §7](../PLAN.md#7-configuration) for the full table. Highlights:

| Var | Default | Notes |
|---|---|---|
| `ZB_ZAI_API_KEY` | (required) | z.ai bearer |
| `ZB_BRIDGE_SECRET` | (required) | Local-caller auth |
| `ZB_UPSTREAM_URL` | `api.z.ai/api/coding/paas/v4/chat/completions` | Change to point at other z.ai plans / mirrors |
| `ZB_DEFAULT_MODEL` | `glm-5.3` | Any GLM id |
| `ZB_MODEL_ALIAS_JSON` | (built-in for claude-*) | Flat JSON dict |
| `ZB_BUFFER_AND_RETRY` | `1` | 0 = passthrough |
| `ZB_STREAM_LOG_PATH` | (unset) | Enable JSONL tee |
| `ZB_THINKING_SIG_KEY` | empty | Set for reproducible trajectory bytes |
| `ZB_CACHE_WRITE_ATTRIBUTION` | `block` | `block` = model `cache_creation_input_tokens` from z.ai's 64-token cache granularity; `none` = always report 0. z.ai exposes no cache-write counter, so this is modelled. See [TRANSLATION.md](./TRANSLATION.md#usage--token-accounting) |
| `ZB_CACHE_BLOCK_TOKENS` | `64` | Cache block granularity used by `block` attribution |

## Point clients at zbridge

```bash
export ANTHROPIC_API_BASE=http://127.0.0.1:8766
export ANTHROPIC_API_KEY=$ZB_BRIDGE_SECRET
# Anthropic SDK / litellm / aider / your harness — no code change needed
```

## Inspect the stream tee

If `ZB_STREAM_LOG_PATH=/tmp/trajectory.jsonl` was set:

```bash
tail -f /tmp/trajectory.jsonl | jq
jq -r '.event' /tmp/trajectory.jsonl | sort | uniq -c
```

Schema per line:
```
{ts, seq, source, request_id, model, kind, event, delta}
```
Byte-identical to ccbridge's [stream_tee.py](../../ccbridge/claude_oauth/stream_tee.py).

## Test suite

```bash
# Unit
.venv/bin/python -m pytest tests/unit/ -q

# Integration (respx-mocked upstream — no real API call)
.venv/bin/python -m pytest tests/integration/ -q -m integration

# Real-endpoint smoke (needs ZB_ZAI_API_KEY)
.venv/bin/python -m pytest tests/smoke/ -q -m smoke
```

## Data residency

**All traffic hits z.ai's Chinese infrastructure.** Do not send:
- Confidential company data
- Regulated data (HIPAA / PCI / GDPR-sensitive PII)
- Non-public source code you don't have data-egress approval for

Use ccbridge (Anthropic OAuth) or another provider for those workloads.

## Known upstream quirks

- **`reasoning_content` may be dropped on multi-turn responses** — [zai-org/GLM-4.5#100](https://github.com/zai-org/GLM-4.5/issues/100). We surface whatever GLM returns; no synthesis.
- **WAF 1301 false-positives on tool-call syntax in message text.** Keep function-call syntax inside the structured `tools[]` field.
- **`tool_choice != "auto"` REJECTED with 400** — GLM Coding Plan has no forced-tool knob today.
- **`cache_control` markers on Anthropic system blocks are silently dropped in v1.**
- **`n>1`, `logprobs`, `seed`, `response_format=json_schema`, `service_tier` REJECTED with 400.**

## Common failure modes

| Symptom | Diagnosis | Fix |
|---|---|---|
| `401 authentication_error` | Missing/wrong `x-zbridge-secret` | Check `ZB_BRIDGE_SECRET` on both sides |
| `500 authentication_error` (in body) | `ZB_ZAI_API_KEY` unset | Export it |
| `429 rate_limit_error` + `zbridge-error-kind: subscription_cap` | Coding Plan quota exhausted | Wait for reset; upgrade plan |
| `400 invalid_request_error` + hint about tool syntax | WAF 1301 | Move code-like text out of user message content |
| `502 api_error` + `zbridge: upstream stream incomplete after retries` | z.ai kept dropping mid-stream after buffer retries | Retry the request; check z.ai status |
