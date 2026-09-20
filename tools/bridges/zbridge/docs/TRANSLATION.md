# Translation Contract — Anthropic ⇄ GLM

Full row-by-row tables live in [PLAN.md §5.1–§5.3](../PLAN.md#5-translation-contract). This doc is the **frozen** short form for downstream consumers.

## Silent v1 deviations (READ THIS)

Callers should know these translation choices — they are silent, intentional, and not fixed in v1:

1. **`cache_control` markers on Anthropic `system` blocks are DROPPED.** z.ai's coding endpoint caches automatically, so caching still happens — you just cannot steer it. `usage` *does* report cache tokens: `cache_read_input_tokens` comes from upstream, while `cache_creation_input_tokens` is **modelled** by zbridge (z.ai reports no write counter). See [Usage / token accounting](#usage--token-accounting).
2. **`tool_choice != "auto"` REJECTED with HTTP 400.** GLM Coding Plan has no forced-tool knob today. Agents that require `tool_choice={"type":"tool","name":X}` cannot use zbridge as-is.
3. **Thinking block `signature` is EMPTY by default.** Set `ZB_THINKING_SIG_KEY` to a stable value for reproducible trajectory bytes. Ephemeral (per-process) keys break replay determinism.
4. **These request fields are REJECTED with HTTP 400:** `n>1`, `logprobs`, `seed`, `response_format={type:"json_schema"}`, `service_tier`.

## Request (Anthropic `/v1/messages` → GLM `/chat/completions`)

| Anthropic field | GLM field | Notes |
|---|---|---|
| `model` | `model` | Alias map — see `ZB_MODEL_ALIAS_JSON` |
| `messages[].role=user` (string) | `messages[].role=user` (string) | direct |
| `messages[].role=user` (block list) | `messages[].role=user` (multimodal or joined) | see PLAN.md for mixed cases |
| `messages[].role=user` w/ tool_result | `messages[].role=tool` + tool_call_id | one tool msg per result; residual text → trailing user msg |
| `messages[].role=assistant` w/ thinking | dropped (unless `ZB_PRESERVE_THINKING_IN_CONTEXT=1`) | GLM only accepts thinking on current-turn response |
| `messages[].role=assistant` w/ tool_use | assistant + `tool_calls[]` | tool_use.id → tool_calls[].id |
| `system` (string or list) | prepended `role=system` | `cache_control` dropped (deviation #1) |
| `stop_sequences[]` | `stop[]` (truncated to 4) | z.ai schema max=4 |
| `temperature` | `temperature` | direct (both use [0,1]) |
| `top_p` | `top_p` (clamped to [0.01, 1]) | |
| `top_k` | dropped | not supported by GLM |
| `tools[]` | `tools[]` with `type:"function"` wrapper | `input_schema` → `parameters` |
| `tool_choice` | REJECT unless `"auto"` or omitted | deviation #2 |
| `thinking:{type:enabled,budget_tokens:N}` | `thinking:{type:enabled}` + `reasoning_effort` | budget→effort gated by model family |
| `metadata.user_id` | `user_id` (if 6..128 chars) | length gate |

### reasoning_effort mapping

Gated by model family — GLM-5.x rejects `medium`:

| Model family | <2k | <4k | <8k | ≥8k |
|---|---|---|---|---|
| GLM-5.x | `low` | `low` | `high` | `max` |
| GLM-4.x | `low` | `medium` | `high` | `max` |

## Response (GLM non-stream → Anthropic Message)

| GLM | Anthropic |
|---|---|
| `id` | `id` (prefixed `msg_` if missing) |
| `model` | `model` |
| `message.reasoning_content` | `content[].type="thinking"` + signature |
| `message.content` (str) | `content[].type="text"` |
| `message.tool_calls[]` | `content[].type="tool_use"` (input parsed JSON) |
| `finish_reason:"stop"` | `stop_reason:"end_turn"` |
| `finish_reason:"tool_calls"` | `stop_reason:"tool_use"` |
| `finish_reason:"length"` | `stop_reason:"max_tokens"` |
| `finish_reason:"model_context_window_exceeded"` | `stop_reason:"max_tokens"` |
| `usage.completion_tokens` | `usage.output_tokens` |
| `usage.prompt_tokens`, `usage.prompt_tokens_details.cached_tokens` | `usage.input_tokens`, `usage.cache_read_input_tokens`, `usage.cache_creation_input_tokens` — **not** a field-for-field copy, see [Usage / token accounting](#usage--token-accounting) |

## Usage / token accounting

The two APIs disagree about cached tokens, and this is the one place where a naive
field-for-field mapping silently corrupts every downstream token count and cost figure:

| | meaning of the prompt total |
|---|---|
| OpenAI / z.ai | `prompt_tokens` **INCLUDES** `prompt_tokens_details.cached_tokens` |
| Anthropic | `input_tokens` **EXCLUDES** `cache_read_input_tokens` |

Verified live against `api.z.ai` by `scripts/probe_usage.py`: a byte-identical prompt
reports `prompt_tokens=5382` on both a forced cache miss (`cached_tokens=0`) and a cache
hit (`cached_tokens=5376`). The prompt never changed, so the cached portion is inside
`prompt_tokens`.

So zbridge guarantees this invariant instead of copying fields:

```
input_tokens + cache_read_input_tokens + cache_creation_input_tokens == prompt_tokens
```

Consumers that reconstruct the prompt total by summing those three — LiteLLM's Anthropic
transform does exactly this — would otherwise count every cached token twice. On a
164-turn GLM-5.3 trajectory that inflated reported input from 13.6M to 27.1M tokens and
cost from ~$10 to $77.

### Cache-write attribution

z.ai reports **no cache-write counter** — the only cache key in its usage object is
`prompt_tokens_details.cached_tokens`. Cache creation is therefore *modelled*, not
measured, controlled by `ZB_CACHE_WRITE_ATTRIBUTION`:

| value | behaviour |
|---|---|
| `block` (default) | Fresh tokens filling a whole cache block are reported as `cache_creation_input_tokens`; the trailing partial block stays in `input_tokens`. |
| `none` | `cache_creation_input_tokens` is always 0; the whole fresh remainder is `input_tokens`. |

`block` rests on a measured property of z.ai's cache: it stores the prompt in
**64-token blocks** and never caches the trailing partial block. Across a 164-turn
trajectory every observed `cached_tokens` value is divisible by 64 (gcd exactly 64), and
`floor((prompt_tokens - 1) / 64) * 64` predicts the *next* turn's `cached_tokens` on
156/163 turns. Block size is overridable via `ZB_CACHE_BLOCK_TOKENS`.

Because the block-aligned cacheable prefix is at most `prompt_tokens - 1`,
**`input_tokens` is never driven to 0** by this split.

Deviations worth knowing:

- `cache_creation_input_tokens` is a *model*, not upstream ground truth. Where the model
  disagreed with observation it under-reported writes (z.ai occasionally caches a few
  blocks more than the previous prompt), so treat it as a lower bound.
- Cache fields are emitted only when upstream actually reports `cached_tokens`. A
  provider silent about caching is never given invented numbers.
- `completion_tokens_details.reasoning_tokens` has no Anthropic equivalent and is
  dropped; reasoning tokens are already inside `completion_tokens` → `output_tokens`.

## SSE stream translation

GLM emits flat `data: {choices:[{delta:{content|reasoning_content|tool_calls}}]}` frames plus a terminal frame with `finish_reason` and `usage`, followed by `data: [DONE]`.

zbridge emits the Anthropic event sequence:
```
message_start
  content_block_start (thinking) / delta+ / content_block_stop
  content_block_start (text)     / delta+ / content_block_stop
  content_block_start (tool_use) / delta+ / content_block_stop
message_delta
message_stop
```

**`[DONE]` without a prior terminal frame** → zbridge synthesises `message_delta(stop_reason=end_turn)` + `message_stop` so the client sees a complete stream.

**Mid-stream drop:**
- Buffered mode (default): full upstream capture + transparent retry, client sees ping keepalives during retry, then a complete stream.
- Passthrough mode: closes with `event: error`.

## Error mapping

| GLM code | HTTP | Anthropic error type | Retriable |
|---|---|---|---|
| 1000/1001/1003 | 401 | authentication_error | ✗ |
| 1301 (WAF) | 400 | invalid_request_error | ✗ |
| 1302 | 429 | rate_limit_error | ✓ (inline retry) |
| 1305 | 429 | overloaded_error | ✓ |
| 1308–1321 | 429 | rate_limit_error (+`zbridge-error-kind: subscription_cap`) | ✗ |
| 1113 | 403 | permission_error | ✗ |
| 1211 | 404 | not_found_error | ✗ |
| 500 / 1234 | 500 | api_error | ✓ |

## Upstream header forwarding

All `x-request-id`, `x-ratelimit-*`, and `retry-after` headers z.ai returns get forwarded as `zbridge-upstream-*` (or in the case of `retry-after`, forwarded unprefixed for client back-off compatibility). Error responses also carry `zbridge-error-kind: <ErrorKind>`.
