# ccbridge: Claude subscription as an Anthropic endpoint

A local FastAPI proxy (`claude_oauth`) that exposes an Anthropic-compatible
`/v1/messages` on `127.0.0.1:8765` and forwards every request to
`api.anthropic.com` signed with the **Claude Code OAuth login already on this
machine**. There is no API key. It is how the OpenHands agent
(`tools/openhands_agent`) reaches Claude from inside a task container.

Vendored from the reference harness's `ccbridge/claude_oauth`. The changes are
listed at the bottom of this file.

## How a request becomes a subscription request

1. **Auth in.** The caller presents the bridge's shared secret as its API key
   (`x-api-key`, `Authorization: Bearer`, or `x-ccbridge-secret`). Anything
   else gets a 401. The secret is what stops any local process or container
   from spending the subscription.
2. **Credentials.** In this order:
   - `CLAUDE_CODE_CREDENTIALS` (inline JSON) or `CCBRIDGE_CREDS_PATH`;
   - `CLAUDE_CODE_OAUTH_TOKEN`, a bare token such as `claude setup-token`
     prints. It is what the harness's `.env` carries and what `run_task.sh`
     checks for, and it cannot be refreshed;
   - the `claude` CLI's own stores: the macOS Keychain entry
     `Claude Code-credentials` or `~/.claude/.credentials.json`, whichever
     expires last.
3. **Headers.** The inbound key, `user-agent`, `x-app` and every
   `x-stainless-*` header are stripped. The bridge sends
   `Authorization: Bearer <oauth token>`, `anthropic-beta: oauth-2025-04-20`,
   and the Claude CLI's user agent. Anthropic routes OAuth traffic by that
   fingerprint, and a third-party one draws from "extra usage" instead of the
   plan.
4. **Body.** It makes these changes to `POST /v1/messages`:
   - It puts `You are Claude Code, Anthropic's official CLI for Claude.` at
     the start of `system`. OAuth requests without it are rejected.
   - It moves the caller's own system prompt into the first user message and
     puts a billing-attribution block in `system[0]`, so the request bills to
     the plan.
   - It prefixes every tool name with `mcp__ccb__` and strips the prefix from
     the response. Seven or more bare lowercase tool names trip the OAuth
     validator.
   - It passes `thinking: {type: adaptive}` and `output_config` through
     unchanged, including the caller's `display` choice. The reference rewrote
     them into the `enabled` + `budget_tokens` form, which `claude-opus-5`
     answers with empty thinking text. `CCBRIDGE_ADAPTIVE_THINKING=convert`
     restores the rewrite. An `enabled` budget is kept below `max_tokens`.
5. **Resilience.**
   - It retries 429/529/5xx errors inline when the wait is short.
   - It classifies a 429 as a subscription cap or a transient throttle.
   - It buffers streamed responses and re-sends them if the upstream drops
     mid-stream.
   - With `CCBRIDGE_ACCOUNT_POOL`, it fails over to another account.

## Running it

`scripts/run_task.sh` runs it for `AGENT=openhands`, the default, and hands
the agent `OPENHANDS_LLM_BASE_URL=http://host.docker.internal:<port>` and
`OPENHANDS_LLM_API_KEY=<secret>`. The lifecycle is described below.

To run it by hand:

```bash
cd tools/bridges/ccbridge
export CCBRIDGE_SECRET="$(tr -d '[:space:]' < .bridge_secret)"
uv run python -m claude_oauth --check                     # credentials load? spends nothing
uv run python -m claude_oauth --host 127.0.0.1 --port 8765
```

### Lifecycle

**One bridge per run** (the default). The harbor stage of `run_task.sh`:
1. picks a free port and a fresh secret, which is never written to disk;
2. proves the Claude login with `python -m claude_oauth --check`, which
   fails in seconds with the reason on a machine with no usable login;
3. starts the bridge from this checkout's venv and makes one `max_tokens: 1`
   call through it;
4. stops it when the stage exits: normally, on an error, or on Ctrl-C (an
   `EXIT` trap).

Nothing outlives the run, so there is no stale bridge on old code, no port
clash, and no secret that another checkout has to match. Concurrent runs each
get their own bridge, and each run's squid config opens only its own port.
Logs go to `logs/run-<job>-<pid>.log`.

**One shared bridge** (`CCBRIDGE_SHARED=1`): the older arrangement.
- It runs on `CCBRIDGE_PORT` (8765), with the secret in `.bridge_secret`.
- It is started on demand, stays up across runs, and is stopped only by
  `bash scripts/stop_harness.sh`.
- `run_task.sh` records the fingerprint of the code it started and warns when
  the running bridge is older than the checkout.
- Logs go to `logs/ccbridge.log`.

When it fails on another machine, check in this order:

1. No Claude login there: `run_task.sh` says so. Run `claude login`, or put
   `CLAUDE_CODE_OAUTH_TOKEN` in `.env`.
2. No `uv` on PATH: the bridge runs from its own uv project, and `uv sync`
   rebuilds a venv copied from another machine.
3. Linux: containers reach the host through the docker bridge gateway, so the
   bridge binds `0.0.0.0` there. A host firewall (ufw, firewalld) must allow
   the docker subnet to reach the host.
4. Shared mode only: port 8765 is taken, or a bridge started with a different
   secret is running. `run_task.sh` reports the mismatch.

| Variable | Default | Meaning |
|---|---|---|
| `CCBRIDGE_SHARED` | `0` | `1` = one long-lived bridge instead of one per run |
| `CCBRIDGE_PORT` | `8765` | shared mode's port (a per-run bridge picks a free one) |
| `CCBRIDGE_HOST` | `127.0.0.1` on macOS, `0.0.0.0` on Linux | bind address (see below) |
| `CCBRIDGE_SECRET` | `.bridge_secret` | shared mode's secret (a per-run bridge makes its own) |
| `CCBRIDGE_READ_TIMEOUT` / `CCBRIDGE_REQUEST_TIMEOUT` | 1800 via run_task.sh | non-streaming timeouts; the OpenHands SDK does not stream |
| `CCBRIDGE_ACCOUNT_POOL` | unset | colon-separated credential files / `keychain:<service>` for failover |
| `CCBRIDGE_ADAPTIVE_THINKING` | `passthrough` | `convert` restores the reference's adaptive -> enabled rewrite |
| `CCBRIDGE_DEBUG_LOG_BODY` | `0` | `1` logs request shape and dumps the body to `CCBRIDGE_BODY_DUMP_DIR` |

**Bind address.** Docker Desktop and OrbStack deliver `host.docker.internal`
to the host's loopback, so `127.0.0.1` is reachable from containers and from
nothing else. Plain Linux docker delivers it to the docker bridge gateway,
where a loopback listener cannot be reached, so on Linux the bridge binds
`0.0.0.0`. The secret is then the only thing guarding it, so keep the port
firewalled from the network.

**Under network isolation** the agent has no route of its own.
`tools/network/egress-proxy/squid-ccbridge.conf` adds exactly one destination to
squid's allowlist: `host.docker.internal` on the bridge's port. The squid
access log records every model call, and the internet audit is told
(`--proxy-allow host.docker.internal`) that these calls are the model, not a
breach.

## Caveats

- Every call spends the Claude **subscription** the host is logged into, with
  its 5-hour and weekly caps. A capped account returns 429; the agent reports
  it as `ApiUsageLimitError`.
- **Token refresh rotates the refresh token.** The `claude` CLI on the same
  machine holds that token too. The bridge first re-reads the credential
  stores and uses a live token if one is there. It refreshes only when every
  store has expired, and then writes the result to
  `~/.cache/yuji-ccbridge/claude_creds.json`, never to the Keychain. If the
  bridge had to refresh, the CLI may need `claude login` afterwards.
  `run_task.sh` refuses to start a run on an already-expired token for the
  same reason.
- The OpenHands agent asks for readable (`summarized`) thinking. On a toy
  task, echoed `summarized` blocks drew `stop_reason: "refusal"` (empty
  content) in 2 of 4 runs; on the real task measured, none did (see
  `tools/openhands_agent/README.md`, "Thinking"). The bridge does not work
  around refusals: it forwards whatever the caller asked for.

## Changes from the reference

- Adaptive thinking is passed through, with no rewrite and no forced `display`.
  Measured 2026-09-22 on the OAuth path, `claude-opus-5`, with and without tools:
  `adaptive` + `display: summarized` returns readable thinking, while the
  reference's `enabled` + `budget_tokens` rewrite returns it empty. The reference
  believed Anthropic-direct rejects `adaptive`; it accepts it today, alongside the
  billing-attribution and tool-rename transforms that fixed the 400 the rewrite
  was blamed for. `CCBRIDGE_ADAPTIVE_THINKING=convert` restores the old behaviour.
- `clamp_thinking_budget`: an `enabled` request with no budget gets 32000, and
  Anthropic refuses a budget that is not below `max_tokens` (OpenHands asks for
  16384 by default). The budget is halved against `max_tokens`, and thinking is
  dropped when even the 1024 floor does not fit.
- Credential selection: explicit overrides still win. Otherwise the
  **freshest** token across the stores is used, and the stores are re-read
  before any refresh (see Caveats).
- `CLAUDE_CODE_OAUTH_TOKEN` is accepted, so a machine with only a
  `claude setup-token` token in `.env` works; `--check` proves the login
  before a run starts.
- Names are this project's own: `CCBRIDGE_*` variables, `X-CCBridge-*`
  headers, the `mcp__ccb__` tool prefix, and the `~/.cache/yuji-ccbridge`
  refresh cache. Only the modules the bridge imports are kept.
