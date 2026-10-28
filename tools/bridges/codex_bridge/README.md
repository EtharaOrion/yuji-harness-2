# codex-bridge

Dependency-free Python bridge that exposes an Anthropic-compatible
`/v1/messages` endpoint backed by the operator's `codex` login. Every request
becomes one `codex exec --json` process; callers never hold the Codex account
credentials, only the per-process keys this bridge mints.

Mirrors `kiro_bridge/` in shape and HTTP contract so the same callers (the
harness launcher, `regrade_output.sh`, Reward Kit, LiteLLM, OpenHands) can
swap one for the other.

## Install

```bash
python -m pip install .
codex-bridge doctor --json
codex-bridge models
```

## Python API

```python
from codex_bridge import AnthropicServer, shared_bridge

with AnthropicServer(shared_bridge(), host="127.0.0.1", port=0) as server:
    agent_key = server.add_key("agent")                     # serves whatever model the request names
    judge_key = server.add_key("judge", "gpt-5.6-sol")      # pinned
    print(server.base_url)
    ...
```

## Standalone server

```bash
codex-bridge serve --host 127.0.0.1 --port 0 --pin-model gpt-5.6-sol
```

Prints `{host, port, agent_key, judge_key, judge_model}` as JSON on stdout and
runs until SIGINT.

## Environment

| Variable | Default | Meaning |
|---|---|---|
| `CODEX_BIN` | `codex` on PATH | Path to the Codex CLI |
| `CODEX_JUDGE_MODEL` | `gpt-5.6-sol` | Model pinned on the judge key |
| `CODEX_REASONING_EFFORT` | request's own | `none\|minimal\|low\|medium\|high\|xhigh\|max` (gpt-5.6-sol rejects `minimal`) |
| `CODEX_BRIDGE_CONCURRENCY` | `8` | Max parallel `codex exec` processes |
| `CODEX_RUN_TIMEOUT` | `1800` | Seconds before a run is cancelled |
| `CODEX_MAX_ATTEMPTS` | `3` | Attempts on retryable startup errors |

## Lifecycle in the harness

`tools/bridges/codex_bridge_runner.py` wraps this package and is launched
per-run by `scripts/regrade_output.sh --judge codex-bridge` and by
`scripts/run_task.sh` under `JUDGE_TRANSPORT=codex-bridge`. The runner picks a free
port, writes `{host, port, agent_key, judge_key, judge_model}` atomically to
its `--config-out` path, and appends one JSON line per call to its
`--calls-out` path (label, requested model, served model, usage, UTC
timestamp, ok/error).

## Caveats

- Rewardkit bundles (`tests/reward/rubric/judge.toml`) reach this proxy through
  rewardkit's `claude-code` transport, which only accepts `claude-*` model
  names: with the judge key pinned to `gpt-5.6-sol` the claude CLI inside the
  verifier exits with `[claude-code:unrecognized_model]` before any request
  leaves the container (checked 2026-10-27 with `harbor trial regrade`). For
  those bundles use rewardkit's own codex judge (`JUDGE_TRANSPORT=codex`,
  `regrade_output.sh --judge codex`), which is also what every harness run
  scored with, so a regrade through it is like-for-like. This bridge serves
  legacy `grade.py` bundles that call the Anthropic wire directly.

- Every call spends the ChatGPT subscription that `codex login` was signed in
  with. There is no API key path.
- `codex exec` is run with `-s read-only --ignore-user-config --ignore-rules
  --ephemeral --skip-git-repo-check` and a throwaway cwd, so a tool-less judge
  call has nothing to touch on the host. The flags are deliberate; a reader
  changing them should understand each one first.
- `codex exec --json` reports real token counts on `turn.completed`, so the
  byte-based estimate in `messages_api.estimate_text_tokens` fires only when a
  run never reaches `turn.completed` (e.g. a transport error). Estimated
  fields are flagged with `usage_estimated: true` on the wire.
