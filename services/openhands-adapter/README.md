# openhands-adapter

Drop-in replacement for the claude-code agent when running yuji-atlas tasks
through OpenHands instead of Anthropic's Claude CLI.

## Design

- Same auth: reads `CLAUDE_CODE_OAUTH_TOKEN` (routed through LiteLLM to
  `anthropic/claude-opus-5`)
- Same trajectory format: writes to `/logs/agent/claude-code.txt` in
  Anthropic Messages API JSONL shape, so downstream parsers
  (`test.sh` step 1, `judge_client.py`) work unchanged
- Same MCP servers: reads bundle `task.toml` `[environment.mcp_servers]`
  and passes them to OpenHands as `mcp_config`
- Same output: writes agent report to `/workspace/out/report.md`

No changes are required in `test.sh`, `judge_client.py`, `grade.py`,
`grade_coverage.py`, or `combine_channels.py`.

## Build

From `harness/` root:

```bash
docker build \
    -t harness/openhands-adapter:latest \
    services/openhands-adapter/
```

Optional pin override:

```bash
docker build \
    --build-arg OPENHANDS_SDK_VERSION=0.31.2 \
    -t harness/openhands-adapter:latest \
    services/openhands-adapter/
```

## Run standalone (bypass harness)

```bash
docker run --rm \
    -e CLAUDE_CODE_OAUTH_TOKEN="$CLAUDE_CODE_OAUTH_TOKEN" \
    -v /Users/macbookpro/Documents/yuji/staging/ed8fbb42-.../tests:/tests:ro \
    -v /tmp/yuji-openhands-workspace:/workspace \
    -v /tmp/yuji-openhands-logs:/logs \
    --network mcp-atlas_default \
    harness/openhands-adapter:latest
```

Result:
- `/tmp/yuji-openhands-workspace/out/report.md` — the agent report
- `/tmp/yuji-openhands-logs/agent/claude-code.txt` — JSONL trajectory

## Run via harness (once integrated)

```bash
cd harness
AGENT=openhands ./scripts/run_task.sh /path/to/bundle
```

## Environment variables

| Var | Default | Purpose |
|---|---|---|
| `CLAUDE_CODE_OAUTH_TOKEN` | (required) | Claude OAuth token for opus-5 |
| `MODEL` | `anthropic/claude-opus-5` | LiteLLM model routing |
| `MAX_ITERATIONS` | `100` | OpenHands agent iteration cap |
| `TASK_INSTRUCTION_PATH` | `/tests/instruction.md` | Agent task prompt source |
| `TASK_TOML_PATH` | derived from `/tests/..` | Where to read MCP server list |
| `AGENT_LOG_PATH` | `/logs/agent/claude-code.txt` | JSONL trajectory output |
| `REPORT_OUTPUT_PATH` | `/workspace/out/report.md` | Report deliverable path |

## Trajectory format

OpenHands `LLMConvertibleEvent.to_llm_message()` produces `Message` objects
which the entrypoint serializes as Anthropic Messages API JSONL lines:

```
{"type": "assistant", "content": [{"type": "text", "text": "..."}, {"type": "tool_use", "id": "...", "name": "LightXero.list_invoices", "input": {...}}]}
{"type": "user", "content": [{"type": "tool_result", "tool_use_id": "...", "content": "..."}]}
{"type": "assistant", "content": [{"type": "text", "text": "..."}]}
```

Matches what `test.sh` step 1's `parse_anthropic()` function expects.

## Deviations from claude-code path

| Aspect | claude-code | openhands-adapter |
|---|---|---|
| Auth transport | Native Anthropic OAuth | LiteLLM wraps OAuth token |
| Agent loop | Anthropic Claude CLI | OpenHands Python SDK |
| Trajectory shape | Anthropic Messages JSONL | Same shape (via serializer) |
| Cost accounting | claude CLI result.json | `llm.metrics.accumulated_cost` |
| Iteration control | Task timeout | `MAX_ITERATIONS` env |

## Governance

This adapter is a FORGE-authored deliverable pending contract update at
`.seed/contract.yaml` (declare `harness.agents[openhands]` — see
`.seed/staging/openhands_integration/recon.md`). Requires Ankit or Chirayu
temper-gate sign before production use.
