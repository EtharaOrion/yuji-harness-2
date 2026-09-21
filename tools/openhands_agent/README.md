# OpenHands agent

The harness's default agent. OpenHands, driven by its SDK inside the task
container, on the host's **Claude subscription** through the ccbridge
(`tools/bridges/ccbridge`). It replaces Harbor's `claude-code` agent in the
same way the reference harness replaced its `claude -p` lane. `claude-code`
is still available as `AGENT=claude-code`.

```bash
scripts/run_task.sh tasks/<task>                   # openhands + claude-opus-5
CC_MODE=zbridge scripts/run_task.sh tasks/<task>   # openhands + glm-5.3
AGENT=claude-code scripts/run_task.sh tasks/<t>    # the previous default
```

## The pieces

| File | Runs where | Does |
|---|---|---|
| `agent.py` | harbor process on the host | `OpenHandsAgent`, imported as `tools.openhands_agent.agent:OpenHandsAgent` (run_task.sh puts the harness on harbor's `PYTHONPATH`). Checks the runtime is mounted, uploads the runner and the instruction (as a file, never argv), runs the runner, reads its trajectory back into harbor's `AgentContext`. |
| `runner.py` | the task's `main` container | One OpenHands SDK conversation: terminal, file_editor, task_tracker and the task's MCP servers. Writes the stream, the ATIF trajectory and the SDK's event log to `/logs/agent`. |
| `Dockerfile` | built on the host | `openhands-runtime:latest`: a self-contained `/opt/openhands-runtime`, with uv's managed CPython 3.12 and a venv of `requirements.lock`. |
| `overlay.yaml` | compose overlay | A sidecar from that image fills a named volume, and `main` mounts it read-only at the same path. It also adds `host.docker.internal` and takes the Claude OAuth token out of `main`'s environment. |

```
host                                            main (task container)
────────────────────────────────────────        ─────────────────────────────────────────
ccbridge :<per-run port> ◄── host.docker.internal ── runner.py (OpenHands SDK, LiteLLM)
  secret check → OAuth bearer                        api_key = per-run secret, never the token
  → api.anthropic.com                                MCP → light-servers:<port>/mcp (NO_PROXY)
  (or zbridge :8766 → z.ai, on CC_MODE=zbridge)      /opt/openhands-runtime  (ro, from sidecar)
```

**Where each piece runs.** The agent itself (the SDK's loop, its terminal and
file editor, its MCP clients) runs **inside `main`**, as the task's user.
Two things deliberately do not:

- **The Python runtime** comes from a sidecar's volume rather than from the
  bundle's image. Bundles stay agent-agnostic and need no rebuild; a bundle
  that bakes in `COPY --from=openhands-runtime:latest /opt/openhands-runtime
  /opt/openhands-runtime` would work too, but every bundle would then pin one
  SDK build.
- **The model proxy** runs on the host. Running it in `main` would put the
  Claude OAuth token (or the z.ai key) in the container the agent has root in,
  which is the one thing this design keeps out.

Under network isolation the `main → host` hop goes through squid, which allows
exactly `host.docker.internal:<bridge port>` in addition to the usual list
(`tools/network/egress-proxy/squid-ccbridge.conf`). LiteLLM honours
`HTTP_PROXY`, so the agent needs no configuration of its own for it.

## What the runner writes, and why in that shape

Every grader here reads the agent's **stream**:

- the bundle's `tests/test.sh` parses `/logs/agent/*.txt`;
- `tools/delivery_utils/harbor_to_output.py` parses the same file;
- the judge sees whatever `test.sh` extracted.

They understand Claude Code's stream-json dialect, so `runner.py` translates the
SDK's events into it, in `/logs/agent/openhands.txt`:

- **MCP tools are named `mcp__<Server>__<tool>`**, the way Claude Code names
  them. Graders attribute a call to an app by splitting that name. The SDK's
  default provider names them `Server_tool` and fails the whole agent when one
  server is down, so `NamespacedMCPToolProvider` connects each server on its
  own instead. Only the name the model sees changes; the MCP call is unchanged.
- **A tool result is the server's raw payload.** The SDK prepends
  `[Tool 'x' executed.]`, which the graders cannot json-parse.
- **Arguments are the tool's own.** The SDK adds `security_risk` and `summary`
  to every tool schema and consumes them itself. They are split off into the
  ATIF tool call's `extra`, except where the tool really declares a `summary`
  (Jira's `create_issue`).
- **Usage uses Anthropic's semantics.** `input_tokens` excludes cache reads and
  writes, because the reshaper adds them back itself.
- **The result line uses Claude Code's subtypes:** `success`,
  `error_max_turns`, and `error_during_execution` with an `API Error: ...`
  result.

Alongside the stream:

- `trajectory.json` is ATIF-v1.8, with per-call metrics joined on the LLM
  response id. The finance reporter reads its `final_metrics`.
- `openhands/<conversation>/events/` is the SDK's own persisted event log.

**Continuations.** An agent that ends its turn on a message, rather than the
`finish` tool, has finished only in the SDK's sense, since nobody will answer
it. It is sent one fixed sentence that carries no task content (the
reference harness's `CONTINUATION_NOTICE`). This happens at most
`max_continuations` times.

## Models

| | Claude (default) | GLM (`CC_MODE=zbridge`) |
|---|---|---|
| `MODEL` default | `claude-opus-5` | `glm-5.3` |
| Proxy | a ccbridge this run starts and stops (`CCBRIDGE_SHARED=1` for one long-lived bridge) | zbridge on `:8766`, shared, started on demand |
| Auth held on the host | Claude OAuth login | `ZB_ZAI_API_KEY` |
| Thinking | adaptive, `display` per `OPENHANDS_THINKING_DISPLAY` | GLM reasons by default; zbridge returns it as readable thinking blocks |
| Cost in trajectories | LiteLLM's API-equivalent price | `0.0`: LiteLLM has no price for `glm-5.3` |

zbridge speaks the same Anthropic protocol as the ccbridge, so the agent is
identical on both; only the URL it is handed differs. Under isolation, squid
opens zbridge's port instead of the ccbridge's.

## Options

These are `run_task.sh` environment variables and become Harbor `--ak` agent
kwargs:

| Env | `--ak` | Default |
|---|---|---|
| `OPENHANDS_MAX_ITERATIONS` | `max_iterations` | 500 |
| `OPENHANDS_MAX_CONTINUATIONS` | `max_continuations` | 6 |
| `OPENHANDS_REASONING_EFFORT` | `reasoning_effort` | SDK default (`high`) |
| `OPENHANDS_MAX_OUTPUT_TOKENS` | `max_output_tokens` | 32000 (SDK default 16384) |
| `OPENHANDS_THINKING_DISPLAY` | `thinking_display` | `summarized` (or `omitted`, see "Thinking") |

`MODEL` keeps its bare name (`claude-opus-5`). It keys every report and
trajectory directory, and `agent.py` maps it to LiteLLM's `anthropic/claude-opus-5`.

## Thinking

`claude-opus-5` thinks adaptively: it decides per turn whether to think. The
runner sends the same `thinking: {type: adaptive, display: <option>}` on every
request, for two reasons found on 2026-09-22:

- **Empty thinking in early runs.** Every block reached the trajectory as a
  signature with empty text: 20 blocks and 4,472 thinking tokens in a 94-turn
  run, with no text. Two things caused it, and both are fixed:
  1. The reference bridge rewrote `adaptive` into `enabled` + `budget_tokens`,
     and `claude-opus-5` ignores `display` in that form. The bridge now passes
     adaptive through (`tools/bridges/ccbridge/README.md`).
  2. LiteLLM removes `thinking` when `modify_params` is on (the SDK turns it on)
     and the previous tool-calling turn had no thinking block. That rule exists
     for manual thinking. Under adaptive thinking one quiet turn stripped
     `thinking` from the rest of the run, and the API's default display then
     omitted the text. `runner.py` `pin_adaptive_thinking` restores it; manual
     thinking keeps LiteLLM's rule.
- **`display` is an option, measured on `claude-opus-5` (2026-09-22):**

  | Run | `display` | Thinking blocks | Readable | Refusals | Reward |
  |---|---|---|---|---|---|
  | `ed8fbb42`, 2 runs | `omitted` | 31 | 0 | 0 | 93.11 mean |
  | `ed8fbb42`, 1 run | `summarized` | 10 | 10 | 0 | 93.06 |
  | toy MCP task, 4 runs | `summarized` | 5 | 5 | 2 runs | - |

  - `summarized` (default): readable thinking in `openhands.jsonl`
    (`thinking` blocks) and `trajectory.json` (`reasoning_content`).
  - `omitted`: the model thinks and the tokens are counted, but each block is a
    signature with no text. Set `OPENHANDS_THINKING_DISPLAY=omitted` to go back.

  In the toy task, echoed `summarized` blocks drew `stop_reason: "refusal"`
  with empty content. That is Anthropic's safety classifier. Dropping the
  echoed block made the same request go through, but neither the bridge nor the
  runner works around a refusal. Instead `count_refusals` records each one, and
  a run that stops after one ends as `error_refusal` (with `refusals: N` on the
  result line) rather than as an agent that got `error_stuck`. Another harness
  (OpenHands V0, its own bridge) reports 37 of 37 readable blocks over a
  300-iteration run with no refusals. If `error_refusal` shows up on real
  tasks, compare against `omitted` before trusting either number.

**What summarized thinking is, and is not.** Use this wording when a
deliverable quotes it:

- It is Anthropic's server-side **summary** of the model's reasoning, written
  by a separate summarizer model. No `display` setting returns the raw chain
  of thought.
- The runner copies it verbatim from the response and never writes or edits
  it. An empty block stays empty.
- The block's signature carries the encrypted full reasoning and authenticates
  where the block came from. It does **not** bind the visible text: Anthropic
  ignores edited text in an echoed block rather than rejecting it.
- Billing is for the full thinking, so `thinking_tokens` exceeds what the
  summary's length suggests. A block with no text and `thinking_tokens > 0`
  means the reasoning was withheld (`omitted`), not that it never happened.

## Constraints

- `main` must be glibc-based and no older than Debian bookworm, which is what
  the runtime's wheels are resolved against. Every bundle here builds on
  `python:3.12-slim`; an alpine (musl) image cannot load the runtime.
- The SDK makes non-streaming calls, so one long thinking turn is one silent
  wait. The agent, the bridge and squid all allow 30 minutes per call.
- The SDK has no equivalent of Claude Code's `--disallowedTools` or PreToolUse
  hook, so the egress guard hook is not installed. The browser tool is not
  loaded, so there are no web tools to withhold, and the routing table remains
  the enforcement boundary. `detect_internet_use.py` audits the `terminal`
  tool's commands the same way it audits `Bash`.

## Updating the SDK

Edit `requirements.in`, then run:

```bash
uv pip compile --universal --python-version 3.12 --generate-hashes \
  requirements.in -o requirements.lock
docker build -t openhands-runtime:latest tools/openhands_agent
```

The next preflight rebuilds the image anyway (`ensure_image`).
