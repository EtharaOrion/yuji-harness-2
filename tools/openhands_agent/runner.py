#!/usr/bin/env python3
"""Drive one OpenHands SDK conversation inside a task container.

Uploaded by tools/openhands_agent/agent.py to /installed-agent/ and executed
with the runtime venv that tools/openhands_agent/Dockerfile builds and
overlay.yaml mounts at /opt/openhands-runtime. Nothing here is installed at
trial time: the agent runs with the container's egress closed, so anything
missing at that point stays missing.

WHAT IT TALKS TO

The model call leaves the container as an ordinary Anthropic Messages request
to LLM_BASE_URL -- the ccbridge on the host (tools/bridges/ccbridge), reached as
http://host.docker.internal:<port>. The bridge swaps the shared secret this
process holds for the host's Claude subscription token. The subscription token
itself never enters the container.

WHAT IT WRITES, AND WHY IN THAT SHAPE

  /logs/agent/openhands.txt     JSON lines in Claude Code's stream-json dialect
  /logs/agent/trajectory.json   ATIF, built from the same events
  /logs/agent/openhands/        the SDK's own persisted conversation

Every grader in this repo reads the agent's stream, not the agent: a bundle's
tests/test.sh parses /logs/agent/*.txt (claude-code.txt first, then the largest
*.txt), and tools/delivery_utils/harbor_to_output.py walks the same file. Both
understand exactly two dialects, and the stream-json one is the one that carries
thinking, tool errors and a usage block. So the SDK's events are translated into
it here rather than every consumer learning a third dialect.

Two details of that translation are load-bearing:

  * MCP tools are named mcp__<Server>__<tool>, the way Claude Code names them.
    The graders attribute a call to an app by splitting that name
    (tests/test_outputs.py::_split, tests/state_dump.py); a bare `update_item`
    is attributed to no app at all, and two servers exposing the same bare name
    would collide. See NamespacedMCPToolProvider.
  * A tool_result carries the MCP server's raw payload, not the SDK's
    "[Tool 'x' executed.]" preamble. Graders json-parse the response to read
    its `status`; a preamble in front of it reads as unparseable text.

The file is named openhands.txt, not claude-code.txt: it is this agent's log in
a shared dialect, and the system/init line says which agent wrote it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STREAM_FILENAME = "openhands.txt"
TRAJECTORY_FILENAME = "trajectory.json"
EVENTS_DIRNAME = "openhands"
AGENT_NAME = "openhands"

MCP_PREFIX = "mcp__"

# The same sentence for every prompt and every model, carried over from the
# reference harness's adapters/openhands.py. It carries no task content and
# nothing derived from a score, so it cannot steer a draw; what it removes is an
# artefact of running an interactive agent with nobody on the other end. An
# agent that ends its turn by asking a question has FINISHED in the SDK's sense
# while the work is untouched, and a draw that stops there measures the absence
# of an operator rather than the solver.
CONTINUATION_NOTICE = (
    "Continue working on the task from where you left off. No operator is "
    "available to answer questions or approve a plan, so proceed on your own "
    "best judgement and record any assumption you make in your final answer. "
    "If the task is genuinely complete, call the finish tool to end the session."
)


# ---------------------------------------------------------------------------
# small helpers (no SDK imports, so the translation is testable on its own)
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _env_int(name: str, default: int | None) -> int | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"[openhands] ignoring {name}={raw!r}: not an integer", file=sys.stderr)
        return default


def _env_float(name: str) -> float | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        print(f"[openhands] ignoring {name}={raw!r}: not a number", file=sys.stderr)
        return None


def _texts(content: Any) -> list[str]:
    """Plain text out of an SDK content list (TextContent objects or dumps)."""
    out: list[str] = []
    if isinstance(content, str):
        return [content] if content else []
    for block in content or []:
        text = block.get("text") if isinstance(block, dict) else getattr(block, "text", None)
        if isinstance(text, str) and text:
            out.append(text)
    return out


def _is_mcp_preamble(text: str) -> bool:
    return text.startswith("[Tool '") and text.rstrip().endswith("executed.]")


def observation_text(observation: Any, *, is_mcp: bool) -> str:
    """What a tool returned, as the graders expect to read it.

    MCP results drop the SDK's "[Tool 'x' executed.]" line so the payload is the
    server's own JSON. Everything else is the text the model was shown.
    """
    if observation is None:
        return ""
    texts = _texts(getattr(observation, "content", None))
    if is_mcp:
        texts = [t for t in texts if not _is_mcp_preamble(t)]
        return "\n".join(texts)
    to_llm = getattr(observation, "to_llm_content", None)
    if to_llm is not None:
        try:
            llm_texts = _texts(to_llm)
            if llm_texts:
                return "\n".join(llm_texts)
        except Exception:  # noqa: BLE001 -- a render failure falls back to raw content
            pass
    return "\n".join(texts)


def tool_arguments(event: Any) -> dict[str, Any]:
    """The arguments the model sent, as a dict.

    Read from the raw tool call first -- that is what the model emitted -- and
    only then from the parsed action, which the SDK may have normalised.
    """
    call = getattr(event, "tool_call", None)
    raw = getattr(call, "arguments", None)
    if raw is None:
        fn = getattr(call, "function", None)
        raw = getattr(fn, "arguments", None)
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
            return {"_raw": parsed}
        except json.JSONDecodeError:
            return {"_raw": raw}
    if isinstance(raw, dict):
        return raw
    action = getattr(event, "action", None)
    if action is not None and hasattr(action, "model_dump"):
        try:
            dumped = action.model_dump(mode="json", exclude_none=True)
            dumped.pop("kind", None)
            data = dumped.get("data")
            return data if isinstance(data, dict) else dumped
        except Exception:  # noqa: BLE001
            pass
    return {}


# Fields the SDK adds to EVERY tool's schema and consumes itself before the tool
# runs (agent.py _extract_security_risk / _extract_summary). They are the
# agent's bookkeeping, not the call: a grader comparing a call's arguments must
# not see them, so they are split off and kept on the ATIF tool call's `extra`.
SDK_META_KEYS = ("security_risk", "summary")


def _executed_keys(event: Any) -> set[str]:
    """Argument names the tool itself accepts, read off the parsed action."""
    action = getattr(event, "action", None)
    if action is None:
        return set()
    data = getattr(action, "data", None)  # MCPToolAction carries the MCP call here
    if isinstance(data, dict):
        return set(data)
    return set(getattr(type(action), "model_fields", {}) or {})


def split_meta_arguments(args: dict[str, Any], event: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """(the tool's own arguments, the SDK's meta fields).

    A meta name the tool ALSO declares stays with the tool -- Jira's
    create_issue takes a real `summary`, and the SDK leaves that one in place.
    """
    executed = _executed_keys(event)
    clean = dict(args)
    meta: dict[str, Any] = {}
    for key in SDK_META_KEYS:
        if key in clean and key not in executed:
            meta[key] = clean.pop(key)
    return clean, meta


def claude_usage(prompt: int, completion: int, cache_read: int, cache_write: int,
                 reasoning: int) -> dict[str, Any]:
    """Anthropic usage semantics from LiteLLM's.

    LiteLLM folds cache reads and writes INTO prompt_tokens; Anthropic (and so
    Claude Code's stream) reports input_tokens WITHOUT them.
    harbor_to_output.py sums the three back together, so reporting LiteLLM's
    prompt_tokens as input_tokens would count the cache twice.
    """
    return {
        "input_tokens": max(int(prompt) - int(cache_read) - int(cache_write), 0),
        "output_tokens": int(completion),
        "cache_read_input_tokens": int(cache_read),
        "cache_creation_input_tokens": int(cache_write),
        "output_tokens_details": {"thinking_tokens": int(reasoning)},
    }


# ---------------------------------------------------------------------------
# the stream writer
# ---------------------------------------------------------------------------

class StreamWriter:
    """Append-only JSON-lines writer, flushed per line.

    Flushed per event so a trial killed mid-run (timeout, OOM) still leaves
    every event it produced on disk for the verifier and the reshaper.
    """

    def __init__(self, path: Path, *, model: str, session_id: str):
        self.path = path
        self.model = model
        self.session_id = session_id
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("a", encoding="utf-8")

    def write(self, obj: dict[str, Any]) -> None:
        obj.setdefault("session_id", self.session_id)
        self._fh.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")
        self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.close()
        except OSError:
            pass

    # -- event shapes -------------------------------------------------------

    def system_init(self, *, tools: list[str], mcp_servers: list[dict[str, str]],
                    cwd: str, sdk_version: str) -> None:
        self.write({
            "type": "system", "subtype": "init", "agent": AGENT_NAME,
            "agent_version": sdk_version, "model": self.model, "cwd": cwd,
            "tools": tools, "mcp_servers": mcp_servers,
        })

    def user_text(self, text: str) -> None:
        self.write({"type": "user", "message": {"role": "user", "content": text}})

    def assistant(self, blocks: list[dict[str, Any]], *, message_id: str | None) -> None:
        if not blocks:
            return
        self.write({"type": "assistant", "message": {
            "id": message_id, "type": "message", "role": "assistant",
            "model": self.model, "content": blocks,
        }})

    def tool_result(self, tool_use_id: str, text: str, *, is_error: bool) -> None:
        self.write({"type": "user", "message": {"role": "user", "content": [{
            "type": "tool_result", "tool_use_id": tool_use_id,
            "content": [{"type": "text", "text": text}], "is_error": bool(is_error),
        }]}})

    def result(self, **fields: Any) -> None:
        self.write({"type": "result", **fields})


def thinking_blocks(event: Any) -> list[dict[str, Any]]:
    """Anthropic thinking blocks (or bare reasoning text) carried on an event."""
    blocks: list[dict[str, Any]] = []
    for tb in getattr(event, "thinking_blocks", None) or []:
        text = tb.get("thinking") if isinstance(tb, dict) else getattr(tb, "thinking", None)
        if text is None:
            continue
        sig = tb.get("signature") if isinstance(tb, dict) else getattr(tb, "signature", None)
        blocks.append({"type": "thinking", "thinking": text, "signature": sig or ""})
    if not blocks:
        reasoning = getattr(event, "reasoning_content", None)
        if isinstance(reasoning, str) and reasoning.strip():
            blocks.append({"type": "thinking", "thinking": reasoning, "signature": ""})
    return blocks


# ---------------------------------------------------------------------------
# the event -> stream/trajectory translator
# ---------------------------------------------------------------------------

class Recorder:
    """Turns SDK events into the stream and the ATIF steps, as they arrive."""

    def __init__(self, stream: StreamWriter, *, model: str):
        self.stream = stream
        self.model = model
        self.steps: list[dict[str, Any]] = []
        self._agent_step_by_response: dict[str, dict[str, Any]] = {}
        self._step_by_call: dict[str, dict[str, Any]] = {}
        self._thought_seen: set[str] = set()
        self.final_text = ""
        self.last_acting: str | None = None  # "message" | "action"
        self.actions = 0
        self.errors = 0
        # (code, detail) of the last ConversationErrorEvent. The SDK reports a
        # run limit (MaxIterationsReached, budget) this way rather than raising.
        self.conversation_error: tuple[str, str] | None = None

    # The kinds are matched by class name, not isinstance, so the translator
    # does not have to import the SDK and stays testable without it.
    def __call__(self, event: Any) -> None:
        try:
            self._dispatch(event)
        except Exception:  # noqa: BLE001 -- a logging fault must never kill the run
            print("[openhands] could not record event:\n" + traceback.format_exc(),
                  file=sys.stderr)

    def _dispatch(self, event: Any) -> None:
        kind = type(event).__name__
        if kind == "SystemPromptEvent":
            prompt = getattr(getattr(event, "system_prompt", None), "text", "") or ""
            self.steps.append({"source": "system", "message": prompt,
                               "timestamp": getattr(event, "timestamp", None)})
        elif kind == "MessageEvent":
            self._message(event)
        elif kind == "ActionEvent":
            self._action(event)
        elif kind == "ObservationEvent":
            name = str(getattr(event, "tool_name", "") or "")
            obs = getattr(event, "observation", None)
            self._result(str(getattr(event, "tool_call_id", "") or ""),
                         observation_text(obs, is_mcp=name.startswith(MCP_PREFIX)),
                         bool(getattr(obs, "is_error", False)))
        elif kind == "AgentErrorEvent":
            self.errors += 1
            self._result(str(getattr(event, "tool_call_id", "") or ""),
                         str(getattr(event, "error", "") or ""), True)
        elif kind == "UserRejectObservation":
            self._result(str(getattr(event, "tool_call_id", "") or ""),
                         "rejected: " + str(getattr(event, "rejection_reason", "") or ""),
                         True)
        elif kind == "ConversationErrorEvent":
            code = str(getattr(event, "code", "") or "")
            detail = str(getattr(event, "detail", "") or "")
            self.conversation_error = (code, detail)
            # Both parsers skip `system` lines, so this is for a human reader.
            self.stream.write({"type": "system", "subtype": "error",
                               "code": code, "detail": detail[:2000]})

    def _message(self, event: Any) -> None:
        msg = getattr(event, "llm_message", None)
        text = "\n".join(_texts(getattr(msg, "content", None)))
        source = getattr(event, "source", "")
        ts = getattr(event, "timestamp", None)
        if source == "user":
            self.stream.user_text(text)
            self.steps.append({"source": "user", "message": text, "timestamp": ts})
            return
        if source != "agent":
            return
        blocks = thinking_blocks(msg) if msg is not None else []
        if text:
            blocks.append({"type": "text", "text": text})
            self.final_text = text
        self.stream.assistant(blocks, message_id=getattr(event, "llm_response_id", None))
        step = {"source": "agent", "message": text, "timestamp": ts,
                "model_name": self.model,
                "_response_id": getattr(event, "llm_response_id", None)}
        reasoning = "\n".join(b["thinking"] for b in blocks if b["type"] == "thinking")
        if reasoning:
            step["reasoning_content"] = reasoning
        self.steps.append(step)
        self.last_acting = "message"

    def _action(self, event: Any) -> None:
        response_id = str(getattr(event, "llm_response_id", "") or "")
        call_id = str(getattr(event, "tool_call_id", "") or "")
        name = str(getattr(event, "tool_name", "") or "")
        args, meta = split_meta_arguments(tool_arguments(event), event)
        ts = getattr(event, "timestamp", None)

        # One LLM response may call several tools; the SDK splits it into one
        # ActionEvent per call and puts the thought on the first. Emit the
        # thinking and the text once per response, and one tool_use per call.
        blocks: list[dict[str, Any]] = []
        thought = ""
        if response_id not in self._thought_seen:
            self._thought_seen.add(response_id)
            blocks.extend(thinking_blocks(event))
            thought = "\n".join(_texts(getattr(event, "thought", None)))
            if thought:
                blocks.append({"type": "text", "text": thought})
        blocks.append({"type": "tool_use", "id": call_id, "name": name, "input": args})
        self.stream.assistant(blocks, message_id=response_id or None)

        step = self._agent_step_by_response.get(response_id) if response_id else None
        if step is None:
            step = {"source": "agent", "message": thought, "timestamp": ts,
                    "model_name": self.model, "tool_calls": [],
                    "_response_id": response_id or None}
            reasoning = "\n".join(b["thinking"] for b in blocks if b["type"] == "thinking")
            if reasoning:
                step["reasoning_content"] = reasoning
            self.steps.append(step)
            if response_id:
                self._agent_step_by_response[response_id] = step
        call: dict[str, Any] = {"tool_call_id": call_id, "function_name": name,
                                "arguments": args}
        if meta:
            call["extra"] = meta
        step.setdefault("tool_calls", []).append(call)
        if call_id:
            self._step_by_call[call_id] = step

        if name == "finish":
            message = args.get("message")
            if isinstance(message, str) and message.strip():
                self.final_text = message
        self.last_acting = "action"
        self.actions += 1

    def _result(self, call_id: str, text: str, is_error: bool) -> None:
        self.stream.tool_result(call_id, text, is_error=is_error)
        step = self._step_by_call.get(call_id)
        if step is None:
            return
        obs = step.setdefault("observation", {"results": []})
        entry: dict[str, Any] = {"source_call_id": call_id, "content": text}
        if is_error:
            entry["extra"] = {"is_error": True}
        obs["results"].append(entry)


def build_trajectory(recorder: Recorder, *, session_id: str, sdk_version: str,
                     tool_definitions: list[dict[str, Any]] | None,
                     per_response: dict[str, dict[str, Any]],
                     totals: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    """ATIF-v1.8 from the recorded steps, with per-call metrics joined by response id."""
    steps: list[dict[str, Any]] = []
    for i, raw in enumerate(recorder.steps, 1):
        step = {k: v for k, v in raw.items() if not k.startswith("_") and v is not None}
        step["step_id"] = i
        if step.get("tool_calls") == []:
            step.pop("tool_calls")
        usage = per_response.get(raw.get("_response_id") or "")
        if usage and step.get("source") == "agent":
            step["metrics"] = {
                "prompt_tokens": usage["prompt_tokens"],
                "completion_tokens": usage["completion_tokens"],
                "cached_tokens": usage["cache_read_tokens"],
                "extra": {"cache_creation_input_tokens": usage["cache_write_tokens"],
                          "reasoning_tokens": usage["reasoning_tokens"]},
            }
        steps.append(step)
    if not steps:
        steps.append({"step_id": 1, "source": "system", "message": ""})
    return {
        "schema_version": "ATIF-v1.8",
        "session_id": session_id,
        "agent": {"name": AGENT_NAME, "version": sdk_version, "model_name": recorder.model,
                  "tool_definitions": tool_definitions or None, "extra": extra},
        "steps": steps,
        "final_metrics": {
            "total_prompt_tokens": totals["prompt_tokens"],
            "total_completion_tokens": totals["completion_tokens"],
            "total_cached_tokens": totals["cache_read_tokens"],
            "total_cost_usd": totals["cost_usd"],
            "total_steps": len(steps),
            "extra": {
                "total_cache_read_input_tokens": totals["cache_read_tokens"],
                "total_cache_creation_input_tokens": totals["cache_write_tokens"],
                "total_reasoning_tokens": totals["reasoning_tokens"],
            },
        },
    }


# ---------------------------------------------------------------------------
# MCP: one client per server, tools named mcp__<Server>__<tool>
# ---------------------------------------------------------------------------

def namespaced_name(server: str, tool: str) -> str:
    return f"{MCP_PREFIX}{server}__{tool}"


class _ToolSet:
    """What LocalConversation reads off an MCP provider's return: `.tools`."""

    def __init__(self, tools: list[Any]):
        self.tools = tools


class NamespacedMCPToolProvider:
    """Connect each MCP server on its own and name its tools like Claude Code.

    The SDK's default provider puts every server behind ONE FastMCP client, which
    prefixes tool names in its own style (`Server_tool`) and fails the whole
    agent if any single server is down. Here each server gets its own client, so
    its tools keep their bare MCP names on the wire and a dead server costs only
    its own tools -- recorded in `status` and in the stream's init line, which
    is what Claude Code does with a server that fails to start.

    The rename touches only what the MODEL sees. The executor keeps the tool's
    real MCP name, so the call that reaches the server is unchanged.
    """

    def __init__(self) -> None:
        self.status: list[dict[str, str]] = []

    def create_tools(self, mcp_config: dict[str, Any], timeout: float = 30.0,
                     **_: Any) -> _ToolSet:
        from openhands.sdk.mcp import create_mcp_tools

        tools: list[Any] = []
        self.status = []
        for server, spec in mcp_config.items():
            try:
                client = create_mcp_tools({server: spec}, timeout)
            except Exception as exc:  # noqa: BLE001 -- one dead server is not a dead run
                print(f"[openhands] MCP server {server!r} failed: "
                      f"{type(exc).__name__}: {str(exc)[:300]}", file=sys.stderr)
                self.status.append({"name": server, "status": "failed"})
                continue
            for tool in client.tools:
                tools.append(_rename_mcp_tool(tool, server))
            self.status.append({"name": server, "status": "connected"})
        return _ToolSet(tools)


def _rename_mcp_tool(tool: Any, server: str) -> Any:
    original = tool.mcp_tool.name
    if original.startswith(MCP_PREFIX):
        return tool
    renamed = tool.mcp_tool.model_copy(update={"name": namespaced_name(server, original)})
    return tool.model_copy(update={"mcp_tool": renamed})


def mcp_config_from_env(raw: str | None) -> dict[str, dict[str, Any]]:
    """Harbor's [[environment.mcp_servers]] list -> the SDK's name-keyed map."""
    if not raw:
        return {}
    config: dict[str, dict[str, Any]] = {}
    for entry in json.loads(raw):
        name = entry.get("name") or "mcp-server"
        transport = entry.get("transport") or "stdio"
        spec: dict[str, Any] = {}
        if transport == "stdio":
            if entry.get("command"):
                spec["command"] = entry["command"]
            if entry.get("args"):
                spec["args"] = entry["args"]
        else:
            if entry.get("url"):
                spec["url"] = entry["url"]
            spec["transport"] = transport
        config[name] = spec
    return config


# ---------------------------------------------------------------------------
# LiteLLM: keep adaptive thinking on every turn
# ---------------------------------------------------------------------------

THINKING_DISPLAYS = ("omitted", "summarized")


def pin_adaptive_thinking(display: str = "summarized") -> bool:
    """Send the same adaptive `thinking`, with the chosen `display`, on every turn.

    Two things in LiteLLM (1.102, AnthropicConfig.transform_request) decide
    what the trajectory shows of the model's thinking, and neither is a
    decision anyone here made:

    1. LiteLLM REMOVES `thinking` whenever litellm.modify_params is on -- the
       OpenHands SDK turns it on at import -- and the last assistant turn that
       called a tool carried no thinking block. That guard is for MANUAL
       thinking ({type: enabled}), where Anthropic rejects such a turn. Under
       ADAPTIVE thinking the model may skip thinking on a turn, and the next
       request with thinking re-attached is accepted (measured, claude-opus-5).
       So one quiet first turn used to strip `thinking` from the whole rest of
       the run, and opus-5 went on thinking with the API's default display.
    2. LiteLLM itself asks for display "summarized".

    `display`, measured on claude-opus-5 through the ccbridge
    (tools/openhands_agent/README.md, "Thinking"):

      summarized  (default) readable thinking in the stream and trajectory. On
                  the real task ed8fbb42: 10 of 10 blocks readable, no refusals,
                  the same reward as omitted. On a toy task, echoed blocks drew
                  stop_reason "refusal" in 2 of 4 runs; count_refusals reports
                  those as error_refusal instead of error_stuck.
      omitted     the model thinks and the tokens are billed; each block comes
                  back as a signature with empty text.

    Manual thinking is left entirely to LiteLLM. Returns whether the patch was
    installed.
    """
    if display not in THINKING_DISPLAYS:
        print(f"[openhands] unknown thinking display {display!r}; using 'summarized'",
              file=sys.stderr)
        display = "summarized"
    try:
        from litellm.llms.anthropic.chat.transformation import AnthropicConfig
    except Exception as exc:  # noqa: BLE001
        print(f"[openhands] thinking patch not installed: {exc}", file=sys.stderr)
        return False
    original = getattr(AnthropicConfig.transform_request, "_pinned_original",
                       AnthropicConfig.transform_request)

    def transform_request(self, model, messages, optional_params, litellm_params, headers):
        thinking = optional_params.get("thinking")
        data = original(self, model, messages, optional_params, litellm_params, headers)
        if (isinstance(thinking, dict) and thinking.get("type") == "adaptive"
                and isinstance(data, dict)):
            data["thinking"] = {**thinking, "display": display}
        return data

    transform_request._pinned_original = original  # type: ignore[attr-defined]
    AnthropicConfig.transform_request = transform_request  # type: ignore[method-assign]
    return True


REFUSALS: list[str] = []
"""Response ids Anthropic answered with stop_reason "refusal", in order."""


def count_refusals(on_refusal=None) -> bool:
    """Record every model response that was a refusal, instead of letting it
    pass as an agent that went quiet.

    Anthropic's safety classifiers end a response with stop_reason "refusal"
    and no content; LiteLLM reports it as finish_reason "content_filter". The
    SDK sees an empty message, nudges ("Your last response did not include a
    function call"), and after a few more empty answers its stuck detector ends
    the run -- recorded as `error_stuck`, which names the wrong cause. Observed
    with summarized thinking (README, "Thinking").

    Observe-only: the response is returned unchanged and nothing is retried or
    rewritten. Returns whether the hook was installed.
    """
    try:
        from litellm.llms.anthropic.chat.transformation import AnthropicConfig
    except Exception as exc:  # noqa: BLE001
        print(f"[openhands] refusal counter not installed: {exc}", file=sys.stderr)
        return False
    original = getattr(AnthropicConfig.transform_response, "_counted_original",
                       AnthropicConfig.transform_response)

    def transform_response(self, *args, **kwargs):
        response = original(self, *args, **kwargs)
        try:
            if response.choices and response.choices[0].finish_reason == "content_filter":
                rid = str(getattr(response, "id", "") or "")
                REFUSALS.append(rid)
                if on_refusal is not None:
                    on_refusal(rid)
        except Exception:  # noqa: BLE001 -- bookkeeping must never break a call
            pass
        return response

    transform_response._counted_original = original  # type: ignore[attr-defined]
    AnthropicConfig.transform_response = transform_response  # type: ignore[method-assign]
    return True


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def _status(conversation: Any) -> str:
    status = getattr(getattr(conversation, "state", None), "execution_status", "")
    return str(getattr(status, "value", status) or "")


def _usage_rows(llm: Any) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    metrics = llm.metrics
    acc = metrics.accumulated_token_usage
    totals = {
        "prompt_tokens": int(getattr(acc, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(acc, "completion_tokens", 0) or 0),
        "cache_read_tokens": int(getattr(acc, "cache_read_tokens", 0) or 0),
        "cache_write_tokens": int(getattr(acc, "cache_write_tokens", 0) or 0),
        "reasoning_tokens": int(getattr(acc, "reasoning_tokens", 0) or 0),
        "cost_usd": float(metrics.accumulated_cost or 0.0),
    }
    per_response: dict[str, dict[str, Any]] = {}
    for usage in getattr(metrics, "token_usages", None) or []:
        rid = getattr(usage, "response_id", None)
        if not rid:
            continue
        per_response[rid] = {
            "prompt_tokens": int(usage.prompt_tokens or 0),
            "completion_tokens": int(usage.completion_tokens or 0),
            "cache_read_tokens": int(usage.cache_read_tokens or 0),
            "cache_write_tokens": int(usage.cache_write_tokens or 0),
            "reasoning_tokens": int(usage.reasoning_tokens or 0),
        }
    return totals, per_response


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run one OpenHands SDK conversation")
    ap.add_argument("--instruction-file", required=True)
    ap.add_argument("--logs-dir", default=os.environ.get("AGENT_LOGS_DIR", "/logs/agent"))
    ap.add_argument("--workspace", default=os.environ.get("OPENHANDS_WORKSPACE") or None)
    args = ap.parse_args(argv)

    os.environ.setdefault("OPENHANDS_SUPPRESS_BANNER", "1")
    # LiteLLM fetches its price map and Anthropic beta-header table from
    # raw.githubusercontent.com when imported. Under isolation squid denies
    # that host, and the internet audit (tools/network/detect_internet_use.py)
    # rightly reports raw.githubusercontent.com as a content host the run tried
    # to reach. The bundled copies are what the pinned runtime was built with.
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    os.environ.setdefault("LITELLM_LOCAL_ANTHROPIC_BETA_HEADERS", "True")
    instruction = Path(args.instruction_file).read_text(encoding="utf-8")
    logs_dir = Path(args.logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)

    model = os.environ.get("LLM_MODEL", "").strip()
    base_url = os.environ.get("LLM_BASE_URL", "").strip() or None
    # Popped, not read: the terminal tool's shell inherits this process's
    # environment, and the agent has no use for the bridge secret in it.
    api_key = os.environ.pop("LLM_API_KEY", "").strip()
    if not model or not api_key:
        print("[openhands] LLM_MODEL and LLM_API_KEY must both be set", file=sys.stderr)
        return 2

    # /workspace is where every bundle here keeps the task's files; the
    # container's WORKDIR is usually `/`, which the SDK would then scan as the
    # project root.
    workspace = args.workspace or ("/workspace" if Path("/workspace").is_dir() else os.getcwd())
    session_id = os.environ.get("SESSION_ID") or uuid.uuid4().hex
    max_iterations = _env_int("MAX_ITERATIONS", 500) or 500
    max_continuations = _env_int("MAX_CONTINUATIONS", 6) or 0

    stream = StreamWriter(logs_dir / STREAM_FILENAME, model=model, session_id=session_id)
    recorder = Recorder(stream, model=model)
    started = time.monotonic()
    sdk_version = "unknown"
    llm = None
    conversation = None
    provider = NamespacedMCPToolProvider()
    error: BaseException | None = None
    continuations = 0

    try:
        from openhands.sdk import LLM, Agent, AgentContext, LocalConversation, Tool
        from openhands.sdk import __version__ as sdk_version  # noqa: F811
        from openhands.tools.file_editor import FileEditorTool
        from openhands.tools.task_tracker import TaskTrackerTool
        from openhands.tools.terminal import TerminalTool

        pin_adaptive_thinking(os.environ.get("LLM_THINKING_DISPLAY", "summarized").strip() or "summarized")
        count_refusals(lambda rid: stream.write({
            "type": "system", "subtype": "refusal", "response_id": rid,
            "detail": "Anthropic returned stop_reason \"refusal\" with no content"}))

        llm_kwargs: dict[str, Any] = {
            "model": model, "api_key": api_key, "base_url": base_url,
            "usage_id": "agent",
            "num_retries": _env_int("LLM_NUM_RETRIES", 5),
            # A single thinking turn on a large context runs for minutes; the
            # SDK's 300s default cut healthy turns off. The bridge's own read
            # timeout is raised to match (run_task.sh ensure_ccbridge).
            "timeout": _env_int("LLM_TIMEOUT", 1800),
        }
        if (effort := os.environ.get("LLM_REASONING_EFFORT", "").strip()):
            llm_kwargs["reasoning_effort"] = effort
        if (max_out := _env_int("LLM_MAX_OUTPUT_TOKENS", None)):
            llm_kwargs["max_output_tokens"] = max_out
        if (temperature := _env_float("LLM_TEMPERATURE")) is not None:
            llm_kwargs["temperature"] = temperature
        llm = LLM(**llm_kwargs)

        mcp_config = mcp_config_from_env(os.environ.get("MCP_SERVERS_JSON"))
        agent_kwargs: dict[str, Any] = {
            "llm": llm,
            "tools": [Tool(name=TerminalTool.name), Tool(name=FileEditorTool.name),
                      Tool(name=TaskTrackerTool.name)],
            # No skills: nothing in these containers is a skill, and loading
            # the defaults would scan the home directory for other agents'.
            "agent_context": AgentContext(skills=[]),
        }
        if mcp_config:
            agent_kwargs["mcp_config"] = mcp_config
        agent = Agent(**agent_kwargs)

        conversation = LocalConversation(
            agent=agent,
            workspace=workspace,
            persistence_dir=str(logs_dir / EVENTS_DIRNAME),
            conversation_id=uuid.UUID(hex=session_id) if len(session_id) == 32 else None,
            max_iteration_per_run=max_iterations,
            callbacks=[recorder],
            visualizer=None,
            mcp_tool_provider=provider,
        )

        print(f"[openhands] model={model} base_url={base_url} workspace={workspace} "
              f"max_iterations={max_iterations} mcp_servers={list(mcp_config)}")
        conversation.send_message(instruction)
        conversation.run()

        # An agent that stopped to talk to the operator is FINISHED only in the
        # SDK's sense; nobody will answer it. Bounded, and never past the budget.
        while continuations < max_continuations:
            if _status(conversation) != "finished":
                break
            if recorder.last_acting != "message":
                break
            if recorder.actions >= max_iterations:
                break
            continuations += 1
            print(f"[openhands] agent ended on a message; continuation {continuations}")
            conversation.send_message(CONTINUATION_NOTICE)
            conversation.run()
    except BaseException as exc:  # noqa: BLE001 -- recorded, then re-signalled below
        error = exc
        print("[openhands] conversation failed:\n" + traceback.format_exc(), file=sys.stderr)

    status = _status(conversation) if conversation is not None else "error"
    totals, per_response = _usage_rows(llm) if llm is not None else (
        {"prompt_tokens": 0, "completion_tokens": 0, "cache_read_tokens": 0,
         "cache_write_tokens": 0, "reasoning_tokens": 0, "cost_usd": 0.0}, {})

    # The tool list and server status are known only once the agent has
    # initialised, which happens inside run(); the init line is therefore
    # written at the end, and put FIRST in the file by the rewrite below.
    tool_names: list[str] = []
    tool_definitions: list[dict[str, Any]] = []
    try:
        for name, tool in (conversation.agent.tools_map.items() if conversation else []):
            tool_names.append(name)
            try:
                tool_definitions.append(tool.to_openai_tool())
            except Exception:  # noqa: BLE001
                pass
    except Exception:  # noqa: BLE001
        pass

    # Claude Code's subtypes, so harbor_to_output.py's termination_reason reads
    # the same way for both agents.
    code = recorder.conversation_error[0] if recorder.conversation_error else ""
    result_text = recorder.final_text
    if error is not None:
        subtype = "error_during_execution"
        result_text = f"API Error: {type(error).__name__}: {str(error)[:1500]}"
    elif code == "MaxIterationsReached":
        subtype = "error_max_turns"
    elif status == "finished":
        subtype = "success"
    elif code:
        subtype = f"error_{code}"
    else:
        subtype = f"error_{status or 'unknown'}"
    if REFUSALS and subtype not in ("success", "error_during_execution"):
        # The run stopped after the model was refused; say that, not "stuck".
        subtype = "error_refusal"
        result_text = (f"Model refusal: Anthropic returned stop_reason \"refusal\" "
                       f"{len(REFUSALS)} time(s); the conversation ended {status}. "
                       + (recorder.final_text or "")).strip()

    stream.result(
        subtype=subtype, is_error=error is not None, result=result_text,
        num_turns=recorder.actions, duration_ms=int((time.monotonic() - started) * 1000),
        total_cost_usd=totals["cost_usd"], stop_reason=status or None,
        continuations=continuations, refusals=len(REFUSALS),
        usage=claude_usage(totals["prompt_tokens"], totals["completion_tokens"],
                           totals["cache_read_tokens"], totals["cache_write_tokens"],
                           totals["reasoning_tokens"]),
        modelUsage={model: {
            "inputTokens": claude_usage(totals["prompt_tokens"], 0,
                                        totals["cache_read_tokens"],
                                        totals["cache_write_tokens"], 0)["input_tokens"],
            "outputTokens": totals["completion_tokens"],
            "cacheReadInputTokens": totals["cache_read_tokens"],
            "cacheCreationInputTokens": totals["cache_write_tokens"],
            "costUSD": totals["cost_usd"],
        }},
    )
    stream.close()
    _prepend_init(stream.path, {
        "type": "system", "subtype": "init", "agent": AGENT_NAME,
        "agent_version": str(sdk_version), "model": model, "cwd": workspace,
        "tools": tool_names, "mcp_servers": provider.status, "session_id": session_id,
    })

    trajectory = build_trajectory(
        recorder, session_id=session_id, sdk_version=str(sdk_version),
        tool_definitions=tool_definitions, per_response=per_response, totals=totals,
        extra={"base_url": base_url, "status": status, "subtype": subtype,
               "continuations": continuations, "mcp_servers": provider.status,
               "refusals": len(REFUSALS)},
    )
    (logs_dir / TRAJECTORY_FILENAME).write_text(json.dumps(trajectory, indent=2, default=str),
                                                encoding="utf-8")
    if REFUSALS:
        print(f"[openhands] {len(REFUSALS)} response(s) were Anthropic refusals "
              "(stop_reason=refusal)", file=sys.stderr)
    print(f"[openhands] {subtype}: {recorder.actions} tool calls, "
          f"{totals['prompt_tokens']} prompt / {totals['completion_tokens']} completion tokens, "
          f"${totals['cost_usd']:.4f} (API-equivalent)")
    # A run that died on the model call is an infrastructure failure, not an
    # answer; exit non-zero so harbor records it as one.
    return 1 if error is not None else 0


def _prepend_init(path: Path, init: dict[str, Any]) -> None:
    try:
        body = path.read_text(encoding="utf-8")
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(init, ensure_ascii=False) + "\n" + body, encoding="utf-8")
        os.replace(tmp, path)
    except OSError as exc:
        print(f"[openhands] could not write the init line: {exc}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
