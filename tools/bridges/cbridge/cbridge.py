from __future__ import annotations

import asyncio
import base64
import json
import logging
import logging.handlers
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from claude_agent_sdk import (
    query as sdk_query,
    ClaudeAgentOptions,
    AssistantMessage,
    ResultMessage,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
    create_sdk_mcp_server,
    tool as sdk_tool_decorator,
    SdkMcpTool,
)

CLAUDE_CLI = os.environ.get("CC_BRIDGE_CLAUDE_BIN", "/opt/homebrew/bin/claude")

MODEL_ALIASES = {
    "claude-opus-5":   "claude-opus-5",
    "claude-opus-4.8": "claude-opus-5",
    "claude-opus-4-8": "claude-opus-5",
    "claude-opus-4-7": "claude-opus-4-7",
    "claude-sonnet-5": "claude-sonnet-5",
    "claude-sonnet-4.5": "claude-sonnet-4-6",
    "claude-sonnet-4-6": "claude-sonnet-4-6",
}

# Forces the shell-out to always produce a discriminated response we can map to
# either a plain assistant message or an OpenAI tool_call. Enum-of-strings at
# the root is the only shape the CLI accepts — it wraps the schema into an
# Anthropic tool's input_schema which must have type=object.
TOOL_ROUTER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "description": (
        "One-shot router. Set type='tool_call' (with name+arguments) to invoke a sandbox tool — "
        "this is the DEFAULT choice when the user has asked you to do something. "
        "Set type='text' (with text) ONLY when the task is fully complete or you have a specific "
        "question for the user. Never use type='text' to refuse or describe what you would do."
    ),
    "properties": {
        "type": {
            "type": "string",
            "enum": ["tool_call", "text"],
            "description": "tool_call = invoke a sandbox tool (prefer this); text = final reply to user (only when done)",
        },
        "name": {"type": "string", "description": "Tool name (required when type=tool_call)"},
        "arguments": {"type": "object", "description": "Tool arguments matching its input schema (required when type=tool_call)"},
        "text": {"type": "string", "description": "Plain-text reply (required when type=text)"},
    },
    "required": ["type"],
}

_seen_aliases: set[str] = set()


def _configure_logging() -> logging.Logger:
    here = Path(__file__).resolve().parent
    log_dir = here / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("cc_bridge")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / "cc-bridge.log", maxBytes=10 * 1024 * 1024, backupCount=5
        )
        file_handler.setFormatter(fmt)
        stream_handler = logging.StreamHandler(sys.stderr)
        stream_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
    return logger


logger = _configure_logging()


def resolve_model(requested: str) -> str:
    mapped = MODEL_ALIASES.get(requested, requested)
    if mapped != requested and requested not in _seen_aliases:
        _seen_aliases.add(requested)
        logger.info("Model alias substitution: %s -> %s", requested, mapped)
    return mapped


# The Messages API *requires* max_tokens — there is no "unlimited" value — so an
# uncapped request means asking for the model's own maximum output. The Models
# API is authoritative and self-updating (`max_tokens` on the model object), so
# resolve from it once per model and cache; the table below is the offline
# fallback. Unknown models fall back to 64000, which is <= every current model's
# ceiling, so it can never 400 for being too large.
_MODEL_MAX_OUTPUT_FALLBACK = {
    "claude-opus-5": 128000,
    "claude-fable-5": 128000,
    "claude-mythos-5": 128000,
    "claude-opus-4-8": 128000,
    "claude-opus-4-7": 128000,
    "claude-opus-4-6": 128000,
    "claude-sonnet-5": 128000,
    "claude-sonnet-4-6": 128000,
    "claude-haiku-4-5": 64000,
}
_DEFAULT_MAX_OUTPUT = 64000
_max_output_cache: dict[str, int] = {}


def _model_max_output(client: Any, model: str) -> int:
    """The model's own output ceiling — the largest max_tokens it will accept."""
    cached = _max_output_cache.get(model)
    if cached is not None:
        return cached
    ceiling = _MODEL_MAX_OUTPUT_FALLBACK.get(model, _DEFAULT_MAX_OUTPUT)
    try:
        reported = getattr(client.models.retrieve(model), "max_tokens", None)
        if isinstance(reported, int) and reported > 0:
            ceiling = reported
    except Exception as exc:                      # offline / unknown id / old SDK
        # DO NOT CACHE A FALLBACK WE NEVER CONFIRMED. A single transient failure
        # at process start would otherwise pin this model to the fallback
        # ceiling for the life of the bridge, silently capping max_tokens on
        # every later request even after the network came back. Returning
        # uncached costs one extra lookup per call while the API is down, and
        # self-heals on the first call that succeeds.
        logger.info("Models API lookup failed for %s (%s); using %d uncached",
                    model, exc, ceiling)
        return ceiling
    _max_output_cache[model] = ceiling
    return ceiling


def _select_mode() -> str:
    forced = os.environ.get("CC_MODE")
    if forced in {"subscription", "api_key"}:
        return forced
    has_api_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    has_cli = shutil.which(CLAUDE_CLI) is not None or Path(CLAUDE_CLI).exists()
    if has_api_key and not has_cli:
        return "api_key"
    return "subscription"


MODE = _select_mode()
logger.info("cc-bridge starting in mode=%s", MODE)


def _extract_text_and_media(content: Any) -> tuple[str, list[dict[str, Any]]]:
    if isinstance(content, str):
        return content, []
    if not isinstance(content, list):
        return "", []
    text_parts: list[str] = []
    media: list[dict[str, Any]] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        ptype = part.get("type")
        if ptype == "text":
            text_parts.append(part.get("text", ""))
        elif ptype == "image_url":
            url = (part.get("image_url") or {}).get("url", "")
            media.append({"kind": "image", "url": url})
        elif ptype == "input_audio":
            audio = part.get("input_audio") or {}
            media.append(
                {
                    "kind": "audio",
                    "data": audio.get("data", ""),
                    "format": audio.get("format", "mp3"),
                }
            )
    return "\n".join(t for t in text_parts if t), media


def _parse_data_url(url: str) -> tuple[str, bytes] | None:
    if not url.startswith("data:"):
        return None
    try:
        header, b64 = url.split(",", 1)
        media_type = header[5:].split(";")[0] or "application/octet-stream"
        return media_type, base64.b64decode(b64)
    except Exception:
        return None


def _build_subscription_prompt(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> tuple[str, str, list[str]]:
    system_chunks: list[str] = []
    transcript: list[str] = []
    tmp_files: list[str] = []

    for msg in messages:
        role = msg.get("role", "user")
        text, media = _extract_text_and_media(msg.get("content"))

        for m in media:
            if m["kind"] == "image":
                parsed = _parse_data_url(m["url"])
                if parsed is not None:
                    media_type, blob = parsed
                    ext = media_type.split("/")[-1] or "bin"
                    fd, path = tempfile.mkstemp(suffix=f".{ext}", prefix="ccbridge-img-")
                    os.write(fd, blob)
                    os.close(fd)
                    tmp_files.append(path)
                    text = f"{text}\n[IMAGE attached at {path} media_type={media_type}]"
                else:
                    text = f"{text}\n[IMAGE URL: {m['url']}]"
            elif m["kind"] == "audio":
                try:
                    blob = base64.b64decode(m["data"])
                except Exception:
                    blob = b""
                fd, path = tempfile.mkstemp(suffix=f".{m['format']}", prefix="ccbridge-audio-")
                os.write(fd, blob)
                os.close(fd)
                tmp_files.append(path)
                text = f"{text}\n[AUDIO attached at {path} format={m['format']}]"

        if role == "system":
            if text:
                system_chunks.append(text)
        elif role == "tool":
            tool_call_id = msg.get("tool_call_id", "?")
            transcript.append(f"[TOOL RESULT id={tool_call_id}]\n{text}")
        elif role == "assistant":
            tool_calls = msg.get("tool_calls") or []
            if tool_calls:
                rendered = []
                for tc in tool_calls:
                    fn = tc.get("function") or {}
                    rendered.append(
                        f"[ASSISTANT TOOL CALL id={tc.get('id','?')} name={fn.get('name','?')} arguments={fn.get('arguments','{}')}]"
                    )
                transcript.append("Assistant:\n" + (text + "\n" if text else "") + "\n".join(rendered))
            else:
                transcript.append(f"Assistant:\n{text}")
        else:
            transcript.append(f"User:\n{text}")

    system_prompt = "\n\n".join(system_chunks) if system_chunks else "You are a helpful assistant."

    if tools:
        tools_json = json.dumps(tools, indent=2)
        system_prompt = (
            system_prompt
            + "\n\n"
              "=== TOOL USE PROTOCOL (READ CAREFULLY) ===\n"
              "You are an AI agent working inside a sandbox. The tools listed below are "
              "REAL tools connected to the sandbox — they WILL execute. Your job is to "
              "COMPLETE the user's task by calling these tools.\n\n"
              "RULES:\n"
              '1. To call a tool, respond with EXACTLY: {"type":"tool_call","name":"<tool_name>","arguments":{...}}. '
              "You may make only ONE tool call per response — the caller will run it and give you the result, then you decide the next step.\n"
              '2. To reply with plain text (only when the task is done OR you have a direct question for the user), respond with: {"type":"text","text":"..."}.\n'
              "3. Do NOT respond with text claiming a tool 'is not available', 'returns no such tool', or similar. "
              "If a tool is listed below, it is available — CALL IT. If it errors, you'll see the error in the tool result and can react then.\n"
              "4. Do NOT respond with a plan or preamble before calling a tool. Just call the tool. "
              "The result comes back and you continue.\n"
              "5. Prefer calling a tool over asking the user to enable something. If you truly cannot proceed after trying, "
              "say WHAT tool you tried and WHAT error came back — not what you assume.\n\n"
              "The task may take many tool calls (dozens). That is normal. Do not summarize progress — just keep calling tools.\n\n"
              f"AVAILABLE TOOLS (call any by name):\n{tools_json}"
        )

    user_prompt = "\n\n".join(transcript) if transcript else ""
    return system_prompt, user_prompt, tmp_files


_NESTED_SESSION_ENV_VARS = (
    "CLAUDECODE",
    "CLAUDE_CODE_SESSION_ID",
    "CLAUDE_CODE_ENTRYPOINT",
    "CLAUDE_CODE_EXECPATH",
    "AI_AGENT",
    "CLAUDE_EFFORT",
)


def _child_env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in _NESTED_SESSION_ENV_VARS}
    return env


async def _run_claude_cli(system_prompt: str, user_prompt: str, model: str, use_schema: bool) -> dict[str, Any]:
    cmd = [
        CLAUDE_CLI,
        "-p",
        "--output-format", "json",
        "--model", model,
        "--setting-sources", "",
        "--tools", "",
        "--system-prompt", system_prompt,
        "--no-session-persistence",
    ]
    if use_schema:
        cmd.extend(["--json-schema", json.dumps(TOOL_ROUTER_SCHEMA)])

    workdir = tempfile.mkdtemp(prefix="ccbridge-cwd-")
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=workdir,
            env=_child_env(),
        )
        stdout_bytes, stderr_bytes = await proc.communicate(input=user_prompt.encode("utf-8"))
    finally:
        try:
            shutil.rmtree(workdir, ignore_errors=True)
        except Exception:
            pass

    if proc.returncode != 0:
        stderr_txt = stderr_bytes.decode("utf-8", errors="replace")[:2000]
        stdout_txt = stdout_bytes.decode("utf-8", errors="replace")[:2000]
        raise RuntimeError(
            f"claude CLI exit={proc.returncode} stderr={stderr_txt!r} stdout={stdout_txt!r}"
        )
    try:
        return json.loads(stdout_bytes.decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(f"claude CLI produced non-JSON output: {exc}: {stdout_bytes[:500]!r}") from None


def _openai_response(
    model_requested: str,
    content: str | None,
    tool_calls: list[dict[str, Any]] | None,
    usage: dict[str, int],
    *,
    reasoning_content: str | None = None,
    reasoning_signatures: list[str] | None = None,
    stop_reason: str | None = None,
) -> dict[str, Any]:
    finish_reason = stop_reason or ("tool_calls" if tool_calls else "stop")
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    if reasoning_content:
        message["reasoning_content"] = reasoning_content
    if reasoning_signatures:
        # Non-standard passthrough — consumed by litellm-strategy → agent-eval → trajectory.
        message["x_reasoning_signatures"] = reasoning_signatures
    if stop_reason:
        message["x_stop_reason"] = stop_reason
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_requested,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0),
            "cache_read_input_tokens": usage.get("cache_read_input_tokens", 0),
            "cache_creation_input_tokens": usage.get("cache_creation_input_tokens", 0),
            "reasoning_tokens": usage.get("reasoning_tokens", 0),
            "prompt_tokens_details": {
                "cached_tokens": usage.get("cache_read_input_tokens", 0),
                "cache_creation_input_tokens": usage.get("cache_creation_input_tokens", 0),
            },
            "completion_tokens_details": {
                "reasoning_tokens": usage.get("reasoning_tokens", 0),
            },
        },
    }


def _build_sdk_prompt(messages: list[dict[str, Any]]) -> tuple[str, str]:
    system_chunks: list[str] = []
    transcript: list[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        text, _media = _extract_text_and_media(msg.get("content"))
        if role == "system":
            if text:
                system_chunks.append(text)
        elif role == "tool":
            tool_call_id = msg.get("tool_call_id", "?")
            transcript.append(f"[TOOL RESULT id={tool_call_id}]\n{text}")
        elif role == "assistant":
            tool_calls = msg.get("tool_calls") or []
            if tool_calls:
                rendered = []
                for tc in tool_calls:
                    fn = tc.get("function") or {}
                    rendered.append(
                        f"[ASSISTANT TOOL CALL id={tc.get('id','?')} name={fn.get('name','?')} arguments={fn.get('arguments','{}')}]"
                    )
                transcript.append("Assistant:\n" + (text + "\n" if text else "") + "\n".join(rendered))
            else:
                transcript.append(f"Assistant:\n{text}")
        else:
            transcript.append(f"User:\n{text}")
    system_prompt = "\n\n".join(system_chunks) if system_chunks else ""
    user_prompt = "\n\n".join(transcript) if transcript else ""
    return system_prompt, user_prompt


def _build_bridge_mcp_and_allowlist(openai_tools: list[dict[str, Any]], captured: list[dict[str, Any]]):
    """Create an in-process MCP server exposing the caller's OpenAI tools.

    Each tool's handler records the invocation into `captured` and returns a
    short marker. Because `max_turns=1`, the SDK stops after the first agent
    turn — one tool_use per bridge call, exactly what OpenAI-shape expects.
    """
    sdk_tools: list[SdkMcpTool[Any]] = []
    allowed: list[str] = []
    for t in openai_tools:
        fn = t.get("function") or t
        name = fn.get("name")
        if not name:
            continue
        desc = fn.get("description", "") or ""
        params = fn.get("parameters") or {"type": "object", "properties": {}}

        def _make_handler(_name: str):
            async def _handler(args: dict[str, Any]) -> dict[str, Any]:
                captured.append({"name": _name, "input": dict(args)})
                return {"content": [{"type": "text", "text": "[bridge] captured — result will be supplied by caller"}]}
            return _handler

        sdk_tools.append(sdk_tool_decorator(name=name, description=desc, input_schema=params)(_make_handler(name)))
        allowed.append(f"mcp__bridge__{name}")

    mcp = create_sdk_mcp_server("bridge", tools=sdk_tools)
    return mcp, allowed


async def handle_subscription(body: dict[str, Any]) -> dict[str, Any]:
    messages = body.get("messages") or []
    tools = body.get("tools") or []
    model_requested = body.get("model", "claude-sonnet-4-6")
    model = resolve_model(model_requested)

    system_prompt, user_prompt = _build_sdk_prompt(messages)
    if not user_prompt.strip():
        user_prompt = "(no user content)"

    captured_calls: list[dict[str, Any]] = []
    mcp_kwargs: dict[str, Any] = {}
    if tools:
        mcp, allowed = _build_bridge_mcp_and_allowlist(tools, captured_calls)
        mcp_kwargs = {"mcp_servers": {"bridge": mcp}, "allowed_tools": allowed}

    _thinking_display = os.environ.get("CC_BRIDGE_THINKING_DISPLAY", "summarized").strip()
    thinking_param = body.get("thinking")
    if _thinking_display:
        if not isinstance(thinking_param, dict) or thinking_param.get("type") != "disabled":
            thinking_param = {"type": "adaptive", "display": _thinking_display}

    options = ClaudeAgentOptions(
        model=model,
        system_prompt=system_prompt if system_prompt else None,
        max_turns=1,
        permission_mode="bypassPermissions",
        setting_sources=[],  # no CLAUDE.md / project settings interference
        tools=[],  # disable Claude Code built-ins (Read/Write/Bash/ToolSearch/etc.)
        thinking=thinking_param if thinking_param else None,
        **mcp_kwargs,
    )

    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_use_blocks: list[dict[str, Any]] = []
    usage: dict[str, int] = {
        "prompt_tokens": 0, "completion_tokens": 0,
        "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0,
        "reasoning_tokens": 0,
    }
    stop_reason: str | None = None

    async def _drain():
        nonlocal stop_reason
        async for msg in sdk_query(prompt=user_prompt, options=options):
            if isinstance(msg, AssistantMessage):
                if msg.stop_reason:
                    stop_reason = msg.stop_reason
                if msg.usage:
                    usage["prompt_tokens"] += int(msg.usage.get("input_tokens", 0) or 0)
                    usage["completion_tokens"] += int(msg.usage.get("output_tokens", 0) or 0)
                    usage["cache_read_input_tokens"] += int(msg.usage.get("cache_read_input_tokens", 0) or 0)
                    usage["cache_creation_input_tokens"] += int(msg.usage.get("cache_creation_input_tokens", 0) or 0)
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        text_parts.append(block.text)
                    elif isinstance(block, ThinkingBlock):
                        reasoning_parts.append(block.thinking)
                    elif isinstance(block, ToolUseBlock):
                        name = block.name.removeprefix("mcp__bridge__")
                        tool_use_blocks.append({
                            "id": block.id,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(block.input or {}),
                            },
                        })
            elif isinstance(msg, ResultMessage):
                if getattr(msg, "usage", None):
                    u = msg.usage
                    if isinstance(u, dict):
                        usage["prompt_tokens"] += int(u.get("input_tokens", 0) or 0)
                        usage["completion_tokens"] += int(u.get("output_tokens", 0) or 0)

    try:
        await _drain()
    except Exception as exc:
        # SDK raises on several benign end-states:
        #   - "Reached maximum number of turns" — by design when max_turns=1
        #     fires after a tool_use (our intentional stop).
        #   - "Claude Code returned an error result: success" — quirk of the
        #     result-message error path; the actual response is fine.
        # In both cases, if we captured usable output (text OR tool_calls),
        # treat it as a success. Only re-raise on real transport/API errors.
        emsg = str(exc).lower()
        benign = ("maximum number of turns" in emsg) or ("error result: success" in emsg)
        if benign and (tool_use_blocks or text_parts):
            stop_reason = stop_reason or ("tool_calls" if tool_use_blocks else "stop")
        else:
            raise RuntimeError(f"claude-agent-sdk query failed: {exc.__class__.__name__}: {exc}") from exc

    content_text = "\n".join(t for t in text_parts if t) or None
    reasoning_text = "\n".join(r for r in reasoning_parts if r) or None
    tool_calls = tool_use_blocks or None

    return _openai_response(
        model_requested,
        content_text,
        tool_calls,
        usage,
        reasoning_content=reasoning_text,
        stop_reason=stop_reason,
    )


def _openai_content_to_anthropic(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if not isinstance(content, list):
        return [{"type": "text", "text": ""}]
    out: list[dict[str, Any]] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        ptype = part.get("type")
        if ptype == "text":
            out.append({"type": "text", "text": part.get("text", "")})
        elif ptype == "image_url":
            url = (part.get("image_url") or {}).get("url", "")
            parsed = _parse_data_url(url)
            if parsed is not None:
                media_type, blob = parsed
                out.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": base64.b64encode(blob).decode("ascii"),
                        },
                    }
                )
            else:
                out.append({"type": "image", "source": {"type": "url", "url": url}})
        elif ptype == "input_audio":
            out.append({"type": "text", "text": "[audio omitted — Anthropic API has no audio input]"})
    return out


def _openai_messages_to_anthropic(messages: list[dict[str, Any]]) -> tuple[str | None, list[dict[str, Any]]]:
    system_chunks: list[str] = []
    out: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role")
        if role == "system":
            text, _ = _extract_text_and_media(msg.get("content"))
            if text:
                system_chunks.append(text)
            continue
        if role == "tool":
            out.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg.get("tool_call_id", ""),
                            "content": msg.get("content") if isinstance(msg.get("content"), str) else json.dumps(msg.get("content")),
                        }
                    ],
                }
            )
            continue
        if role == "assistant":
            blocks: list[dict[str, Any]] = []
            text, _ = _extract_text_and_media(msg.get("content"))
            if text:
                blocks.append({"type": "text", "text": text})
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function") or {}
                try:
                    args = json.loads(fn.get("arguments", "{}") or "{}")
                except json.JSONDecodeError:
                    args = {}
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": tc.get("id", ""),
                        "name": fn.get("name", ""),
                        "input": args,
                    }
                )
            if not blocks:
                blocks = [{"type": "text", "text": ""}]
            out.append({"role": "assistant", "content": blocks})
            continue
        out.append({"role": "user", "content": _openai_content_to_anthropic(msg.get("content"))})
    system_prompt = "\n\n".join(system_chunks) if system_chunks else None
    return system_prompt, out


def _openai_tools_to_anthropic(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for t in tools:
        if t.get("type") != "function":
            continue
        fn = t.get("function") or {}
        out.append(
            {
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters") or {"type": "object", "properties": {}},
            }
        )
    return out


def _create_message(client: Any, **kwargs: Any) -> Any:
    """Blocking Messages call, streamed.

    Not messages.create(): at the model's full max_tokens the SDK refuses a
    non-streaming request outright (ValueError — it estimates the call will
    outlive the ~10 minute HTTP timeout), and an idle connection that long is
    liable to drop anyway. get_final_message() returns the same Message object
    create() would have, so everything downstream is unchanged.
    """
    with client.messages.stream(**kwargs) as stream:
        return stream.get_final_message()


async def handle_api_key(body: dict[str, Any]) -> dict[str, Any]:
    from anthropic import Anthropic

    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    model_requested = body.get("model", "claude-sonnet-4-6")
    model = resolve_model(model_requested)
    system_prompt, anthropic_msgs = _openai_messages_to_anthropic(body.get("messages") or [])
    tools = _openai_tools_to_anthropic(body.get("tools") or [])

    # Uncapped: default to the model's full output ceiling rather than a fixed
    # number, and clamp a caller-supplied value to it so an over-large request
    # is trimmed instead of 400ing.
    ceiling = await asyncio.to_thread(_model_max_output, client, model)
    requested_max = body.get("max_tokens")
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": anthropic_msgs,
        "max_tokens": min(int(requested_max), ceiling) if requested_max else ceiling,
    }
    if system_prompt:
        kwargs["system"] = system_prompt
    if tools:
        kwargs["tools"] = tools
    if body.get("temperature") is not None:
        kwargs["temperature"] = body["temperature"]
    # Extended thinking: normalize to {"type": "adaptive", "display": <mode>} so the
    # API actually returns populated thinking blocks. "enabled" with budget_tokens gives
    # 0-char thinking even with display:"summarized"; "adaptive" does not.
    _thinking_display = os.environ.get("CC_BRIDGE_THINKING_DISPLAY", "summarized").strip()
    thinking_param = body.get("thinking")
    if _thinking_display:
        if not isinstance(thinking_param, dict) or thinking_param.get("type") != "disabled":
            thinking_param = {"type": "adaptive", "display": _thinking_display}
    if thinking_param:
        kwargs["thinking"] = thinking_param

    resp = await asyncio.to_thread(_create_message, client, **kwargs)

    text_parts: list[str] = []
    thinking_parts: list[str] = []
    reasoning_signatures: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for block in resp.content:
        btype = getattr(block, "type", None)
        if btype == "text":
            text_parts.append(getattr(block, "text", ""))
        elif btype in ("thinking", "redacted_thinking"):
            # Anthropic extended-thinking block. `thinking` carries the visible
            # reasoning text; `signature` is the verifiable signed marker.
            thinking_text = getattr(block, "thinking", "") or getattr(block, "text", "")
            if thinking_text:
                thinking_parts.append(thinking_text)
            sig = getattr(block, "signature", None) or getattr(block, "data", None)
            if sig:
                reasoning_signatures.append(sig)
        elif btype == "tool_use":
            tool_calls.append(
                {
                    "id": getattr(block, "id", f"call_{uuid.uuid4().hex[:20]}"),
                    "type": "function",
                    "function": {
                        "name": getattr(block, "name", ""),
                        "arguments": json.dumps(getattr(block, "input", {}) or {}),
                    },
                }
            )

    usage = {
        "prompt_tokens": getattr(resp.usage, "input_tokens", 0),
        "completion_tokens": getattr(resp.usage, "output_tokens", 0),
        "cache_read_input_tokens": getattr(resp.usage, "cache_read_input_tokens", 0) or 0,
        "cache_creation_input_tokens": getattr(resp.usage, "cache_creation_input_tokens", 0) or 0,
        # Anthropic doesn't split thinking tokens out of output_tokens; approximate.
        "reasoning_tokens": 0,
    }
    content_text = "".join(text_parts) if text_parts else None
    reasoning_content = "\n".join(thinking_parts) if thinking_parts else None
    stop_reason = getattr(resp, "stop_reason", None)
    return _openai_response(
        model_requested,
        content_text,
        tool_calls or None,
        usage,
        reasoning_content=reasoning_content,
        reasoning_signatures=reasoning_signatures or None,
        stop_reason=stop_reason,
    )


app = FastAPI(title="cc-bridge")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "mode": MODE}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> JSONResponse:
    started = time.time()
    auth = request.headers.get("authorization", "")
    if not auth.lower().startswith("bearer ") or not auth[7:].strip():
        raise HTTPException(status_code=401, detail="missing bearer token")

    try:
        body = await request.json()
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="invalid JSON body")
    if not isinstance(body, dict) or not isinstance(body.get("messages"), list):
        raise HTTPException(status_code=400, detail="body must include messages: []")

    model_requested = body.get("model", "claude-sonnet-4-6")
    tools_present = bool(body.get("tools"))

    try:
        if MODE == "api_key":
            result = await handle_api_key(body)
        else:
            result = await handle_subscription(body)
    except HTTPException:
        raise
    except Exception as exc:
        latency = time.time() - started
        logger.exception(
            "chat_completion failed mode=%s model_requested=%s tools=%s latency=%.2fs",
            MODE, model_requested, tools_present, latency,
        )
        raise HTTPException(status_code=502, detail=f"upstream failure: {exc}") from None

    latency = time.time() - started
    logger.info(
        "chat_completion ok mode=%s model_requested=%s resolved=%s tools=%s latency=%.2fs usage=%s",
        MODE, model_requested, resolve_model(model_requested), tools_present, latency, result.get("usage"),
    )
    return JSONResponse(result)


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("CC_BRIDGE_PORT", "4000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
