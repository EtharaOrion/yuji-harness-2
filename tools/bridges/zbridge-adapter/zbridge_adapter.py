from __future__ import annotations

import asyncio
import base64
import json
import logging
import logging.handlers
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse


MODEL_ALIASES = {
    "glm-5.3": "glm-5.3",
    "claude-opus-5":    "claude-opus-5",
    "claude-opus-4-8":  "claude-opus-5",
    "claude-opus-4-7":  "claude-opus-4-7",
    "claude-sonnet-5":  "claude-sonnet-5",
    "claude-sonnet-4.5": "claude-sonnet-4-6",
    "claude-sonnet-4-6": "claude-sonnet-4-6",
}

ZBRIDGE_URL = os.environ.get("ZBRIDGE_URL", "http://127.0.0.1:8766")
ZBRIDGE_SECRET = os.environ.get("ZB_BRIDGE_SECRET", "local")


def _configure_logging() -> logging.Logger:
    here = Path(__file__).resolve().parent
    log_dir = here / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("zbridge_adapter")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        fh = logging.handlers.RotatingFileHandler(
            log_dir / "zbridge-adapter.log", maxBytes=10 * 1024 * 1024, backupCount=5
        )
        fh.setFormatter(fmt)
        sh = logging.StreamHandler(sys.stderr)
        sh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(sh)
    return logger


logger = _configure_logging()


def resolve_model(requested: str) -> str:
    return MODEL_ALIASES.get(requested, requested)


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
                out.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": base64.b64encode(blob).decode("ascii"),
                    },
                })
            else:
                out.append({"type": "image", "source": {"type": "url", "url": url}})
    return out


def _openai_messages_to_anthropic(
    messages: list[dict[str, Any]],
) -> tuple[str | None, list[dict[str, Any]]]:
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
            out.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": msg.get("tool_call_id", ""),
                    "content": (
                        msg.get("content")
                        if isinstance(msg.get("content"), str)
                        else json.dumps(msg.get("content"))
                    ),
                }],
            })
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
                blocks.append({
                    "type": "tool_use",
                    "id": tc.get("id", ""),
                    "name": fn.get("name", ""),
                    "input": args,
                })
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
        out.append({
            "name": fn.get("name", ""),
            "description": fn.get("description", ""),
            "input_schema": fn.get("parameters") or {"type": "object", "properties": {}},
        })
    return out


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
        message["x_reasoning_signatures"] = reasoning_signatures
    if stop_reason:
        message["x_stop_reason"] = stop_reason
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_requested,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
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


def _create_message(client: Any, **kwargs: Any) -> Any:
    with client.messages.stream(**kwargs) as stream:
        return stream.get_final_message()


async def handle(body: dict[str, Any]) -> dict[str, Any]:
    from anthropic import Anthropic

    client = Anthropic(api_key=ZBRIDGE_SECRET, base_url=ZBRIDGE_URL)
    model_requested = body.get("model", "glm-5.3")
    model = resolve_model(model_requested)
    system_prompt, anthropic_msgs = _openai_messages_to_anthropic(body.get("messages") or [])
    tools = _openai_tools_to_anthropic(body.get("tools") or [])

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": anthropic_msgs,
    }
    # anthropic SDK requires max_tokens; 999999 lets z.ai upstream cap take over.
    kwargs["max_tokens"] = int(body["max_tokens"]) if body.get("max_tokens") else 999999
    if system_prompt:
        kwargs["system"] = system_prompt
    if tools:
        kwargs["tools"] = tools
    if body.get("temperature") is not None:
        kwargs["temperature"] = body["temperature"]

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
            thinking_text = getattr(block, "thinking", "") or getattr(block, "text", "")
            if thinking_text:
                thinking_parts.append(thinking_text)
            sig = getattr(block, "signature", None) or getattr(block, "data", None)
            if sig:
                reasoning_signatures.append(sig)
        elif btype == "tool_use":
            tool_calls.append({
                "id": getattr(block, "id", f"call_{uuid.uuid4().hex[:20]}"),
                "type": "function",
                "function": {
                    "name": getattr(block, "name", ""),
                    "arguments": json.dumps(getattr(block, "input", {}) or {}),
                },
            })

    usage = {
        "prompt_tokens": getattr(resp.usage, "input_tokens", 0),
        "completion_tokens": getattr(resp.usage, "output_tokens", 0),
        "cache_read_input_tokens": getattr(resp.usage, "cache_read_input_tokens", 0) or 0,
        "cache_creation_input_tokens": getattr(resp.usage, "cache_creation_input_tokens", 0) or 0,
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


app = FastAPI(title="zbridge-adapter")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


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

    model_requested = body.get("model", "glm-5.3")
    tools_present = bool(body.get("tools"))

    try:
        result = await handle(body)
    except HTTPException:
        raise
    except Exception as exc:
        latency = time.time() - started
        logger.exception(
            "chat_completion failed model=%s tools=%s latency=%.2fs",
            model_requested, tools_present, latency,
        )
        raise HTTPException(status_code=502, detail=f"upstream failure: {exc}") from None

    latency = time.time() - started
    logger.info(
        "chat_completion ok model=%s tools=%s latency=%.2fs usage=%s",
        model_requested, tools_present, latency, result.get("usage"),
    )
    return JSONResponse(result)


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("ZBRIDGE_ADAPTER_PORT", "4001"))
    uvicorn.run(app, host="0.0.0.0", port=port)
