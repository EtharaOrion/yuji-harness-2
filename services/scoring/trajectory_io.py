"""ATIF-v1.7 trajectory serialization.

Layout:

    line 1  : {"schema_version":"ATIF-v1.7","session_id":"...","trajectory_id":"...","agent":{...},"final_response":"...","termination_reason":"..."}
    line 2  : {"step_id":0,"source":"user","message":"..."}
    line 3  : {"step_id":1,"source":"agent","reasoning_content":"...","message":"","model_name":"...","tool_calls":[...],"observation":{"results":[...]},"timestamp":"...","metrics":{...},"llm_call_count":1,"extra":{"stop_reason":"..."}}
    ...
    line N+1: {"step_id":N,"source":"agent","message":"<final assistant reply>","model_name":"..."}

`load_trajectory` reads either:
  - .jsonl  ATIF-JSONL (this format)
  - .json   ATIF single-blob (`trajectory.json`)
  - .json   legacy `{task_id, messages, response}`  (Round 1/2)
and always returns:
    {schema_version, session_id, agent, steps, messages (OpenAI-shape shim), response, termination_reason}
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "ATIF-v1.7"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.") + f"{datetime.now(timezone.utc).microsecond // 1000:03d}Z"


def _extract_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text":
                    parts.append(part.get("text", ""))
                elif part.get("type") == "image_url":
                    parts.append("[image]")
                elif part.get("type") == "input_audio":
                    parts.append("[audio]")
        return "".join(parts)
    return ""


def _openai_messages_to_atif_steps(messages: list[dict], model: str) -> list[dict]:
    """Group OpenAI-format messages into ATIF steps.

    Rules:
      - user/system message → step with source matching role
      - assistant message → new agent step; tool_calls lifted in; observation empty
      - tool message (role='tool') → results appended to the preceding agent step's observation
      - _metrics/_extra/_timestamp on assistant messages (harness enrichment) → lifted into step
    """
    steps: list[dict] = []
    for msg in messages:
        role = msg.get("role")
        if role == "user":
            steps.append({
                "step_id": len(steps),
                "source": "user",
                "message": _extract_text_content(msg.get("content")),
            })
        elif role == "system":
            steps.append({
                "step_id": len(steps),
                "source": "system",
                "message": _extract_text_content(msg.get("content")),
            })
        elif role == "assistant":
            tool_calls_out = []
            for tc in (msg.get("tool_calls") or []):
                fn = tc.get("function") or {}
                raw_args = fn.get("arguments") or "{}"
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except json.JSONDecodeError:
                    args = {"_raw": raw_args}
                tool_calls_out.append({
                    "tool_call_id": tc.get("id"),
                    "function_name": fn.get("name"),
                    "arguments": args,
                    "extra": {
                        "raw_arguments": args,
                        "tool_use_name": fn.get("name"),
                    },
                })
            step = {
                "step_id": len(steps),
                "source": "agent",
                "reasoning_content": msg.get("reasoning_content"),
                "message": _extract_text_content(msg.get("content")),
                "model_name": model,
                "tool_calls": tool_calls_out,
                "observation": {"results": []},
                "timestamp": msg.get("_timestamp") or _now_iso(),
                "metrics": msg.get("_metrics") or {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "cached_tokens": 0,
                    "extra": {
                        "cache_creation_input_tokens": 0,
                        "cache_read_input_tokens": 0,
                        "reasoning_tokens": 0,
                    },
                },
                "llm_call_count": 1,
                "extra": msg.get("_extra") or {},
            }
            steps.append(step)
        elif role == "tool":
            content = msg.get("content")
            if isinstance(content, list):
                text = "".join(c.get("text", "") for c in content if isinstance(c, dict))
            else:
                text = str(content or "")
            result = {
                "source_call_id": msg.get("tool_call_id"),
                "content": text,
                "extra": {
                    "tool_result_metadata": {"raw_tool_result": content},
                    "tool_result_is_error": False,
                },
            }
            # Attach to the most-recent agent step; synthesize one if none exists (malformed trace).
            if steps and steps[-1]["source"] == "agent":
                steps[-1]["observation"]["results"].append(result)
            else:
                steps.append({
                    "step_id": len(steps),
                    "source": "agent",
                    "message": "",
                    "model_name": model,
                    "tool_calls": [],
                    "observation": {"results": [result]},
                    "timestamp": _now_iso(),
                    "extra": {"synthesized": True},
                })
    return steps


def _resolve_final_response(steps: list[dict]) -> tuple[str, str]:
    """Return (final_response, termination_reason).

    final_response = last agent step's `message` that has no pending tool_calls
                     (i.e. the model's terminal plain-text reply).
    termination_reason: 'natural' when the model ended with a message,
                        'no_terminal_reply' for max_turns / max_tool_calls / error.
    """
    for step in reversed(steps):
        if step.get("source") != "agent":
            continue
        message = (step.get("message") or "").strip()
        has_pending_tool_calls = bool(step.get("tool_calls"))
        if message and not has_pending_tool_calls:
            return message, "natural"
    # Fall back: last non-empty agent message even if it had tool_calls.
    for step in reversed(steps):
        if step.get("source") == "agent" and (step.get("message") or "").strip():
            return step["message"], "no_terminal_reply"
    return "", "no_terminal_reply"


def _aggregate_final_metrics(steps: list[dict]) -> dict:
    total_prompt = 0
    total_completion = 0
    total_cached = 0
    total_cache_creation = 0
    total_cache_read = 0
    total_reasoning = 0
    for step in steps:
        if step.get("source") != "agent":
            continue
        m = step.get("metrics") or {}
        total_prompt += int(m.get("prompt_tokens", 0) or 0)
        total_completion += int(m.get("completion_tokens", 0) or 0)
        total_cached += int(m.get("cached_tokens", 0) or 0)
        extra = m.get("extra") or {}
        total_cache_creation += int(extra.get("cache_creation_input_tokens", 0) or 0)
        total_cache_read += int(extra.get("cache_read_input_tokens", 0) or 0)
        total_reasoning += int(extra.get("reasoning_tokens", 0) or 0)
    return {
        "total_prompt_tokens": total_prompt,
        "total_completion_tokens": total_completion,
        "total_cached_tokens": total_cached,
        "total_cost_usd": None,
        "total_steps": len(steps),
        "extra": {
            "total_cache_creation_input_tokens": total_cache_creation,
            "total_cache_read_input_tokens": total_cache_read,
            "total_reasoning_tokens": total_reasoning,
        },
        "usage": {
            "input_tokens": total_prompt,
            "output_tokens": total_completion,
            "cache_read_tokens": total_cache_read,
            "cache_creation_tokens": total_cache_creation,
            "reasoning_tokens": total_reasoning,
            "cost_usd": 0.0,
        },
    }


def _build_trajectory(
    *,
    task_id: str,
    run: int | str | None,
    model: str,
    messages: list[dict[str, Any]],
    response: str,
    session_id: str | None,
) -> tuple[dict, list[dict]]:
    steps = _openai_messages_to_atif_steps(messages, model)
    final_response, termination = _resolve_final_response(steps)
    if response and not final_response:
        final_response = response
        termination = "natural"

    meta = {
        "schema_version": SCHEMA_VERSION,
        "session_id": session_id or str(uuid.uuid4()),
        "trajectory_id": task_id,
        "task_id": task_id,
        "run": run,
        "agent": {
            "name": task_id,
            "version": "1.0.0",
            "model_name": model,
            "extra": {},
        },
        "final_response": final_response,
        "termination_reason": termination,
        "step_count": len(steps),
    }
    return meta, steps


def write_trajectory_jsonl(
    path: Path,
    *,
    task_id: str,
    run: int | str | None,
    model: str,
    messages: list[dict[str, Any]],
    response: str = "",
    session_id: str | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    meta, steps = _build_trajectory(
        task_id=task_id, run=run, model=model,
        messages=messages, response=response, session_id=session_id,
    )
    meta["final_metrics"] = _aggregate_final_metrics(steps)

    with path.open("w") as f:
        f.write(json.dumps(meta) + "\n")
        for step in steps:
            f.write(json.dumps(step) + "\n")


def write_trajectory_json(
    path: Path,
    *,
    task_id: str,
    run: int | str | None,
    model: str,
    messages: list[dict[str, Any]],
    response: str = "",
    session_id: str | None = None,
) -> None:
    """Write a single-blob ATIF-v1.7 trajectory (steps array + final_metrics at top level)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    meta, steps = _build_trajectory(
        task_id=task_id, run=run, model=model,
        messages=messages, response=response, session_id=session_id,
    )
    blob = {
        "schema_version": meta["schema_version"],
        "session_id": meta["session_id"],
        "trajectory_id": meta["trajectory_id"],
        "agent": meta["agent"],
        "steps": steps,
        "final_metrics": _aggregate_final_metrics(steps),
        "final_response": meta["final_response"],
        "termination_reason": meta["termination_reason"],
    }
    path.write_text(json.dumps(blob, indent=4))


def _atif_steps_to_openai_messages(steps: list[dict]) -> list[dict]:
    """Reverse of the writer — reconstructs OpenAI message list for pytest/rubric consumers."""
    out: list[dict] = []
    for step in steps:
        src = step.get("source")
        if src == "user":
            out.append({"role": "user", "content": step.get("message", "")})
        elif src == "system":
            out.append({"role": "system", "content": step.get("message", "")})
        elif src == "agent":
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": step.get("message", "") or None,
            }
            tool_calls = step.get("tool_calls") or []
            if tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.get("tool_call_id"),
                        "type": "function",
                        "function": {
                            "name": tc.get("function_name"),
                            "arguments": json.dumps(tc.get("arguments") or {}),
                        },
                    }
                    for tc in tool_calls
                ]
            reasoning = step.get("reasoning_content")
            if reasoning:
                assistant_msg["reasoning_content"] = reasoning
            out.append(assistant_msg)
            for result in (step.get("observation") or {}).get("results", []):
                out.append({
                    "role": "tool",
                    "tool_call_id": result.get("source_call_id"),
                    "content": [{"type": "text", "text": str(result.get("content", ""))}],
                })
    return out


def load_trajectory(path: Path) -> dict[str, Any]:
    """Read ATIF-JSONL, ATIF single-blob JSON, or legacy {task_id, messages, response} JSON."""
    path = Path(path)
    if path.suffix == ".jsonl":
        return _load_atif_jsonl(path)
    if path.suffix == ".json":
        return _load_json_any(path)
    if path.is_file():
        first = path.read_text().lstrip()[:1]
        if first == "{":
            return _load_json_any(path)
    raise ValueError(f"unrecognized trajectory format: {path}")


def _load_atif_jsonl(path: Path) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    steps: list[dict] = []
    with path.open() as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno} invalid JSON: {e}") from e
            if not meta and obj.get("schema_version"):
                meta = obj
            elif "step_id" in obj:
                steps.append(obj)
            elif obj.get("type") == "meta":
                # Legacy Round-2 JSONL: convert to synthetic ATIF header.
                meta = {
                    "schema_version": SCHEMA_VERSION,
                    "session_id": str(uuid.uuid4()),
                    "trajectory_id": obj.get("task_id"),
                    "task_id": obj.get("task_id"),
                    "run": obj.get("run"),
                    "agent": {"name": obj.get("task_id"), "version": "1.0.0",
                              "model_name": obj.get("model"), "extra": {}},
                    "final_response": obj.get("final_response", ""),
                    "termination_reason": "unknown",
                }
            elif obj.get("type") == "message":
                # Legacy Round-2 message line — collect as raw OpenAI msg for batch conversion below.
                msg = {k: v for k, v in obj.items() if k not in {"type", "index"}}
                steps.append({"_legacy_openai_msg": msg})
    if steps and any("_legacy_openai_msg" in s for s in steps):
        legacy_msgs = [s["_legacy_openai_msg"] for s in steps if "_legacy_openai_msg" in s]
        steps = _openai_messages_to_atif_steps(legacy_msgs, meta.get("agent", {}).get("model_name", ""))
    return _finalize_load(meta, steps)


def _load_json_any(path: Path) -> dict[str, Any]:
    with path.open() as f:
        data = json.load(f)
    if data.get("schema_version", "").startswith("ATIF"):
        meta = {k: v for k, v in data.items() if k != "steps"}
        steps = data.get("steps", [])
        return _finalize_load(meta, steps)
    # Legacy round-1 shape: {task_id, messages, response}
    return _finalize_load({
        "schema_version": SCHEMA_VERSION,
        "task_id": data.get("task_id"),
        "trajectory_id": data.get("task_id"),
        "run": data.get("run"),
        "agent": {"name": data.get("task_id"), "version": "1.0.0",
                  "model_name": data.get("model"), "extra": {}},
        "final_response": data.get("response", ""),
        "termination_reason": "unknown",
    }, _openai_messages_to_atif_steps(data.get("messages", []), data.get("model", "")))


def _finalize_load(meta: dict, steps: list[dict]) -> dict[str, Any]:
    messages = _atif_steps_to_openai_messages(steps)
    return {
        "schema_version": meta.get("schema_version"),
        "session_id": meta.get("session_id"),
        "trajectory_id": meta.get("trajectory_id"),
        "task_id": meta.get("task_id") or meta.get("trajectory_id"),
        "run": meta.get("run"),
        "agent": meta.get("agent"),
        "steps": steps,
        "messages": messages,
        "response": meta.get("final_response", ""),
        "termination_reason": meta.get("termination_reason", "unknown"),
    }


def resolve_trajectory_path(run_dir: Path) -> Path:
    run_dir = Path(run_dir)
    jsonl = run_dir / "trajectory.jsonl"
    if jsonl.exists():
        return jsonl
    return run_dir / "trajectory.json"
