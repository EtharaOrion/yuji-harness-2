#!/usr/bin/env python3
"""Turn a raw agent event log into the {steps, final_message} shape the judge reads.

Every bundle's `tests/test.sh` carries an inline copy of this conversion, which
it runs in-container against /logs/agent. This module exists because the same
conversion is now needed on the HOST: the rubric judge grades through the local
`codex` CLI, and codex cannot run inside the task container -- there is no
binary in python:3.12-slim, and mounting the operator's ChatGPT credential into
a container that just ran an agent under bypassPermissions is not an acceptable
way to put one there.

Kept deliberately identical to the in-container copy, including the dialects it
accepts. Two agents write two different logs and both have to grade:

  * Anthropic stream-json (`claude-code`) -- assistant/user events carrying
    tool_use and tool_result blocks, correlated by tool_use_id.
  * OpenAI-shaped (the oracle and parity agents) -- message events with
    tool_calls and tool_call_id.

A tool call is only emitted once its result arrives, so a call still in flight
when the log ends is dropped rather than shown to the judge as a call with a
null response -- which reads as a failed call rather than an unfinished one.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _text_of(raw: Any) -> Any:
    if isinstance(raw, list) and raw:
        return raw[0].get("text", "") if isinstance(raw[0], dict) else str(raw[0])
    return raw


def _parse_json(raw: Any) -> Any:
    try:
        return json.loads(raw) if isinstance(raw, str) else raw
    except (json.JSONDecodeError, TypeError):
        return {"text": str(raw)}


def find_agent_log(agent_dir: Path) -> Path | None:
    """Pick the agent's event log, preferring the known names.

    Harbor writes claude-code.jsonl on the host and claude-code.txt in the
    container -- same content, different extension -- so both are candidates.
    Anything else falls back to the largest log present, on the assumption that
    the biggest file is the real transcript rather than a stub.
    """
    if not agent_dir.is_dir():
        return None
    named = [agent_dir / n for n in
             ("claude-code.jsonl", "claude-code.txt", "oracle.jsonl", "oracle.txt")]
    rest = sorted(
        (p for p in list(agent_dir.glob("*.jsonl")) + list(agent_dir.glob("*.txt"))
         if p not in named),
        key=lambda f: f.stat().st_size, reverse=True,
    )
    return next((p for p in named + rest if p.exists() and p.stat().st_size), None)


def build_trajectory(log_path: Path | None) -> dict[str, Any]:
    """Parse one agent log into {"steps": [...], "final_message": "..."}."""
    steps: list[dict[str, Any]] = []
    final_message = ""
    pending: dict[str, dict[str, Any]] = {}

    for line in (log_path.read_text(errors="replace").splitlines() if log_path else []):
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(ev, dict):
            continue
        etype = ev.get("type")
        msg = ev.get("message") or {}

        # ---- Anthropic stream-json (claude-code) ---------------------------
        if etype == "assistant":
            content = msg.get("content") or []
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_use":
                    pending[block.get("id", "")] = {
                        "tool": block.get("name", ""),
                        "arguments": block.get("input") or {},
                        "response": None,
                    }
                elif block.get("type") == "text" and block.get("text"):
                    final_message = block["text"]

        elif etype == "user":
            content = msg.get("content") or []
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "tool_result":
                    continue
                tid = block.get("tool_use_id", "")
                if tid in pending:
                    pending[tid]["response"] = _parse_json(_text_of(block.get("content")))
                    steps.append(pending.pop(tid))

        # ---- OpenAI-shaped dialect (oracle / parity agent) -----------------
        elif etype == "message" and msg.get("role"):
            role = msg.get("role")
            if role == "assistant":
                for tc in msg.get("tool_calls") or []:
                    fn = tc.get("function") or {}
                    args = fn.get("arguments")
                    if isinstance(args, str):
                        args = _parse_json(args)
                    pending[tc.get("id", "")] = {
                        "tool": fn.get("name", ""),
                        "arguments": args if isinstance(args, dict) else {},
                        "response": None,
                    }
                if isinstance(msg.get("content"), str) and msg["content"].strip() \
                        and not msg.get("tool_calls"):
                    final_message = msg["content"]
            elif role == "tool":
                tid = msg.get("tool_call_id", "")
                if tid in pending:
                    pending[tid]["response"] = _parse_json(msg.get("content"))
                    steps.append(pending.pop(tid))

        elif etype == "result":
            r = ev.get("result", "")
            if isinstance(r, str) and r:
                final_message = r

    return {"steps": steps, "final_message": final_message}
