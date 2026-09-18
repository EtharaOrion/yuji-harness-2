"""
services/scoring/traj_asserts.py

Channel A helper library: trajectory assertions a task's tests/test_outputs.py
can call to check *which tools the agent called, with what arguments* — never
the content of its final answer (that's Channel B / GTFA_CLAIMS, graded
separately by services/scoring/score_claims.py).

Reads the trajectory to check from the MCPATLAS_TRAJECTORY env var, which
must point at a JSON file holding either:

  * a full RunTrajectory object, matching
    services/agent-harness/src/mcp-agent/schema.ts's RunTrajectorySchema
    (schema_version: "mcp-atlas-trajectory-v1", steps: [...]), or
  * a flat list of OpenAI-style chat messages — the shape already used by
    run_eval.py's `raw_conversation_history` CSV column — with assistant
    messages carrying `tool_calls` and tool messages carrying
    `tool_call_id` + `content`.

Both shapes are normalized into the same internal call/result list at load
time, so a test author doesn't need to know which pipeline produced the
trajectory they're grading.

Every assertion here is phrased *positively* ("X happened") — whether that's
good news or bad news is decided entirely by the sign of this test's weight
in tests/test_weights.json, not by writing `assert not ...` in the test body.
See weighted_judge.py's module docstring for why that convention matters.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

_ENV_VAR = "MCPATLAS_TRAJECTORY"


@dataclass
class ToolCall:
    step_id: int | None
    id: str | None
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResult:
    step_id: int | None
    tool_call_id: str | None
    text: str
    is_error: bool


@dataclass
class _Trajectory:
    calls: list[ToolCall] = field(default_factory=list)
    results: list[ToolResult] = field(default_factory=list)
    # assistant messages with non-empty content, in step order
    assistant_texts: list[str] = field(default_factory=list)


_cache: dict[str, _Trajectory] = {}


def reset_cache() -> None:
    """Drop cached trajectory parses. Called between grading runs in the same
    process so state never leaks across tasks/attempts."""
    _cache.clear()


def _parse_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {"_raw": parsed}
        except json.JSONDecodeError:
            return {"_raw": raw}
    return {}


def _result_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join(parts)
    return ""


def _is_error_result(content: Any) -> bool:
    text = _result_text(content)
    return text.startswith("Error:") or text.startswith("Error ")


def _from_run_trajectory(data: dict) -> _Trajectory:
    traj = _Trajectory()
    for step in data.get("steps", []):
        step_id = step.get("step_id")
        message = step.get("message") or {}
        if message.get("role") == "assistant":
            content = message.get("content")
            if content:
                traj.assistant_texts.append(content)
            for tc in message.get("tool_calls") or []:
                fn = tc.get("function", {})
                traj.calls.append(ToolCall(
                    step_id=step_id,
                    id=tc.get("id"),
                    name=fn.get("name", ""),
                    arguments=_parse_arguments(fn.get("arguments")),
                ))
        for tr in step.get("tool_results") or []:
            traj.results.append(ToolResult(
                step_id=step_id,
                tool_call_id=tr.get("tool_call_id"),
                text=_result_text(tr.get("content")),
                is_error=_is_error_result(tr.get("content")),
            ))
    return traj


def _from_flat_messages(data: list) -> _Trajectory:
    traj = _Trajectory()
    for i, message in enumerate(data):
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "assistant":
            content = message.get("content")
            if content:
                traj.assistant_texts.append(content)
            for tc in message.get("tool_calls") or []:
                fn = tc.get("function", {})
                traj.calls.append(ToolCall(
                    step_id=i,
                    id=tc.get("id"),
                    name=fn.get("name", ""),
                    arguments=_parse_arguments(fn.get("arguments")),
                ))
        elif role == "tool":
            traj.results.append(ToolResult(
                step_id=i,
                tool_call_id=message.get("tool_call_id"),
                text=_result_text(message.get("content")),
                is_error=_is_error_result(message.get("content")),
            ))
    return traj


def _load(path: str) -> _Trajectory:
    if path in _cache:
        return _cache[path]
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "steps" in data:
        traj = _from_run_trajectory(data)
    elif isinstance(data, list):
        traj = _from_flat_messages(data)
    else:
        raise ValueError(
            f"{_ENV_VAR}={path!r} is neither a RunTrajectory object (dict "
            f"with a 'steps' key) nor a flat message list"
        )
    _cache[path] = traj
    return traj


def _trajectory() -> _Trajectory:
    path = os.environ.get(_ENV_VAR)
    if not path:
        raise RuntimeError(
            f"{_ENV_VAR} is not set — traj_asserts must be run under "
            f"weighted_judge.py's pytest runner, which stages the "
            f"trajectory and sets this env var before invoking pytest."
        )
    return _load(path)


# ============================================================================
# Public assertions
# ============================================================================

def calls() -> list[ToolCall]:
    """Every tool call the agent issued, in step order."""
    return list(_trajectory().calls)


def call_count(tool_name: str | None = None) -> int:
    """Number of tool calls, optionally filtered to one tool name."""
    cs = calls()
    if tool_name is None:
        return len(cs)
    return sum(1 for c in cs if c.name == tool_name)


def distinct_tools_called() -> set[str]:
    return {c.name for c in calls()}


def called_any(*tool_names: str) -> bool:
    """True if any of the given tools were called at least once."""
    names = distinct_tools_called()
    return any(n in names for n in tool_names)


def never_called(*tool_names: str) -> bool:
    """True if none of the given tools were called."""
    return not called_any(*tool_names)


def called_with(tool_name: str, **matchers: Any) -> bool:
    """True if `tool_name` was called at least once, and — for every keyword
    matcher given — the call's parsed arguments satisfy it. A matcher value
    that's callable is used as a predicate (`arg_value -> bool`); anything
    else is compared with `==`.

        called_with("mongodb_find")                        # called at all?
        called_with("mongodb_find", collection="orders")    # exact match
        called_with("web_search", query=lambda q: "2024" in q)  # predicate
    """
    for c in calls():
        if c.name != tool_name:
            continue
        ok = True
        for key, expected in matchers.items():
            actual = c.arguments.get(key)
            if callable(expected):
                if not expected(actual):
                    ok = False
                    break
            elif actual != expected:
                ok = False
                break
        if ok:
            return True
    return False


def tool_errored(tool_name: str | None = None) -> bool:
    """True if any tool result (optionally scoped to one tool's calls) came
    back as an error. Matches agent-eval.ts's `Error: ...` tool-message
    convention."""
    traj = _trajectory()
    if tool_name is None:
        return any(r.is_error for r in traj.results)
    call_ids = {c.id for c in traj.calls if c.name == tool_name and c.id is not None}
    return any(r.is_error for r in traj.results if r.tool_call_id in call_ids)


def final() -> str:
    """The agent's last non-empty assistant message text. Only useful for
    presence/shape checks (e.g. `final() != ""`) — reading it for factual
    correctness belongs in Channel B (tests/rubric.json), not here."""
    texts = _trajectory().assistant_texts
    return texts[-1] if texts else ""
