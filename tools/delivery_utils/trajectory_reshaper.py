#!/usr/bin/env python3
"""Reshape OpenHands ATIF-v1.8 trajectories, in place, to the ATIF-v1.5 shape of
Harbor's openhands-sdk writer (the shape the yuji-trajectories delivery carries).

    trajectory_reshaper.py <dir> [<dir> ...] [--to 1.5]

Every */run_N/agent/trajectory.json under each <dir> is rewritten in place,
except under .raw/, which keeps the run as it was generated.
Only ATIF-v1.8 files from the openhands agent are reshaped; files already in the
target shape, and other agents, are left alone, so a rerun is a no-op.

run_task.sh calls this as its last stage when ATIF_VERSION is 1.5 (the default).
ATIF_VERSION=1.8 skips it and ships the runner's own trajectory.

The target shape is data: SPECS[version] names the fields each level keeps, in
output order. Adding a field to a version is one more name in its spec; a value
that has to be derived from the source goes in its STEP_FIELDS entry.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

RUN_DIR = re.compile(r"^[Rr]un_\d+$")
SOURCE = ("ATIF-v1.8", "openhands")

SPECS = {
    "1.5": {
        "schema_version": "ATIF-v1.5",
        "agent_name": "openhands-sdk",
        "top": ("schema_version", "session_id", "agent", "steps", "final_metrics"),
        "agent": ("name", "tool_definitions", "version"),
        "step": ("step_id", "timestamp", "source", "message", "model_name", "thought",
                 "tool_calls", "observation"),
        "tool_call": ("tool_call_id", "function_name", "arguments"),
        "result": ("source_call_id", "content"),
        "final_metrics": ("total_prompt_tokens", "total_completion_tokens",
                          "total_cached_tokens", "total_cost_usd"),
        # One tool call per agent step, as Harbor's openhands-sdk writer emits.
        "split_tool_calls": True,
        # The answer closes the run as an agent text step, never a finish() call.
        "finish_as_message": True,
    },
}

# Step fields whose target value is derived rather than copied: source step -> value.
STEP_FIELDS = {
    "thought": lambda s: s.get("reasoning_content"),
}


def _pick(src: dict, fields: tuple[str, ...]) -> dict:
    return {k: src[k] for k in fields if src.get(k) is not None}


def _step(src: dict, spec: dict) -> dict:
    derived = {k: fn(src) for k, fn in STEP_FIELDS.items() if k in spec["step"]}
    out = _pick({**src, **derived}, spec["step"])
    if "tool_calls" in out:
        out["tool_calls"] = [_pick(c, spec["tool_call"]) for c in out["tool_calls"]]
    if "observation" in out:
        out["observation"] = {"results": [_pick(r, spec["result"])
                                          for r in out["observation"].get("results") or []]}
    return out


def _finish_as_message(step: dict) -> dict:
    """A finish() step becomes the closing text step: its message is the answer."""
    call = step["tool_calls"][0]
    out = {k: v for k, v in step.items() if k not in ("tool_calls", "observation")}
    out["message"] = (call.get("arguments") or {}).get("message") or ""
    return out


def _split(step: dict) -> list[dict]:
    """One step per tool call, each with only its own results. The step's text and
    reasoning go on the first as its thought; tool steps carry message ""."""
    calls = step.get("tool_calls") or []
    results = (step.get("observation") or {}).get("results") or []
    ids = {c.get("tool_call_id") for c in calls}
    orphans = [r for r in results if r.get("source_call_id") not in ids]
    thought = "\n\n".join(t for t in (step.get("reasoning_content"), step.get("message"))
                          if isinstance(t, str) and t.strip())
    out = []
    for i, call in enumerate(calls):
        own = [r for r in results if r.get("source_call_id") == call.get("tool_call_id")]
        new = {k: v for k, v in step.items() if k not in ("tool_calls", "observation")}
        new["message"] = ""
        new["reasoning_content"] = thought if i == 0 and thought else None
        new["tool_calls"] = [call]
        if own or (i == 0 and orphans):
            new["observation"] = {"results": own + (orphans if i == 0 else [])}
        out.append(new)
    return out


def reshape(traj: dict, version: str = "1.5") -> dict:
    spec = SPECS[version]
    steps = []
    finish = spec.get("finish_as_message")
    for step in traj.get("steps") or []:
        # The runner's own closing copy of the finish message; finish() becomes it below.
        if finish and (step.get("extra") or {}).get("origin") == "finish_tool_message":
            continue
        parts = (_split(step) if spec.get("split_tool_calls")
                 and step.get("source") == "agent" and step.get("tool_calls") else [step])
        if finish:
            parts = [_finish_as_message(p) if [c.get("function_name") for c in p.get("tool_calls") or []]
                     == ["finish"] else p for p in parts]
        steps.extend(_step(p, spec) for p in parts)
    for i, step in enumerate(steps, start=1):
        step["step_id"] = i

    agent = traj.get("agent") or {}
    built = {
        "schema_version": spec["schema_version"],
        "session_id": traj.get("session_id"),
        "agent": {"name": spec["agent_name"], "tool_definitions": None,
                  "version": agent.get("version", "unknown")},
        "steps": steps,
        "final_metrics": {k: (traj.get("final_metrics") or {}).get(k, 0)
                          for k in spec["final_metrics"]},
    }
    built["agent"] = {k: built["agent"].get(k) for k in spec["agent"]}
    return {k: built[k] for k in spec["top"]}


def find(root: Path) -> list[Path]:
    # .raw/ is the developers' untouched mirror of each run; it stays as generated.
    return sorted(p for p in root.rglob("trajectory.json")
                  if p.parent.name == "agent" and RUN_DIR.match(p.parent.parent.name)
                  and ".raw" not in p.relative_to(root).parts)


def reshape_tree(root: Path, version: str = "1.5") -> tuple[int, int]:
    """Reshape every run trajectory under root in place; (reshaped, left alone)."""
    done = left = 0
    for path in find(root):
        try:
            traj = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            print(f"[atif] skip {path}: unreadable ({exc})", file=sys.stderr)
            left += 1
            continue
        if (traj.get("schema_version"), (traj.get("agent") or {}).get("name")) != SOURCE:
            left += 1
            continue
        new = reshape(traj, version)
        path.write_text(json.dumps(new, indent=2, ensure_ascii=False) + "\n")
        print(f"[atif] {path}: {len(traj.get('steps') or [])} -> {len(new['steps'])} steps")
        done += 1
    return done, left


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("roots", type=Path, nargs="+", help="output folders to search")
    ap.add_argument("--to", default="1.5", choices=sorted(SPECS), help="target ATIF version")
    args = ap.parse_args(argv)
    for root in args.roots:
        if not root.is_dir():
            continue
        done, left = reshape_tree(root, args.to)
        print(f"[atif] {root}: {done} reshaped to ATIF-v{args.to}, {left} left as they are")
    return 0


if __name__ == "__main__":
    sys.exit(main())
