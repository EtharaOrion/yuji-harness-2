#!/usr/bin/env python3
"""Write verifier/judge_tokens.json from the Kiro bridge call log.

Some bundles (genkit/rewardkit) judge through the bridge but never record the
judge's token usage, so finance_reporter.py posts `judge_lines: []`. The bridge
logs every call it serves (kiro_bridge_runner.py --calls-out); this sums the
judge-key calls that landed inside each run's verifier window and writes the
same judge_tokens.json shape the yuji rubric_judge_cli.py writes.

A run that already has judge_tokens.json is left alone: the bundle's own
numbers win. Runs with no judge calls in their window get no file.

Usage: kiro_judge_tokens.py --calls CALLS.jsonl RUN_DIR [RUN_DIR ...]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# USD per million tokens, Bedrock Global cross-Region (CRIS) prices, which for
# Claude are Anthropic's list prices. Regional and geo profiles (us., in., ...)
# cost 10% more; these rates assume the application inference profiles in
# BEDROCK_MODEL_IDS wrap global.* profiles. A valuation only for Kiro, which
# bills credits, not these rates.
#
# gpt-5.6-sol: Bedrock model card, Global CRIS, short context (<=272K input);
# in ap-south-1 Sol is offered on Global CRIS only. AWS calls the $4/$20 cut
# promotional "at least through November 21, 2026" -- recheck after that date.
# The yuji judge image (judge-1.1.0) still carries the old $5/$30 Sol rate in
# its own _JUDGE_RATES, so judge lines it writes itself value Sol higher.
RATES = {
    "gpt-5.6-sol":       {"input": 4.0, "output": 20.0, "cache_read": 0.4, "cache_write": 5.00},
    "claude-opus-5":     {"input": 5.0, "output": 25.0, "cache_read": 0.5, "cache_write": 6.25},
    "claude-opus-4-5":   {"input": 5.0, "output": 25.0, "cache_read": 0.5, "cache_write": 6.25},
    "claude-sonnet-5":   {"input": 2.0, "output": 10.0, "cache_read": 0.2, "cache_write": 2.50},
    "claude-sonnet-4-5": {"input": 3.0, "output": 15.0, "cache_read": 0.3, "cache_write": 3.75},
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0, "cache_read": 0.3, "cache_write": 3.75},
    "claude-haiku-4-5":  {"input": 1.0, "output": 5.0,  "cache_read": 0.1, "cache_write": 1.25},
    # glm-5.3: z.ai API list price (docs.z.ai/guides/overview/pricing); cache storage is free for now.
    "glm-5.3":           {"input": 1.4, "output": 4.4,  "cache_read": 0.26, "cache_write": 0.0},
}
RATE_ENV = {
    "input": "JUDGE_PRICE_INPUT_PER_MTOK",
    "output": "JUDGE_PRICE_OUTPUT_PER_MTOK",
    "cache_read": "JUDGE_PRICE_CACHED_INPUT_PER_MTOK",
    "cache_write": "JUDGE_PRICE_CACHE_WRITE_PER_MTOK",
}


def _ts(raw: str) -> datetime | None:
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None


def _int(usage: dict, key: str) -> int:
    try:
        return max(0, int(usage.get(key) or 0))
    except (TypeError, ValueError):
        return 0


def _cost(model: str, fresh: int, output: int, cache_read: int, cache_write: int) -> float:
    rates = dict(RATES.get(model, {}))
    for key, var in RATE_ENV.items():
        if os.getenv(var, "").strip():
            try:
                rates[key] = float(os.environ[var])
            except ValueError:
                rates[key] = 0.0
    return round(fresh / 1e6 * rates.get("input", 0.0)
                 + output / 1e6 * rates.get("output", 0.0)
                 + cache_read / 1e6 * rates.get("cache_read", 0.0)
                 + cache_write / 1e6 * rates.get("cache_write", 0.0), 6)


def load_calls(path: Path) -> list[dict]:
    calls = []
    if not path.is_file():
        return calls
    for line in path.read_text().splitlines():
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        ts = _ts(rec.get("ts", ""))
        if ts is not None and rec.get("label") == "judge":
            rec["_ts"] = ts
            calls.append(rec)
    return calls


def lines_for_window(calls: list[dict], start: datetime, end: datetime) -> list[dict]:
    by_model: dict[str, dict] = {}
    for rec in calls:
        if not (start <= rec["_ts"] <= end):
            continue
        usage = rec.get("usage") or {}
        model = str(rec.get("served") or rec.get("requested") or "")
        agg = by_model.setdefault(model, {
            "model_name": model, "judge_wire": "anthropic",
            "judge_input_tokens": 0, "judge_output_tokens": 0,
            "judge_input_cache_tokens": 0, "judge_cache_write_tokens": 0,
            "judge_turns": 0, "judge_failed_calls": 0,
            "judge_usage_source": "kiro_bridge_call_log",
        })
        if not rec.get("ok"):
            agg["judge_failed_calls"] += 1
            continue
        agg["judge_input_tokens"] += _int(usage, "input_tokens")
        agg["judge_output_tokens"] += _int(usage, "output_tokens")
        agg["judge_input_cache_tokens"] += _int(usage, "cache_read_input_tokens")
        agg["judge_cache_write_tokens"] += _int(usage, "cache_creation_input_tokens")
        agg["judge_turns"] += 1
        if usage.get("usage_estimated"):
            agg["judge_usage_estimated"] = True
    out = []
    for agg in by_model.values():
        if agg["judge_turns"] == 0:
            continue
        agg["judge_cost_usd"] = _cost(agg["model_name"], agg["judge_input_tokens"],
                                      agg["judge_output_tokens"], agg["judge_input_cache_tokens"],
                                      agg["judge_cache_write_tokens"])
        out.append(agg)
    return out


def backfill(run_dir: Path, calls: list[dict]) -> str:
    target = run_dir / "verifier" / "judge_tokens.json"
    if target.exists():
        return "kept (bundle wrote its own)"
    try:
        verifier = json.loads((run_dir / "result.json").read_text()).get("verifier") or {}
    except (OSError, ValueError):
        return "skipped (no readable result.json)"
    start, end = _ts(verifier.get("started_at", "")), _ts(verifier.get("finished_at", ""))
    if start is None or end is None:
        return "skipped (no verifier window)"
    lines = lines_for_window(calls, start, end)
    if not lines:
        return "skipped (no judge calls in verifier window)"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(lines, indent=2) + "\n")
    turns = sum(line["judge_turns"] for line in lines)
    return f"wrote {turns} judge call(s)"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--calls", required=True, help="bridge call log (JSONL)")
    ap.add_argument("run_dirs", nargs="+", type=Path)
    args = ap.parse_args()
    calls = load_calls(Path(args.calls))
    for run_dir in args.run_dirs:
        if run_dir.is_dir():
            print(f"[judge_tokens] {run_dir.name}: {backfill(run_dir, calls)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
