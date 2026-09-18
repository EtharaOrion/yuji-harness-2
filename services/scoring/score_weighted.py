#!/usr/bin/env python3
"""
services/scoring/score_weighted.py

CSV-pipeline entry point for the weighted grader. Purely additive alongside
score_claims.py's mean-coverage scoring — a task only gets a weighted_reward
if it has a tests/test_weights.json opting a component in; every other row
gets weighted_reward = "" (blank), so running this over a CSV that has no
weighted tasks at all changes nothing that score_claims.py already reports.

Input: a merged CSV in the shape score_claims.py's merge_gtfa_with_model_data()
produces (or run_eval.py's raw output, which already has task_id/response/
raw_conversation_history/run_trajectory_json) — this script only needs
task_id, a response column, and a trajectory column; GTFA_CLAIMS is read too
as the legacy Channel-B fallback when a task has no tests/rubric.json.

Per task_id, this script looks for an opt-in tests/ dir at
<tasks-dir>/<task_id>/tests/ (the same layout convert_tasks_to_harbor.py /
adapters/mcp_atlas/adapter.py already generate for Harbor bundles):

    tests/test_weights.json   — component weights (see weighted_judge.py)
    tests/test_outputs.py     — Channel A pytest assertions
    tests/rubric.json         — Channel B polarity-tagged criteria (optional;
                                 falls back to the CSV's own GTFA_CLAIMS column
                                 as legacy weight=1/is_positive=True criteria
                                 if rubric.json is absent but rubric.weight > 0)

A task with no tests/ dir at all, or a test_weights.json with both component
weights at 0, is left alone entirely (weighted_reward stays blank).

Trajectory resolution per row: prefers the run_trajectory_json column (the
richer RunTrajectory shape, see run_eval.py) and falls back to
raw_conversation_history (the flat OpenAI-message-list shape) when that's
blank — services/scoring/traj_asserts.py accepts both.

Usage:
    python services/scoring/score_weighted.py \\
        --merged-csv scoring_results/gpt-4o_n100_.../merged_gpt-4o.csv \\
        --tasks-dir output/harbor \\
        --model-name gpt-4o \\
        --output scoring_results/gpt-4o_n100_.../weighted_gpt-4o.csv

    # Channel B (rubric) needs a judge model, same env vars as score_claims.py:
    #   EVAL_LLM_BASE_URL / EVAL_LLM_API_KEY (or LLM_BASE_URL / LLM_API_KEY)
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

csv.field_size_limit(sys.maxsize)

_SCORING_DIR = Path(__file__).resolve().parent
if str(_SCORING_DIR) not in sys.path:
    sys.path.insert(0, str(_SCORING_DIR))

import weighted_judge as wj  # noqa: E402
import rubric_weighted as rw  # noqa: E402
# score_claims.py pulls in a heavy chain (dotenv, matplotlib, aiohttp,
# tenacity, nest_asyncio) at import time — only import from it lazily, right
# where it's needed, so a traj-tests-only run doesn't require any of that.


def _response_column(row: dict[str, Any], model_name: str) -> str:
    for col in (f"{model_name}_response", "script_model_response", "response"):
        val = row.get(col)
        if val is not None and str(val).strip():
            return str(val)
    return ""


def _trajectory_for_row(row: dict[str, Any]) -> dict | list | None:
    raw = row.get("run_trajectory_json")
    if raw and str(raw).strip():
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
    raw = row.get("raw_conversation_history")
    if raw and str(raw).strip():
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
    return None


def _rubric_criteria_for_task(tests_dir: Path, row: dict[str, Any]) -> list[rw.RubricCriterion]:
    rubric_path = tests_dir / "rubric.json"
    if rubric_path.exists():
        return rw.parse_rubric(json.loads(rubric_path.read_text(encoding="utf-8")))
    # Legacy fallback: today's GTFA_CLAIMS column, each claim weight=1/goal.
    from score_claims import extract_claims
    return rw.parse_rubric(extract_claims(row.get("GTFA_CLAIMS", "")))


async def score_row(
    row: dict[str, Any],
    *,
    tasks_dir: Path | None,
    model_name: str,
    rubric_evaluator: Any | None,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    task_id = str(row.get("task_id") or row.get("TASK") or "").strip()
    out: dict[str, Any] = {
        "task_id": task_id,
        "weighted_reward": "",
        "traj_tests_value": "",
        "rubric_value": "",
        "weighted_ledger_json": "",
        "weighted_detail_json": "",
    }
    if not task_id or tasks_dir is None:
        return out

    tests_dir = tasks_dir / task_id / "tests"
    weights_file = tests_dir / "test_weights.json"
    if not weights_file.exists():
        return out  # task never opted in — leave every weighted_* column blank

    weights = wj.load_weights(weights_file)
    if not weights.traj_tests.weight and not weights.rubric.weight:
        return out  # test_weights.json present but inert (e.g. flat legacy shape)

    response = _response_column(row, model_name)
    traj_results: dict[str, bool] | None = None
    if weights.traj_tests.weight:
        trajectory = _trajectory_for_row(row)
        test_outputs = tests_dir / "test_outputs.py"
        if trajectory is not None and test_outputs.exists():
            async with semaphore:
                traj_results = await asyncio.to_thread(wj._run_traj_pytest, trajectory, test_outputs)

    rubric_value = None
    rubric_rows: list[dict[str, Any]] = []
    if weights.rubric.weight and rubric_evaluator is not None and response:
        criteria = _rubric_criteria_for_task(tests_dir, row)
        async with semaphore:
            rb = await rw.evaluate_rubric(rubric_evaluator, criteria, response)
        rubric_value = rb["value"]
        rubric_rows = rb["rows"]

    result = wj.judge_weighted(
        test_weights_file=weights_file,
        traj_results=traj_results,
        rubric_value=rubric_value,
        rubric_weight=weights.rubric.weight,
        rubric_rows=rubric_rows,
    )
    out["weighted_reward"] = result["reward"] if result["reward"] is not None else ""
    out["traj_tests_value"] = result["ledger"].get("traj_tests", {}).get("value", "")
    out["rubric_value"] = result["ledger"].get("rubric", {}).get("value", "")
    out["weighted_ledger_json"] = json.dumps(result["ledger"])
    out["weighted_detail_json"] = json.dumps(result)
    return out


async def run(args: argparse.Namespace) -> None:
    import pandas as pd

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                         format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    df = pd.read_csv(args.merged_csv)
    tasks_dir = Path(args.tasks_dir) if args.tasks_dir else None

    rubric_evaluator = None
    if args.enable_rubric:
        from score_claims import AsyncLiteLLMClient, CoverageEvaluator, EvaluatorConfig
        config = EvaluatorConfig(model_name=args.evaluator_model, api_key=args.api_key, base_url=args.base_url)
        rubric_evaluator = CoverageEvaluator(AsyncLiteLLMClient(config), config)

    semaphore = asyncio.Semaphore(args.concurrency)
    rows = df.to_dict(orient="records")

    results = await asyncio.gather(*[
        score_row(row, tasks_dir=tasks_dir, model_name=args.model_name,
                  rubric_evaluator=rubric_evaluator, semaphore=semaphore)
        for row in rows
    ])

    for col in ("weighted_reward", "traj_tests_value", "rubric_value", "weighted_ledger_json", "weighted_detail_json"):
        df[col] = [r[col] for r in results]

    out_path = args.output or args.merged_csv
    df.to_csv(out_path, index=False)

    opted_in = [r for r in results if r["weighted_reward"] != ""]
    logger.info(f"Wrote {out_path}")
    logger.info(f"{len(opted_in)}/{len(results)} tasks opted into weighted grading")
    if opted_in:
        mean_reward = sum(float(r["weighted_reward"]) for r in opted_in) / len(opted_in)
        logger.info(f"Mean weighted_reward (opted-in tasks only): {mean_reward:.4f}")

    if args.write_artifacts and tasks_dir is not None:
        artifacts_root = Path(args.write_artifacts)
        for row in results:
            if row["weighted_reward"] == "":
                continue
            detail = json.loads(row["weighted_detail_json"])
            wj.write_verifier_artifacts(artifacts_root / row["task_id"] / "verifier", detail)
        logger.info(f"Wrote per-task verifier artifacts under {artifacts_root}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--merged-csv", required=True, help="CSV with task_id, response, and trajectory columns")
    parser.add_argument("--tasks-dir", default=None,
                         help="Root dir of <task_id>/tests/{test_weights.json,test_outputs.py,rubric.json} bundles. "
                              "Without this, every row is left unscored (weighted_reward stays blank).")
    parser.add_argument("--model-name", required=True, help="Same model-name used when the CSV was produced")
    parser.add_argument("--output", default=None, help="Output CSV path (default: overwrite --merged-csv)")
    parser.add_argument("--enable-rubric", action="store_true",
                         help="Actually make Channel B (rubric) LLM-judge calls. Without this flag, "
                              "any task with rubric.weight > 0 is skipped for Channel B (traj_tests still runs).")
    parser.add_argument("--evaluator-model", default=os.getenv("EVAL_LLM_MODEL", "gemini/gemini-3.1-pro-preview"))
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--write-artifacts", default=None,
                         help="If set, also write Harbor-shaped verifier/{reward.json,reward.txt,ctrf.json,detail.json} "
                              "per opted-in task under this directory")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
