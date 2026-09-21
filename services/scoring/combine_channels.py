"""Three-channel reward combiner: Channel A + Channel B + Channel C.

Reads the three per-channel reward files produced by the grading pipeline
and produces the final reward.json with both a raw signed reward (for RL
training) and a normalized [0, 1] reward (for legacy consumers).

Channels:
  A = state assertions (rule-based, [0, 1])              -> reward_channel_a.json
  B = rubric criteria (LLM 11-trial, [0, 1])             -> reward_channel_b.json OR rubric_breakdown.json
  C = claim coverage (rule-based, signed unbounded)      -> reward_channel_c.json

Weights come from tests/test_weights.json under `components.{traj_tests,
rubric, coverage}.weight`. If a channel file is missing OR its component is
inert (weight 0 or graded=false), that channel is dropped from the reward.

Reward formulas:

  reward_raw = (wA * A + wB * B + wC * C_raw) / (wA + wB + wC)

  max_possible_raw = (wA * 1 + wB * 1 + wC * C_range_max) / (wA + wB + wC)
  min_possible_raw = (wA * 0 + wB * 0 + wC * C_range_min) / (wA + wB + wC)
  reward_normalized = max(0, min(1, (reward_raw - min_possible_raw) /
                                    (max_possible_raw - min_possible_raw)))

Backward compat: if `coverage` component is absent from test_weights.json,
the combiner falls back to Channel A + Channel B only and reward_raw ==
reward_normalized (both in [0, 1]).

Invocation (from container):
    python3 /harness/scoring/combine_channels.py

Or standalone:
    LOGS_DIR=/tmp/rc TEST_WEIGHTS=/path/to/test_weights.json \\
        python3 combine_channels.py

Output: writes /logs/verifier/reward.json (overwriting any prior contents)
with the three-channel combined reward.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

LOGS = Path(os.environ.get("LOGS_DIR", "/logs/verifier"))
TEST_WEIGHTS = Path(os.environ.get("TEST_WEIGHTS", "/tests/test_weights.json"))
OUTPUT = Path(os.environ.get("REWARD_OUTPUT", str(LOGS / "reward.json")))

CHANNEL_A_FILE = LOGS / "reward_channel_a.json"
CHANNEL_B_FILE = LOGS / "reward_channel_b.json"
CHANNEL_C_FILE = LOGS / "reward_channel_c.json"
RUBRIC_BREAKDOWN = LOGS / "rubric_breakdown.json"


def load_weights_components() -> dict:
    if not TEST_WEIGHTS.exists():
        return {}
    try:
        data = json.loads(TEST_WEIGHTS.read_text())
    except Exception:
        return {}
    return data.get("components", {})


def load_channel_a() -> float | None:
    """Channel A in [0, 1].

    Bundle-local grade.py writes it as `score`; the harness grader
    (tests/grade.py, run by evaluate.sh) writes it as `channel_a`. Reading only
    `score` silently dropped traj_tests from the ledger for every bundle graded
    by the harness, and the combined reward was computed without it.
    """
    if not CHANNEL_A_FILE.exists():
        return None
    try:
        data = json.loads(CHANNEL_A_FILE.read_text())
        value = data.get("score")
        if value is None:
            value = data.get("channel_a")
        return float(value)
    except (ValueError, TypeError, KeyError):
        return None


def load_channel_b() -> float | None:
    for path in (CHANNEL_B_FILE, RUBRIC_BREAKDOWN):
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        for key in ("score", "value", "reward"):
            if key in data:
                try:
                    return float(data[key])
                except (ValueError, TypeError):
                    continue
    return None


def load_channel_c() -> dict | None:
    if not CHANNEL_C_FILE.exists():
        return None
    try:
        return json.loads(CHANNEL_C_FILE.read_text())
    except Exception:
        return None


def combine() -> dict:
    components = load_weights_components()
    wA = float(components.get("traj_tests", {}).get("weight", 0.0))
    wB = float(components.get("rubric", {}).get("weight", 0.0))
    wC = float(components.get("coverage", {}).get("weight", 0.0))

    channel_a = load_channel_a() if wA else None
    channel_b = load_channel_b() if wB else None
    channel_c_data = load_channel_c() if wC else None

    channel_c_raw = None
    if channel_c_data and channel_c_data.get("available", True):
        channel_c_raw = float(channel_c_data.get("raw", 0.0))

    active_weights = 0.0
    weighted_sum = 0.0
    ledger = {}

    if channel_a is not None and wA:
        active_weights += wA
        weighted_sum += wA * channel_a
        ledger["traj_tests"] = {"weight": wA, "value": round(channel_a, 4)}

    if channel_b is not None and wB:
        active_weights += wB
        weighted_sum += wB * channel_b
        ledger["rubric"] = {"weight": wB, "value": round(channel_b, 4)}

    if channel_c_raw is not None and wC:
        active_weights += wC
        weighted_sum += wC * channel_c_raw
        ledger["coverage"] = {
            "weight": wC,
            "raw": round(channel_c_raw, 2),
            "range_min": channel_c_data.get("range_min", -20.0),
            "range_max": channel_c_data.get("range_max", 12.0),
            "pos_hits": channel_c_data.get("pos_hits"),
            "pos_max": channel_c_data.get("pos_max"),
            "neg_hits": channel_c_data.get("neg_hits"),
        }

    if active_weights <= 0:
        return {
            "reward_raw": None,
            "reward_normalized": None,
            "reward": None,
            "passed": False,
            "quadrant": "UNSCORED",
            "unscored_reason": "no active channels; all weights zero or all channel files missing",
            "ledger": ledger,
        }

    reward_raw = round(weighted_sum / active_weights, 4)

    max_possible = 0.0
    min_possible = 0.0
    for name, weight in (("traj_tests", wA), ("rubric", wB)):
        if name in ledger:
            max_possible += weight * 1.0
    if "coverage" in ledger:
        max_possible += wC * ledger["coverage"]["range_max"]
        min_possible += wC * ledger["coverage"]["range_min"]

    if max_possible > min_possible:
        normalized = (reward_raw * active_weights - min_possible) / (max_possible - min_possible)
        reward_normalized = round(max(0.0, min(1.0, normalized)), 4)
    else:
        reward_normalized = round(max(0.0, min(1.0, reward_raw)), 4)

    passed = reward_normalized >= 0.5
    quadrant = "PASSED" if passed else "FAILED"

    return {
        "reward_raw": reward_raw,
        "reward_normalized": reward_normalized,
        "reward": reward_normalized,
        "passed": passed,
        "quadrant": quadrant,
        "comparable": True,
        "threshold": 0.5,
        "ledger": ledger,
        "reward_formula_raw": "(wA*A + wB*B + wC*C_raw) / (wA + wB + wC)",
        "reward_formula_normalized": "max(0, min(1, (reward_raw - min_possible) / (max_possible - min_possible)))",
        "active_weights_sum": active_weights,
        "max_possible_normalized_input": max_possible,
        "min_possible_normalized_input": min_possible,
    }


def main() -> int:
    result = combine()
    LOGS.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2))
    if result.get("reward_raw") is None:
        print("combine_channels: UNSCORED - no active channels")
        return 1
    print(
        f"combine_channels: reward_raw={result['reward_raw']:+.4f} "
        f"reward_normalized={result['reward_normalized']:.4f} "
        f"quadrant={result['quadrant']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
