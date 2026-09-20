#!/usr/bin/env python3
"""Grade a finished trial's rubric channel on the HOST, then recompute reward.

WHY THIS EXISTS

`tests/test.sh` grades every channel inside the task container. That works for
Channel A and the state channel, which need the live light-servers the agent
wrote to. It does not work for the rubric channel when the judge is codex:
python:3.12-slim has no `codex` binary, and the only way to put a working one
there is to mount the operator's ChatGPT credential into a container that just
ran an agent under --permission-mode=bypassPermissions. Harbor's verifier
cannot be isolated from that container either -- environment_mode='separate'
brings up "a fresh copy of the top-level [environment]", which restarts
light-servers clean and destroys the world the state channel exists to read.

So the rubric is graded here instead, on the host, where codex is already
installed and logged in and the credential never leaves the machine it belongs
to. The container keeps grading everything that needs the live world.

WHAT IT RECOMPUTES, AND WHY IT CANNOT JUST PATCH THE NUMBER

test_outputs.py accumulates the denominator only over components it actually
scored (`den += w` sits after the `v is None` guard). An unscored rubric is
therefore excluded from the divisor rather than counted as zero, so folding a
rubric score in afterwards changes both halves of the fraction. The ledger math
below mirrors test_outputs.py exactly for that reason -- including the gates,
which can zero a reward that the arithmetic alone would leave positive.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools" / "delivery"))
from harbor_to_output import fmt_reward, norm_reward  # noqa: E402

sys.path.insert(0, str(REPO / "services" / "scoring"))

import agent_log_to_trajectory as alt  # noqa: E402


def _load(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def recompute_reward(weights: dict, chan_a, rubric_value, rc_val, rb_val,
                     guards_tripped: list) -> tuple[float, dict]:
    """The ledger from test_outputs.py, kept line-for-line equivalent."""
    comps = weights.get("components") or {}
    values = {"traj_tests": chan_a, "rubric": rubric_value,
              "state_completion": rc_val}
    num, den, ledger = 0.0, 0.0, {}
    for name, spec in comps.items():
        spec = spec or {}
        w = spec.get("weight") or 0
        if name == "state_misbehave":
            continue
        if not spec.get("graded") or w <= 0:
            ledger[name] = {"status": "retired", "weight": w, "value": None}
            continue
        v = values.get(name)
        if v is None:
            ledger[name] = {"status": "unscored", "weight": w, "value": None}
            continue
        # `value` is what the ledger publishes; the arithmetic below stays on
        # the full float. Cutting only the published copy keeps the report
        # readable at the tree's two places without the reward inheriting a
        # truncation on every component it sums.
        ledger[name] = {"status": "scored", "weight": w, "value": norm_reward(float(v))}
        num += w * float(v)
        den += w

    mis = comps.get("state_misbehave") or {}
    mis_w = mis.get("weight") or 0
    if mis.get("graded") and mis_w < 0 and rb_val is not None:
        num -= abs(mis_w) * float(rb_val)
        ledger["state_misbehave"] = {"status": "scored", "weight": mis_w,
                                     "severity": norm_reward(float(rb_val))}
    else:
        ledger["state_misbehave"] = {
            "status": "retired" if not mis.get("graded") else "unscored",
            "weight": mis_w, "severity": None}

    reward = norm_reward(max(0.0, num / den)) if den else 0.0

    gate = weights.get("gate") or {}
    if gate.get("require_state_exact") and rc_val is not None \
            and (rc_val < 1.0 or (rb_val or 0) > 0):
        reward = 0.0
    if gate.get("require_no_guards") and guards_tripped:
        reward = 0.0
    floor = gate.get("floor")
    if floor is not None and reward < float(floor):
        reward = 0.0
    return reward, ledger


def _comparability(ledger: dict, verifier: Path | None = None) -> tuple[bool, list[str]]:
    """Name the graded components the ledger had to leave out, if any.

    recompute_reward drops an unscored component from BOTH halves of the
    fraction, so a reward missing one is a different quantity from a full grade
    -- same 0..1 scale, different divisor, and the number itself cannot say so.

    `verifier` adds the one distinction the ledger cannot make on its own. A
    state channel reads "unscored" both when the bundle retired it and when the
    capture ran and failed, and those are not the same claim: the first is a
    grading decision, the second is an infrastructure fault that silently cost
    a channel worth w=5. test.sh writes state_capture_failed.txt into this same
    directory when it ends with no end_env.json, so the marker is the only
    evidence that separates them -- and until now nothing read it.
    """
    caveats = []
    marker = (verifier / "state_capture_failed.txt") if verifier else None
    if marker is not None and marker.exists():
        try:
            detail = marker.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            detail = ""
        caveats.append(
            "state capture FAILED: the state channels below are missing because "
            "the post-run world could not be read, not because this task retired "
            "them" + (f" ({detail})" if detail else "")
        )
    for name, row in sorted(ledger.items()):
        if (row or {}).get("status") != "unscored":
            continue
        if name == "state_misbehave":
            caveats.append(f"{name} unscored: no misbehaviour penalty was applied")
        else:
            caveats.append(f"{name} unscored: dropped from the reward's divisor")
    return not caveats, caveats


def _sync_harbor_result(trial: Path, reward: float) -> None:
    """Rewrite the stale reward Harbor recorded before the host pass ran.

    Harbor finalises its own result.json at the end of the verify phase, which
    is necessarily before this pass grades the rubric. Both the trial-level and
    job-level records therefore hold the pre-rubric number. Best-effort: a
    stale summary is bad, but it is not worth failing a graded run over.
    """
    for path, mutate in (
        (trial / "result.json", lambda d: (d.get("verifier_result") or {}).get("rewards")),
        (trial.parent / "result.json", None),
    ):
        try:
            if not path.exists():
                continue
            doc = json.loads(path.read_text())
            touched = 0
            if mutate is not None:
                rewards = mutate(doc)
                if isinstance(rewards, dict):
                    rewards["reward"] = reward
                    touched += 1
            else:
                for ev in ((doc.get("stats") or {}).get("evals") or {}).values():
                    for metric in ev.get("metrics") or []:
                        if isinstance(metric, dict) and "reward" in metric:
                            metric["reward"] = reward
                            touched += 1
                    # reward_stats buckets trial names under the reward as a
                    # STRING KEY, so a stale entry here is a second, contradictory
                    # copy of the score in the same file -- and it kept the raw
                    # float even after metrics[] was synced. Re-key only this
                    # trial's entry; sibling trials own their own buckets.
                    rstats = (ev.get("reward_stats") or {}).get("reward")
                    if isinstance(rstats, dict):
                        for key in list(rstats):
                            rest = [t for t in rstats[key] if t != trial.name]
                            if len(rest) != len(rstats[key]):
                                if rest:
                                    rstats[key] = rest
                                else:
                                    del rstats[key]
                                rstats.setdefault(str(reward), []).append(trial.name)
                                touched += 1
            if not touched:
                # No reward field exists to correct -- the trial died before the
                # verify phase, so there is no stale number here. Say that
                # instead of printing "synced", which claimed a write that never
                # happened. The graded reward still lives in verifier/reward.json,
                # which is what the output pipeline actually reads.
                print(f"[host-rubric] nothing to sync in {path.name} "
                      "(no verifier reward recorded); reward.json remains the source")
                continue
            path.write_text(json.dumps(doc, indent=4))
            print(f"[host-rubric] synced {path.name} -> {fmt_reward(reward)} ({touched} field(s))")
        except (OSError, ValueError, AttributeError) as exc:
            print(f"[host-rubric] could not sync {path}: {exc!r}", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial", required=True, help="output/<job>/<job>__<id>/")
    ap.add_argument("--task", required=True, help="tasks/<task>/")
    ap.add_argument("--model", default=None, help="judge model (default: JUDGE_MODEL, else codex)")
    ap.add_argument("--dry-run", action="store_true", help="convert and report, do not call the judge")
    ap.add_argument("--rejudge", action="store_true",
                    help="call the judge here even when the judge container already wrote verdicts")
    a = ap.parse_args()

    trial, task = Path(a.trial), Path(a.task)
    verifier = trial / "verifier"
    verifier.mkdir(parents=True, exist_ok=True)

    rubric = task / "tests" / "rubric.json"
    weights_path = task / "tests" / "test_weights.json"
    if not rubric.exists():
        print(f"[host-rubric] no rubric at {rubric}; nothing to grade", file=sys.stderr)
        return 0

    log = alt.find_agent_log(trial / "agent")
    traj = alt.build_trajectory(log)
    print(f"[host-rubric] agent log: {log}")
    print(f"[host-rubric] {len(traj['steps'])} tool calls, "
          f"final_message {len(traj['final_message'])} chars")
    if not traj["steps"] and not traj["final_message"]:
        print("[host-rubric] empty trajectory; refusing to grade nothing", file=sys.stderr)
        return 1

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(traj, fh)
        traj_path = Path(fh.name)

    breakdown = verifier / "rubric_breakdown.json"
    if a.dry_run:
        print(f"[host-rubric] dry run; trajectory at {traj_path}")
        return 0

    # Verdicts the bundle's judge container already wrote (tools/judge). Reused,
    # not re-bought: for such a trial the host's only job is the ledger, which
    # the bundle's own reward may leave the rubric out of.
    graded_in = "host"
    container = _load(verifier / "judge_container.json", {}) or {}
    prior = _load(breakdown, {}) or {}
    prior_rows = prior.get("per_criterion") or prior.get("results")
    if container.get("ok") is True and prior_rows and not a.rejudge:
        graded_in = "judge-container"
        traj_path.unlink(missing_ok=True)
        print(f"[host-rubric] rubric already graded in the judge container "
              f"({container.get('model')}); recomputing the ledger only")
    else:
        cmd = [sys.executable, str(REPO / "services" / "scoring" / "rubric_judge_cli.py"),
               "--rubric", str(rubric), "--trajectory", str(traj_path),
               "--output", str(breakdown),
               "--token-output", str(verifier / "judge_tokens.json")]
        if a.model:
            cmd += ["--model", a.model]
        print(f"[host-rubric] judging with {a.model or os.getenv('JUDGE_MODEL') or 'codex default'} ...")
        proc = subprocess.run(cmd)
        traj_path.unlink(missing_ok=True)
        if proc.returncode != 0:
            print("[host-rubric] judge failed; rubric channel stays UNSCORED", file=sys.stderr)
            return proc.returncode

    rb = _load(breakdown, {}) or {}
    rubric_value = rb.get("score")
    rubric_value = float(rubric_value) if isinstance(rubric_value, (int, float)) else None

    weights = _load(weights_path, {}) or {}
    _chan_a_path = verifier / "reward_channel_a.json"
    prior = _load(_chan_a_path, {}) or {}
    state = _load(verifier / "state_channel.json", {}) or {}
    chan_a = prior.get("channel_a")
    if chan_a is None:
        print(f"[host-rubric] WARNING: no channel_a in {_chan_a_path.name}; "
              "traj_tests is DROPPED from the ledger and the reward comes from "
              "the rubric alone -- not comparable to a full in-pipeline grade",
              file=sys.stderr)
    rc_val = state.get("completion") if state.get("available") else None
    rb_val = state.get("misbehave") if state.get("available") else None
    guards = prior.get("guards_tripped") or []

    reward, ledger = recompute_reward(weights, chan_a, rubric_value, rc_val, rb_val, guards)
    comparable, caveats = _comparability(ledger, verifier)

    out = dict(prior)
    out.update({"reward": reward, "rubric": norm_reward(rubric_value),
                "ledger": ledger,
                "rubric_graded_on": graded_in, "grader": "weighted_ledger"})
    (verifier / "reward_channel_a.json").write_text(json.dumps(out, indent=2))
    (verifier / "reward.json").write_text(json.dumps(
        {"reward": reward,
         "completion_rate": norm_reward(out.get("completion_rate", 0.0)),
         "misbehave_rate": norm_reward(out.get("misbehave_rate", 0.0)),
         "comparable": comparable, "caveats": caveats,
         "producer": "host_rubric_pass"}, indent=2))

    # Harbor recorded its own reward before this pass ran, so its result.json
    # still holds the rubric-less number. Leaving both on disk means one trial
    # with two different scores and no indication which is current. Update the
    # job-level record to match the ledger.
    _sync_harbor_result(trial, reward)

    print(f"[host-rubric] rubric={rubric_value} channel_a={chan_a} "
          f"state_completion={rc_val} -> reward={fmt_reward(reward)}"
          f"{'' if comparable else '  NOT COMPARABLE'}")
    for caveat in caveats:
        print(f"[host-rubric]   {caveat}", file=sys.stderr)
    for name, row in ledger.items():
        print(f"    {name:18} {row.get('status'):9} w={row.get('weight')} "
              f"v={row.get('value', row.get('severity'))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
