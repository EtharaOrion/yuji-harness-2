#!/bin/bash
# Everything that scores one run: the state dump, Channel A, the rubric and the
# ledger. It runs INSIDE the judge container (tools/judge), never in `main` --
# the container the agent had root in.
#
# This is the harness's copy, shared by every bundle. It used to be copied into
# each bundle as tests/evaluate.sh and drifted between them; one file means the
# numbers cannot move because a bundle shipped an older script. The judge mounts
# it with the rest of services/scoring at /harness/scoring, and codexbridge.py
# runs it as `bash /harness/scoring/tests/evaluate.sh` after writing the
# trajectory main posted to /tmp/agent_trajectory.json.
#
# What stays per-bundle, read from /tests: test_outputs.py, test_weights.json,
# rubric.json, state_dump.py and the answer files. Reports go to /logs/verifier,
# harbor's own per-trial directory, mounted here directly.
set -u

LOGS=/logs/verifier
TRAJ="${COMPLEXMCP_TRAJECTORY:-${ATLAS_TRAJECTORY:-/tmp/agent_trajectory.json}}"
mkdir -p "$LOGS"

# Every step reads the trajectory from the environment, and bundles differ on
# which name they read. Export all three once rather than per command.
export COMPLEXMCP_TRAJECTORY="$TRAJ" ATLAS_TRAJECTORY="$TRAJ" MCPATLAS_TRAJECTORY="$TRAJ"
export PYTHONPATH="/tests:/harness/scoring"

# -------------------------------------------------------------------- 2. state
if [ -f /tests/state_dump.py ]; then
    python3 /tests/state_dump.py \
        || echo "WARNING: state dump failed; no state diagnostic for this run." >&2
else
    echo "[2/5 state] no tests/state_dump.py; state channel not captured"
fi

# The state CHANNEL is retired, but positive Channel A tests read the snapshot
# this script writes. Without it they return False and cost real score with no
# error anywhere. Say so loudly rather than scoring it.
if [ ! -s "$LOGS/end_env.json" ]; then
    echo "ERROR: no end_env.json captured. Every schedule assertion in" >&2
    echo "       test_outputs.py will report False for want of a world snapshot," >&2
    echo "       not for want of agent work. Channel A is NOT comparable to a" >&2
    echo "       run that captured one. Investigate before reading the reward." >&2
    echo "state capture produced no end_env.json" > "$LOGS/state_capture_failed.txt"
fi

# ------------------------------------------------------------------- 3. rubric
JUDGE=/harness/scoring/rubric_judge_cli.py
if [ ! -f "$JUDGE" ]; then
    echo "FATAL: scoring harness not mounted -- $JUDGE is absent." >&2
    echo "       Check the /harness/scoring volume in environment/docker-compose.yaml;" >&2
    echo "       the ../ depth must reach the repo root exactly." >&2
    exit 2
fi
if python3 "$JUDGE" \
        --rubric /tests/rubric.json \
        --trajectory "$TRAJ" \
        --output "$LOGS/rubric_breakdown.json" \
        --token-output "$LOGS/judge_tokens.json"; then
    echo "[3/5 rubric] judged -> $LOGS/rubric_breakdown.json"
else
    echo "ERROR: rubric judge ran but failed; rubric channel is UNSCORED." >&2
    echo "rubric judge failed at run time" > "$LOGS/rubric_judge_failed.txt"
fi

# ------------------------------------------------------------------- 4. pytest
# Reporting only for bundles that grade in step 4b. It may fail softly for that
# reason. Older bundles still write the reward from inside this run, from
# test_write_reward_json in test_outputs.py; step 4b sees that and stands down.
python3 -m pytest /tests/test_outputs.py \
        -p ctrf_pytest_plugin \
        --junitxml="$LOGS/junit.xml" -v -p no:cacheprovider \
        -W ignore::pytest.PytestReturnNotNoneWarning \
        > "$LOGS/test-stdout.txt" 2>&1 || true
tail -n 30 "$LOGS/test-stdout.txt"
echo "[4/5 pytest] -> $LOGS/ctrf.json (junit.xml as fallback)"

# ---------------------------------------------------------------- 4b. grade
# The score. It was test_write_reward_json inside test_outputs.py, where pytest
# collected and reported it as a test and swallowed its exceptions. Now an
# explicit step, so a grader that dies fails the verifier instead of leaving a
# stale or zero reward behind. Deliberately NOT `|| true`.
#
# A bundle that still carries its own writer has already produced the file in
# step 4, under its own rules. Regrading it here would score its assert-style
# checks as False (an assert that passes returns None), so leave it alone.
if [ -s "$LOGS/reward_channel_a.json" ]; then
    echo "[4b/5 grade] reward written by the bundle's own pytest step; grade.py stands down"
elif grep -q 'def test_write_reward_json' /tests/test_outputs.py 2>/dev/null; then
    # Same bundle, but its writer did not run: its module skipped or errored in
    # step 4. Grading it here would score its assert-style checks False, so leave
    # the reward to step 5, which publishes zeros exactly as this bundle did
    # before the grading moved out of it.
    echo "ERROR: the bundle's own reward writer produced nothing; Channel A is NOT" >&2
    echo "       graded for this run. See $LOGS/test-stdout.txt." >&2
    echo "the bundle's own test_write_reward_json did not run" > "$LOGS/channel_a_not_graded.txt"
else
    python3 /harness/scoring/tests/grade.py \
        || { echo "ERROR: Channel A grader failed; reward is NOT trustworthy." >&2; exit 1; }
    echo "[4b/5 grade] -> $LOGS/reward_channel_a.json + grade_report.md"
fi

# ------------------------------------------------------------------- 5. reward
python3 - <<'PYEOF'
import json, pathlib
ca = pathlib.Path("/logs/verifier/reward_channel_a.json")
out = pathlib.Path("/logs/verifier/reward.json")
out.parent.mkdir(parents=True, exist_ok=True)
d = json.loads(ca.read_text()) if ca.exists() else {}
doc = {"reward": float(d.get("reward", 0.0)),
       "completion_rate": float(d.get("completion_rate", 0.0)),
       "misbehave_rate": float(d.get("misbehave_rate", 0.0))}
# This reward was computed here, in the judge container, and it folds in a rubric
# graded here too. Say so in a file of its own: harbor validates reward.json as
# numbers only (VerifierResult.rewards: dict[str, float | int]) and fails the
# whole trial on a string. scripts/run_task.sh carries the label into reward.json
# on the host, where harbor_to_output.py reads it.
#
# Claim it only when this reward really carries the rubric: a bundle whose ledger
# leaves the rubric out (a binary traj_pytest reward) must not stamp, or the host
# pass skips it and the rubric channel is silently lost.
#
# The test is the reward and the rubric's own verdicts, never judge_container.json:
# that marker is written by judge_client.py back in main, AFTER this script
# returns, so it is never here yet. Checking it meant the label was never written
# at all, the host pass re-graded every run, and reshape failed on runs it could
# not re-grade.
try:
    _rb = json.loads(pathlib.Path("/logs/verifier/rubric_breakdown.json").read_text())
    if isinstance(d.get("rubric"), (int, float)) and (_rb.get("per_criterion") or _rb.get("results")):
        pathlib.Path("/logs/verifier/reward_producer.json").write_text(
            json.dumps({"producer": "judge_container"}))
except (OSError, ValueError):
    pass
out.write_text(json.dumps(doc, indent=2))
print("[5/5 reward]", doc)
PYEOF
