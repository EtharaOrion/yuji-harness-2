#!/usr/bin/env bash
# Re-run ONLY the verifier on delivery-shaped bundles and write the new score
# back IN PLACE into each run's verifier/, one task at a time.
#
#   scripts/regrade_delivery.sh <task-path> [--judge codex] [--dry-run]
#
#   <task-path> is either:
#     - one task dir (holds task.toml + trajectories/<model>/run_N), or
#     - a folder of task dirs: every task dir holding trajectories/<model>/run_N
#       is regraded one by one, in name order; a failure is reported and the
#       batch continues. Per-task logs and a summary land under
#       /tmp/regrade-dataset/batch-<ts>/.
#
# These are delivery-shaped bundles: <uuid>/trajectories/<model>/run_N with
# agent/trajectory.json, artifacts/manifest.json + artifacts/workspace/, and a
# verifier/ holding the captured world snapshot end_env.json. The agent is NOT
# re-run. The verifier runs offline in a separate container: it reads the world
# state from the snapshot on disk, never dialling the Light MCP servers; only
# the judge LLM needs the network.
#
#   What this does per run:
#     1. reshape the bundle into the harbor trial layout harbor regrade expects
#        (artifacts/manifest.json declares /logs/agent -> artifacts/logs/agent and
#        /logs/artifacts -> artifacts/logs/artifacts), placing verifier/end_env.json
#        at artifacts/logs/artifacts/atlas/end_env.json where ATLAS_END_ENV resolves;
#     2. `harbor trial regrade` (NopAgent, separate verifier only);
#     3. write the new verifier/ back over the run's verifier/, keeping a backup.
#
#   NOTE: delivery bundles ship no server_tool_calls.log. Checks that need MCP
#   calls seen only in that log (not in agent/trajectory.json) fail closed. In
#   practice the trajectory already carries the calls, so the pytest channel is
#   reproduced; the rubric/judge channel is what the regrade refreshes.
#
#   --judge codex   (default) rewardkit's own codex judge inside the verifier
#                   (JUDGE_TRANSPORT=codex) on your `codex login` (ChatGPT
#                   subscription), shipped as CODEX_AUTH_B64; model
#                   $CODEX_JUDGE_MODEL (default gpt-5.6-sol).
#   --dry-run       regrade only; print the new score, change nothing in <task-path>.
#
# A backup of each run's verifier/ is kept under
# /tmp/regrade-dataset/<task-id>/<model>-run_N-backup-<ts>/ before write-back.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

J="" JUDGE="${REGRADE_JUDGE:-codex}" DRY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --judge) JUDGE="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) sed -n '2,45p' "$0"; exit 0 ;;
    *) [ -z "$J" ] && J="$1" || { echo "unexpected arg: $1" >&2; exit 2; }; shift ;;
  esac
done
[ -n "$J" ] || { echo "usage: $0 <task-path> [--judge codex] [--dry-run]" >&2; exit 2; }
[ "$JUDGE" = codex ] || { echo "only --judge codex is supported for delivery bundles (rewardkit codex transport on gpt-5.6-sol)" >&2; exit 2; }

[ -d "$J" ] || { echo "no such dir: $J (pass a task dir or a folder of task dirs)" >&2; exit 2; }
TARGET="$(cd "$J" && pwd)"

JUDGE_MODEL="${CODEX_JUDGE_MODEL:-gpt-5.6-sol}"

# --- batch: a folder of task dirs -> regrade each, sequentially ---------------
# A task dir has task.toml + trajectories/ (plural) or trajectory/ (singular);
# a folder of task dirs is a dir of those. A dir that is itself a task is single-mode.
if [ ! -d "$TARGET/trajectories" ] && [ ! -d "$TARGET/trajectory" ]; then
  TASKS_LIST=() SKIPPED=()
  for d in "$TARGET"/*/; do
    d="${d%/}"
    if { [ -d "$d/trajectories" ] || [ -d "$d/trajectory" ]; } && [ -f "$d/task.toml" ]; then TASKS_LIST+=("$d")
    elif [ -f "$d/task.toml" ]; then SKIPPED+=("$(basename "$d"): task definition only, no trajectory run")
    fi   # anything else (.DS_Store, .raw, ...) is ignored
  done
  if [ ${#SKIPPED[@]} -gt 0 ]; then
    echo "[batch] skipping ${#SKIPPED[@]} dir(s) that cannot be regraded:"; printf '  %s\n' "${SKIPPED[@]}"
  fi
  [ ${#TASKS_LIST[@]} -gt 0 ] || { echo "no task dirs (with trajectories/<model>/run_N) under $TARGET" >&2; exit 1; }
  FLAGS=(--judge "$JUDGE"); [ "$DRY" = 1 ] && FLAGS+=(--dry-run)
  BLOG="/tmp/regrade-dataset/batch-$(date +%Y%m%d-%H%M%S)"; mkdir -p "$BLOG"
  echo "[batch] ${#TASKS_LIST[@]} tasks under $TARGET  judge=$JUDGE ($JUDGE_MODEL)  logs: $BLOG"
  RESULTS=() FAILED=0 i=0
  for t in "${TASKS_LIST[@]}"; do
    i=$((i+1)); slug="$(basename "$t")"
    echo "[batch] ($i/${#TASKS_LIST[@]}) $slug"
    if "$0" "$t" "${FLAGS[@]}" 2>&1 | tee "$BLOG/$slug.log"; then rc=0; else rc=${PIPESTATUS[0]}; fi
    line="$(grep -E '^\[regrade\] (reward now|new reward)' "$BLOG/$slug.log" | tail -1 || true)"
    RESULTS+=("rc=$rc  $slug  ${line#\[regrade\] }")
    [ "$rc" = 0 ] || FAILED=$((FAILED+1))
  done
  echo; echo "[batch] summary (rc: 0 ok, 1 not regradeable, 4 judge failed/no reward):"
  printf '  %s\n' "${RESULTS[@]}" | tee "$BLOG/summary.txt"
  if [ ${#SKIPPED[@]} -gt 0 ]; then printf '  skipped  %s\n' "${SKIPPED[@]}" | tee -a "$BLOG/summary.txt"; fi
  echo "[batch] $((${#TASKS_LIST[@]}-FAILED)) ok, $FAILED failed, ${#SKIPPED[@]} skipped; per-task logs in $BLOG"
  [ "$FAILED" = 0 ]; exit $?
fi

# --- single task dir ----------------------------------------------------------
TASK="$TARGET"
J="$(basename "$TASK")"
[ -f "$TASK/task.toml" ] || { echo "no $TASK/task.toml" >&2; exit 1; }

# latest run under EITHER shape:
#   plural   trajectories/<model>/run_N   (delivery bundles; snapshot at verifier/end_env.json)
#   singular trajectory/run_N             (harness output; snapshot at artifacts/atlas/end_env.json)
SHAPE=plural
RUN_PATH=""
if [ -d "$TASK/trajectories" ]; then
  RUN_PATH="$(find "$TASK/trajectories" -maxdepth 2 -type d -name 'run_*' 2>/dev/null | sort -V | tail -1)"
fi
if [ -z "$RUN_PATH" ] && [ -d "$TASK/trajectory" ]; then
  RUN_PATH="$(find "$TASK/trajectory" -maxdepth 1 -type d -name 'run_*' 2>/dev/null | sort -V | tail -1)"
  SHAPE=singular
fi
[ -n "$RUN_PATH" ] || { echo "no trajectories/<model>/run_N or trajectory/run_N under $TASK" >&2; exit 1; }
if [ "$SHAPE" = plural ]; then MODEL="$(basename "$(dirname "$RUN_PATH")")"; else MODEL="$(basename "$TASK/trajectory")"; fi
RUN="$(basename "$RUN_PATH")"

# where the captured world snapshot lives, per shape
if [ "$SHAPE" = plural ]; then SNAP_SRC="$RUN_PATH/verifier/end_env.json"; else SNAP_SRC="$RUN_PATH/artifacts/atlas/end_env.json"; fi
# the verifier dir to update in place (both shapes use verifier/)
VDIR="$RUN_PATH/verifier"

[ -f "$RUN_PATH/agent/trajectory.json" ] || { echo "$RUN_PATH has no agent/trajectory.json (agent never ran) -- cannot regrade" >&2; exit 1; }
[ -f "$RUN_PATH/artifacts/manifest.json" ] || { echo "$RUN_PATH has no artifacts/manifest.json -- cannot regrade" >&2; exit 1; }
[ -s "$SNAP_SRC" ] || { echo "$RUN_PATH has no end_env.json snapshot ($SNAP_SRC) -- cannot regrade offline" >&2; exit 1; }
[ -f "$VDIR/state_capture_failed.txt" ] && { echo "$RUN_PATH: state capture failed in the original run (state_capture_failed.txt) -- cannot regrade offline" >&2; exit 1; }

old_reward() { python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(json.dumps({k:d.get(k) for k in ("pytest","rubric","reward")}))' "$1/verifier/reward.json" 2>/dev/null || echo "n/a"; }
echo "[regrade] task $J  $SHAPE $MODEL/$RUN  judge=$JUDGE ($JUDGE_MODEL)  current reward: $(old_reward "$RUN_PATH")"

TS="$(date +%Y%m%d-%H%M%S)"
WORKROOT="/tmp/regrade-dataset/$J"
WORK="$WORKROOT/trial-$TS"
NEWOUT="$WORKROOT/out-$TS"
rm -rf "$WORK" "$NEWOUT"; mkdir -p "$WORK" "$NEWOUT"

# --- 1. reshape delivery bundle -> harbor trial shape -------------------------
# harbor regrade seeds from <trial>/agent and <trial>/artifacts, validating each
# against <trial>/artifacts/manifest.json. The manifest declares:
#   /logs/agent      -> artifacts/logs/agent
#   /logs/artifacts  -> artifacts/logs/artifacts     (atlas snapshot lives here)
#   /workspace       -> artifacts/workspace
cp -R "$RUN_PATH/agent" "$WORK/agent"
cp -R "$RUN_PATH/artifacts" "$WORK/artifacts"
[ -f "$RUN_PATH/result.json" ] && cp "$RUN_PATH/result.json" "$WORK/result.json"
[ -f "$RUN_PATH/config.json" ] && cp "$RUN_PATH/config.json" "$WORK/config.json"

mkdir -p "$WORK/artifacts/logs/agent" "$WORK/artifacts/logs/artifacts/atlas"
cp -R "$RUN_PATH/agent/." "$WORK/artifacts/logs/agent/"
# place the captured snapshot where ATLAS_END_ENV (/logs/artifacts/atlas/end_env.json) resolves
cp "$SNAP_SRC" "$WORK/artifacts/logs/artifacts/atlas/end_env.json"
# carry the server tool-call log if present (plural: verifier/, singular: artifacts/atlas/)
SLOG=""
for cand in "$VDIR/server_tool_calls.log" "$RUN_PATH/artifacts/atlas/server_tool_calls.log"; do
  [ -f "$cand" ] && SLOG="$cand" && break
done
if [ -n "$SLOG" ]; then
  cp "$SLOG" "$WORK/artifacts/logs/artifacts/atlas/server_tool_calls.log"
else
  echo "[regrade] note: no server_tool_calls.log in bundle; state checks needing it fall back to the trajectory"
fi

# --- 2. regrade (verifier only), rewardkit codex judge on your codex login ----
[ -f "$HOME/.codex/auth.json" ] || { echo "no ~/.codex/auth.json -- run \`codex login\` first" >&2; exit 3; }
CODEX_AUTH_B64="$(base64 < "$HOME/.codex/auth.json" | tr -d '\n')"
[ -n "$CODEX_AUTH_B64" ] || { echo "~/.codex/auth.json did not encode" >&2; exit 3; }

# Force the verifier to build from source (tests/Dockerfile) rather than pull the
# prebuilt ECR image (tests/Dockerfile.ecr -> yuji-local/judge, not available here).
# Forwarded explicitly so harbor's `docker compose` subprocess expands it.
JUDGE_DOCKERFILE="${JUDGE_DOCKERFILE:-Dockerfile}" \
JUDGE_TRANSPORT=codex JUDGE_MODEL="$JUDGE_MODEL" CODEX_AUTH_B64="$CODEX_AUTH_B64" \
PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}" \
  harbor trial regrade "$WORK" --task-path "$TASK" -o "$NEWOUT"

NEW="$(ls -dt "$NEWOUT"/*/ 2>/dev/null | head -1)"; NEW="${NEW%/}"
[ -n "$NEW" ] && [ -f "$NEW/verifier/reward.json" ] || { echo "[regrade] no verifier/reward.json in new trial -- nothing replaced" >&2; exit 4; }
echo "[regrade] new reward: $(old_reward "$NEW")"
if [ -f "$NEW/verifier/rubric_judge_failed.txt" ]; then
  echo "[regrade] judge FAILED (verifier/rubric_judge_failed.txt): $(cat "$NEW/verifier/rubric_judge_failed.txt")" >&2
  echo "[regrade] nothing replaced" >&2
  exit 4
fi
if [ "$DRY" = 1 ]; then echo "[regrade] --dry-run: $TASK untouched. new trial kept at $NEW"; exit 0; fi

# --- 3. write back IN PLACE into the run's verifier/ --------------------------
# Back up everything the write-back touches: verifier/ and the two run-level
# reward-bearing files result.json and report.json.
BK="$WORKROOT/$MODEL-${RUN}-backup-$TS"
mkdir -p "$BK"
cp -R "$RUN_PATH/verifier" "$BK/verifier"
[ -f "$RUN_PATH/result.json" ] && cp "$RUN_PATH/result.json" "$BK/result.json"
[ -f "$RUN_PATH/report.json" ] && cp "$RUN_PATH/report.json" "$BK/report.json"
echo "[regrade] backup: $BK"

# Schema-preserving IN-PLACE update. The regrade must be invisible: no new file
# is created, no file is deleted, and no JSON key set changes. For every file
# that already exists in the bundle AND the fresh verifier reproduced, we keep
# the original file's exact keys and overwrite ONLY the values of keys that
# already exist (recursively), taking new values from the regrade output.
# Everything the fresh run produced but the bundle never had is ignored.
python3 - "$RUN_PATH" "$NEW" <<'PY'
import json, sys
from pathlib import Path

run = Path(sys.argv[1]); new = Path(sys.argv[2])
ver = run / "verifier"; nver = new / "verifier"

def load(p):
    try: return json.loads(Path(p).read_text())
    except (OSError, ValueError): return None

def dump(p, obj):
    Path(p).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")

def merge(orig, fresh):
    """Return orig with the VALUES of its existing keys overwritten from fresh,
    recursively. Never add a key orig didn't have; never drop one. Lists of
    dicts are merged element-wise by position (same length only)."""
    if isinstance(orig, dict) and isinstance(fresh, dict):
        out = {}
        for k, v in orig.items():
            out[k] = merge(v, fresh[k]) if k in fresh else v
        return out
    if isinstance(orig, list) and isinstance(fresh, list) and len(orig) == len(fresh):
        return [merge(a, b) for a, b in zip(orig, fresh)]
    # scalar (or shape mismatch): take fresh when it is a compatible scalar,
    # else keep orig untouched.
    if type(orig) is type(fresh) or (isinstance(orig,(int,float)) and isinstance(fresh,(int,float))):
        return fresh
    return orig

# The regrade's canonical scores (0-1) from the fresh reward.json.
fresh_reward = load(nver / "reward.json") or {}
new_pytest = fresh_reward.get("pytest")
new_rubric = fresh_reward.get("rubric")
new_final  = fresh_reward.get("reward")   # 0-1 combined

updated_files = []

# 1. verifier/*.json: only files present in BOTH bundle and fresh output, merged
#    key-preservingly. Non-JSON (junit.xml etc.) are copied only if they already
#    exist (format is stable), handled after.
for f in ("reward.json","reward-details.json","ctrf.json"):
    op, fp = ver / f, nver / f
    if op.is_file() and fp.is_file():
        o, fr = load(op), load(fp)
        if o is not None and fr is not None:
            # special case: bundle reward.json may store the combined score on a
            # 0-100 scale under 'reward'. Detect from the ORIGINAL value.
            if f == "reward.json" and isinstance(o, dict) and "reward" in o:
                ov = o.get("reward")
                scale = 100.0 if isinstance(ov,(int,float)) and ov and ov > 1.5 else 1.0
                merged = dict(o)
                if "pytest" in o and new_pytest is not None: merged["pytest"] = new_pytest
                if "rubric" in o and new_rubric is not None: merged["rubric"] = new_rubric
                if new_final is not None:
                    merged["reward"] = round(new_final*scale, 6) if scale != 1.0 else new_final
                dump(op, merged)
            else:
                dump(op, merge(o, fr))
            updated_files.append(f)

# 2. non-JSON verifier files: overwrite only if already present (stable schema)
for f in ("junit.xml","final_message.md"):
    if (ver / f).is_file() and (nver / f).is_file():
        (ver / f).write_bytes((nver / f).read_bytes()); updated_files.append(f)

# per-criterion verdicts from the fresh reward-details (id -> satisfied, reasoning)
details = load(nver / "reward-details.json") or {}
verdicts = {}
for c in ((details.get("rubric") or {}).get("criteria") or []):
    if isinstance(c, dict):
        raw = str(c.get("raw") or "").strip().lower()
        try: val = float(c.get("value") or 0)
        except (TypeError, ValueError): val = 0.0
        sat = (raw == "yes") if raw in ("yes","no") else (val > 0)
        verdicts[str(c.get("id"))] = (sat, c.get("reasoning"))

def apply_scalar_scores(obj, pyt, rub, fin):
    """Update only pre-existing reward-ish scalar keys in a dict, preserving scale."""
    if not isinstance(obj, dict): return
    for k in list(obj.keys()):
        v = obj[k]
        kl = k.lower()
        if isinstance(v,(int,float)):
            sc = 100.0 if v and v > 1.5 else 1.0
            if kl in ("final_reward","reward") and fin is not None:
                obj[k] = round(fin*sc,6) if sc!=1.0 else fin
            elif ("pytest" in kl) and ("score" in kl or kl=="pytest") and pyt is not None:
                obj[k] = round(pyt*sc,6) if sc!=1.0 else pyt
            elif ("rubric" in kl) and ("score" in kl or kl=="rubric") and rub is not None:
                obj[k] = round(rub*sc,6) if sc!=1.0 else rub

# 3. result.json: update only existing reward keys under verifier_result.rewards
res = load(run / "result.json")
if isinstance(res, dict):
    vr = (res.get("verifier_result") or {})
    rewards = vr.get("rewards") if isinstance(vr, dict) else None
    if isinstance(rewards, dict):
        apply_scalar_scores(rewards, new_pytest, new_rubric, new_final)
        dump(run / "result.json", res); updated_files.append("result.json")

# 4. report.json: update only existing score keys + refresh existing rubric
#    entry fields (satisfied/passed/justification) by id-match. Add nothing.
rep = load(run / "report.json")
if isinstance(rep, dict):
    apply_scalar_scores(rep, new_pytest, new_rubric, new_final)
    if isinstance(rep.get("channels"), dict): apply_scalar_scores(rep["channels"], new_pytest, new_rubric, new_final)
    if isinstance(rep.get("pytest"), dict): apply_scalar_scores(rep["pytest"], new_pytest, None, None)
    if isinstance(rep.get("rubric"), list):
        for c in rep["rubric"]:
            if isinstance(c, dict):
                v = verdicts.get(str(c.get("id")))
                if v is None: continue
                sat, reason = v
                if "satisfied" in c: c["satisfied"] = sat
                if "passed" in c: c["passed"] = sat
                if "justification" in c and reason is not None: c["justification"] = reason
    dump(run / "report.json", rep); updated_files.append("report.json")

print("[sync] updated in place (keys preserved): " + ", ".join(updated_files))
PY

echo "[regrade] reward now: $(old_reward "$RUN_PATH")"
echo "[regrade] done. To undo:"
echo "  rm -rf '$RUN_PATH/verifier' && cp -R '$BK/verifier' '$RUN_PATH/verifier'"
echo "  [ -f '$BK/result.json' ] && cp '$BK/result.json' '$RUN_PATH/result.json'"
echo "  [ -f '$BK/report.json' ] && cp '$BK/report.json' '$RUN_PATH/report.json'"
