#!/usr/bin/env bash
# Run one mcp-atlas Harbor task end-to-end and emit the per-task output/ tree
# (same layout complex-mcp's --layout harbor writer produces).
#
#   scripts/run_task.sh tasks/xenon-atomic-cube                 # openhands + opus-5 (defaults)
#   AGENT=claude-code scripts/run_task.sh tasks/foo             # harbor's claude-code agent
#   CC_MODE=zbridge N=3 scripts/run_task.sh tasks/foo           # openhands on GLM-5.3 (zbridge)
#   CC_MODE=bedrock MODEL=claude-opus-5 scripts/run_task.sh tasks/foo  # openhands on AWS Bedrock
#   CC_MODE=kiro KIRO_MODEL=claude-opus-5 KIRO_JUDGE_MODEL=gpt-5.6-sol \
#       scripts/run_task.sh tasks/foo                           # openhands on Kiro (dynamic model + judge)
#   AGENT=oracle scripts/run_task.sh tasks/foo                  # oracle gate
#   COPY_TO=/some/dir scripts/run_task.sh tasks/foo           # optional extra mirror
#
#   scripts/run_task.sh --stage reshape tasks/foo               # one stage only
#
# The default agent is OpenHands (tools/openhands_agent). CC_MODE picks the
# model route and, where the route exposes a second key, also the judge wire
# that the bundle's rubric runs against:
#
#   unset (default) ccbridge -> host's Claude subscription (OAuth). Agent only;
#                   judge takes CALLER_ANTHROPIC_API_KEY (or ANTHROPIC_API_KEY
#                   from .env) direct to api.anthropic.com.
#   CC_MODE=zbridge zbridge on :8766 -> GLM for the agent. Judge wire unchanged
#                   (same ANTHROPIC_API_KEY -> api.anthropic.com path).
#   CC_MODE=bedrock no host bridge; the agent calls AWS Bedrock directly from
#                   the container with AWS_BEARER_TOKEN_BEDROCK and AWS_REGION
#                   forwarded in. The judge also runs on Bedrock in-process via
#                   litellm.acompletion against JUDGE_MODEL (BEDROCK_MODEL_IDS
#                   alias) using the same AWS creds.
#   CC_MODE=kiro    kiro_bridge on 127.0.0.1 (ephemeral port) exposes two keys:
#                   agent_key pinned to KIRO_MODEL, judge_key pinned to
#                   KIRO_JUDGE_MODEL. Both sides hit kiro-cli; agent and judge
#                   models can be set independently per run.
#
# See tools/openhands_agent/README.md and tools/bridges/<mode>/README.md.
#
# Env overrides: AGENT (openhands) MODEL (claude-opus-5) N (1) JOB (<task slug>)
#                OUTPUT_DIR (<repo>/output) COPY_TO (unset) BUILD_MULT (3) AT (auto)
#                STAGE (all) RUN_OFFSET (auto) SETUP_MULT (6)
#                DISALLOWED_TOOLS (WebSearch,WebFetch)
#                CC_MODE (unset -> ccbridge on Claude subscription;
#                         "zbridge" -> glm-5.3 via :8766;
#                         "bedrock" -> AWS Bedrock direct, no host bridge;
#                         "kiro"    -> kiro_bridge, two-key agent+judge split)
#   bedrock:     AWS_BEARER_TOKEN_BEDROCK (required) BEDROCK_AWS_REGION (required)
#                BEDROCK_MODEL_IDS (alias1=id1,alias2=id2 map, resolves MODEL
#                and JUDGE_MODEL to full Bedrock inference-profile ARNs)
#                JUDGE_MODEL (defaults to MODEL; alias from BEDROCK_MODEL_IDS)
#                JUDGE_MODE (bedrock | kiro | anthropic | openai; selects the
#                judge wire independently of the agent. default: bedrock)
#                  JUDGE_MODE=kiro      needs kiro-cli login + KIRO_JUDGE_MODEL
#                  JUDGE_MODE=anthropic needs ANTHROPIC_API_KEY (direct call)
#                  JUDGE_MODE=openai    needs OPENAI_API_KEY (direct call)
#                AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY (optional, enables
#                the pre-flight bedrock:ListFoundationModels credential check;
#                set BEDROCK_CHECK_OFF=1 to skip)
#                CC_BRIDGE_ENABLED (0)
#                AGENT_HEADROOM_ENABLED (false)
#   kiro:        KIRO_MODEL (default: $MODEL; pinned on the agent key)
#                KIRO_JUDGE_MODEL (default: $MODEL; pinned on the judge key,
#                injected into the verifier as JUDGE_MODEL via --ve)
#                KIRO_BRIDGE_START_TIMEOUT_SEC (60) JUDGE_WIRE (anthropic)
#                kiro-cli must be logged in on the host; the bridge mints a
#                per-run agent_key and judge_key and tears both down at exit.
#   judge:       ANTHROPIC_API_KEY (ccbridge/zbridge: the key the rubric judge
#                calls api.anthropic.com with; auto-set for bedrock/kiro by
#                this script)
#                JUDGE_MODEL (bedrock/kiro: which model the judge runs on;
#                defaults to the agent MODEL when unset)
#                JUDGE_WIRE (anthropic|openai; kiro default anthropic)
#   openhands:   CCBRIDGE_SHARED (0: a ccbridge per run, on a free port, stopped
#                at exit; 1: one long-lived bridge on CCBRIDGE_PORT (8765) with
#                CCBRIDGE_SECRET or tools/bridges/ccbridge/.bridge_secret)
#                CCBRIDGE_HOST (127.0.0.1 on macOS when every phase is public,
#                else 0.0.0.0, gated by the bridge's secret)
#                OPENHANDS_MAX_ITERATIONS (500) OPENHANDS_MAX_CONTINUATIONS (6)
#                OPENHANDS_REASONING_EFFORT (SDK default) OPENHANDS_MAX_OUTPUT_TOKENS (32000)
#                OPENHANDS_THINKING_DISPLAY (summarized | omitted)
#                OPENHANDS_NUM_RETRIES (5 attempts per model call, backing off 8-64s)
#
# Values may also come from <repo>/.env, which is read as DEFAULTS only: anything
# already in the environment wins over it.
#
# Scoring and network policy belong to the bundle, not to this script. Harbor
# runs the bundle's own tests/ in the verifier environment its task.toml
# declares ([verifier] environment_mode = "separate"), and enforces each phase's
# declared network_mode ([environment], [agent], [verifier]) with its docker
# egress-control sidecar, refusing a trial whose policy it cannot enforce. This
# script adds no grader, no judge and no proxy of its own; the only network
# input it makes is --allow-agent-host for its own model bridge on the host.
#
# Stages. The default STAGE=all runs the four below in order, which is the
# original one-shot behaviour. They are separable because their costs differ by
# orders of magnitude: the agent phase takes minutes and real money, reshaping
# takes seconds, and reporting is one HTTP call. A crash between them must not
# re-run the agent, so scripts/run_batch.py drives them one at a time and
# checkpoints in between.
#
#   preflight  auth, docker, image build/pull               (idempotent)
#   harbor     harbor run -> output/<job>/                  (NOT idempotent: makes a trial)
#   reshape    harbor_to_output.py -> trajectory/Run_N/      (idempotent)
#   finance    finance_reporter.py -> Odoo                  (NOT idempotent: external POST)
#   mask       make_delivery.py --mask-only -> strips host   (idempotent; runs
#              paths; automatic at the end of reshape + finance)
#   atif       trajectory_reshaper.py -> ATIF_VERSION shape (idempotent; runs
#              (default 1.5) in output/ and delivery_output/  after finance)
#
# State that crosses a stage boundary (which Run_N this invocation owns, where
# earlier runs were stashed) is written to output/<job>/.run_state.json so a
# later stage in a separate process can pick it up.
set -euo pipefail
# Clear anything in the caller's environment that would redirect the agent's
# `claude` CLI away from the API. Running this script from inside a Claude Code
# session exports ANTHROPIC_BASE_URL=http://127.0.0.1:<port> for that session's
# own proxy; inherited into the task container, 127.0.0.1 is the container and
# nothing listens there, so every agent turn dies with
# "API Error: Connection refused (ConnectionRefused)" -- an infrastructure
# failure that reads exactly like the agent refusing the task.
# CLAUDE_CODE_OAUTH_TOKEN is deliberately NOT cleared: it is what the run
# authenticates with, and the verifier needs it too.
unset AWS_BEARER_TOKEN_BEDROCK 2>/dev/null || true
# Saved before the unset: resolve_auth hands it back when nothing else
# authenticates the run, rather than reporting it missing.
CALLER_ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-}"
unset ANTHROPIC_BASE_URL ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN 2>/dev/null || true
unset CLAUDECODE CLAUDE_CODE_ENTRYPOINT CLAUDE_CODE_SSE_PORT 2>/dev/null || true
unset CLAUDE_CODE_MESSAGING_SOCKET CLAUDE_CODE_MESSAGING_TOKEN 2>/dev/null || true

REPO="$(cd "$(dirname "$0")/.." && pwd)"
# The harbor pin and the install command for THIS machine (uv vs pipx). Sourced
# so a version refusal can print the command that actually fixes the box it is
# running on.
# shellcheck source=lib/harbor.sh
. "$REPO/scripts/lib/harbor.sh"
cd "$REPO"

# .env supplies DEFAULTS for everything below, so it has to be read BEFORE the
# first setting that depends on it. It used to be sourced down in the dispatch
# section, ~980 lines after CC_MODE picks the model default -- so the
# `CC_MODE=zbridge` line env.template tells operators to add was invisible to
# that decision, and a GLM run asked harbor for claude-opus-5 while the
# credential check (which runs after the source) reported zbridge. One script,
# two views of the same variable.
#
# Fill-if-unset, never `set -a; source`: the CALLER must win. Sourcing assigned
# unconditionally, so `KEY= scripts/run_task.sh ...` lost to the .env value and
# no .env key could be overridden for a single run.
# Presence is what is tested (${!k+set}), not emptiness, so an explicitly-empty
# override is honoured rather than refilled from the file.
load_dotenv() {
  [ -f "$REPO/.env" ] || return 0
  local line key val skipped="" quoted=""
  while IFS= read -r line; do
    key="${line%%=*}"
    val="${line#*=}"
    # Balanced quotes are taken verbatim: the charset filter guards a BARE value
    # against re-splitting, which quoting already does. Both styles, because
    # finance_reporter.py:119 strips both -- a value dropped here and loaded there
    # is how the Odoo gate warns "unauthenticated" about a token that is set.
    # Unbalanced quotes fall through to the filter rather than half-parsing.
    # What the filter still skips is NAMED, which is how ZB_MODEL_ALIAS_JSON sat
    # in .env doing nothing while looking like configuration.
    case "$val" in
      \"*\") val="${val#\"}"; val="${val%\"}"; quoted="$quoted $key" ;;
      \'*\') val="${val#\'}"; val="${val%\'}"; quoted="$quoted $key" ;;
      *[!A-Za-z0-9_./:@~-]*) skipped="$skipped $key"; continue ;;
    esac
    if [ -z "${!key+set}" ]; then
      export "$key=$val"
    fi
  done < <(sed 's/[[:space:]]*$//' "$REPO/.env" | grep -E '^[A-Za-z_][A-Za-z0-9_]*=')
  if [ -n "$skipped" ]; then
    echo "[run_task] .env: skipped (value has unsupported characters):$skipped" >&2
  fi
  if [ -n "$quoted" ]; then
    echo "[run_task] .env: loaded with quotes stripped:$quoted" >&2
  fi
  return 0
}
load_dotenv

# Settings that used to switch harness-side grading or networking. Both now
# belong to the bundle, so each would be silently ignored -- and configuration
# that vanishes looks exactly like configuration that works. Refuse instead.
for _retired in NETWORK_ISOLATION_OFF EGRESS_SQUID_CONF FORCE_HOST_RUBRIC SCORING_DIR; do
  if [ -n "${!_retired+set}" ]; then
    echo "[run_task] ERROR: $_retired is set, and it no longer does anything." >&2
    echo "[run_task]   Network policy and scoring are the bundle's: task.toml declares each" >&2
    echo "[run_task]   phase's network_mode, and Harbor runs the bundle's own tests/ to score." >&2
    echo "[run_task]   Unset it (check .env too)." >&2
    exit 2
  fi
done

# Docker bind mounts are colon-delimited (source:target:mode), so a checkout or
# jobs directory whose path contains a ":" cannot be mounted at all -- compose
# rejects it as "too many colons", and the structured long syntax fails the same
# way because it is serialised back to that string for the daemon. Measured on a
# clone under ~/Downloads/feat:new_eval_container: the trial died in
# "starting environment..." with the compose error forty lines below the cause,
# and the run looked like a harness bug.
#
# Checked here, before anything is built or spent, because every mount this
# harness adds -- the graders, the bundle tests, harbor's own per-trial log dirs
# -- carries one of these paths.
refuse_uncmountable_paths() {
  local bad="" p
  for p in "$REPO" "${OUTPUT_DIR:-$REPO/output}"; do
    case "$p" in *:*) bad="$bad $p" ;; esac
  done
  [ -z "$bad" ] && return 0
  echo "[run_task] ERROR: a path here contains a colon, which docker cannot bind-mount:" >&2
  for p in $bad; do echo "[run_task]   $p" >&2; done
  echo "[run_task]   Move the checkout (and OUTPUT_DIR) somewhere without a ':' in the name." >&2
  exit 2
}

STAGE="${STAGE:-all}"
TASK=""
while [ $# -gt 0 ]; do
  case "$1" in
    --stage)   STAGE="${2:?--stage needs a value}"; shift 2;;
    --stage=*) STAGE="${1#*=}"; shift;;
    -h|--help) awk 'NR==1{next} /^#/{sub(/^# ?/,""); print; next} {exit}' "$0"; exit 0;;
    *)         TASK="$1"; shift;;
  esac
done

TASK="${TASK:?usage: scripts/run_task.sh [--stage all|preflight|harbor|reshape|finance|mask|atif] <task-dir>}"
[ -f "$TASK/task.toml" ] || { echo "not a task dir (no task.toml): $TASK" >&2; exit 2; }
case "$STAGE" in
  all|preflight|harbor|reshape|finance|mask|atif) ;;
  *) echo "unknown stage: $STAGE (want all|preflight|harbor|reshape|finance|mask|atif)" >&2; exit 2;;
esac
SLUG="$(basename "$TASK")"

AGENT="${AGENT:-openhands}"
if [ "${CC_MODE:-}" = "zbridge" ]; then
  MODEL="${MODEL:-glm-5.3}"
elif [ "${CC_MODE:-}" = "bedrock" ]; then
  MODEL="${MODEL:-claude-opus-5}"
elif [ "${CC_MODE:-}" = "kiro" ]; then
  MODEL="${MODEL:-${KIRO_MODEL:-claude-opus-5}}"
else
  MODEL="${MODEL:-claude-opus-5}"
fi

# BEDROCK_MODEL_IDS maps a short alias (what MODEL carries, and what every
# report, trajectory dir and finance record is keyed on) to a Bedrock model id
# or inference-profile ARN. Format: "alias1=id1,alias2=id2". A MODEL that is
# already a bedrock/* string, a bare arn:, or a dotted model id (anthropic.*)
# is passed through untouched. The resolved value becomes a bedrock/<id>
# string, which LiteLLM routes directly to Bedrock.
resolve_bedrock_model() {
  local m="$MODEL"
  case "$m" in
    bedrock/*) printf '%s\n' "$m"; return 0 ;;
    arn:aws:bedrock:*|arn:aws:iam:*) printf 'bedrock/%s\n' "$m"; return 0 ;;
  esac
  if [ -n "${BEDROCK_MODEL_IDS:-}" ]; then
    local pair key val
    IFS=',' read -ra _pairs <<< "$BEDROCK_MODEL_IDS"
    for pair in "${_pairs[@]}"; do
      key="${pair%%=*}"
      val="${pair#*=}"
      key="${key// /}"
      if [ "$key" = "$m" ] && [ -n "$val" ]; then
        printf 'bedrock/%s\n' "$val"
        return 0
      fi
    done
  fi
  case "$m" in
    anthropic.*|amazon.*|meta.*|cohere.*|mistral.*|ai21.*|deepseek.*)
      printf 'bedrock/%s\n' "$m"; return 0 ;;
  esac
  return 1
}

# The OpenHands agent is not one of harbor's built-ins: harbor imports it from
# this checkout, which is why stage_harbor puts $REPO on harbor's PYTHONPATH.
OPENHANDS_AGENT_IMPORT="tools.openhands_agent.agent:OpenHandsAgent"
harbor_agent_ref() {
  case "$AGENT" in
    openhands) echo "$OPENHANDS_AGENT_IMPORT" ;;
    *)         echo "$AGENT" ;;
  esac
}

# Combinations that would run, and would not mean what they say. Refused at
# second zero rather than discovered in the agent phase -- and only by the
# stages that start an agent, so reshaping an older GLM job is unaffected.
if [ "$AGENT" = "openhands" ] && case "$STAGE" in preflight|harbor|all) true ;; *) false ;; esac; then
  if [ "${AGENT_HEADROOM_ENABLED:-false}" = "true" ]; then
    echo "[run_task] ERROR: AGENT_HEADROOM_ENABLED=true compresses the claude-code agent's" >&2
    echo "[run_task]   traffic (it reroutes ANTHROPIC_BASE_URL); the openhands agent never" >&2
    echo "[run_task]   reads that variable, so the run would say compressed and not be." >&2
    echo "[run_task]   Unset it, or use AGENT=claude-code." >&2
    exit 2
  fi
fi
N="${N:-1}"
BUILD_MULT="${BUILD_MULT:-3}"
SETUP_MULT="${SETUP_MULT:-6}"
# Agent-path Headroom prompt compression, OFF unless asked for: a headroom proxy
# container in front of main. It does not change the task's network policy.
AGENT_HEADROOM_ENABLED="${AGENT_HEADROOM_ENABLED:-false}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO/output}"
AT="${AT:-auto}"   # pass@k ks for the reshaper; auto = every k from 1..N runs
JOB="${JOB:-$SLUG}"   # job dir == output/<task>/ (reshaped in place by the converter)

# "summarized" = readable thinking summaries in message text; "" = signature-only blocks with empty content (measured 2026-09-03)
THINKING="${THINKING:-adaptive}"
THINKING_DISPLAY="${THINKING_DISPLAY:-summarized}"

# Web tools withheld from the agent while the task's agent phase is closed
# (network_mode "allowlist" or "no-network"). Harbor turns this into
# `--disallowedTools`, so the tools are absent from the tool list rather than
# present-and-failing.
#
# These two are the whole list on purpose. Bash is NOT here: the tasks need it
# for real local work, and its egress is already refused by Harbor's egress
# sidecar and still audited by detect_internet_use.py afterwards. Denying it
# would break tasks to buy nothing.
#
# Set to empty to pass no --disallowedTools at all.
DISALLOWED_TOOLS="${DISALLOWED_TOOLS-WebSearch,WebFetch}"

# The reshaped output dir is NOT always output/<bundle-dir>. harbor_to_output.py
# names it from task.toml's `name` (last path segment), falling back to the
# bundle dir -- so tasks/Input_1 with name="complexmcp/larkmoor-depot-false-
# alarm-attribution" reshapes into output/larkmoor-depot-false-alarm-attribution
# while Harbor's raw job stays in output/Input_1. $JOB is right for the raw job;
# everything that reads the reshaped tree has to use this instead. Matches the
# converter's own regex so the two cannot disagree.
resolve_out_slug() {
  local name=""
  if [ -f "$TASK/task.toml" ]; then
    name="$(grep -m1 -E '^[[:space:]]*name[[:space:]]*=[[:space:]]*"' "$TASK/task.toml" 2>/dev/null \
            | sed -E 's/^[[:space:]]*name[[:space:]]*=[[:space:]]*"([^"]+)".*/\1/')"
  fi
  name="${name##*/}"
  [ -n "$name" ] || name="$SLUG"
  printf '%s\n' "$name"
}
OUT_SLUG="$(resolve_out_slug)"

TRAJ_DIR="$OUTPUT_DIR/$OUT_SLUG/trajectory"
STATE_FILE="$OUTPUT_DIR/$JOB/.run_state.json"
# Kept outside output/<job>/ on purpose: harbor may wipe the job dir wholesale,
# which would take a stash living inside it with it.
STASH_DIR="$OUTPUT_DIR/.stash/$JOB"

state_get() {  # state_get <key> -> value on stdout, empty if absent
  [ -f "$STATE_FILE" ] || return 0
  # `or ""` would fold a real 0 into "absent", and run_offset is 0 on every
  # first run -- callers then fell back to counting Run_* dirs and aimed one
  # run too high. Only None/missing may read as empty.
  python3 -c 'import json,sys
try:
    v = json.load(open(sys.argv[1])).get(sys.argv[2])
    print("" if v is None else v)
except Exception:
    pass' "$STATE_FILE" "$1"
}

state_has() {  # state_has <key> -> exit 0 if the key is present at all
  [ -f "$STATE_FILE" ] || return 1
  python3 -c 'import json,sys
try:
    sys.exit(0 if sys.argv[2] in json.load(open(sys.argv[1])) else 1)
except Exception:
    sys.exit(1)' "$STATE_FILE" "$1"
}

state_put() {  # state_put <key> <value>
  mkdir -p "$(dirname "$STATE_FILE")"
  python3 -c 'import json,os,sys
path, key, value = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    doc = json.load(open(path))
except Exception:
    doc = {}
doc[key] = int(value) if value.lstrip("-").isdigit() else value
tmp = path + ".tmp"
open(tmp, "w").write(json.dumps(doc, indent=2) + "\n")
os.replace(tmp, path)' "$STATE_FILE" "$1" "$2"
}

# ---------------------------------------------------------------- teardown ---
# Harbor deletes its own compose project when `harbor run` RETURNS: --delete is
# its default, and that teardown is `down --rmi local --volumes
# --remove-orphans`, so a run that finishes takes its containers and its
# workspace_data volume with it.
#
# Nothing does that when the process does not return. A Ctrl-C, a SIGTERM, or a
# harbor that dies mid-trial leaves every container of that project up and the
# volume attached to them -- and `make stop-harness` could not clear those
# either, because a running container holds its volume open. Two interrupted
# runs on this machine left 8 containers and 2 volumes behind for 50 minutes.
#
# So: record which compose projects exist before harbor starts, and on the way
# out remove exactly the ones that appeared since. Scoped by the compose project
# LABEL and by harbor's "__" trial marker, so a concurrent run of another task,
# and anything on this machine that is not harbor's, is never touched.
HARBOR_PROJECTS_BEFORE=""

compose_projects_now() {
  docker ps -a --filter "label=com.docker.compose.project" \
      --format '{{.Label "com.docker.compose.project"}}' 2>/dev/null | sort -u
}

snapshot_compose_projects() {
  docker info >/dev/null 2>&1 || return 0
  HARBOR_PROJECTS_BEFORE="$(compose_projects_now)"
}

# Remove the compose projects this invocation created. Safe to call twice, and
# safe to call when harbor already cleaned up: the delta is then empty.
teardown_this_runs_projects() {
  docker info >/dev/null 2>&1 || return 0
  local new p
  new="$(comm -13 <(printf '%s\n' "${HARBOR_PROJECTS_BEFORE:-}" | grep . | sort) \
                  <(compose_projects_now | grep . | sort) 2>/dev/null \
         | grep '__' || true)"
  [ -n "$new" ] || return 0
  local ids vols
  while IFS= read -r p; do
    [ -n "$p" ] || continue
    echo "[run_task] teardown: removing compose project $p" >&2
    ids="$(docker ps -aq --filter "label=com.docker.compose.project=$p" 2>/dev/null)"
    # Read the mounts BEFORE the containers go. workspace_data carries the
    # compose project label and can be found again by it, but an image that
    # declares VOLUME gets an ANONYMOUS volume -- 64 hex characters, no labels.
    # Once its container is gone nothing can attribute it to this run again,
    # and it sits on the disk forever.
    vols=""
    if [ -n "$ids" ]; then
      vols="$(printf '%s\n' "$ids" | xargs -r docker inspect \
          --format '{{range .Mounts}}{{if eq .Type "volume"}}{{println .Name}}{{end}}{{end}}' \
          2>/dev/null | grep . | sort -u)"
      printf '%s\n' "$ids" | xargs -r docker rm -f >/dev/null 2>&1 || true
    fi
    docker volume ls -q --filter "label=com.docker.compose.project=$p" 2>/dev/null \
      | xargs -r docker volume rm -f >/dev/null 2>&1 || true
    [ -n "$vols" ] && printf '%s\n' "$vols" \
      | xargs -r docker volume rm -f >/dev/null 2>&1 || true
    docker network ls -q --filter "label=com.docker.compose.project=$p" 2>/dev/null \
      | xargs -r docker network rm >/dev/null 2>&1 || true
  done <<< "$new"
}

# SIGINT/SIGTERM. Without these the script had no trap at all: Ctrl-C killed
# harbor, the `|| { ... }` around it swallowed the 130, and the script walked on
# into reshape, mask and finance and exited 0 with the whole compose project
# still up.
on_interrupt() {
  trap - INT TERM
  echo "" >&2
  echo "[run_task] interrupted -- tearing down this run's containers and volumes" >&2
  teardown_this_runs_projects
  # 128 + signal, so run_batch.py can tell an interrupt from a task failure.
  exit 130
}
on_terminate() {
  trap - INT TERM
  echo "[run_task] SIGTERM -- tearing down this run's containers and volumes" >&2
  teardown_this_runs_projects
  exit 143
}

list_trial_dirs() {
  local d="$OUTPUT_DIR/$JOB" p
  [ -d "$d" ] || return 0
  # Both places harbor_to_output.py looks for trials: the job root (where harbor
  # puts them -- each trial config.json records trials_dir as the job dir) and
  # trajectory/, which older reshaped layouts left non-run_N trial dirs in. The
  # two lists must agree or the delta below would miss a trial and convert none.
  for p in "$d"/*__*/ "$d"/trajectory/*__*/; do
    [ -d "$p" ] || continue          # no match: the glob stays literal
    p="${p%/}"
    printf '%s\n' "${p##*/}"
  done
}

# Which Run_N this invocation owns. An explicit RUN_OFFSET (what run_batch.py
# passes) wins, because the driver already decided the numbering; otherwise
# count what is on disk, which is what a bare run_task.sh has always done.
resolve_run_offset() {
  if [ -n "${RUN_OFFSET:-}" ]; then echo "$RUN_OFFSET"; return; fi
  local last=""
  if [ -d "$TRAJ_DIR" ]; then
    last="$(ls "$TRAJ_DIR" 2>/dev/null | grep -E '^run_[0-9]+$' | sed 's/run_//' | sort -n | tail -1 || true)"
  fi
  echo "${last:-0}"
}

# --- stages -------------------------------------------------------------------

# Harbor's agent needs a token. Every stage resolves this, not just preflight:
# stages run as separate processes now, so an export in one is gone by the time
# the next starts.
# How much validity a token must still have for the run to be worth starting.
AUTH_MIN_REMAINING_SEC="${AUTH_MIN_REMAINING_SEC:-900}"
# expiresAt (ms) from whichever store the token came out of; "" = not known,
# which check_credentials treats as no opinion rather than as a problem.
CLAUDE_TOKEN_EXPIRES_MS=""

resolve_auth() {
  [ -n "${CLAUDE_CODE_OAUTH_TOKEN:-}" ] && return 0
  [ -n "${ANTHROPIC_API_KEY:-}" ] && return 0
  SRC=""
  local _raw=""
  _raw="$(security find-generic-password -s "Claude Code-credentials" -w 2>/dev/null || true)"
  [ -n "$_raw" ] && SRC="keychain"

  # Linux has no keychain: the claude CLI writes the same JSON structure to
  # ~/.claude/.credentials.json instead. Read it here rather than expecting the
  # operator to export the token by hand -- the access token is short-lived
  # (hours) and the CLI refreshes it in place, so a value exported once into a
  # shell profile goes stale mid-batch. That surfaces as the agent failing to
  # authenticate deep inside a paid trial, which reads like a bad agent rather
  # than an expired credential. Resolving per invocation always picks up
  # whatever the CLI last refreshed.
  if [ -z "$_raw" ] && [ -s "$HOME/.claude/.credentials.json" ]; then
    _raw="$(cat "$HOME/.claude/.credentials.json" 2>/dev/null || true)"
    [ -n "$_raw" ] && SRC="~/.claude/.credentials.json"
  fi

  # One read, both fields. expiresAt travels with the token the CLI wrote, so
  # reading it here costs nothing and needs no network.
  TOKEN=""
  if [ -n "$_raw" ]; then
    local _parsed=""
    _parsed="$(printf '%s' "$_raw" | python3 -c 'import sys,json
try:
    d = json.load(sys.stdin)["claudeAiOauth"]
except Exception:
    raise SystemExit(0)
print(d.get("accessToken") or "")
print(d.get("expiresAt") or "")' 2>/dev/null || true)"
    TOKEN="$(printf '%s\n' "$_parsed" | sed -n '1p')"
    CLAUDE_TOKEN_EXPIRES_MS="$(printf '%s\n' "$_parsed" | sed -n '2p')"
  fi

  if [ -n "$TOKEN" ]; then
    export CLAUDE_CODE_OAUTH_TOKEN="$TOKEN"
    echo "[run_task] using Claude Code OAuth token from $SRC"
    return 0
  fi
  if [ -n "${CALLER_ANTHROPIC_API_KEY:-}" ]; then
    export ANTHROPIC_API_KEY="$CALLER_ANTHROPIC_API_KEY"
    echo "[run_task] using ANTHROPIC_API_KEY from the caller's environment"
    return 0
  fi
  echo "[run_task] WARNING: no CLAUDE_CODE_OAUTH_TOKEN / ANTHROPIC_API_KEY — the agent will fail auth" >&2
}

# Is the Claude token we resolved still valid? Only a value in the PAST is a
# failure; a missing or unreadable expiresAt says nothing and blocks nothing.
check_claude_token_expiry() {   # -> 0 ok, 1 expired
  [ -n "${CLAUDE_TOKEN_EXPIRES_MS:-}" ] || return 0
  local verdict
  verdict="$(python3 -c 'import sys,time
try:
    exp = int(sys.argv[1])
except Exception:
    raise SystemExit(0)
left = exp / 1000.0 - time.time()
if left <= 0:
    print("expired")
elif left < float(sys.argv[2]):
    print("soon %d" % int(left / 60))
else:
    print("ok")' "$CLAUDE_TOKEN_EXPIRES_MS" "$AUTH_MIN_REMAINING_SEC" 2>/dev/null || true)"
  case "$verdict" in
    expired)
      echo "[run_task] ERROR: the Claude Code token has EXPIRED" >&2
      echo "[run_task]   Harbor forwards it into the container, which cannot refresh it," >&2
      echo "[run_task]   so every agent turn would fail auth. Run: claude login" >&2
      return 1 ;;
    soon*)
      echo "[run_task] WARNING: the Claude Code token expires in ${verdict#soon } min;" >&2
      echo "[run_task]   a long run may lose auth part-way. Consider: claude login" >&2 ;;
  esac
  return 0
}

check_credentials() {
  case "$STAGE" in reshape|finance|mask) return 0 ;; esac
  local fail=0

  if [ "${CC_MODE:-}" = "zbridge" ]; then
    if [ -z "${ZB_ZAI_API_KEY:-}" ]; then
      echo "[run_task] ERROR: CC_MODE=zbridge but ZB_ZAI_API_KEY is not set" >&2
      echo "[run_task]   Add ZB_ZAI_API_KEY=<your-z.ai-key> to harness/.env" >&2
      fail=1
    else
      echo "[run_task] zbridge: ZB_ZAI_API_KEY OK"
    fi
    # ZB_BRIDGE_SECRET is deliberately NOT required. ensure_zbridge starts the
    # bridge with it set to "" (bridge.py:134 reads empty as auth-disabled),
    # because the containerised agent sends no x-zbridge-secret header and a
    # live gate would 401 every call. Demanding a value here only to discard it
    # at launch failed runs for a credential that changes nothing.
  elif [ "${CC_MODE:-}" = "bedrock" ]; then
    # 1. AWS auth env present (bearer token or SigV4 keys).
    if [ -z "${AWS_BEARER_TOKEN_BEDROCK:-}" ] \
         && [ -z "${AWS_ACCESS_KEY_ID:-}" ]; then
      echo "[run_task] bedrock ERROR: neither AWS_BEARER_TOKEN_BEDROCK nor AWS_ACCESS_KEY_ID is set." >&2
      echo "[run_task]   Add one to harness/.env." >&2
      fail=1
    fi
    # 2. Region.
    local _bedrock_region="${BEDROCK_AWS_REGION:-${AWS_REGION:-${AWS_DEFAULT_REGION:-}}}"
    if [ -z "$_bedrock_region" ]; then
      echo "[run_task] bedrock ERROR: no AWS region is set." >&2
      echo "[run_task]   Set BEDROCK_AWS_REGION or AWS_REGION in harness/.env" >&2
      echo "[run_task]   (e.g. BEDROCK_AWS_REGION=ap-south-1)." >&2
      fail=1
    else
      echo "[run_task] bedrock: region = $_bedrock_region"
    fi
    # 3. Agent MODEL alias -> ARN resolves.
    if ! _resolved="$(resolve_bedrock_model)"; then
      echo "[run_task] bedrock ERROR: cannot resolve MODEL='$MODEL' to a Bedrock id." >&2
      echo "[run_task]   Pass MODEL as a bedrock/* id, an ARN, a dotted provider id" >&2
      echo "[run_task]   (anthropic.*), or an alias from BEDROCK_MODEL_IDS." >&2
      fail=1
    else
      echo "[run_task] bedrock: agent MODEL '$MODEL' -> $_resolved"
      # Region embedded in the ARN vs the env region — warn on mismatch.
      case "$_resolved" in
        bedrock/arn:aws:bedrock:*)
          local _arn_region
          _arn_region="$(printf '%s' "$_resolved" | sed -E 's|^bedrock/arn:aws:bedrock:([^:]+):.*$|\1|')"
          if [ -n "$_arn_region" ] && [ -n "$_bedrock_region" ] \
               && [ "$_arn_region" != "$_bedrock_region" ]; then
            echo "[run_task] bedrock WARNING: ARN region ($_arn_region) != env region ($_bedrock_region);" >&2
            echo "[run_task]   LiteLLM uses env region; cross-region may fail or add latency." >&2
          fi ;;
      esac
    fi
    # 4. Judge model: if EVAL_LLM_MODEL starts with bedrock/, verify its alias
    # resolves too. Keeps a bedrock agent + bedrock judge consistent.
    if [ -n "${EVAL_LLM_MODEL:-}" ]; then
      case "${EVAL_LLM_MODEL}" in
        bedrock/arn:aws:bedrock:*)
          echo "[run_task] bedrock: judge EVAL_LLM_MODEL uses a direct ARN" ;;
        bedrock/*)
          local _judge_alias="${EVAL_LLM_MODEL#bedrock/}" _judge_found=0 _judge_arn=""
          if [ -n "${BEDROCK_MODEL_IDS:-}" ]; then
            local IFS=',' pair key val
            for pair in $BEDROCK_MODEL_IDS; do
              key="${pair%%=*}"; val="${pair#*=}"
              if [ "$key" = "$_judge_alias" ] && [ -n "$val" ]; then
                _judge_found=1; _judge_arn="$val"; break
              fi
            done
          fi
          if [ "$_judge_found" = "1" ]; then
            echo "[run_task] bedrock: judge EVAL_LLM_MODEL '$_judge_alias' -> $_judge_arn"
          else
            echo "[run_task] bedrock ERROR: EVAL_LLM_MODEL='$EVAL_LLM_MODEL' alias not in BEDROCK_MODEL_IDS." >&2
            echo "[run_task]   Add the mapping, or use a direct bedrock/arn:aws:bedrock:... value." >&2
            fail=1
          fi ;;
      esac
    fi
    # 5. Live auth check: aws sts get-caller-identity. Verifies SigV4 creds end
    # to end -- bearer-token-only mode has no sts equivalent, so skip it and
    # note that bedrock will exercise the token on the first call. Set
    # BEDROCK_CHECK_OFF=1 to skip entirely.
    if [ -z "${BEDROCK_CHECK_OFF:-}" ] && [ -n "${AWS_ACCESS_KEY_ID:-}" ] \
         && command -v aws >/dev/null 2>&1; then
      local _sts_out
      if _sts_out="$(aws sts get-caller-identity --region "$_bedrock_region" --output json 2>&1)"; then
        local _account
        _account="$(printf '%s' "$_sts_out" | python3 -c 'import json,sys
try: print(json.load(sys.stdin).get("Account",""))
except Exception: pass' 2>/dev/null)"
        if [ -n "$_account" ]; then
          echo "[run_task] bedrock: AWS SigV4 creds verified (account $_account)"
        else
          echo "[run_task] bedrock: AWS SigV4 creds verified"
        fi
      else
        echo "[run_task] bedrock ERROR: aws sts get-caller-identity failed:" >&2
        printf '%s\n' "$_sts_out" | head -3 | sed 's/^/[run_task]   /' >&2
        echo "[run_task]   Fix the AWS keys in .env, or set BEDROCK_CHECK_OFF=1 to skip this gate." >&2
        fail=1
      fi
    elif [ -n "${AWS_BEARER_TOKEN_BEDROCK:-}" ]; then
      echo "[run_task] bedrock: bearer-token mode -- live check deferred to first agent call"
    fi
  elif [ "${CC_MODE:-}" = "kiro" ]; then
    # Kiro routes through the local kiro-cli login. The credential check is a
    # single `kiro-cli whoami`: no network to the model, no spend. It fails
    # for the two reasons that actually matter -- cli not installed and no
    # login -- and names them separately so the operator fixes the right one.
    if ! command -v kiro-cli >/dev/null 2>&1; then
      echo "[run_task] kiro ERROR: kiro-cli not found on PATH." >&2
      echo "[run_task]   Install from the Kiro distribution and re-run \`kiro-cli login\`." >&2
      fail=1
    else
      local _who
      if _who="$(kiro-cli whoami 2>&1)"; then
        echo "[run_task] kiro: logged in as $(printf '%s' "$_who" | head -1)"
      else
        echo "[run_task] kiro ERROR: kiro-cli is installed but no valid login:" >&2
        printf '%s\n' "$_who" | head -3 | sed 's/^/[run_task]   /' >&2
        echo "[run_task]   Run: kiro-cli login" >&2
        fail=1
      fi
    fi
    # The agent model (KIRO_MODEL / MODEL) and the judge model (KIRO_JUDGE_MODEL)
    # must both be in the login's model list -- otherwise the bridge will 404
    # mid-run. Fetching the list is cheap (kiro-cli --list-models) and names
    # the typo at second zero.
    if [ "$fail" = "0" ]; then
      local _models_json
      if _models_json="$(kiro-cli chat --list-models --format json 2>/dev/null)"; then
        local _bare_model="${MODEL#kiro/}"
        _bare_model="${_bare_model#*/}"
        local _judge_model="${KIRO_JUDGE_MODEL:-$_bare_model}"
        local _agent_ok _judge_ok
        _agent_ok="$(printf '%s' "$_models_json" | python3 -c '
import json, sys
try: m = json.load(sys.stdin)
except Exception: sys.exit(0)
entries = m.get("models", []) if isinstance(m, dict) else (m if isinstance(m, list) else [])
ids = {x.get("model_id") or x.get("id") or x.get("model_name") for x in entries if isinstance(x, dict)}
print("ok" if sys.argv[1] in ids else "miss")' "$_bare_model" 2>/dev/null)"
        _judge_ok="$(printf '%s' "$_models_json" | python3 -c '
import json, sys
try: m = json.load(sys.stdin)
except Exception: sys.exit(0)
entries = m.get("models", []) if isinstance(m, dict) else (m if isinstance(m, list) else [])
ids = {x.get("model_id") or x.get("id") or x.get("model_name") for x in entries if isinstance(x, dict)}
print("ok" if sys.argv[1] in ids else "miss")' "$_judge_model" 2>/dev/null)"
        case "$_agent_ok" in
          ok) echo "[run_task] kiro: agent MODEL '$_bare_model' is available" ;;
          *)  echo "[run_task] kiro ERROR: agent MODEL '$_bare_model' not in kiro-cli model list" >&2; fail=1 ;;
        esac
        case "$_judge_ok" in
          ok) echo "[run_task] kiro: judge KIRO_JUDGE_MODEL '$_judge_model' is available" ;;
          *)  echo "[run_task] kiro ERROR: judge KIRO_JUDGE_MODEL '$_judge_model' not in kiro-cli model list" >&2; fail=1 ;;
        esac
      else
        echo "[run_task] kiro WARNING: could not list models to validate selections; continuing" >&2
      fi
    fi
  else
    if [ -z "${CLAUDE_CODE_OAUTH_TOKEN:-}" ] && [ -z "${ANTHROPIC_API_KEY:-}" ]; then
      echo "[run_task] ERROR: no Claude Code credentials found" >&2
      echo "[run_task]   Log in with: claude login   or set CLAUDE_CODE_OAUTH_TOKEN / ANTHROPIC_API_KEY" >&2
      fail=1
    else
      if check_claude_token_expiry; then
        echo "[run_task] claude: credentials OK"
      else
        fail=1
      fi
    fi
  fi

  [ "$fail" = "0" ] || { echo "[run_task] credential check FAILED — fix the above and retry" >&2; exit 4; }
}

# --- finance attribution ------------------------------------------------------
# The ODOO_*/FINANCE_* keys are validated HERE, at second zero, and not in
# stage_finance where they are used.
#
# stage_finance is the LAST thing a run does -- after hours of paid agent time --
# and it calls finance_reporter.py WITHOUT --strict, which makes `fail = 0`
# (finance_reporter.py:354). A bad enum therefore raises inside build_payload,
# prints "cannot build payload: ..." to stderr, and returns 0. run_task.sh sees
# a clean exit, the run reports success, and nothing was ever posted. The loss is
# invisible until someone reconciles the ledger against the trials that actually
# ran. Everything below is string comparison over variables already in the
# environment: it costs nothing, and it moves that failure from hour five to
# second zero.
#
# An empty ODOO_URL is the documented opt-out (finance_reporter.py:382-385), so
# it skips the rest rather than forcing operators who never report to fill in
# attribution they do not use.
#
# Bypass with FINANCE_ENV_CHECK_OFF=1.
FINANCE_ENV_BAD=0
FINANCE_DEFAULT_PROJECT_TYPE="Technical"   # finance_reporter.py:276
FINANCE_DEFAULT_TEAM_TYPE="Projects"       # finance_reporter.py:279

# Warn when a value is the documented one in the wrong case. The reporter
# forwards project_type and team_type to Odoo verbatim and validates neither, so
# a casing slip passes every check on this machine and surfaces only as a wrong
# or rejected record on the server. Anything genuinely different is left alone --
# the server owns that vocabulary, not this script.
warn_finance_case() {   # warn_finance_case <VAR_NAME> <documented-value>
  local name="$1" want="$2" have="${!1:-}"
  [ -n "$have" ] || return 0
  [ "$have" = "$want" ] && return 0
  [ "$(printf '%s' "$have" | tr 'A-Z' 'a-z')" = "$(printf '%s' "$want" | tr 'A-Z' 'a-z')" ] || return 0
  echo "[run_task] WARNING: $name='$have' differs only in CASE from the documented '$want'" >&2
  echo "[run_task]   Odoo is sent the value verbatim; use '$want' unless you mean otherwise." >&2
}

check_finance_env() {
  # reshape never reports; finance runs as its own stage under run_batch.py and
  # is checked on its own invocation.
  case "$STAGE" in reshape|mask) return 0 ;; esac
  [ -z "${FINANCE_ENV_CHECK_OFF:-}" ] || return 0

  if [ -z "${ODOO_URL:-}" ]; then
    echo "[run_task] finance: ODOO_URL empty — usage reporting disabled, skipping checks"
    return 0
  fi

  local fail=0
  case "$ODOO_URL" in
    http://*|https://*) ;;
    *) echo "[run_task] ERROR: ODOO_URL must start with http:// or https:// — got '$ODOO_URL'" >&2
       echo "[run_task]   (a value with a space or a quote is dropped by load_dotenv above)" >&2
       fail=1 ;;
  esac

  # The one key with no default anywhere: build_payload raises outright.
  if [ -z "${FINANCE_PROJECT_ID:-}" ]; then
    echo "[run_task] ERROR: FINANCE_PROJECT_ID is empty or unset — required (e.g. PRJ-512)" >&2
    fail=1
  fi

  # An EMPTY value is legal wherever the reporter has a default: env() is
  # `(os.environ.get(name) or default)`, which treats "" and unset identically.
  # Only a non-empty WRONG value is an error, so the defaults are resolved here
  # exactly as the reporter resolves them and the result is what gets checked.
  #
  # Comparisons are case-SENSITIVE because the reporter's are
  # (finance_reporter.py:72-73, 251-263). "testing" is not "Testing", and that
  # one difference is the entire failure this function exists to catch.
  local budget="${FINANCE_BUDGET_TYPE:-}"
  [ -n "$budget" ] || budget="RFP"          # finance_reporter.py:251
  case "$budget" in
    RFP|Production) ;;
    *) echo "[run_task] ERROR: FINANCE_BUDGET_TYPE must be exactly 'RFP' or 'Production' — got '$budget'" >&2
       fail=1 ;;
  esac

  local detail=""
  if [ "$budget" = "RFP" ]; then
    local sub="${FINANCE_RFP_SUB_TYPE:-}"
    [ -n "$sub" ] || sub="Testing"          # finance_reporter.py:255
    case "$sub" in
      Testing|Sampling) detail="rfp_sub_type=$sub" ;;
      *) echo "[run_task] ERROR: FINANCE_RFP_SUB_TYPE must be exactly 'Testing' or 'Sampling' — got '$sub'" >&2
         echo "[run_task]   Case matters: 'testing' is rejected by finance_reporter.py:256." >&2
         fail=1 ;;
    esac
  elif [ "$budget" = "Production" ]; then
    # No default on this one -- env("FINANCE_PRODUCTION_MODE") is called with no
    # fallback -- so empty is a hard error rather than a silent default.
    case "${FINANCE_PRODUCTION_MODE:-}" in
      Singlephase|Multiphase) detail="production_mode=${FINANCE_PRODUCTION_MODE}" ;;
      "") echo "[run_task] ERROR: FINANCE_PRODUCTION_MODE is required when FINANCE_BUDGET_TYPE=Production" >&2
          echo "[run_task]   Set it to 'Singlephase' or 'Multiphase'." >&2
          fail=1 ;;
      *)  echo "[run_task] ERROR: FINANCE_PRODUCTION_MODE must be exactly 'Singlephase' or 'Multiphase' — got '${FINANCE_PRODUCTION_MODE}'" >&2
          fail=1 ;;
    esac
  fi

  # Odoo's handler requires phase_number and wants it as a string of digits.
  case "${FINANCE_PHASE_NUMBER:-1}" in
    ''|*[!0-9]*) echo "[run_task] ERROR: FINANCE_PHASE_NUMBER must be a number — got '${FINANCE_PHASE_NUMBER:-}'" >&2
                 fail=1 ;;
  esac

  warn_finance_case FINANCE_PROJECT_TYPE "$FINANCE_DEFAULT_PROJECT_TYPE"
  warn_finance_case FINANCE_TEAM_TYPE    "$FINANCE_DEFAULT_TEAM_TYPE"

  # Bad attribution stops the Odoo POST, not the agent phase. Said at second
  # zero either way; only the stage that actually reports refuses to start.
  if [ "$fail" != "0" ]; then
    echo "[run_task] finance env check FAILED — fix <repo>/.env." >&2
    echo "[run_task]   Clear ODOO_URL to disable reporting, or set FINANCE_ENV_CHECK_OFF=1 to skip this gate." >&2
    if [ "$STAGE" = "finance" ]; then
      exit 4
    fi
    echo "[run_task]   Continuing: the run proceeds, usage reporting will be skipped." >&2
    FINANCE_ENV_BAD=1
    return 0
  fi
  echo "[run_task] finance: OK (project=$FINANCE_PROJECT_ID budget=$budget${detail:+ $detail} -> $ODOO_URL)"
}

# --- images -------------------------------------------------------------------
# Nothing below ever asks the operator to have run `docker pull` or
# `make build-light-servers` first. Preflight is the cheap idempotent stage
# run_batch.py re-runs on every resume, so it is the right place to make the
# world match what the bundle declares.

# A python that can parse YAML. The repo venv has pyyaml; a bare system python3
# usually does not. Empty when nothing on the host can.
python_with_yaml() {
  local c
  for c in "$REPO/.venv/bin/python" python3 python; do
    if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import yaml' >/dev/null 2>&1; then
      echo "$c"; return 0
    fi
  done
}

# Images the bundle needs that NOTHING in the run will produce on its own: a
# compose service carrying an `image:` and no `build:`. `main` has a build:
# section, so compose builds it and it is deliberately not listed here.
compose_unbuilt_images() {
  local compose="$1" py
  [ -f "$compose" ] || return 0
  py="$(python_with_yaml)"
  if [ -n "$py" ]; then
    "$py" -c '
import sys, yaml
doc = yaml.safe_load(open(sys.argv[1])) or {}
for svc in (doc.get("services") or {}).values():
    if isinstance(svc, dict) and svc.get("image") and not svc.get("build"):
        print(svc["image"])
' "$compose" && return 0
  fi
  # No pyyaml anywhere on the host. Fall back to a structural scan: service keys
  # sit at exactly two spaces and their fields deeper, which holds for every
  # bundle in this repo. Depth is what keeps `depends_on:`'s nested
  # `light-servers:` from being read as a service of its own.
  awk '
    /^services:[[:space:]]*$/ { in_s = 1; next }
    in_s && /^[^[:space:]#]/  { in_s = 0 }
    in_s && /^  [A-Za-z0-9_.-]+:[[:space:]]*$/ {
      if (img != "" && !built) print img
      img = ""; built = 0; next
    }
    in_s && /^[[:space:]]+build:/ { built = 1 }
    in_s && /^[[:space:]]+image:[[:space:]]*/ {
      v = $0
      sub(/^[[:space:]]+image:[[:space:]]*/, "", v)
      sub(/[[:space:]]*#.*$/, "", v)
      gsub(/["'"'"']/, "", v)
      img = v
    }
    END { if (img != "" && !built) print img }
  ' "$compose"
}

# How to obtain an image the local daemon does not have. Images built out of
# this checkout are in NO registry -- `docker pull light-servers:latest` 404s --
# so a pull-only preflight cannot fix the one image every bundle here needs.
# Anything not named is treated as a registry image and pulled.
#
# agent-environment is deliberately NOT here, and dropping it changed nothing.
# The image that context builds is PUBLISHED as ghcr.io/scaleapi/mcp-atlas:<ver>
# (`make push`), and that is the name adapter-generated bundles pin for their
# `mcp-server` sidecar (adapters/mcp_atlas/adapter.py:27,92). `${1%%:*}` on it is
# "ghcr.io/scaleapi/mcp-atlas", which never matched this case -- so those bundles
# have always taken the registry-pull path above, which is the correct one.
#
# The only string this branch could ever match was the LOCAL `agent-environment`
# alias, which exists solely for the :1984 REST sandbox that run_all.sh /
# run_eval.py drive (`make run-docker`, `make shell`, run_all.sh:15) and which
# nothing on the Harbor path uses. Keeping it only meant preflight could be asked
# to build a 5.4 GB image no `harbor run` would ever start. Re-add it only if a
# bundle actually pins that bare name.
image_build_context() {
  case "${1%%:*}" in
    light-servers) echo "$REPO/services/light-servers" ;;
    headroom-compress) echo "$REPO/tools/headroom" ;;
    openhands-runtime) echo "$REPO/tools/openhands_agent" ;;
  esac
}

# Extra `docker build` flags, one per line, for images whose Dockerfile needs
# more than its own context. headroom-compress copies the compression library
# from services/scoring/grader_compress.py (tools/headroom/Dockerfile).
image_build_args() {
  case "${1%%:*}" in
    headroom-compress)
      printf '%s\n' --build-context "scoring=$REPO/services/scoring" ;;
  esac
}

# Make one image usable, however it has to be obtained. Never asks the operator
# to have run `docker pull` or `make build-light-servers` first, which is what a
# fresh clone otherwise required: light-servers:latest is declared only in
# compose, with no build: section, so nothing in the run produces it and the
# compose up inside harbor_run dies on
# "pull access denied for light-servers, repository does not exist" -- minutes
# into the one stage that must never be re-run.
#
# For images built out of this checkout the refresh is an unconditional
# `docker build`, not a timestamp comparison, because there is nothing on the
# image to compare against: BuildKit does NOT advance .Created when every layer
# is cached (the config blob is byte-identical, so it is reused as-is). An
# mtime-vs-.Created check therefore never converges -- it reports stale,
# rebuilds, sees the same old .Created, and rebuilds again on every single run.
# BuildKit's own cache already answers the real question exactly, and answers it
# in about a second when nothing changed.
ensure_image() {
  local img="$1" ctx a extra=()
  ctx="$(image_build_context "$img")"
  while IFS= read -r a; do extra+=("$a"); done < <(image_build_args "$img")

  # Registry image: presence is the entire question.
  if [ -z "$ctx" ]; then
    docker image inspect "$img" >/dev/null 2>&1 && return 0
    echo "[run_task] $img is not present locally — pulling"
    docker pull "$img" || { echo "[run_task] failed to pull $img" >&2; exit 3; }
    return 0
  fi

  # First build on this machine -- the fresh-clone case. Minutes, with nothing
  # cached, so let the build print its own progress.
  if ! docker image inspect "$img" >/dev/null 2>&1; then
    echo "[run_task] $img is not present locally — building from $ctx"
    docker build ${extra[@]+"${extra[@]}"} -t "$img" "$ctx" \
      || { echo "[run_task] failed to build $img from $ctx" >&2; exit 3; }
    return 0
  fi

  # Present. Rebuild anyway so edits under $ctx reach the run: compose pins the
  # image by tag, so without this they simply do not, and nothing errors -- the
  # container boots clean, serves the previous world, and the agent is graded
  # against a bundle that no longer exists on disk. Quiet, because the common
  # case is fully cached and prints one line.
  [ -z "${SKIP_IMAGE_REFRESH:-}" ] || return 0
  docker build -q ${extra[@]+"${extra[@]}"} -t "$img" "$ctx" >/dev/null \
    || { echo "[run_task] failed to refresh $img from $ctx" >&2; exit 3; }
}

# The interpreter that can import harbor. `harbor` is installed as a uv tool, so
# it has its own venv; the repo .venv cannot import it and neither can the system
# python3. Without this the network preflight would degrade to a warn on the one
# machine it most needs to run on.
harbor_python() {
  local bin real
  bin="$(command -v harbor 2>/dev/null)" || return 1
  real="$(python3 -c "import os,sys;print(os.path.realpath(sys.argv[1]))" "$bin" 2>/dev/null)" || return 1
  local cand="$(dirname "$real")/python"
  [ -x "$cand" ] && { echo "$cand"; return 0; }
  return 1
}

stage_preflight() {
  # FIRST, before the image build and long before the agent phase: prove Harbor
  # can enforce this task's network policy. Harbor itself only checks this in
  # Trial.__init__, i.e. after Trial.create() has made the trial directory and
  # after the environment image has been built -- so an unenforceable policy
  # costs a full build, then aborts, then leaves a trial dir with no config.json
  # that harbor_to_output.py silently skips. Needs no docker, costs ~0.3s.
  #
  # PREFLIGHT_NETWORK_OFF=1 bypasses it.
  if [ -z "${PREFLIGHT_NETWORK_OFF:-}" ]; then
    # No silent fallback to python3: without harbor importable the checker
    # degrades to a warn and exits 0, so the gate would report "go" having
    # verified nothing -- the exact silent skip it exists to prevent.
    local _hpy
    _hpy="$(harbor_python)" || {
      echo "[run_task] cannot locate harbor's python (is harbor installed?);" >&2
      echo "           refusing to run an unverified network policy." >&2
      echo "           bypass with PREFLIGHT_NETWORK_OFF=1" >&2
      exit 2
    }
    # AGENT decides which image checks apply (the claude CLI pre-bake is a
    # claude-code requirement); it is a shell variable here, not an export.
    # The model-route hosts are the ones stage_harbor passes as
    # --allow-agent-host, so the plan checked is the plan harbor will enforce.
    local _host _route=()
    while IFS= read -r _host; do
      [ -n "$_host" ] && _route+=(--allow-agent-host "$_host")
    done < <(agent_route_hosts)
    AGENT="$AGENT" "$_hpy" "$REPO/tools/network/preflight_network.py" "$TASK" \
      ${_route[@]+"${_route[@]}"} || {
      echo "[run_task] network policy preflight failed — refusing to build or run" >&2
      exit 2
    }
  fi

  # GLM runs: prove the key works before the image builds, not after. Idempotent
  # -- stage_harbor calls it again and finds the bridge already up.
  [ "${CC_MODE:-}" = "zbridge" ] && ensure_zbridge
  # OpenHands runs on Claude: the same, for the subscription behind the
  # ccbridge. A per-run bridge does not exist yet, so preflight proves the login
  # with --check (no model call); the harbor stage starts the bridge.
  # Bedrock runs skip the bridge entirely: LiteLLM calls bedrock-runtime from
  # inside the container with AWS_BEARER_TOKEN_BEDROCK forwarded via harbor env.
  # Kiro runs skip ccbridge too: ensure_kiro_bridge starts its own proxy and
  # sets OPENHANDS_LLM_* / ANTHROPIC_* from the keys it mints.
  if [ "$AGENT" = "openhands" ] \
       && [ "${CC_MODE:-}" != "zbridge" ] \
       && [ "${CC_MODE:-}" != "bedrock" ] \
       && [ "${CC_MODE:-}" != "kiro" ]; then
    if [ "${CCBRIDGE_SHARED:-0}" = "1" ]; then ensure_shared_ccbridge; else ccbridge_prepare; fi
  fi

  if ! docker info >/dev/null 2>&1; then
    echo "[run_task] docker not running — starting OrbStack/Docker"
    open -a OrbStack 2>/dev/null || open -a Docker 2>/dev/null || true
    for _ in $(seq 1 60); do docker info >/dev/null 2>&1 && break; sleep 2; done
    docker info >/dev/null 2>&1 || { echo "[run_task] docker still down" >&2; exit 3; }
  fi
  # Every image this bundle needs, from both places one can be declared: the
  # (rare) top-level `image =` in task.toml, and the compose services that pin
  # an image with no build: section. No task.toml in this repo sets the former,
  # which is why the task.toml-only grep this replaces pulled nothing, ever --
  # while light-servers:latest, the image every bundle here actually depends on,
  # is declared only in compose and had to be built by hand before each run.
  for _img in \
    "$(grep -E '^image *= *"' "$TASK/task.toml" 2>/dev/null | head -1 | sed -E 's/.*"([^"]+)".*/\1/' || true)" \
    $(compose_unbuilt_images "$TASK/environment/docker-compose.yaml")
  do
    [ -n "$_img" ] || continue
    ensure_image "$_img"
  done

  headroom_memory_check

  # The OpenHands SDK tree tools/openhands_agent/overlay.yaml mounts into main.
  # Declared in an overlay, not the bundle, so the loop above never sees it.
  [ "$AGENT" = "openhands" ] && ensure_image "openhands-runtime:latest"

  # The optional Headroom image, built here and only when this run asks for it.
  # It is declared in an overlay rather than in the bundle's compose file, so
  # the loop above never sees it.
  if [ "${AGENT_HEADROOM_ENABLED:-false}" = "true" ]; then
    ensure_image "$HEADROOM_IMAGE"
  fi
}

# Harbor validates output/<job>/result.json against its own JobResult model
# before it will start a trial (harbor/job.py:80), and it dies in pydantic when
# that fails -- no trial, no logs, just a traceback. So a job dir written by an
# older harness is not merely stale, it is unopenable.
#
# One such dir is in the wild: harbor_to_output.py used to write our labelled
# "k=<k>" pass@k keys into that file, where the field is typed dict[int, float].
# The writer is fixed, but the dirs it already wrote are still on disk, and a
# fix nobody can reach without hand-editing JSON is not a fix. Normalise what we
# know how to normalise; leave anything else for harbor to rule on.
normalise_stale_job_result() {
  [ -f "$1" ] || return 0
  python3 - "$1" <<'PY' || true
import json, sys

p = sys.argv[1]
try:
    d = json.loads(open(p).read())
except Exception:
    sys.exit(0)             # unreadable is harbor's call to report, not ours

changed = False
for e in ((d.get("stats") or {}).get("evals") or {}).values():
    pk = e.get("pass_at_k")
    if not isinstance(pk, dict) or not any(str(k).startswith("k=") for k in pk):
        continue
    fixed = {}
    for k, v in pk.items():
        s = str(k).removeprefix("k=")
        if not s.isdigit():
            break           # not the shape we know; leave the whole map alone
        fixed[int(s)] = v
    else:
        e["pass_at_k"] = fixed
        changed = True

if changed:
    json.dump(d, open(p, "w"), indent=2)
    print(f"[run_task] repaired legacy pass_at_k keys in {p}")
PY
}

stage_harbor() {
  local offset; offset="$(resolve_run_offset)"
  state_put slug "$SLUG"
  state_put job "$JOB"
  state_put run_offset "$offset"

  # Preserve earlier Run_* dirs: harbor owns output/<job>/ and will happily
  # clear it, and those runs are other units' trajectories.
  if [ -d "$TRAJ_DIR" ]; then
    mkdir -p "$STASH_DIR"
    cp -r "$TRAJ_DIR"/run_* "$STASH_DIR/" 2>/dev/null || true
    local _job_out="$OUTPUT_DIR/$JOB"
    [ -f "$_job_out/summary.json" ]      && cp "$_job_out/summary.json"      "$STASH_DIR/.summary.json"      2>/dev/null || true
    [ -f "$_job_out/pass_summary.json" ] && cp "$_job_out/pass_summary.json" "$STASH_DIR/.pass_summary.json" 2>/dev/null || true
    state_put stash_dir "$STASH_DIR"
  fi

  # The default agent changed from claude-code to openhands, and a job dir keeps
  # numbering runs across invocations. Appending one agent's runs to another's
  # would publish a pass@k over two different solvers under one name. Read the
  # agent the job was last run with before harbor's config is cleared below.
  local _prev_agent=""
  _prev_agent="$(python3 -c 'import json,sys
print(((json.load(open(sys.argv[1])).get("agents") or [{}])[0] or {}).get("name") or "")' \
    "$OUTPUT_DIR/$JOB/config.json" 2>/dev/null || true)"
  if [ -n "$_prev_agent" ] && [ "$_prev_agent" != "$(harbor_agent_ref)" ]; then
    echo "[run_task] WARNING: $OUTPUT_DIR/$JOB holds runs from agent '$_prev_agent';" >&2
    echo "[run_task]   this run uses '$(harbor_agent_ref)' and will be numbered after them," >&2
    echo "[run_task]   so pass@k and summary.json will mix the two. Use JOB=<new name> to keep" >&2
    echo "[run_task]   them apart, or AGENT=claude-code to continue the earlier series." >&2
  fi

  rm -f "$OUTPUT_DIR/$JOB/lock.json"
  rm -f "$OUTPUT_DIR/$JOB/config.json"
  normalise_stale_job_result "$OUTPUT_DIR/$JOB/result.json"
  # SETUP_MULT multiplies Harbor's agent-setup timeout (base 360s). Setup is
  # `apt-get install curl procps` followed by
  # `curl downloads.claude.ai/.../bootstrap.sh | bash`, so it is network-bound
  # and unrelated to how long the task itself takes. A slow mirror or a cold
  # CDN blows the default and the trial dies as AgentSetupTimeoutError with an
  # empty /logs/agent -- which reads like an agent that produced nothing rather
  # than an agent that never started. Observed on a run that had succeeded
  # three times prior with no task change.
  # Must precede harbor: it exports ANTHROPIC_BASE_URL, which harbor's
  # claude_code agent reads from this environment and forwards into the
  # container (empty values are dropped, so a failed proxy start is a no-op).
  ensure_cc_bridge
  [ "${CC_MODE:-}" = "zbridge" ] && ensure_zbridge
  [ "${CC_MODE:-}" = "bedrock" ] && select_bedrock_judge
  # Exports OPENHANDS_LLM_BASE_URL / OPENHANDS_LLM_API_KEY, which the agent
  # class reads from harbor's environment: zbridge on a GLM run, else this
  # run's own ccbridge (stopped by the EXIT trap). Called directly, not in
  # $(...): the exports and the bridge must belong to this shell.
  [ "$AGENT" = "openhands" ] && ensure_openhands_model_route
  # After ensure_zbridge: on a GLM run both set ANTHROPIC_BASE_URL, and with
  # agent-path Headroom on it is the headroom container that must win, with
  # zbridge behind it as the upstream.
  route_agent_through_headroom
  local args=(run -y --path "$TASK" --agent "$(harbor_agent_ref)" --jobs-dir "$OUTPUT_DIR" --job-name "$JOB" \
              --environment-build-timeout-multiplier "$BUILD_MULT" \
              --agent-setup-timeout-multiplier "$SETUP_MULT" --n-attempts "$N")
  # The model route this run adds on the host (a bridge on host.docker.internal,
  # or the headroom container) has to stay reachable during agent.run(). Harbor
  # merges --allow-agent-host into the task's agent-phase allowlist and ignores
  # it against a public policy; the task's own hosts are never widened here.
  # Called after the proxy helpers above: they decide which route is in use.
  local _host
  while IFS= read -r _host; do
    [ -n "$_host" ] && args+=(--allow-agent-host "$_host")
  done < <(agent_route_hosts)
  # OpenHands runs: the SDK tree for main, and -- only while harbor's egress
  # sidecar is off -- the host-gateway mapping the ccbridge is reached through.
  # With the sidecar on, docker refuses extra_hosts on main and the sidecar
  # resolves host.docker.internal itself (tools/openhands_agent/overlay-host-gateway.yaml).
  if [ "$AGENT" = "openhands" ]; then
    args+=(--extra-docker-compose "$REPO/tools/openhands_agent/overlay.yaml")
    [ "$(harbor_egress_control)" = "off" ] \
      && args+=(--extra-docker-compose "$REPO/tools/openhands_agent/overlay-host-gateway.yaml")
  fi
  [ "${AGENT_HEADROOM_ENABLED:-false}" = "true" ] \
    && args+=(--extra-docker-compose "$REPO/tools/headroom/overlay.yaml")
  # Whether the task closes the agent phase. Read through Harbor's own resolver
  # so it is the policy `harbor run` enforces, not a reading of task.toml.
  local _closed=""
  [ "$(agent_phase_network_mode)" != "public" ] && _closed=1
  [ "$AGENT" != "oracle" ] && args+=(--model "$MODEL")
  # Kiro mode: inject the bridge's judge creds directly on the verifier via
  # --ve KEY=VALUE. Does not depend on the bundle's [verifier.env] block
  # forwarding ${ANTHROPIC_API_KEY:-} -- a bundle that forgot the forwarding
  # still gets a judged run.
  if [ "${#KIRO_VERIFIER_ARGS[@]}" -gt 0 ]; then
    echo "[run_task] appending ${#KIRO_VERIFIER_ARGS[@]} KIRO verifier args to harbor" >&2
    args+=("${KIRO_VERIFIER_ARGS[@]}")
  elif [ "${CC_MODE:-}" = "kiro" ]; then
    echo "[run_task] WARNING: CC_MODE=kiro but KIRO_VERIFIER_ARGS is empty; judge wire not injected" >&2
  fi
  if [ "${#BEDROCK_VERIFIER_ARGS[@]}" -gt 0 ]; then
    echo "[run_task] appending ${#BEDROCK_VERIFIER_ARGS[@]} BEDROCK verifier args to harbor" >&2
    args+=("${BEDROCK_VERIFIER_ARGS[@]}")
  fi
  ensure_rewardkit_judge_env || exit 4
  if [ "$AGENT" = "claude-code" ]; then
    [ -n "$THINKING" ] && args+=(--ak "thinking=$THINKING")
    [ -n "$THINKING_DISPLAY" ] && args+=(--ak "thinking_display=$THINKING_DISPLAY")
    # Second layer under Harbor's egress block, not a replacement for it. The
    # sidecar already makes these tools fail, but failing costs a turn: the
    # model picks WebFetch, waits out a connection error, and reasons about it
    # before trying the sidecars. Denying them means they never appear in the
    # tool list, so that turn is never spent.
    #
    # Deliberately NOT the whole defence. This is agent configuration, and an
    # agent can be run without it (AGENT=oracle, a future adapter, a hand
    # `harbor run`); the network block holds regardless. Both layers stay.
    #
    # Only when the task closes the agent phase: a task that declares a public
    # agent phase has asked for an open run, and silently keeping the web tools
    # off would make that run mean something different from what it declares.
    [ -n "$_closed" ] && [ -n "$DISALLOWED_TOOLS" ] \
      && args+=(--ak "disallowed_tools=$DISALLOWED_TOOLS")

    # Third layer, and the only one the model can actually READ. The sidecar
    # refuses the connection and the tool list removes the web tools; neither
    # says anything back when the model reaches for `apt-get`, so it retries.
    # This one refuses the call and explains what to use instead, in the same
    # turn. Same gate as above: an open run stays open.
    if [ -n "$_closed" ]; then
      require_harbor_agent_config
      local _guard; _guard="$(egress_guard_settings)"
      [ -n "$_guard" ] && args+=(--ak "config=$_guard")
    fi
  fi
  # OpenHands options (tools/openhands_agent/agent.py OpenHandsAgentOptions),
  # passed only when set so the agent's own defaults stay the defaults. None of
  # the claude-code layers above apply: there are no web tools to withhold
  # (the browser tool is not loaded) and no PreToolUse hook to install. The
  # routing table is the enforcement boundary for this agent, as for any other.
  if [ "$AGENT" = "openhands" ]; then
    [ -n "${OPENHANDS_MAX_ITERATIONS:-}" ] && args+=(--ak "max_iterations=$OPENHANDS_MAX_ITERATIONS")
    [ -n "${OPENHANDS_MAX_CONTINUATIONS:-}" ] && args+=(--ak "max_continuations=$OPENHANDS_MAX_CONTINUATIONS")
    [ -n "${OPENHANDS_REASONING_EFFORT:-}" ] && args+=(--ak "reasoning_effort=$OPENHANDS_REASONING_EFFORT")
    [ -n "${OPENHANDS_MAX_OUTPUT_TOKENS:-}" ] && args+=(--ak "max_output_tokens=$OPENHANDS_MAX_OUTPUT_TOKENS")
    # summarized (default, readable thinking in the trajectory) or omitted:
    # tools/openhands_agent/README.md, "Thinking".
    [ -n "${OPENHANDS_THINKING_DISPLAY:-}" ] && args+=(--ak "thinking_display=$OPENHANDS_THINKING_DISPLAY")
    # How long a model call rides out an unreachable model (a bridge 502, a
    # 503, a timeout): each attempt past the fourth adds 64s of waiting.
    [ -n "${OPENHANDS_NUM_RETRIES:-}" ] && args+=(--ak "num_retries=$OPENHANDS_NUM_RETRIES")
  fi
  # Trial dirs left behind by EARLIER invocations. Harbor never removes a trial
  # that died (docker build failure, Ctrl-C, agent-setup timeout) and
  # stage_harbor only clears lock.json/config.json, so those directories sit in
  # the job dir indefinitely. harbor_to_output.convert_job globs EVERY *__* dir
  # holding a config.json and emits one trajectory/run_N per hit -- so the next
  # `N=1` invocation materialised 1 + <stale count> runs in a single shot,
  # numbered in ASCII order of the random suffix rather than chronologically,
  # and every one of them counted as an attempt in summary.json and pass@k.
  # Snapshot what is already here; only the delta belongs to this invocation.
  local _pre; _pre="$(list_trial_dirs)"
  echo "[run_task] harbor ${args[*]}"
  # Must be immediately before harbor: everything that appears after this point
  # and carries harbor's "__" marker belongs to this invocation.
  snapshot_compose_projects
  if [ "${CC_MODE:-}" = "kiro" ]; then
    echo "[run_task] kiro env snapshot (keys set that the verifier.env will pull):"
    for _k in ANTHROPIC_API_KEY ANTHROPIC_BASE_URL OPENAI_API_KEY OPENAI_BASE_URL JUDGE_WIRE JUDGE_MODEL; do
      if [ -n "${!_k:-}" ]; then
        echo "[run_task]   $_k=<SET>"
      else
        echo "[run_task]   $_k=<EMPTY>"
      fi
    done
  fi
  local _hrc=0
  # harbor resolves --agent tools.openhands_agent.agent:OpenHandsAgent with
  # importlib, so the checkout has to be importable from harbor's interpreter.
  # pipx-installed harbor has shebang `python -E`; -E strips PYTHONPATH, so
  # simply exporting PYTHONPATH before `command harbor` is a no-op. Invoke the
  # venv's Python directly (no -E) instead, keeping the same argv shape.
  local _pypath="${PYTHONPATH:-}"
  local _harbor_bin _harbor_py
  _harbor_bin="$(command -v harbor)"
  if [ "$AGENT" = "openhands" ] && [ -n "$_harbor_bin" ]; then
    _pypath="$REPO${PYTHONPATH:+:$PYTHONPATH}"
    _harbor_py="$(head -n 1 "$_harbor_bin" | sed 's|^#!||' | awk '{print $1}')"
    if [ -n "$_harbor_py" ] && [ -x "$_harbor_py" ]; then
      PYTHONPATH="$_pypath" HARBOR_OUTPUT_OFF=1 "$_harbor_py" "$_harbor_bin" "${args[@]}" \
        || { _hrc=$?; echo "[run_task] harbor exited $_hrc; checking whether a trial actually ran" >&2; }
    else
      echo "[run_task] WARN: cannot locate harbor's Python from shebang; falling back to \`command harbor\` (PYTHONPATH may be stripped by -E)" >&2
      PYTHONPATH="$_pypath" HARBOR_OUTPUT_OFF=1 command harbor "${args[@]}" \
        || { _hrc=$?; echo "[run_task] harbor exited $_hrc; checking whether a trial actually ran" >&2; }
    fi
  else
    PYTHONPATH="$_pypath" HARBOR_OUTPUT_OFF=1 command harbor "${args[@]}" \
      || { _hrc=$?; echo "[run_task] harbor exited $_hrc; checking whether a trial actually ran" >&2; }
  fi

  # An interrupted harbor is not a failed trial. Continuing into reshape, mask
  # and finance would publish a half-run, and -- because harbor never reached
  # its own teardown -- would leave the compose project up behind it. Clean up
  # and stop, propagating the signal's exit code.
  #
  # The trap above catches the usual case, where the signal reaches this shell
  # too. This catches the rest: a signal delivered only to harbor, and a harbor
  # that died of one on its own.
  case "$_hrc" in
    130|143)
      echo "[run_task] harbor was interrupted (exit $_hrc); tearing down and stopping" >&2
      teardown_this_runs_projects
      exit "$_hrc"
      ;;
  esac

  # A trial DIRECTORY is not a trial that RAN. Harbor creates it in
  # Trial.create(), before the environment is built and before
  # Trial.__init__ validates the network policy -- so an aborted trial leaves a
  # directory containing nothing.
  #
  # Nothing downstream notices. harbor_to_output.py:1129 selects trials with
  # `(p / "config.json").exists()`, so an empty trial dir is SILENTLY SKIPPED:
  # convert_job returns [], reshape exits 0, and stage_finance then looks for a
  # trajectory/run_N that was never written. The visible symptom is a task that
  # gained no Run_N for no stated reason, thirty lines below the real error.
  #
  # stage_harbor removes $JOB/config.json before running, so what is checked
  # here is always this invocation's.
  local _job_dir="$OUTPUT_DIR/$JOB"

  # What THIS invocation produced: everything in the job dir now, minus what was
  # there before harbor ran. Recorded for stage_reshape, which may be a separate
  # process (run_batch.py drives the stages one at a time).
  local _new _stale_n
  _new="$(comm -13 <(printf '%s\n' "$_pre" | grep . | sort) \
                   <(list_trial_dirs | sort) | grep . || true)"
  state_put trials "$(printf '%s' "$_new" | tr '\n' ',' | sed 's/,$//')"
  _stale_n="$(printf '%s\n' "$_pre" | grep -c . || true)"
  if [ "$_stale_n" -gt 0 ]; then
    echo "[run_task] NOTE: $_stale_n trial dir(s) from earlier invocations are still in" >&2
    echo "           $_job_dir. They are NOT part of this run and will not be reshaped:" >&2
    echo "             $(printf '%s' "$_pre" | tr '\n' ' ')" >&2
    echo "           Delete them once inspected. To convert one deliberately (it was a" >&2
    echo "           real attempt whose reshape never finished):" >&2
    echo "             python3 tools/delivery_utils/harbor_to_output.py $_job_dir \\" >&2
    echo "               --output-dir $OUTPUT_DIR --run-offset <n> --only-trials <name>" >&2
  fi

  # The guard below asks whether the trial harbor just made actually started, so
  # it looks at THIS invocation's trials. Reading the newest dir by mtime instead
  # picked up a stale one whenever harbor aborted before creating any -- the
  # stale dir has a config.json, so the guard stayed silent on the very case it
  # exists to catch. Every new trial is checked, not just one: with N>1 a single
  # empty dir among several good ones would otherwise slip through.
  local _trial="" _n
  if [ -n "$_new" ] && [ -d "$_job_dir" ]; then
    while IFS= read -r _n; do
      [ -n "$_n" ] || continue
      [ -f "$_job_dir/$_n/config.json" ] || { _trial="$_job_dir/$_n"; break; }
    done <<< "$_new"
  fi
  if [ -n "$_trial" ]; then
    echo >&2
    echo "==> AGENT PHASE DID NOT RUN" >&2
    echo "    $_trial exists but has no config.json, so Harbor aborted before the" >&2
    echo "    trial started. The cause is in the harbor output above -- scroll to the" >&2
    echo "    LAST line of the traceback, which names it." >&2
    echo >&2
    echo "    Most common: [agent].network_mode differs from [environment]." >&2
    echo "    network_mode and the docker provider cannot switch policy after start." >&2
    echo "    Diagnose with:" >&2
    echo "      tools/network/preflight_network.py $TASK" >&2
    echo >&2
    echo "    Refusing to reshape: an empty trial dir is skipped without comment and" >&2
    echo "    would leave this task silently missing a run." >&2
    # `${_hrc:-1}` only substituted when _hrc was unset/empty, and _hrc is 0
    # whenever harbor believed it succeeded -- so this guard printed REFUSING TO
    # RESHAPE and then exited 0, which run_batch.py records as a completed unit.
    # A run that did not happen must not report success.
    [ "${_hrc:-0}" -ne 0 ] || _hrc=1
    exit "$_hrc"
  fi
  # Harbor exiting NON-ZERO with no trial directory at all: nothing ran and
  # nothing can be graded, so stop rather than let reshape find nothing.
  # A zero exit with no trial directory is a different animal -- harbor believes
  # it succeeded -- so that one is reported loudly and left to proceed, because
  # blocking on it would mean asserting a harbor contract this script does not
  # own. Loud either way; the thing being prevented is silence, not progress.
  if [ -z "$_new" ] && [ ! -d "$_job_dir/trajectory" ]; then
    if [ "$_hrc" -ne 0 ]; then
      echo >&2
      echo "==> NO TRIAL DIRECTORY: harbor exited $_hrc and produced nothing to grade." >&2
      echo "    Check the harbor output above." >&2
      exit "$_hrc"
    fi
    echo "[run_task] WARNING: harbor exited 0 but left no trial directory in" >&2
    echo "           $_job_dir — reshape will have nothing to convert." >&2
  fi
  state_put harbor_done 1
}

# ------------------------------------------------------------- agent Headroom
# Compress the AGENT's prompts with a Headroom proxy that runs IN THE BUNDLE, on
# the same network as main. OFF unless AGENT_HEADROOM_ENABLED=true.
#
# A container in the bundle's own compose project, so Harbor's egress sidecar
# governs its traffic like main's (tools/headroom/overlay.yaml).
#
# Still opt-in, and still a change to what the benchmark measures: compression
# rewrites the prompt prefix, so prompt caching misses and the trajectory is no
# longer the record of what the model saw. See services/scoring/grader_compress.py.
HEADROOM_SERVICE_PORT=8787            # tools/headroom/Dockerfile EXPOSEs this
HEADROOM_IMAGE="headroom-compress:latest"
HEADROOM_MIN_DOCKER_GB="${HEADROOM_MIN_DOCKER_GB:-12}"

# Refuse an agent-path Headroom run on a docker that cannot hold the project.
#
# WHY THIS IS A REFUSAL AND NOT A WARNING
#
# Measured, on an 8 GB docker VM: the project already runs main, light-servers
# and Harbor's egress sidecar, and headroom adds another container. The VM hit a
# GLOBAL out-of-memory (dmesg: `global_oom ... task=headroom`), the kernel
# killed the proxy, and because docker stops resolving the name of a container
# that is gone, Claude Code ended the session with "API Error: Can't reach the
# API server (EAI_AGAIN)" after ten turns. Harbor recorded no exception -- the
# agent process exited 0 -- so the trial was published as a reward of 0, in a
# job whose mean it then dragged down. An infrastructure death that looks like a
# bad answer is worse than a run that never starts, and the cost lands on the
# paid agent phase either way.
#
# HEADROOM_MIN_DOCKER_GB tunes the bar; HEADROOM_IGNORE_MEMORY=1 skips it.
headroom_memory_check() {
  [ "${AGENT_HEADROOM_ENABLED:-false}" = "true" ] || return 0
  if [ -n "${HEADROOM_IGNORE_MEMORY:-}" ]; then
    echo "[run_task] HEADROOM_IGNORE_MEMORY set; not checking docker's memory" >&2
    return 0
  fi
  local total gb
  total="$(docker info --format '{{.MemTotal}}' 2>/dev/null)" || return 0
  case "$total" in ''|*[!0-9]*) return 0 ;; esac   # cannot tell; do not block
  # Rounded, not floored. A VM set to 12 GB reports what is left after the
  # kernel's own reservation -- 12,304,840 kB, i.e. 11.7 GiB -- so flooring
  # called a correctly sized machine 11 GB and refused it. Being told to raise
  # a limit that is already raised is how a guard gets commented out.
  gb=$(( (total + 536870912) / 1073741824 ))
  [ "$gb" -ge "$HEADROOM_MIN_DOCKER_GB" ] && return 0
  echo "[run_task] REFUSING: docker has ${gb} GB of memory; agent-path Headroom needs" >&2
  echo "[run_task]   at least ${HEADROOM_MIN_DOCKER_GB} GB. It adds a proxy container to a project that already" >&2
  echo "[run_task]   runs main, light-servers and Harbor's egress sidecar." >&2
  echo "[run_task]   At 8 GB the VM ran out and the kernel killed the proxy mid-run: the" >&2
  echo "[run_task]   agent lost the API and the trial published a reward of 0." >&2
  echo "[run_task]   Raise the memory in OrbStack/Docker Desktop settings, run without" >&2
  echo "[run_task]   AGENT_HEADROOM_ENABLED, or set HEADROOM_IGNORE_MEMORY=1 to proceed." >&2
  exit 2
}

ensure_cc_bridge() {
  # OFF by default. Nothing this script runs reads :4000 -- the agent talks to
  # api.anthropic.com (or to zbridge) -- so starting it bought nothing, while
  # costing 180s of dead
  # wall-clock per trial on any host where it cannot boot (a missing fastapi in
  # tools/bridges/cbridge is enough, and is the state of this machine).
  #
  # run_eval.py and adapters/ DO use it, via LLM_BASE_URL. Set
  # CC_BRIDGE_ENABLED=1 there, or start it by hand.
  [ "${CC_BRIDGE_ENABLED:-0}" = "1" ] || return 0
  local port="${CC_BRIDGE_PORT:-4000}"
  if curl -sf -m 2 "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
    echo "[run_task] cc-bridge already running on :$port"
    return 0
  fi
  local bridge_dir="$REPO/tools/bridges/cbridge"
  local py
  for py in "$bridge_dir/.venv/bin/python3" "$REPO/.venv/bin/python3" python3; do
    command -v "$py" >/dev/null 2>&1 && break
  done
  if [ ! -f "$bridge_dir/cbridge.py" ]; then
    echo "[run_task] WARNING: cbridge not found at $bridge_dir; agent will run without proxy" >&2
    return 0
  fi
  echo "[run_task] starting cbridge on :$port"
  mkdir -p "$bridge_dir/logs"
  CC_BRIDGE_PORT="$port" nohup "$py" "$bridge_dir/cbridge.py" \
    >"$bridge_dir/logs/cbridge.log" 2>&1 &
  local i=0
  while [ $i -lt 20 ]; do
    sleep 1; i=$((i+1))
    curl -sf -m 1 "http://127.0.0.1:$port/health" >/dev/null 2>&1 && {
      echo "[run_task] cc-bridge ready on :$port"; return 0
    }
  done
  echo "[run_task] WARNING: cbridge did not come up in 20s; check $bridge_dir/logs/cbridge.log" >&2
}

# Which address zbridge binds, and which address this script probes it on.
#
# Not loopback. zbridge lives on the host, but the caller that matters is the
# agent inside the task container, which reaches it through
# host.docker.internal -- the docker gateway, not 127.0.0.1. A loopback bind is
# refused there, and every turn of the run fails while /healthz on 127.0.0.1
# stays green.
#
# ccbridge_host() answers the same question for the ccbridge and picks 0.0.0.0
# on Linux. This one prefers the gateway address, because zbridge runs
# deliberately unauthenticated (ZB_BRIDGE_SECRET is left empty below, since the
# containerised agent sends no x-zbridge-secret header) and 0.0.0.0 would put an
# unmetered z.ai spend endpoint on every interface of the box. 0.0.0.0 remains
# the fallback for hosts with no docker bridge to read, and ZB_HOST overrides.
zbridge_host() {
  if [ -n "${ZB_HOST:-}" ]; then echo "$ZB_HOST"; return; fi
  case "$(uname -s)" in Darwin) echo "127.0.0.1"; return ;; esac
  local gw
  gw="$(docker network inspect bridge -f '{{range .IPAM.Config}}{{.Gateway}}{{end}}' 2>/dev/null)"
  case "$gw" in
    ''|*[!0-9.]*) echo "0.0.0.0" ;;
    *) echo "$gw" ;;
  esac
}

# A 0.0.0.0 bind is not a connectable address; everything else is its own probe.
zbridge_probe_host() {
  case "$(zbridge_host)" in 0.0.0.0) echo "127.0.0.1" ;; *) zbridge_host ;; esac
}

ensure_zbridge() {
  local port="${ZB_PORT:-8766}" host probe
  host="$(zbridge_host)"; probe="$(zbridge_probe_host)"
  # zbridge serves /healthz (bridge.py), NOT /health -- the adapter on :4001 and
  # cc-bridge on :4000 are the ones with /health. Probing /health here 404s on a
  # perfectly healthy bridge, so this guard never fired: every run fell through
  # to the start branch and spawned a second uvicorn, which could not bind the
  # port but did truncate the live bridge's log on its way out (the redirect
  # below is `>`), destroying its startup warnings.
  if curl -sf -m 2 "http://$probe:$port/healthz" >/dev/null 2>&1; then
    echo "[run_task] zbridge already running on $probe:$port"
  elif [ "$probe" != "127.0.0.1" ] && curl -sf -m 2 "http://127.0.0.1:$port/healthz" >/dev/null 2>&1; then
    # A bridge IS up, on loopback only -- the pre-fix bind, or start_zbridge.sh
    # with its 127.0.0.1 default. Starting a second one cannot help: the port is
    # taken, the new uvicorn dies, and the run proceeds against a bridge no
    # container can reach. Say which process to kill instead.
    echo "[run_task] ERROR: a zbridge is listening on 127.0.0.1:$port but not on $probe:$port." >&2
    echo "[run_task]   the agent reaches it via host.docker.internal ($probe), so every" >&2
    echo "[run_task]   agent turn would fail to connect." >&2
    echo "[run_task]   Fix: kill that process and re-run, or start it with ZB_HOST=$probe." >&2
    exit 3
  else
    local zbridge_dir="$REPO/tools/bridges/zbridge"
    if [ ! -f "$zbridge_dir/pyproject.toml" ]; then
      echo "[run_task] ERROR: zbridge not found at $zbridge_dir" >&2
      exit 3
    fi
    echo "[run_task] starting zbridge on :$port"
    local log_dir="$zbridge_dir/logs"
    mkdir -p "$log_dir"
    (cd "$zbridge_dir" && \
      ZB_ZAI_API_KEY="$ZB_ZAI_API_KEY" \
      ZB_BRIDGE_SECRET="" \
      ZB_PORT="$port" ZB_HOST="$host" \
      ZB_THINKING_SIG_KEY="${ZB_THINKING_SIG_KEY:-yuji-harness-zbridge-v1}" \
      ZB_MODEL_ALIAS_JSON='{"claude-opus-5":"glm-5.3","claude-sonnet-5":"glm-5.3","claude-sonnet-4-6":"glm-5.3","claude-sonnet-4-5":"glm-5.3","claude-opus-4-8":"glm-5.3","claude-opus-4-7":"glm-5.3","claude-haiku-4-5-20251001":"glm-5.3","claude-haiku-4-5":"glm-5.3","claude-3-5-sonnet-latest":"glm-5.3","claude-3-opus-latest":"glm-5.3"}' \
      ZB_UPSTREAM_URL="${ZB_UPSTREAM_URL:-https://api.z.ai/api/coding/paas/v4/chat/completions}" \
      nohup uv run python -m zbridge --port "$port" --host "$host" \
        >"$log_dir/zbridge.log" 2>&1 &)
    # Generous, because the first `uv run` on a machine resolves and builds the
    # bridge's venv before anything listens -- minutes, not seconds. The old 15s
    # was under that, so a cold start looked like a broken bridge.
    local i=0 wait_s="${ZB_START_TIMEOUT_SEC:-180}"
    while [ $i -lt "$wait_s" ]; do
      sleep 1; i=$((i+1))
      curl -sf -m 1 "http://$probe:$port/healthz" >/dev/null 2>&1 && {
        echo "[run_task] zbridge ready on $probe:$port"; break
      }
    done
    # Pointing the agent at a port nothing listens on spends the whole trial on
    # connection errors, so this refuses instead of warning.
    curl -sf -m 1 "http://$probe:$port/healthz" >/dev/null 2>&1 || {
      echo "[run_task] ERROR: zbridge did not come up in ${wait_s}s on $probe:$port" >&2
      echo "[run_task]   The agent would fail every turn. See $log_dir/zbridge.log" >&2
      echo "[run_task]   Raise ZB_START_TIMEOUT_SEC if this machine is just slow." >&2
      exit 3
    }
  fi
  zbridge_live_check || exit 4
  export ANTHROPIC_BASE_URL="http://host.docker.internal:$port"
  echo "[run_task] agent routed through zbridge ($ANTHROPIC_BASE_URL)"
}

# /healthz answers ok whatever key the bridge holds, so it cannot tell a working
# GLM login from a dead one. One minimal request does. Only an upstream 401/403
# refuses the run; anything else warns, because it may not be the credential.
zbridge_live_check() {
  local port="${ZB_PORT:-8766}" verdict probe
  probe="$(zbridge_probe_host)"
  verdict="$(python3 - "$probe" "$port" <<'PYEOF' 2>/dev/null || true
import json, sys, urllib.error, urllib.request
body = json.dumps({"model": "claude-opus-5", "max_tokens": 1,
                   "messages": [{"role": "user", "content": "ping"}]}).encode()
req = urllib.request.Request(f"http://{sys.argv[1]}:{sys.argv[2]}/v1/messages",
                             data=body, headers={"content-type": "application/json"})
try:
    with urllib.request.urlopen(req, timeout=60) as r:
        print("ok" if r.status == 200 else f"http {r.status}")
except urllib.error.HTTPError as exc:
    try:
        detail = (exc.read() or b"")[:160].decode("utf-8", "replace").replace("\n", " ")
    except Exception:
        detail = ""
    print(f"{'auth' if exc.code in (401, 403) else 'http'} {exc.code} {detail}")
except Exception as exc:
    print(f"unverified {exc}")
PYEOF
)"
  case "$verdict" in
    ok)
      echo "[run_task] zbridge: GLM credential verified against z.ai" ;;
    auth*)
      echo "[run_task] ERROR: z.ai rejected the GLM credential — $verdict" >&2
      echo "[run_task]   Every agent turn would fail auth. Fix ZB_ZAI_API_KEY in .env." >&2
      return 1 ;;
    *)
      echo "[run_task] WARNING: could not verify the GLM credential (${verdict:-no answer}); continuing" >&2 ;;
  esac
  return 0
}

# ------------------------------------------------------------------ kiro bridge
# tools/bridges/kiro_bridge_runner.py: a per-run local Anthropic-compatible
# proxy backed by the host's kiro-cli login. Zero credentials enter the
# container -- only two short-lived keys the bridge mints at start time:
#   agent key -- serves whatever model each request names (the run's MODEL)
#   judge key -- pinned to KIRO_JUDGE_MODEL, so a task's rubric judge gets a
#                fixed model regardless of what judge.toml asks for
# Both exported through the usual env surface: OPENHANDS_LLM_* for the agent,
# ANTHROPIC_BASE_URL / ANTHROPIC_API_KEY for the judge wire (task bundles must
# forward ANTHROPIC_BASE_URL in [verifier.env] for the judge to find it).
KIRO_BRIDGE_PID=""
KIRO_BRIDGE_CONFIG=""
# Every call the bridge serves, one JSON line each. Named by job, not pid, so a
# later `--stage reshape` finds it; kiro_judge_tokens.py turns the judge calls
# into verifier/judge_tokens.json for bundles whose judge records no usage.
KIRO_CALLS_LOG="$OUTPUT_DIR/.kiro-bridge/calls-${JOB//[^A-Za-z0-9._-]/_}.jsonl"
# Verifier env args to append to the harbor CLI (--ve KEY=VALUE, repeatable).
# Populated by ensure_kiro_bridge. Harbor sets these DIRECTLY on the verifier
# container, bypassing task.toml's ${VAR:-} template resolution -- so a bundle
# that forgot to forward ANTHROPIC_BASE_URL (or similar) still works.
KIRO_VERIFIER_ARGS=()

ensure_kiro_bridge() {
  # kiro_bridge is vendored at tools/bridges/kiro_bridge/ and has zero runtime
  # dependencies (pure stdlib), so no venv or pip-install is needed -- just
  # put its parent on PYTHONPATH and run with host python3.
  local host_py; host_py="$(command -v python3 || true)"
  if [ -z "$host_py" ]; then
    echo "[run_task] kiro ERROR: python3 not found on PATH" >&2
    exit 4
  fi
  local bridge_path="$REPO/tools/bridges"
  if ! PYTHONPATH="$bridge_path" "$host_py" -c 'import kiro_bridge' >/dev/null 2>&1; then
    echo "[run_task] kiro ERROR: cannot import kiro_bridge from $bridge_path/kiro_bridge" >&2
    echo "[run_task]   Check that tools/bridges/kiro_bridge/__init__.py exists." >&2
    exit 4
  fi
  local cfg_dir="$OUTPUT_DIR/.kiro-bridge"
  mkdir -p "$cfg_dir"
  KIRO_BRIDGE_CONFIG="$cfg_dir/run-${JOB//[^A-Za-z0-9._-]/_}-$$.json"
  rm -f "$KIRO_BRIDGE_CONFIG"
  local log="$cfg_dir/run-${JOB//[^A-Za-z0-9._-]/_}-$$.log"
  echo "[run_task] starting kiro bridge (log: $log)"
  (
    KIRO_MODEL="${KIRO_MODEL:-$MODEL}" \
    KIRO_JUDGE_MODEL="${KIRO_JUDGE_MODEL:-}" \
    PYTHONPATH="$bridge_path${PYTHONPATH:+:$PYTHONPATH}" \
    nohup "$host_py" "$REPO/tools/bridges/kiro_bridge_runner.py" \
      --host 127.0.0.1 --port 0 --config-out "$KIRO_BRIDGE_CONFIG" \
      --calls-out "$KIRO_CALLS_LOG" \
      >"$log" 2>&1 &
    echo $!
  ) > "$cfg_dir/.pid.tmp"
  KIRO_BRIDGE_PID="$(cat "$cfg_dir/.pid.tmp" | tr -d '[:space:]')"
  rm -f "$cfg_dir/.pid.tmp"
  local i=0 wait_s="${KIRO_BRIDGE_START_TIMEOUT_SEC:-60}"
  while [ $i -lt "$wait_s" ]; do
    [ -f "$KIRO_BRIDGE_CONFIG" ] && break
    if ! kill -0 "$KIRO_BRIDGE_PID" 2>/dev/null; then
      echo "[run_task] kiro ERROR: bridge runner exited during startup. Last 10 lines of $log:" >&2
      tail -10 "$log" 2>/dev/null | sed 's/^/[run_task]   /' >&2
      exit 3
    fi
    sleep 1; i=$((i+1))
  done
  if [ ! -f "$KIRO_BRIDGE_CONFIG" ]; then
    echo "[run_task] kiro ERROR: bridge did not come up in ${wait_s}s; see $log" >&2
    kill "$KIRO_BRIDGE_PID" 2>/dev/null || true
    exit 3
  fi
  local port agent_key judge_key agent_model judge_model
  port="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("port",""))' "$KIRO_BRIDGE_CONFIG")"
  agent_key="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("agent_key",""))' "$KIRO_BRIDGE_CONFIG")"
  judge_key="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("judge_key",""))' "$KIRO_BRIDGE_CONFIG")"
  agent_model="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("agent_model",""))' "$KIRO_BRIDGE_CONFIG")"
  judge_model="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("judge_model",""))' "$KIRO_BRIDGE_CONFIG")"
  if [ -z "$port" ] || [ -z "$agent_key" ] || [ -z "$judge_key" ]; then
    echo "[run_task] kiro ERROR: bridge config missing fields in $KIRO_BRIDGE_CONFIG" >&2
    exit 3
  fi
  export OPENHANDS_LLM_BASE_URL="http://host.docker.internal:$port"
  export OPENHANDS_LLM_API_KEY="$agent_key"
  export ANTHROPIC_BASE_URL="http://host.docker.internal:$port"
  export ANTHROPIC_API_KEY="$judge_key"
  export JUDGE_WIRE="${JUDGE_WIRE:-anthropic}"
  KIRO_VERIFIER_ARGS=(
    --ve "ANTHROPIC_API_KEY=$judge_key"
    --ve "ANTHROPIC_BASE_URL=http://host.docker.internal:$port"
    --ve "JUDGE_WIRE=${JUDGE_WIRE:-anthropic}"
    # litellm (genkit/rewardkit judges) rejects reasoning_effort for bridge
    # model names it doesn't map (e.g. anthropic/gpt-5.6-sol); drop it.
    --ve "LITELLM_DROP_PARAMS=True"
  )
  # rubric_judge_cli.py reads JUDGE_MODEL from env; without it, it defaults to
  # the wire's canonical model (claude-sonnet-5 for anthropic) which is NOT
  # the model the kiro bridge is serving on the judge key. Pin it so the
  # judge call records the right model and matches what the bridge actually
  # answers.
  # Only pin JUDGE_MODEL when the kiro bridge is actually serving the judge
  # (JUDGE_WIRE=anthropic path) AND the operator didn't set JUDGE_MODEL
  # themselves. With JUDGE_TRANSPORT=codex the operator's JUDGE_MODEL is for
  # a different wire; clobbering it would make rewardkit call codex with
  # gpt-5.6-sol and drop the rubric channel to pytest-only.
  if [ -n "${KIRO_JUDGE_MODEL:-}" ] \
       && [ -z "${JUDGE_MODEL:-}" ] \
       && [ "${JUDGE_WIRE:-anthropic}" = "anthropic" ]; then
    KIRO_VERIFIER_ARGS+=(--ve "JUDGE_MODEL=$KIRO_JUDGE_MODEL")
  fi
  echo "[run_task] kiro bridge ready on 127.0.0.1:$port (pid $KIRO_BRIDGE_PID)"
  echo "[run_task]   agent model = $agent_model"
  echo "[run_task]   judge model = $judge_model"
  echo "[run_task]   openhands  -> $OPENHANDS_LLM_BASE_URL"
  echo "[run_task]   judge wire -> $ANTHROPIC_BASE_URL (JUDGE_WIRE=$JUDGE_WIRE)"
}

stop_kiro_bridge() {
  [ -n "${KIRO_BRIDGE_PID:-}" ] || return 0
  if kill -0 "$KIRO_BRIDGE_PID" 2>/dev/null; then
    kill -TERM "$KIRO_BRIDGE_PID" 2>/dev/null || true
    local i=0
    while [ $i -lt 20 ] && kill -0 "$KIRO_BRIDGE_PID" 2>/dev/null; do
      sleep 0.25; i=$((i+1))
    done
    kill -9 "$KIRO_BRIDGE_PID" 2>/dev/null || true
    echo "[run_task] kiro bridge stopped (pid $KIRO_BRIDGE_PID)" >&2
  fi
  KIRO_BRIDGE_PID=""
  [ -n "${KIRO_BRIDGE_CONFIG:-}" ] && rm -f "$KIRO_BRIDGE_CONFIG"
  return 0
}

# ------------------------------------------------------- bedrock judge proxy
# The bundled rubric_judge_cli.py (in the ECR judge image) only speaks
# JUDGE_WIRE = openai | anthropic | gemini -- no native Bedrock. In CC_MODE
# =bedrock, this proxy serves /v1/messages as an Anthropic Messages API, and
# forwards each call to the Bedrock Converse API via boto3. The judge sees an
# ANTHROPIC_BASE_URL and ANTHROPIC_API_KEY and never touches AWS; the proxy
# process holds AWS_BEARER_TOKEN_BEDROCK and resolves the JUDGE_MODEL alias
# through BEDROCK_MODEL_IDS exactly like the agent's resolver does.
BEDROCK_JUDGE_PID=""
BEDROCK_JUDGE_CONFIG=""
BEDROCK_VERIFIER_ARGS=()

ensure_bedrock_judge_proxy() {
  local host_py; host_py="$(command -v python3 || true)"
  if [ -z "$host_py" ]; then
    echo "[run_task] bedrock judge ERROR: python3 not found on PATH" >&2
    exit 4
  fi
  local bridge_path="$REPO/tools/bridges"
  if ! PYTHONPATH="$bridge_path" "$host_py" -c 'import bedrock_bridge, boto3' >/dev/null 2>&1; then
    echo "[run_task] bedrock judge ERROR: cannot import bedrock_bridge + boto3 from host python" >&2
    echo "[run_task]   Required host modules: bedrock_bridge (vendored), boto3 (pip install boto3)" >&2
    exit 4
  fi
  local judge_alias="${JUDGE_MODEL:-$MODEL}"
  if [ -z "$judge_alias" ]; then
    echo "[run_task] bedrock judge ERROR: JUDGE_MODEL (or MODEL) must be set" >&2
    exit 4
  fi
  local cfg_dir="$OUTPUT_DIR/.bedrock-judge"
  mkdir -p "$cfg_dir"
  BEDROCK_JUDGE_CONFIG="$cfg_dir/run-${JOB//[^A-Za-z0-9._-]/_}-$$.json"
  rm -f "$BEDROCK_JUDGE_CONFIG"
  local log="$cfg_dir/run-${JOB//[^A-Za-z0-9._-]/_}-$$.log"
  echo "[run_task] starting bedrock judge proxy (log: $log)"
  (
    JUDGE_MODEL="$judge_alias" \
    MODEL="${MODEL:-}" \
    BEDROCK_MODEL_IDS="${BEDROCK_MODEL_IDS:-}" \
    AWS_BEARER_TOKEN_BEDROCK="${AWS_BEARER_TOKEN_BEDROCK:-}" \
    AWS_REGION="${BEDROCK_AWS_REGION:-${AWS_REGION:-${AWS_DEFAULT_REGION:-}}}" \
    BEDROCK_AWS_REGION="${BEDROCK_AWS_REGION:-${AWS_REGION:-${AWS_DEFAULT_REGION:-}}}" \
    PYTHONPATH="$bridge_path${PYTHONPATH:+:$PYTHONPATH}" \
    nohup "$host_py" "$REPO/tools/bridges/bedrock_bridge_runner.py" \
      --host 127.0.0.1 --port 0 --config-out "$BEDROCK_JUDGE_CONFIG" \
      >"$log" 2>&1 &
    echo $!
  ) > "$cfg_dir/.pid.tmp"
  BEDROCK_JUDGE_PID="$(cat "$cfg_dir/.pid.tmp" | tr -d '[:space:]')"
  rm -f "$cfg_dir/.pid.tmp"
  local i=0 wait_s="${BEDROCK_JUDGE_START_TIMEOUT_SEC:-30}"
  while [ $i -lt "$wait_s" ]; do
    [ -f "$BEDROCK_JUDGE_CONFIG" ] && break
    if ! kill -0 "$BEDROCK_JUDGE_PID" 2>/dev/null; then
      echo "[run_task] bedrock judge ERROR: proxy exited during startup. Last 10 lines of $log:" >&2
      tail -10 "$log" 2>/dev/null | sed 's/^/[run_task]   /' >&2
      exit 3
    fi
    sleep 1; i=$((i+1))
  done
  if [ ! -f "$BEDROCK_JUDGE_CONFIG" ]; then
    echo "[run_task] bedrock judge ERROR: proxy did not come up in ${wait_s}s; see $log" >&2
    kill "$BEDROCK_JUDGE_PID" 2>/dev/null || true
    exit 3
  fi
  local port api_key judge_model
  port="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("port",""))' "$BEDROCK_JUDGE_CONFIG")"
  api_key="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("api_key",""))' "$BEDROCK_JUDGE_CONFIG")"
  judge_model="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("judge_model",""))' "$BEDROCK_JUDGE_CONFIG")"
  if [ -z "$port" ] || [ -z "$api_key" ]; then
    echo "[run_task] bedrock judge ERROR: proxy config missing fields in $BEDROCK_JUDGE_CONFIG" >&2
    exit 3
  fi
  BEDROCK_VERIFIER_ARGS=(
    --ve "ANTHROPIC_API_KEY=$api_key"
    --ve "ANTHROPIC_BASE_URL=http://host.docker.internal:$port"
    --ve "JUDGE_WIRE=anthropic"
    --ve "JUDGE_MODEL=$judge_model"
  )
  echo "[run_task] bedrock judge proxy ready on 127.0.0.1:$port (pid $BEDROCK_JUDGE_PID)"
  echo "[run_task]   judge model = $judge_model"
  echo "[run_task]   verifier wire -> http://host.docker.internal:$port (JUDGE_WIRE=anthropic)"
}

stop_bedrock_judge_proxy() {
  [ -n "${BEDROCK_JUDGE_PID:-}" ] || return 0
  if kill -0 "$BEDROCK_JUDGE_PID" 2>/dev/null; then
    kill -TERM "$BEDROCK_JUDGE_PID" 2>/dev/null || true
    local i=0
    while [ $i -lt 20 ] && kill -0 "$BEDROCK_JUDGE_PID" 2>/dev/null; do
      sleep 0.25; i=$((i+1))
    done
    kill -9 "$BEDROCK_JUDGE_PID" 2>/dev/null || true
    echo "[run_task] bedrock judge proxy stopped (pid $BEDROCK_JUDGE_PID)" >&2
  fi
  BEDROCK_JUDGE_PID=""
  [ -n "${BEDROCK_JUDGE_CONFIG:-}" ] && rm -f "$BEDROCK_JUDGE_CONFIG"
  return 0
}

# ------------------------------------------------------ rewardkit rubric judge
# Bundles in the TASK-STRUCTURE layout (tests/reward/rubric/judge.toml) are judged
# by rewardkit inside tests/test.sh, which reads JUDGE_TRANSPORT (codex default)
# plus CODEX_AUTH_B64 or CLAUDE_CODE_OAUTH_TOKEN -- not the JUDGE_WIRE/ANTHROPIC_*
# creds the legacy tests/grade.py judge uses. task.toml's [verifier.env] pulls
# them from this shell, so export them here; harbor stores them as ${VAR}
# templates in config.json, never the literal value. Without them test.sh
# silently skips the rubric and the run ships pytest-only, so fail instead.
# JUDGE_SKIP=1 opts out.
ensure_rewardkit_judge_env() {
  [ -f "$TASK/tests/reward/rubric/judge.toml" ] || return 0
  if [ "${JUDGE_SKIP:-0}" = "1" ]; then
    echo "[run_task] JUDGE_SKIP=1: rubric judge disabled, runs will be pytest-only" >&2
    return 0
  fi
  # With a harness judge proxy (kiro bridge / bedrock proxy, passed as
  # --ve ANTHROPIC_BASE_URL=...) route rewardkit through it via the claude CLI,
  # the same proxy legacy grade.py tasks are judged through.
  local a via_proxy=""
  for a in "${args[@]}"; do
    case "$a" in ANTHROPIC_BASE_URL=?*) via_proxy=1 ;; esac
  done
  if [ -n "$via_proxy" ] && [ -z "${JUDGE_TRANSPORT:-}" ]; then
    export JUDGE_TRANSPORT=claude-code
    # An OAuth token would make the claude CLI bypass the proxy's API key.
    export JUDGE_CLAUDE_TOKEN=""
    args+=(--ve JUDGE_TRANSPORT=claude-code)
  fi
  export JUDGE_TRANSPORT="${JUDGE_TRANSPORT:-codex}"
  case "$JUDGE_TRANSPORT" in
    codex)
      if [ -z "${CODEX_AUTH_B64:-}" ]; then
        local auth="${CODEX_AUTH_FILE:-$HOME/.codex/auth.json}"
        if [ ! -s "$auth" ]; then
          echo "[run_task] rubric judge ERROR: JUDGE_TRANSPORT=codex needs CODEX_AUTH_B64 or a codex login at $auth (run 'codex login', or set JUDGE_SKIP=1)" >&2
          return 1
        fi
        CODEX_AUTH_B64="$(base64 < "$auth" | tr -d '\n')"
      fi
      export CODEX_AUTH_B64
      ;;
    claude-code)
      [ -n "$via_proxy" ] || export JUDGE_CLAUDE_TOKEN="${JUDGE_CLAUDE_TOKEN:-${CLAUDE_CODE_OAUTH_TOKEN:-}}"
      if [ -z "$via_proxy$JUDGE_CLAUDE_TOKEN${ANTHROPIC_API_KEY:-}" ]; then
        echo "[run_task] rubric judge ERROR: JUDGE_TRANSPORT=claude-code needs JUDGE_CLAUDE_TOKEN / CLAUDE_CODE_OAUTH_TOKEN or ANTHROPIC_API_KEY" >&2
        return 1
      fi
      ;;
    *)
      echo "[run_task] rubric judge ERROR: unknown JUDGE_TRANSPORT '$JUDGE_TRANSPORT' (codex or claude-code)" >&2
      return 1
      ;;
  esac
  # JUDGE_MODEL may also arrive as a --ve already in the caller's args.
  local has_model=""
  [ -n "${JUDGE_MODEL:-}" ] && has_model=1
  for a in "${args[@]}"; do
    case "$a" in JUDGE_MODEL=?*) has_model=1 ;; esac
  done
  if [ -z "$has_model" ]; then
    echo "[run_task] rubric judge ERROR: JUDGE_MODEL is empty (set JUDGE_MODEL or KIRO_JUDGE_MODEL, or JUDGE_SKIP=1)" >&2
    return 1
  fi
  echo "[run_task] rubric judge: rewardkit via JUDGE_TRANSPORT=$JUDGE_TRANSPORT" >&2
}

# -------------------------------------------------- bedrock judge dispatcher
# JUDGE_MODE picks the judge wire when CC_MODE=bedrock. The agent always runs
# on Bedrock (per CC_MODE); the judge wire is independent so an operator can
# compare judge models without rerunning the agent. Falls back to the Bedrock
# judge proxy when JUDGE_MODE is unset.
select_bedrock_judge() {
  local mode="${JUDGE_MODE:-bedrock}"
  case "$mode" in
    bedrock)
      ensure_bedrock_judge_proxy
      ;;
    anthropic)
      if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
        echo "[run_task] JUDGE_MODE=anthropic ERROR: ANTHROPIC_API_KEY not set" >&2
        exit 4
      fi
      BEDROCK_VERIFIER_ARGS=(
        --ve "ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY"
        --ve "JUDGE_WIRE=anthropic"
      )
      [ -n "${JUDGE_MODEL:-}" ] && BEDROCK_VERIFIER_ARGS+=(--ve "JUDGE_MODEL=$JUDGE_MODEL")
      echo "[run_task] judge wire -> api.anthropic.com (JUDGE_MODE=anthropic, JUDGE_MODEL=${JUDGE_MODEL:-<wire default>})"
      ;;
    openai)
      if [ -z "${OPENAI_API_KEY:-}" ]; then
        echo "[run_task] JUDGE_MODE=openai ERROR: OPENAI_API_KEY not set" >&2
        exit 4
      fi
      BEDROCK_VERIFIER_ARGS=(
        --ve "OPENAI_API_KEY=$OPENAI_API_KEY"
        --ve "JUDGE_WIRE=openai"
      )
      [ -n "${JUDGE_MODEL:-}" ] && BEDROCK_VERIFIER_ARGS+=(--ve "JUDGE_MODEL=$JUDGE_MODEL")
      echo "[run_task] judge wire -> api.openai.com (JUDGE_MODE=openai, JUDGE_MODEL=${JUDGE_MODEL:-<wire default>})"
      ;;
    kiro)
      ensure_kiro_bridge
      # ensure_kiro_bridge populates KIRO_VERIFIER_ARGS and exports
      # OPENHANDS_LLM_*. The OPENHANDS_LLM_* exports would compete with the
      # Bedrock agent env; clear them so the agent stays on Bedrock.
      unset OPENHANDS_LLM_BASE_URL OPENHANDS_LLM_API_KEY
      BEDROCK_VERIFIER_ARGS=("${KIRO_VERIFIER_ARGS[@]}")
      KIRO_VERIFIER_ARGS=()
      echo "[run_task] judge wire -> kiro_bridge (JUDGE_MODE=kiro, KIRO_JUDGE_MODEL=${KIRO_JUDGE_MODEL:-<KIRO_MODEL fallback>})"
      ;;
    *)
      echo "[run_task] JUDGE_MODE ERROR: unknown value '$mode' (expected: bedrock | anthropic | openai | kiro)" >&2
      exit 4
      ;;
  esac
}

# ------------------------------------------------------------------ ccbridge
# The OpenHands agent's route to Claude: tools/bridges/ccbridge, an
# Anthropic-compatible proxy on the host that signs each request with the Claude
# Code OAuth login on this machine (CLAUDE_CODE_OAUTH_TOKEN, the Keychain on
# macOS, ~/.claude/.credentials.json on Linux) and forwards it to
# api.anthropic.com. Not to be confused with the cbridge on :4000 above, which
# serves run_eval.py over the claude-agent-sdk.
#
# PER RUN, by default. Each harbor stage starts its own bridge on a free port
# with a fresh secret that is never written to disk, and stops it when the
# stage exits -- normally, on error, or on Ctrl-C (the EXIT trap at dispatch).
# A bridge that outlives its run is how a machine ends up serving old code,
# holding a port another checkout wants, or refusing a secret it was never
# given; a per-run bridge has none of those states to get into.
#
# CCBRIDGE_SHARED=1 keeps the older arrangement: one long-lived bridge on
# CCBRIDGE_PORT (8765) shared by every run, with its secret in
# tools/bridges/ccbridge/.bridge_secret, stopped only by scripts/stop_harness.sh.
CCBRIDGE_DIR="$REPO/tools/bridges/ccbridge"
CCBRIDGE_RUN_PID=""      # the per-run bridge this invocation started, if any
CCBRIDGE_RUN_PORT=""

# Where the container's traffic arrives. Docker Desktop and OrbStack deliver
# host.docker.internal to the host's loopback; plain Linux docker delivers it
# to the bridge gateway, where a 127.0.0.1 listener is unreachable. The
# secret gates the wider bind.
#
# Behind harbor's egress sidecar the loopback delivery does not hold either:
# measured on harbor 0.24.0 with Docker Desktop, a 127.0.0.1 listener timed out
# from main while a 0.0.0.0 one answered. So a trial the sidecar governs gets
# the wider, secret-gated bind on every OS.
ccbridge_host() {
  if [ -n "${CCBRIDGE_HOST:-}" ]; then echo "$CCBRIDGE_HOST"; return; fi
  case "$(uname -s)" in
    Darwin) [ "$(harbor_egress_control)" = "off" ] && echo "127.0.0.1" || echo "0.0.0.0" ;;
    *) echo "0.0.0.0" ;;
  esac
}

# The shared bridge's secret: CCBRIDGE_SECRET if set, else a per-machine file
# created on first use (0600, gitignored). Never printed. Shared mode only.
ccbridge_secret() {
  if [ -n "${CCBRIDGE_SECRET:-}" ]; then
    printf '%s' "$CCBRIDGE_SECRET"
    return 0
  fi
  local f="$CCBRIDGE_DIR/.bridge_secret"
  if [ ! -s "$f" ]; then
    ( umask 077 && python3 -c 'import secrets; print("ccb-" + secrets.token_hex(24))' > "$f" ) || return 1
  fi
  tr -d '[:space:]' < "$f"
}

# Which bridge code a SHARED bridge was started from. It outlives every run, so
# after the harness is updated it keeps serving the old code with nothing to
# say so. Recorded at start, compared on every later run. (A per-run bridge is
# always this checkout's code.)
ccbridge_source_hash() {
  python3 -c 'import glob,hashlib,sys
h = hashlib.sha256()
for f in sorted(glob.glob(sys.argv[1] + "/claude_oauth/*.py")):
    h.update(open(f, "rb").read())
print(h.hexdigest()[:16])' "$CCBRIDGE_DIR" 2>/dev/null
}

# The bridge's venv, and proof that this machine has a usable Claude login --
# before anything is launched. A bridge that cannot find one exits at once, and
# a wait loop would then poll a port nothing will ever open and report "did not
# come up" instead of the reason. --check loads (and if needed refreshes) the
# token and exits; it makes no model call, so preflight can afford it too.
ccbridge_prepare() {
  if [ ! -f "$CCBRIDGE_DIR/pyproject.toml" ]; then
    echo "[run_task] ERROR: ccbridge not found at $CCBRIDGE_DIR" >&2
    exit 3
  fi
  command -v uv >/dev/null 2>&1 || {
    echo "[run_task] ERROR: uv is not installed; the ccbridge runs from its own uv project" >&2
    exit 3
  }
  mkdir -p "$CCBRIDGE_DIR/logs"
  # The first sync on a machine builds the venv -- minutes when cold. A venv
  # copied from another machine is rebuilt here rather than half-working.
  # Pin UV_PYTHON=3.12 (pyproject requires >=3.12 but a 3.13/3.14 host without
  # 3.12 installed makes uv silently fail the sync). Clear UV_PROJECT_ENVIRONMENT
  # so an operator-set venv path from an outer uv run does not redirect the
  # sync away from CCBRIDGE_DIR/.venv.
  (cd "$CCBRIDGE_DIR" && unset UV_PROJECT_ENVIRONMENT && \
      UV_PYTHON=3.12 uv sync --frozen --quiet) || {
    echo "[run_task] ERROR: could not build the ccbridge venv (uv sync in $CCBRIDGE_DIR)" >&2
    exit 3
  }
  local check_out
  if ! check_out="$(cd "$CCBRIDGE_DIR" && CCBRIDGE_SECRET="${1:-check}" \
                    .venv/bin/python -m claude_oauth --check 2>&1)"; then
    echo "[run_task] ERROR: the ccbridge cannot load a Claude login on this machine:" >&2
    printf '%s\n' "$check_out" | grep -v "UNAUTHENTICATED" | sed 's/^/[run_task]   /' >&2
    echo "[run_task]   Fix: run \`claude login\` here, or put CLAUDE_CODE_OAUTH_TOKEN in .env." >&2
    exit 4
  fi
}

# Launch one bridge in the background from this checkout's venv; echoes its pid.
# exec, so the pid IS the python process: signalling a `uv run` wrapper does not
# reach the server it spawned.
ccbridge_launch() {  # ccbridge_launch <host> <port> <secret> <log>
  (cd "$CCBRIDGE_DIR" && \
    CCBRIDGE_SECRET="$3" \
    CCBRIDGE_READ_TIMEOUT="${CCBRIDGE_READ_TIMEOUT:-1800}" \
    CCBRIDGE_REQUEST_TIMEOUT="${CCBRIDGE_REQUEST_TIMEOUT:-1800}" \
    exec .venv/bin/python -m claude_oauth --host "$1" --port "$2" >>"$4" 2>&1) &
  echo $!
}

# Timeouts, in both modes: the OpenHands SDK calls without streaming, so a long
# thinking turn is one silent wait. The bridge's own defaults (180s between
# bytes, 600s per request) would cut it; the agent allows 1800s (agent.py
# llm_timeout), so the bridge does too.

start_run_ccbridge() {
  local host port secret log pid i=0 wait_s="${CCBRIDGE_START_TIMEOUT_SEC:-180}"
  host="$(ccbridge_host)"
  port="$(python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()')"
  secret="$(python3 -c 'import secrets; print("ccb-run-" + secrets.token_hex(24))')"
  ccbridge_prepare "$secret"
  log="$CCBRIDGE_DIR/logs/run-${JOB//[^A-Za-z0-9._-]/_}-$$.log"
  pid="$(ccbridge_launch "$host" "$port" "$secret" "$log")"
  CCBRIDGE_RUN_PID="$pid"
  CCBRIDGE_RUN_PORT="$port"
  echo "[run_task] ccbridge for this run starting on $host:$port (pid $pid)"
  while [ $i -lt "$wait_s" ]; do
    curl -sf -m 1 "http://127.0.0.1:$port/healthz" >/dev/null 2>&1 && break
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "[run_task] ERROR: the ccbridge exited during startup. Last lines of $log:" >&2
      tail -5 "$log" 2>/dev/null | sed 's/^/[run_task]   /' >&2
      exit 3
    fi
    sleep 1; i=$((i+1))
  done
  curl -sf -m 1 "http://127.0.0.1:$port/healthz" >/dev/null 2>&1 || {
    echo "[run_task] ERROR: the ccbridge did not come up in ${wait_s}s on :$port; see $log" >&2
    exit 3
  }
  export CCBRIDGE_PORT="$port"
  ccbridge_live_check "$port" "$secret" || exit 4
  export OPENHANDS_LLM_BASE_URL="http://host.docker.internal:$port"
  export OPENHANDS_LLM_API_KEY="$secret"
  echo "[run_task] openhands agent routed through this run's ccbridge ($OPENHANDS_LLM_BASE_URL)"
}

# The EXIT trap. Stops only what this invocation started; safe to call when it
# started nothing, and safe to call twice.
stop_run_ccbridge() {
  if [ -n "${CCBRIDGE_RUN_PID:-}" ]; then
    if kill -0 "$CCBRIDGE_RUN_PID" 2>/dev/null; then
      kill "$CCBRIDGE_RUN_PID" 2>/dev/null || true
      local i=0
      while [ $i -lt 20 ] && kill -0 "$CCBRIDGE_RUN_PID" 2>/dev/null; do
        sleep 0.25; i=$((i+1))
      done
      kill -9 "$CCBRIDGE_RUN_PID" 2>/dev/null || true
      echo "[run_task] ccbridge for this run stopped (:$CCBRIDGE_RUN_PORT)" >&2
    fi
    CCBRIDGE_RUN_PID=""
  fi
  return 0
}

ensure_shared_ccbridge() {
  local port="${CCBRIDGE_PORT:-8765}" host secret
  case "$port" in
    ''|*[!0-9]*) echo "[run_task] ERROR: CCBRIDGE_PORT must be a number, got '$port'" >&2; exit 2 ;;
  esac
  host="$(ccbridge_host)"
  secret="$(ccbridge_secret)" || {
    echo "[run_task] ERROR: could not create the ccbridge secret in $CCBRIDGE_DIR" >&2
    exit 4
  }

  local running_src="$CCBRIDGE_DIR/logs/.running-source"
  if curl -sf -m 2 "http://127.0.0.1:$port/healthz" >/dev/null 2>&1; then
    echo "[run_task] shared ccbridge already running on :$port"
    local was now
    was="$(cat "$running_src" 2>/dev/null || true)"
    now="$(ccbridge_source_hash)"
    if [ -z "$was" ]; then
      echo "[run_task] NOTE: the ccbridge on :$port was not started by run_task.sh, so" >&2
      echo "[run_task]   which bridge code it runs is unknown. If in doubt, restart it:" >&2
      echo "[run_task]   bash scripts/stop_harness.sh   (the next run starts a fresh one)" >&2
    elif [ -n "$now" ] && [ "$was" != "$now" ]; then
      echo "[run_task] WARNING: the ccbridge on :$port runs OLDER bridge code than this" >&2
      echo "[run_task]   checkout (started from $was, now $now). It keeps serving the old" >&2
      echo "[run_task]   code until restarted: bash scripts/stop_harness.sh" >&2
    fi
  else
    ccbridge_prepare "$secret"
    echo "[run_task] starting the shared ccbridge on $host:$port"
    # Deliberately detached: it outlives this run. nohup, so a closed terminal
    # does not take it down.
    local pid i=0 wait_s="${CCBRIDGE_START_TIMEOUT_SEC:-180}"
    pid="$(ccbridge_launch "$host" "$port" "$secret" "$CCBRIDGE_DIR/logs/ccbridge.log")"
    while [ $i -lt "$wait_s" ]; do
      curl -sf -m 1 "http://127.0.0.1:$port/healthz" >/dev/null 2>&1 && {
        echo "[run_task] shared ccbridge ready on :$port (pid $pid)"
        ccbridge_source_hash > "$running_src" 2>/dev/null || true
        break
      }
      if ! kill -0 "$pid" 2>/dev/null; then
        echo "[run_task] ERROR: the ccbridge exited during startup; see $CCBRIDGE_DIR/logs/ccbridge.log" >&2
        exit 3
      fi
      sleep 1; i=$((i+1))
    done
    curl -sf -m 1 "http://127.0.0.1:$port/healthz" >/dev/null 2>&1 || {
      echo "[run_task] ERROR: ccbridge did not come up in ${wait_s}s on :$port" >&2
      echo "[run_task]   The agent would fail every turn. See $CCBRIDGE_DIR/logs/ccbridge.log" >&2
      exit 3
    }
  fi
  ccbridge_live_check "$port" "$secret" || exit 4
  export OPENHANDS_LLM_BASE_URL="http://host.docker.internal:$port"
  export OPENHANDS_LLM_API_KEY="$secret"
  echo "[run_task] openhands agent routed through the shared ccbridge ($OPENHANDS_LLM_BASE_URL)"
}

# Where the openhands agent's model calls go, for this run. Call it directly,
# not in $(...): it exports, and a per-run bridge must belong to this shell.
#
#   CC_MODE=zbridge     GLM through zbridge (ensure_zbridge has already started
#                       and verified it). zbridge speaks the same Anthropic
#                       protocol as the ccbridge, so the agent is unchanged;
#                       only where it points is.
#   CC_MODE=bedrock     Bedrock, called directly by LiteLLM from inside the
#                       container; no host bridge. The two placeholders exist
#                       only to satisfy the agent class's preflight -- the
#                       runner drops them for bedrock/* models and reads
#                       AWS_BEARER_TOKEN_BEDROCK / AWS_REGION, which the agent
#                       forwards from harbor's env into the container.
#   CCBRIDGE_SHARED=1   the long-lived shared ccbridge.
#   otherwise           a ccbridge of this run's own.
ensure_openhands_model_route() {
  if [ "${CC_MODE:-}" = "zbridge" ]; then
    # run_task.sh starts zbridge with auth off (ensure_zbridge); a zbridge
    # started by hand with a secret checks x-api-key, which is where LiteLLM
    # puts this value. LiteLLM refuses an empty key, hence the placeholder.
    export OPENHANDS_LLM_BASE_URL="http://host.docker.internal:${ZB_PORT:-8766}"
    export OPENHANDS_LLM_API_KEY="${ZB_BRIDGE_SECRET:-zbridge}"
    echo "[run_task] openhands agent routed through zbridge ($OPENHANDS_LLM_BASE_URL, model $MODEL)"
  elif [ "${CC_MODE:-}" = "bedrock" ]; then
    # Keep the user-facing alias (e.g. "claude-opus-5") as the model string so
    # trajectories, pass_summary.json and finance records are keyed on a short
    # readable name rather than the full inference-profile ARN. The alias is
    # resolved to the ARN inside the container at the LiteLLM call boundary
    # (runner.py:pin_bedrock_alias_resolution). The bedrock/ prefix is still
    # needed so the agent class's is_bedrock check (model.startswith) fires
    # and so LiteLLM routes to the Bedrock provider.
    case "$MODEL" in
      bedrock/*|arn:aws:bedrock:*) ;;
      *) MODEL="bedrock/$MODEL" ;;
    esac
    export OPENHANDS_LLM_BASE_URL="bedrock-direct"
    export OPENHANDS_LLM_API_KEY="bedrock-direct"
    export AWS_BEARER_TOKEN_BEDROCK="${AWS_BEARER_TOKEN_BEDROCK:-}"
    export AWS_REGION="${BEDROCK_AWS_REGION:-${AWS_REGION:-${AWS_DEFAULT_REGION:-}}}"
    [ -n "${AWS_ACCESS_KEY_ID:-}" ] && export AWS_ACCESS_KEY_ID
    [ -n "${AWS_SECRET_ACCESS_KEY:-}" ] && export AWS_SECRET_ACCESS_KEY
    [ -n "${AWS_SESSION_TOKEN:-}" ] && export AWS_SESSION_TOKEN
    echo "[run_task] openhands agent routed to Bedrock directly ($MODEL in $AWS_REGION)"
  elif [ "${CC_MODE:-}" = "kiro" ]; then
    # Kiro bridge: a per-run Anthropic-compatible proxy backed by kiro-cli.
    # Keep MODEL bare (e.g. "claude-opus-5"); litellm_model() in agent.py
    # prepends "anthropic/" for the LiteLLM provider route, and LiteLLM sends
    # the Anthropic Messages request to the bridge at OPENHANDS_LLM_BASE_URL.
    # The bridge reads the bare model name from the request body and forwards
    # it to kiro-cli. A "kiro/" prefix would make LiteLLM 400 with
    # "LLM Provider NOT provided" since it does not recognize kiro as a provider.
    case "$MODEL" in
      kiro/*) MODEL="${MODEL#kiro/}" ;;
    esac
    ensure_kiro_bridge
  elif [ "${CCBRIDGE_SHARED:-0}" = "1" ]; then
    ensure_shared_ccbridge
  else
    start_run_ccbridge
  fi
}

# /healthz answers ok for any caller, so it proves neither that this harness's
# secret is the one the bridge holds nor that the subscription will serve the
# model. Two checks: /healthz WITH the secret (free -- the bridge loads its
# token and says so only to an authorised caller), then one minimal completion
# on the run's model. Only a definite auth/permission answer refuses the run.
ccbridge_live_check() {
  local port="$1" secret="$2" verdict
  verdict="$(CCB_SECRET="$secret" CCB_MODEL="$MODEL" python3 - "$port" <<'PYEOF' 2>/dev/null || true
import json, os, sys, urllib.error, urllib.request
base = f"http://127.0.0.1:{sys.argv[1]}"
secret, model = os.environ["CCB_SECRET"], os.environ["CCB_MODEL"]
try:
    req = urllib.request.Request(f"{base}/healthz", headers={"x-api-key": secret})
    with urllib.request.urlopen(req, timeout=30) as r:
        health = json.loads(r.read() or b"{}")
except urllib.error.HTTPError as exc:
    print(f"creds {exc.code} {(exc.read() or b'')[:160].decode('utf-8', 'replace')}")
    sys.exit(0)
except Exception as exc:
    print(f"unverified {exc}"); sys.exit(0)
if "token_prefix" not in health:
    print("secret the bridge on this port does not accept this harness's secret"); sys.exit(0)
body = json.dumps({"model": model, "max_tokens": 1,
                   "messages": [{"role": "user", "content": "ping"}]}).encode()
req = urllib.request.Request(f"{base}/v1/messages", data=body,
                             headers={"content-type": "application/json", "x-api-key": secret})
try:
    with urllib.request.urlopen(req, timeout=120) as r:
        print("ok" if r.status == 200 else f"http {r.status}")
except urllib.error.HTTPError as exc:
    detail = (exc.read() or b"")[:200].decode("utf-8", "replace").replace("\n", " ")
    print(f"{'auth' if exc.code in (401, 403) else 'http'} {exc.code} {detail}")
except Exception as exc:
    print(f"unverified {exc}")
PYEOF
)"
  case "$verdict" in
    ok)
      echo "[run_task] ccbridge: Claude subscription verified for $MODEL" ;;
    secret*)
      echo "[run_task] ERROR: the bridge on :$port does not accept this harness's secret." >&2
      echo "[run_task]   It was started with another one. Stop it (bash scripts/stop_harness.sh)" >&2
      echo "[run_task]   or set CCBRIDGE_SECRET to the secret it was started with." >&2
      return 1 ;;
    creds*|auth*)
      echo "[run_task] ERROR: the Claude subscription behind the ccbridge refused -- $verdict" >&2
      echo "[run_task]   Every agent turn would fail auth. Run: claude login" >&2
      return 1 ;;
    *)
      echo "[run_task] WARNING: could not verify the ccbridge (${verdict:-no answer}); continuing" >&2 ;;
  esac
  return 0
}

# Point the agent at the Headroom container and tell it which upstream to
# forward to. Call it directly, not in $(...), or the exports are lost.
#
#   Claude:  main -> headroom -> api.anthropic.com
#   GLM:     main -> headroom -> zbridge on the host -> z.ai
#
# The headroom container joins the bundle's compose project, so Harbor's egress
# sidecar governs its traffic exactly as it governs main's: agent_route_hosts
# adds `headroom` (and the host bridge, on GLM) to the agent-phase allowlist.
route_agent_through_headroom() {
  [ "${AGENT_HEADROOM_ENABLED:-false}" = "true" ] || return 0

  if [ "${CC_MODE:-}" = "zbridge" ]; then
    export HEADROOM_UPSTREAM="http://host.docker.internal:${ZB_PORT:-8766}"
  else
    export HEADROOM_UPSTREAM="https://api.anthropic.com"
  fi

  # Built in stage_preflight. Warned about rather than built here: a bare
  # `--stage harbor` on a machine without the image otherwise dies inside
  # compose with "pull access denied", forty lines from the cause.
  if docker info >/dev/null 2>&1 && ! docker image inspect "$HEADROOM_IMAGE" >/dev/null 2>&1; then
    echo "[run_task] ERROR: $HEADROOM_IMAGE is not built; compose up would fail deep inside harbor." >&2
    echo "[run_task]   Build it with: make build-headroom-compress" >&2
    echo "[run_task]   Or run without AGENT_HEADROOM_ENABLED." >&2
    exit 3
  fi

  # What harbor forwards into main (claude_code.py reads it from this
  # environment). Setting it also makes harbor pin every model alias.
  export ANTHROPIC_BASE_URL="http://headroom:$HEADROOM_SERVICE_PORT"
  echo "[run_task] agent-path Headroom ON: main -> headroom -> $HEADROOM_UPSTREAM"
  echo "[run_task]   NOTE: setting ANTHROPIC_BASE_URL makes harbor pin every model"
  echo "[run_task]   alias (sonnet/opus/haiku/subagent) to $MODEL."
}

# The hosts this run's own model route needs during agent.run(), one per line:
# the bridge on the host (ccbridge for openhands, zbridge for GLM), the Bedrock
# endpoint for CC_MODE=bedrock (the agent calls bedrock-runtime directly from
# the container, no host bridge), and the headroom container when it fronts the
# model. stage_harbor passes each to Harbor as --allow-agent-host and
# stage_preflight checks the merged policy.
agent_route_hosts() {
  if [ "${CC_MODE:-}" = "bedrock" ]; then
    local _region="${BEDROCK_AWS_REGION:-${AWS_REGION:-${AWS_DEFAULT_REGION:-us-east-1}}}"
    echo "bedrock-runtime.${_region}.amazonaws.com"
  elif [ "${CC_MODE:-}" = "kiro" ] || [ "$AGENT" = "openhands" ] || [ "${CC_MODE:-}" = "zbridge" ]; then
    echo "host.docker.internal"
  fi
  [ "${AGENT_HEADROOM_ENABLED:-false}" = "true" ] && echo "headroom"
  return 0
}

# "on" when harbor's docker egress sidecar will govern main for this task, "off"
# when every agent-environment policy is public; nothing when harbor cannot be
# imported, which callers treat as on (the side that adds no extra_hosts).
harbor_egress_control() {
  local _hpy
  _hpy="$(harbor_python)" || return 0
  "$_hpy" "$REPO/tools/network/preflight_network.py" "$TASK" --egress-control 2>/dev/null || true
}

# The agent-phase network_mode Harbor will enforce for this task with this
# run's --allow-agent-host hosts merged in: public, allowlist or no-network.
# Resolved by Harbor's own resolver (tools/network/preflight_network.py), so it
# is the policy `harbor run` applies. Prints nothing when harbor cannot be
# imported, which every caller treats as closed: the layers it gates only ever
# take tools away from the agent.
agent_phase_network_mode() {
  local _hpy _host _flags=()
  _hpy="$(harbor_python)" || return 0
  while IFS= read -r _host; do
    [ -n "$_host" ] && _flags+=(--allow-agent-host "$_host")
  done < <(agent_route_hosts)
  "$_hpy" "$REPO/tools/network/preflight_network.py" "$TASK" --agent-mode \
    ${_flags[@]+"${_flags[@]}"} 2>/dev/null || true
}

# Abort the run if the installed harbor will not carry the guard into the
# container. Called from the caller's shell, NOT from inside the $(...) that
# collects egress_guard_settings -- an `exit` there would kill only the subshell
# and the run would walk on unguarded, which is the exact failure being closed.
#
# WHY THIS IS FATAL AND THE OTHER TWO BRANCHES ARE NOT
#
# harbor's agent kwargs have a permissive floor: `--ak config=<path>` on a
# harbor whose ClaudeCode has no `config` parameter falls through to
# BaseAgent.__init__'s **kwargs and is dropped without a word. The claude
# command line then carries no --settings, no hook loads, and this script still
# prints "egress guard ON". A delivered EC2 run of
# 83d7e97e-2aed-4b23-a906-45f5a6a6a4da did exactly that on harbor 0.20: seven
# Bash commands that exit 2 against these rules on the host ran to completion in
# the container, and the session recorded zero PreToolUse events.
#
# So "harbor positively lacks the mechanism" stops the run: it is cheap to
# detect, it is always an install problem (setup.sh pins the version), and the
# alternative is a graded run whose guard was decoration. "Cannot tell" does
# NOT stop it -- the routing table is still the enforcement boundary and a
# probe that cannot read a future harbor must not be able to ground the fleet.
# It says what it could not read and the run continues, unguarded but honest.
#
# EGRESS_GUARD_IGNORE_HARBOR=1 overrides, on the HEADROOM_IGNORE_MEMORY pattern.
require_harbor_agent_config() {
  if [ -n "${EGRESS_GUARD_IGNORE_HARBOR:-}" ]; then
    echo "[run_task] EGRESS_GUARD_IGNORE_HARBOR set; not checking harbor's config support" >&2
    return 0
  fi
  local rc=0
  python3 "$REPO/tools/network/make_guard_settings.py" --check-harbor || rc=$?
  if [ "$rc" -eq 0 ]; then
    HARBOR_DELIVERS_GUARD=yes
    return 0
  fi
  if [ "$rc" -eq 1 ]; then
    echo "[run_task] REFUSING: this harbor drops --ak config=, so the egress guard" >&2
    echo "[run_task]   would not reach the container and the agent would run with no" >&2
    echo "[run_task]   PreToolUse hook while this script reported it ON." >&2
    echo "[run_task]   This machine has harbor $(harbor --version 2>/dev/null || echo '?'); the harness pins ${HARBOR_VERSION}." >&2
    echo "[run_task]   Fix it with:  $(harbor_install_cmd)" >&2
    echo "[run_task]   (the pin lives in scripts/lib/harbor.sh), or set" >&2
    echo "[run_task]   EGRESS_GUARD_IGNORE_HARBOR=1 to run knowingly without the guard." >&2
    exit 2
  fi
  # rc 3, or anything unexpected: the probe already named what it could not
  # read. Not a fault, so not fatal -- but the guard is not claimed either.
  echo "[run_task]   the run continues; the routing table still has no gateway out." >&2
  return 0
}

# Echo the path of a generated Claude Code --settings file whose PreToolUse hook
# refuses egress commands before they run, or nothing if it cannot be built.
#
# WHY A HOOK WHEN HARBOR'S SIDECAR ALREADY BLOCKS
#
# It buys turns, not safety. Under a closed agent phase a `pip install` does not
# fail quickly -- it retries a connection the sidecar refuses until something
# times out, and the model then reasons about the error and tries
# again with a different tool. One recorded run spent seven consecutive Bash
# calls that way (npm, then apt-get, then chromium, then puppeteer) before
# giving up, on a task where every fact it needed was already in the MCP
# sidecars. The hook turns each of those into an immediate refusal that SAYS
# what to use instead, so the model re-plans in one turn.
#
# WHY THE SCRIPT IS INLINED RATHER THAN MOUNTED
#
# tools/network/egress_rules.py is base64'd into the hook command itself. No
# bind mount to add to every bundle's compose file, no second copy to drift, and
# nothing on disk for an agent running as root to overwrite -- the rules the
# hook enforces are byte-identical to the ones detect_internet_use.py imports
# afterwards, because they are the same file.
#
# Gated on the task closing its agent phase, exactly like DISALLOWED_TOOLS
# above: a task that declares an open agent phase must get one.
egress_guard_settings() {
  local guard="$REPO/tools/network/egress_rules.py"
  if [ ! -f "$guard" ]; then
    echo "[run_task] egress guard missing at $guard -- agent runs without it" >&2
    return 0
  fi

  local out_dir="$OUTPUT_DIR/.egress-guard"
  mkdir -p "$out_dir" 2>/dev/null || return 0
  local out="$out_dir/claude-settings.json"

  # Regenerated every run, never cached: a stale settings.json would enforce
  # whatever the rules were the last time someone looked.
  if ! python3 "$REPO/tools/network/make_guard_settings.py" "$guard" "$out" >&2; then
    echo "[run_task] could not build the egress guard -- agent runs without it" >&2
    return 0
  fi
  # Two wordings, because "ON" is a claim about what will happen in the
  # container and only one of these paths has evidence for it. Saying ON either
  # way is how the EC2 run came to report a guard it did not have.
  if [ "${HARBOR_DELIVERS_GUARD:-unknown}" = "yes" ]; then
    echo "[run_task] egress guard ON -- Bash egress is refused with an explanation" >&2
  else
    echo "[run_task] egress guard PASSED TO HARBOR, delivery unconfirmed -- see above" >&2
  fi
  echo "$out"
}

# --- internet audit -----------------------------------------------------------
# These bundles are closed-world: every fact the agent needs is served by the
# light-servers sidecars or sits under the read-only /workspace/data mount. A run
# that answered from the open web did not solve the task -- but it grades exactly
# as though it had, because the reward is computed from the same tool calls
# either way and no channel looks at where a fact came from.
#
# The block is Harbor's: the bundle declares its agent-phase network_mode and
# Harbor's egress sidecar enforces it. The resolved mode is handed to the
# auditor as --enforced-policy, which is what lets an attempt under a closed
# policy read as blocked rather than unwitnessed.
#
# This audit stays anyway, as a backstop. Prevention can regress silently -- a
# bundle left on a public agent phase, an allowlist widened to unblock a run --
# and the trajectory is the one place that shows what the model actually
# reached. It costs nothing on a clean run.
#
# Runs after reshape on purpose. harbor_to_output.py synthesizes agent/trajectory.json
# from the raw stream when Harbor did not publish one (harbor_to_output.py:744),
# so this is the first point where every run is guaranteed to have one to audit.
#
#   INTERNET_AUDIT_OFF=1     skip entirely
#   INTERNET_AUDIT_WARN=1    report findings but never block
#   INTERNET_AUDIT_STRICT=1  also block a run that only ATTEMPTED egress
#
# Default policy: only a run that actually REACHED the internet withholds
# delivery. An attempt the egress guard refused is the block working, and the
# audit says so in those words rather than crediting the network for it.
stage_netaudit() {
  [ -z "${INTERNET_AUDIT_OFF:-}" ] || return 0

  local flags=()
  [ -n "${INTERNET_AUDIT_WARN:-}" ] && flags+=(--warn-only)
  # Reaching for the web is disqualifying in itself, rather than only getting
  # there. Off by default: the guard now refuses the command and tells the model
  # what to use instead, and a model that probes once, is refused, and adapts is
  # the system working -- one recorded run did exactly that and went on to
  # finish. Blocking it would discard good runs.
  [ -n "${INTERNET_AUDIT_STRICT:-}" ] && flags+=(--strict)
  # The agent-phase policy Harbor enforced, with this run's model-route hosts
  # merged in exactly as stage_harbor passed them. Unresolvable (no harbor to
  # ask) passes nothing, and an attempt then has no witness -- the safe side.
  local _policy; _policy="$(agent_phase_network_mode)"
  case "$_policy" in
    allowlist|no-network|public) flags+=(--enforced-policy "$_policy") ;;
  esac

  local traj run_dir dirty=0 seen=0 empty=0
  # Iterate RUN DIRECTORIES, not trajectory files.
  #
  # This globbed [Rr]un_*/agent/trajectory.json and skipped anything that did not
  # match. Measured on an 11-run job: 3 runs carried a trajectory and were
  # audited, 8 did not and were passed over in silence -- and `seen` only warns
  # when it reaches ZERO, so the job reported three clean audits and said nothing
  # about the other eight. An unaudited run read exactly like a clean one.
  #
  # `[Rr]un_*` because the reshaper writes run_N while older stashed trees carry
  # Run_N; auditing only one casing would skip half a resumed task in silence.
  for run_dir in "$TRAJ_DIR"/[Rr]un_*; do
    [ -d "$run_dir" ] || continue
    traj="$run_dir/agent/trajectory.json"

    # A run with no trajectory is an aborted trial that produced nothing --
    # counted and reported below, not audited, because there is nothing to read.
    if [ ! -f "$traj" ]; then
      empty=$((empty+1))
      continue
    fi
    seen=$((seen+1))

    # ${flags[@]+...} is load-bearing under `set -u`: bash 3.2 on macOS treats a
    # bare "${flags[@]}" on an empty array as unbound and kills the script.
    python3 "$REPO/tools/network/detect_internet_use.py" "$traj" \
      --json "$run_dir/internet_audit.json" \
      ${flags[@]+"${flags[@]}"} || dirty=1
  done

  # Nothing auditable at all.
  if [ "$seen" -eq 0 ]; then
    echo "[run_task] internet audit: nothing auditable under $TRAJ_DIR" >&2
    return 0
  fi

  # One tally pass over the per-run JSON. Each run already printed its own
  # verdict line; this only reports what a per-run line cannot say -- how the
  # runs add up, and what happens to the delivery as a result.
  #
  # The vocabulary is detect_internet_use.py's: reached_internet (it got out),
  # attempt_blocked (it tried, nothing left), no_attempt, no_agent_activity,
  # setup_traffic. Only the first is a failure; see _exit_code() there for why.
  local _reached=0 _unwitnessed=0 _attempted=0 _norun=0 _skipped="$empty" _aj _v
  for _aj in "$TRAJ_DIR"/[Rr]un_*/internet_audit.json; do
    [ -f "$_aj" ] || continue
    _v="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("verdict",""))' "$_aj" 2>/dev/null)"
    case "$_v" in
      reached_internet)   _reached=$((_reached+1)) ;;
      attempt_unverified) _unwitnessed=$((_unwitnessed+1)) ;;
      attempt_blocked|setup_traffic) _attempted=$((_attempted+1)) ;;
      no_agent_activity) _norun=$((_norun+1)) ;;
    esac
  done

  [ "$_skipped" -gt 0 ] && \
    echo "[run_task] internet audit: $_skipped run(s) had no trajectory -- not audited, NOT clean" >&2

  # Not an internet finding, so it never blocks -- but pass@k counts these as
  # attempts, so a mean reward over the job is diluted by runs that never ran.
  [ "$_norun" -gt 0 ] && \
    echo "[run_task] internet audit: $_norun run(s) made no tool calls (API error or rate limit); still counted in pass@k" >&2

  [ "$_attempted" -gt 0 ] && [ "$_reached" -eq 0 ] && \
    echo "[run_task] internet audit: $_attempted run(s) reached for the web and were refused; nothing left the sandbox" >&2

  [ "$dirty" -eq 0 ] && return 0

  # A real breach. Withdraw the delivered copy before saying it is withheld:
  # harbor_to_output.py writes delivery_output/ at the end of the reshape, which
  # is BEFORE this stage runs, so the old "Not delivering this run" was printed
  # over a directory that had already been written.
  local _delivered="$(dirname "$OUTPUT_DIR")/delivery_output/$OUT_SLUG"
  local _withdrawn=""
  if [ -d "$_delivered" ]; then
    rm -rf "$_delivered" && _withdrawn=" (delivered copy withdrawn)"
  fi

  echo >&2
  if [ "$_reached" -gt 0 ]; then
    echo "==> DELIVERY WITHHELD: the model reached the open internet$_withdrawn" >&2
    echo "    $_reached of $seen audited run(s) got traffic out of the sandbox." >&2
  else
    echo "==> DELIVERY WITHHELD: egress with no witness$_withdrawn" >&2
    echo "    $_unwitnessed of $seen audited run(s) reached for the web with no" >&2
    echo "    enforced closed agent policy, so nothing shows the attempts failed." >&2
  fi
  echo "    This task is closed-world: the answer comes from the MCP sidecars" >&2
  echo "    and /workspace/data. Per-run detail: <run>/internet_audit.json" >&2
  echo "    Inspect without blocking: INTERNET_AUDIT_WARN=1 scripts/run_task.sh ..." >&2
  echo >&2
  exit 2
}


stage_reshape() {
  local offset; offset="$(state_get run_offset)"
  [ -n "$offset" ] || offset="$(resolve_run_offset)"

  # Restore stash BEFORE reshape so harbor_to_output.py reads the accumulated
  # summary.json and prior Run_N dirs. Harbor may have cleared the job dir,
  # which wipes summary.json; without it the reshaper sees n=1 on every run
  # and the pass@k counts never grow past the latest run.
  local stash; stash="$(state_get stash_dir)"
  [ -n "$stash" ] || stash="$STASH_DIR"
  if [ -d "$stash" ]; then
    mkdir -p "$TRAJ_DIR"
    for run_dir in "$stash"/run_*; do
      [ -d "$run_dir" ] || continue
      run_name="$(basename "$run_dir")"
      [ -d "$TRAJ_DIR/$run_name" ] || cp -r "$run_dir" "$TRAJ_DIR/$run_name"
    done
    local _job_out="$OUTPUT_DIR/$JOB"
    if [ -f "$stash/.summary.json" ]      && [ ! -f "$_job_out/summary.json" ];      then cp "$stash/.summary.json"      "$_job_out/summary.json";      fi
    if [ -f "$stash/.pass_summary.json" ] && [ ! -f "$_job_out/pass_summary.json" ]; then cp "$stash/.pass_summary.json" "$_job_out/pass_summary.json"; fi
    rm -rf "$stash"
  fi

  local conv=(python3 tools/delivery_utils/harbor_to_output.py "$OUTPUT_DIR/$JOB" \
              --output-dir "$OUTPUT_DIR" --at "$AT" --run-offset "$offset")
  # Convert only the trials stage_harbor just made. Without this the reshaper
  # adopts every stale trial dir in the job dir as an extra run (see the
  # snapshot in stage_harbor). Absent key -> convert everything, which is what a
  # hand-driven `--stage reshape` over a job dir with no state expects; present
  # but EMPTY -> convert nothing, because harbor made nothing this time. Those
  # two must not collapse into one value: treating "harbor made nothing" as
  # "convert everything" is exactly the extra-runs bug.
  if state_has trials; then
    local only; only="$(state_get trials)"
    conv+=(--only-trials "$only")
    [ -n "$only" ] || echo "[run_task] WARNING: harbor produced no trial dir this run — nothing to reshape." >&2
  fi
  [ -n "${COPY_TO:-}" ] && conv+=(--copy-to "$COPY_TO")
  "${conv[@]}"
  if [ -f "$KIRO_CALLS_LOG" ]; then
    python3 tools/finance/kiro_judge_tokens.py --calls "$KIRO_CALLS_LOG" "$TRAJ_DIR"/run_* \
      || echo "[run_task] WARNING: judge token backfill failed (non-fatal)" >&2
  fi
  # Relative to the job dir this state file sits in — keeps host-local paths
  # out of everything the pipeline writes (informational only, never read back).
  state_put run_dir "trajectory/run_$((offset+1))"

  # After conversion, before anything is delivered or reported.
  stage_netaudit
  stage_mask
}

# Re-run make_delivery.py's host-path mask over the finished tree.
#
# harbor_to_output.py already masks what it writes, but two things land after
# it: stage_netaudit's internet_audit.json (called just above) and, one stage
# later, finance_receipt.json. Both would ship whatever absolute path they were
# handed. The sweep is idempotent and walks one task's output dir, so calling
# it at the end of reshape AND at the end of finance costs nothing and leaves
# no window where a host path is the last thing written.
stage_mask() {
  local dir="$OUTPUT_DIR/$OUT_SLUG"
  [ -d "$dir" ] || dir="$OUTPUT_DIR/$JOB"
  [ -d "$dir" ] || { echo "[mask] no output dir to mask" >&2; return 0; }
  # Never fatal: a delivered tree with a stray path in it beats a run marked
  # failed after the agent phase already spent its money.
  python3 tools/delivery_utils/make_delivery.py --mask-only "$dir" \
    || echo "[mask] WARNING: path mask failed (non-fatal)" >&2
}

# finance API: report trajectory usage (after the runs are done).
# Skips itself when ODOO_URL is unset; never fails the task run.
# ODOO_URL normally lives in .env, which this script does not source, so read it
# from there when the environment doesn't already provide it.
stage_finance() {
  # The .env may live above the harness when it is vendored into a larger
  # workspace, so search upward the way finance_reporter.py does -- but only
  # when ODOO_URL is UNSET. `ODOO_URL= scripts/run_task.sh ...` is the
  # documented opt-out, and check_finance_env already reports it as "usage
  # reporting disabled"; testing emptiness here instead of presence read the
  # value back out of .env and posted anyway (two test runs reached Odoo that
  # way). Presence, not emptiness, is the rule load_dotenv uses too.
  if [ -z "${ODOO_URL+set}" ]; then
    local d="$REPO"
    while [ -n "$d" ] && [ "$d" != "/" ]; do
      if [ -f "$d/.env" ]; then
        ODOO_URL="$(grep -E '^ODOO_URL=' "$d/.env" | tail -1 | cut -d= -f2- \
                    | sed 's/[[:space:]]*#.*$//' | tr -d '[:space:]' || true)"
        [ -n "$ODOO_URL" ] && break
      fi
      d="$(dirname "$d")"
    done
  fi
  if [ -z "${ODOO_URL:-}" ]; then
    if [ -n "${ODOO_URL+set}" ]; then
      echo "[finance] ODOO_URL set empty -- usage reporting disabled for this run" >&2
    else
      echo "[finance] WARNING: ODOO_URL unset (no .env at or above $REPO) — usage NOT reported" >&2
    fi
    return 0
  fi
  if [ "${FINANCE_ENV_BAD:-0}" = "1" ]; then
    echo "[finance] ERROR: attribution env is invalid (reported at startup) — usage NOT reported" >&2
    stage_mask
    return 0
  fi

  local offset; offset="$(state_get run_offset)"
  [ -n "$offset" ] || offset="$(resolve_run_offset)"
  # Prefer the reshaped tree; fall back to the job dir for bundles whose
  # task.toml name already matches their directory.
  local run_dir="$OUTPUT_DIR/$OUT_SLUG/trajectory/run_$((offset+1))"
  if [ ! -d "$run_dir" ] && [ -d "$OUTPUT_DIR/$JOB/trajectory/run_$((offset+1))" ]; then
    run_dir="$OUTPUT_DIR/$JOB/trajectory/run_$((offset+1))"
  fi
  python3 tools/finance/finance_reporter.py \
    --run-dir "$run_dir" \
    --skip-if-reported \
    --task-id "$OUT_SLUG" || echo "[finance] WARNING: reporting failed (non-fatal)"
  # The receipt lands after reshape masked the tree.
  stage_mask
}

# ATIF_VERSION (.env, default 1.5): reshape the OpenHands ATIF-v1.8 trajectory.json
# in output/ and delivery_output/ to the v1.5 delivery shape, in place. Runs last,
# after netaudit and finance have read the v1.8 file. ATIF_VERSION=1.8 ships it as is.
stage_atif() {
  local want="${ATIF_VERSION:-1.5}"
  if [ "$want" = "1.8" ]; then
    echo "[atif] ATIF_VERSION=1.8: trajectory.json shipped as the runner wrote it" >&2
    return 0
  fi
  local dirs=("$OUTPUT_DIR/$OUT_SLUG" "$(dirname "$OUTPUT_DIR")/delivery_output/$OUT_SLUG")
  [ -n "${COPY_TO:-}" ] && dirs+=("$COPY_TO/$OUT_SLUG")
  python3 tools/delivery_utils/trajectory_reshaper.py --to "$want" "${dirs[@]}" \
    || echo "[atif] WARNING: trajectory reshape failed (non-fatal)" >&2
}

# --- dispatch -----------------------------------------------------------------

# Only the stages that actually drive harbor. reshape/finance/mask move files
# and must not die because harbor drifted or is not installed.
case "$STAGE" in
  preflight|harbor|all) python3 "$REPO/scripts/patch_harbor.py" ;;
esac

trap on_interrupt INT
trap on_terminate TERM
# A per-run ccbridge or kiro bridge dies with the run however the run ends:
# on_interrupt and on_terminate call exit, which runs this too.
_stop_run_bridges() { stop_run_ccbridge; stop_kiro_bridge; stop_bedrock_judge_proxy; }
trap _stop_run_bridges EXIT

refuse_uncmountable_paths
resolve_auth
check_credentials
check_finance_env

case "$STAGE" in
  preflight) stage_preflight ;;
  harbor)    stage_harbor ;;
  reshape)   stage_reshape ;;
  finance)   stage_finance; stage_atif ;;
  mask)      stage_mask ;;
  atif)      stage_atif ;;
  all)       stage_preflight; stage_harbor; stage_reshape; stage_finance; stage_atif
             echo "[run_task] done → $OUTPUT_DIR/$OUT_SLUG" ;;
esac
