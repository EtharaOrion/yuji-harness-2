#!/usr/bin/env bash
# Run one mcp-atlas Harbor task end-to-end and emit the per-task output/ tree
# (same layout complex-mcp's --layout harbor writer produces).
#
#   scripts/run_task.sh tasks/xenon-atomic-cube                 # claude-code + opus-5 (defaults)
#   CC_MODE=zbridge N=3 scripts/run_task.sh tasks/foo            # 3 attempts via GLM-5.3 (zbridge)
#   AGENT=oracle scripts/run_task.sh tasks/foo                  # oracle gate
#   COPY_TO=/some/dir scripts/run_task.sh tasks/foo           # optional extra mirror
#
#   scripts/run_task.sh --stage reshape tasks/foo               # one stage only
#
# Env overrides: AGENT (claude-code) MODEL (claude-opus-5) N (1) JOB (<task slug>)
#                OUTPUT_DIR (<repo>/output) COPY_TO (unset) BUILD_MULT (3) AT (auto)
#                STAGE (all) RUN_OFFSET (auto) SETUP_MULT (6) JUDGE_MODEL (gpt-5.6-sol)
#                NETWORK_ISOLATION_OFF (unset) DISALLOWED_TOOLS (WebSearch,WebFetch)
#                CC_MODE (unset -> claude-opus-5; "zbridge" -> glm-5.3 via :8766)
#                CC_BRIDGE_ENABLED (0)
#                AGENT_HEADROOM_ENABLED (false) GRADER_HEADROOM_ENABLED (false)
#
# Values may also come from <repo>/.env, which is read as DEFAULTS only: anything
# already in the environment wins over it. NETWORK_ISOLATION_OFF is the one key
# .env may NOT set (DOTENV_FORBIDDEN_KEYS): it turns the closed-world guarantee
# off, and a file that does that for every future run is not configuration.
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
# unconditionally, so `NETWORK_ISOLATION_OFF= scripts/run_task.sh ...` lost to
# the .env value and no .env key could be overridden for a single run.
# Presence is what is tested (${!k+set}), not emptiness, so an explicitly-empty
# override is honoured rather than refilled from the file.
# Keys .env may not set, however they are spelled in the file.
#
# NETWORK_ISOLATION_OFF is the only member and it is here because it was the
# whole of a real failure. It is a per-RUN decision -- "open the network, this
# once, and I know what that means" -- and it sat in .env instead, so every run
# on the machine was open and nothing said so. A closed-world task then
# installed Pillow, puppeteer and chromium off the public internet and graded as
# though it had not. A file that turns the guarantee off for all future runs is
# not configuration, it is an unattended decision, so the caller has to make it:
#
#   NETWORK_ISOLATION_OFF=1 scripts/run_task.sh tasks/<task>
#
# Named rather than dropped in silence, for the same reason the charset skip
# below is named: configuration that vanishes looks exactly like configuration
# that works.
DOTENV_FORBIDDEN_KEYS=" NETWORK_ISOLATION_OFF "

load_dotenv() {
  [ -f "$REPO/.env" ] || return 0
  local line key val skipped="" refused="" quoted=""
  while IFS= read -r line; do
    key="${line%%=*}"
    val="${line#*=}"
    case "$DOTENV_FORBIDDEN_KEYS" in
      *" $key "*) refused="$refused $key"; continue ;;
    esac
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
  if [ -n "$refused" ]; then
    echo "[run_task] .env: REFUSED (per-run only, not a file setting):$refused" >&2
    echo "[run_task]   pass it on the command line for a single run instead:" >&2
    echo "[run_task]   ${refused# }=1 scripts/run_task.sh <task>" >&2
  fi
  return 0
}
load_dotenv

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

TASK="${TASK:?usage: scripts/run_task.sh [--stage all|preflight|harbor|reshape|finance|mask] <task-dir>}"
[ -f "$TASK/task.toml" ] || { echo "not a task dir (no task.toml): $TASK" >&2; exit 2; }
case "$STAGE" in
  all|preflight|harbor|reshape|finance|mask) ;;
  *) echo "unknown stage: $STAGE (want all|preflight|harbor|reshape|finance|mask)" >&2; exit 2;;
esac
SLUG="$(basename "$TASK")"

AGENT="${AGENT:-claude-code}"
if [ "${CC_MODE:-}" = "zbridge" ]; then
  MODEL="${MODEL:-glm-5.3}"
else
  MODEL="${MODEL:-claude-opus-5}"
fi
N="${N:-1}"
# Pin the rubric grader for the whole run. Left unset, rubric_judge_cli picks a
# model from whichever CLI the machine has, so the host pass grades on
# gpt-5.6-sol while an in-container pass grades on a Claude model -- one
# recorded job has 42 trials on the former and 1 on the latter, with nothing
# marking the odd one out. The judge is a benchmark property; it should not
# depend on where a stage happened to run.
export JUDGE_MODEL="${JUDGE_MODEL:-gpt-5.6-sol}"
BUILD_MULT="${BUILD_MULT:-3}"
SETUP_MULT="${SETUP_MULT:-6}"
# Headroom prompt compression, both paths OFF unless asked for. Each turns on
# one container: AGENT_ the headroom proxy in front of main, GRADER_ a judge
# image carrying the library. Neither changes the egress allowlist.
AGENT_HEADROOM_ENABLED="${AGENT_HEADROOM_ENABLED:-false}"    # agent-path compression: OFF
GRADER_HEADROOM_ENABLED="${GRADER_HEADROOM_ENABLED:-false}"  # grader-path compression: OFF
OUTPUT_DIR="${OUTPUT_DIR:-$REPO/output}"
AT="${AT:-auto}"   # pass@k ks for the reshaper; auto = every k from 1..N runs
JOB="${JOB:-$SLUG}"   # job dir == output/<task>/ (reshaped in place by the converter)

# "summarized" = readable thinking summaries in message text; "" = signature-only blocks with empty content (measured 2026-09-03)
THINKING="${THINKING:-adaptive}"
THINKING_DISPLAY="${THINKING_DISPLAY:-summarized}"

# Web tools withheld from the agent while network isolation is on. Harbor turns
# this into `--disallowedTools` (claude_code.py:84-86), so the tools are absent
# from the tool list rather than present-and-failing.
#
# These two are the whole list on purpose. Bash is NOT here: the tasks need it
# for real local work, and its egress is already dead at the routing layer and
# still audited by detect_internet_use.py afterwards. Denying it would break
# tasks to buy nothing.
#
# Set to empty to pass no --disallowedTools at all.
DISALLOWED_TOOLS="${DISALLOWED_TOOLS-WebSearch,WebFetch}"

# The absolute pin the compose comment has always claimed existed. Without it,
# compose falls through to ${SCORING_DIR:-../../../services/scoring}, which is
# correct only for bundles at exactly the current depth; at any other depth
# Docker creates the missing directory, mounts /harness/scoring EMPTY, and the
# collect hook dies with "can't open file" -- a trial scored 0 with the agent's
# work intact. Refuse instead of mounting a directory that isn't the grader.
SCORING_DIR="${SCORING_DIR:-$REPO/services/scoring}"
if [ ! -f "$SCORING_DIR/collect_artifacts.py" ]; then
  echo "[run_task] SCORING_DIR does not hold the grader (no collect_artifacts.py): $SCORING_DIR" >&2
  exit 2
fi
export SCORING_DIR

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
    # compose project label and can be found again by it, but egress-proxy's
    # base image (ubuntu/squid) declares VOLUME /var/log/squid and
    # /var/spool/squid, so each proxy also holds two ANONYMOUS volumes -- 64 hex
    # characters, no labels, four per run counting the judge's proxy. Once their
    # container is gone nothing can attribute them to this run again, and they
    # sit on the disk forever.
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

# Harbor's agent and the in-container judge both need a token. Every stage
# resolves this, not just preflight: stages run as separate processes now, so an
# export in one is gone by the time the next starts, and harbor aborts up front
# on a missing [verifier.env] variable.
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
  echo "[run_task] WARNING: no CLAUDE_CODE_OAUTH_TOKEN / ANTHROPIC_API_KEY — agent + judge will fail auth" >&2
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

# Is there a credential in this auth.json at all? Measured: `codex login status`
# answers "Logged in using ChatGPT" for {"tokens": null} and even for {}, so it
# only really detects a MISSING file. This reads the one fact it misses. Silent
# on anything it cannot parse or does not recognise.
codex_auth_is_empty() {   # -> 0 when the file definitely carries no credential
  [ "$(python3 -c 'import json,sys
try:
    d = json.load(open(sys.argv[1]))
    assert isinstance(d, dict)
except Exception:
    raise SystemExit(0)
tok = d.get("tokens") or {}
if not isinstance(tok, dict):
    raise SystemExit(0)
if not (tok.get("access_token") or d.get("OPENAI_API_KEY")):
    print("empty")' "$1" 2>/dev/null)" = "empty" ]
}

# Ask the CLI, not its credential file. Only an explicit "not logged in" is
# treated as a failure -- any other trouble running it warns and lets the run go.
#
# NOTE what this does and does not prove. It catches a missing or emptied
# credential. It does NOT detect an EXPIRED ChatGPT session: the answer comes
# back instantly, so it is a local read, not a call to openai.
check_codex_login() {   # -> 0 ok/unknown, 1 definitely logged out
  if codex_auth_is_empty "${1:-}"; then
    echo "[run_task] ERROR: $1 carries no codex credential (no tokens, no API key)" >&2
    echo "[run_task]   The rubric judge cannot grade. Run: codex login" >&2
    return 1
  fi
  local out rc
  out="$(codex login status 2>&1)" && rc=0 || rc=$?
  case "$(printf '%s' "$out" | tr 'A-Z' 'a-z')" in
    *"not logged in"*)
      echo "[run_task] ERROR: the codex CLI is not logged in — the rubric judge cannot grade" >&2
      echo "[run_task]   Run: codex login" >&2
      return 1 ;;
  esac
  if [ "$rc" != "0" ]; then
    echo "[run_task] WARNING: 'codex login status' exited $rc; could not confirm the judge login" >&2
    echo "[run_task]   $(printf '%s' "$out" | head -1)" >&2
  fi
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
  else
    if [ -z "${CLAUDE_CODE_OAUTH_TOKEN:-}" ] && [ -z "${ANTHROPIC_API_KEY:-}" ]; then
      echo "[run_task] ERROR: no Claude Code credentials found" >&2
      echo "[run_task]   Log in with: claude login   or set CLAUDE_CODE_OAUTH_TOKEN / ANTHROPIC_API_KEY" >&2
      fail=1
    else
      # The agent's CLI is pre-baked into the task image, so a host binary is
      # only needed if the rubric falls back to a Claude judge. Not a blocker.
      command -v claude >/dev/null 2>&1 \
        || echo "[run_task] WARNING: 'claude' not found in PATH (only matters for a Claude-model rubric fallback)" >&2
      if check_claude_token_expiry; then
        echo "[run_task] claude: credentials OK"
      else
        fail=1
      fi
    fi
  fi

  if ! command -v codex >/dev/null 2>&1; then
    echo "[run_task] ERROR: 'codex' not found — rubric judge cannot use gpt-5.6-sol" >&2
    echo "[run_task]   Install: npm install -g @openai/codex" >&2
    fail=1
  else
    local codex_auth="${CODEX_AUTH_FILE:-${CODEX_HOME:-$HOME/.codex}/auth.json}"
    if [ ! -s "$codex_auth" ]; then
      echo "[run_task] ERROR: codex auth file missing or empty: $codex_auth" >&2
      echo "[run_task]   Log in to the ChatGPT desktop app or run: codex login" >&2
      fail=1
    elif ! check_codex_login "$codex_auth"; then
      fail=1
    else
      echo "[run_task] codex: signed in"
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
    egress-proxy)  echo "$REPO/tools/network/egress-proxy" ;;
    codex-judge)   echo "$REPO/tools/judge" ;;
    codex-judge-headroom) echo "$REPO/tools/judge" ;;
    headroom-compress) echo "$REPO/tools/headroom" ;;
  esac
}

# Extra `docker build` flags, one per line, for images whose Dockerfile needs
# more than its own context. codex-judge bakes in the shared grader from
# services/scoring rather than mounting it, so the running judge holds no host
# directory (tools/judge/Dockerfile).
image_build_args() {
  case "${1%%:*}" in
    codex-judge) printf '%s\n' --build-context "scoring=$REPO/services/scoring" ;;
    codex-judge-headroom)
      printf '%s\n' --build-context "scoring=$REPO/services/scoring" \
                    --build-arg WITH_HEADROOM=1 ;;
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
    "$_hpy" "$REPO/tools/network/preflight_network.py" "$TASK" || {
      echo "[run_task] network policy preflight failed — refusing to build or run" >&2
      exit 2
    }
  fi

  # GLM runs: prove the key works before the image builds, not after. Idempotent
  # -- stage_harbor calls it again and finds the bridge already up.
  [ "${CC_MODE:-}" = "zbridge" ] && ensure_zbridge

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

  # The optional Headroom images, built here and only when this run asks for
  # them. They are declared in overlays rather than in the bundle's compose
  # file, so the loop above never sees them; and they are big (the judge image
  # is ~440 MB larger than the plain one), so building them on every run would
  # be a tax on the runs that do not use them.
  if [ "${AGENT_HEADROOM_ENABLED:-false}" = "true" ]; then
    ensure_image "$HEADROOM_IMAGE"
  fi
  if [ "${GRADER_HEADROOM_ENABLED:-false}" = "true" ] && bundle_has_judge; then
    ensure_image "codex-judge-headroom:latest"
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
  # After ensure_zbridge: on a GLM run both set ANTHROPIC_BASE_URL, and with
  # agent-path Headroom on it is the headroom container that must win, with
  # zbridge behind it as the upstream.
  route_agent_through_headroom
  local args=(run -y --path "$TASK" --agent "$AGENT" --jobs-dir "$OUTPUT_DIR" --job-name "$JOB" \
              --environment-build-timeout-multiplier "$BUILD_MULT" \
              --agent-setup-timeout-multiplier "$SETUP_MULT" --n-attempts "$N")
  # Must come after the proxy helpers above: they decide whether the agent is
  # pointed at host.docker.internal, which network isolation cannot route to.
  local _iso; _iso="$(network_isolation_overlay)"
  [ -n "$_iso" ] && args+=(--extra-docker-compose "$_iso")
  # GLM runs: a second overlay lets squid also reach zbridge on the host.
  if [ -n "$_iso" ] && [ "${CC_MODE:-}" = "zbridge" ]; then
    write_zbridge_squid_conf
    args+=(--extra-docker-compose "$REPO/tools/network/egress-proxy/overlay-zbridge.yaml")
  fi
  # Bundles that grade the rubric in a judge container of their own: a per-run
  # token and the login to mount, and under isolation the judge's own
  # allowlisted way out. Called directly, not in $(...), so the exports reach
  # harbor and compose.
  if bundle_has_judge; then
    prepare_judge
    [ -n "$_iso" ] && args+=(--extra-docker-compose "$REPO/tools/network/egress-proxy/overlay-judge.yaml")
    # Grader-path Headroom: a judge image that carries the library, and the flag
    # the grader reads. Nothing else changes -- the judge's own squid, mounts
    # and token are the same, and compression happens inside the process.
    if [ "${GRADER_HEADROOM_ENABLED:-false}" = "true" ]; then
      args+=(--extra-docker-compose "$REPO/tools/network/egress-proxy/overlay-judge-headroom.yaml")
      echo "[run_task] grader-path Headroom ON: judge runs codex-judge-headroom:latest" >&2
    fi
  fi
  # Last, so main's NO_PROXY here wins over the isolated overlay's.
  while IFS= read -r _hr_overlay; do
    if [ -n "$_hr_overlay" ]; then args+=(--extra-docker-compose "$_hr_overlay"); fi
  done < <(agent_headroom_overlays "$_iso")
  [ "$AGENT" != "oracle" ] && args+=(--model "$MODEL")
  if [ "$AGENT" = "claude-code" ]; then
    [ -n "$THINKING" ] && args+=(--ak "thinking=$THINKING")
    [ -n "$THINKING_DISPLAY" ] && args+=(--ak "thinking_display=$THINKING_DISPLAY")
    # Second layer under the routing block, not a replacement for it. The
    # overlay already makes these tools fail, but failing costs a turn: the
    # model picks WebFetch, waits out a connection error, and reasons about it
    # before trying the sidecars. Denying them means they never appear in the
    # tool list, so that turn is never spent.
    #
    # Deliberately NOT the whole defence. This is agent configuration, and an
    # agent can be run without it (AGENT=oracle, a future adapter, a hand
    # `harbor run`); the network block holds regardless. Both layers stay.
    #
    # Only when the block is on: with NETWORK_ISOLATION_OFF=1 the operator has
    # asked for an open run, and silently keeping the web tools off would make
    # that run mean something different from what they asked for.
    [ -n "$_iso" ] && [ -n "$DISALLOWED_TOOLS" ] \
      && args+=(--ak "disallowed_tools=$DISALLOWED_TOOLS")

    # Third layer, and the only one the model can actually READ. The routing
    # table removes the route and the tool list removes the web tools; neither
    # says anything back when the model reaches for `apt-get`, so it retries.
    # This one refuses the call and explains what to use instead, in the same
    # turn. Same gate as above: an open run stays open.
    if [ -n "$_iso" ]; then
      require_harbor_agent_config
      local _guard; _guard="$(egress_guard_settings)"
      [ -n "$_guard" ] && args+=(--ak "config=$_guard")
    fi
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
  local _hrc=0
  HARBOR_OUTPUT_OFF=1 command harbor "${args[@]}" \
    || { _hrc=$?; echo "[run_task] harbor exited $_hrc; checking whether a trial actually ran" >&2; }

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
    echo "             python3 tools/delivery/harbor_to_output.py $_job_dir \\" >&2
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
  state_put host_rubric_done 0
}

# ------------------------------------------------------------- agent Headroom
# Compress the AGENT's prompts with a Headroom proxy that runs IN THE BUNDLE, on
# the same network as main. OFF unless AGENT_HEADROOM_ENABLED=true.
#
# It used to run on the host, with main pointed at host.docker.internal:8787. An
# internal network has no route there, so an isolated Claude run was refused
# outright and an isolated GLM run silently dropped compression: the feature was
# unavailable on precisely the runs this harness makes. A container on the
# bundle's own network is reachable, and its egress is squid's business like
# everything else's (overlay-headroom-isolated.yaml).
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
# Measured, on an 8 GB docker VM: the project already runs main, light-servers,
# the judge and two squids, and headroom makes seven containers. The VM hit a
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
  echo "[run_task]   runs main, light-servers, the judge and two squids." >&2
  echo "[run_task]   At 8 GB the VM ran out and the kernel killed the proxy mid-run: the" >&2
  echo "[run_task]   agent lost the API and the trial published a reward of 0." >&2
  echo "[run_task]   Raise the memory in OrbStack/Docker Desktop settings, run without" >&2
  echo "[run_task]   AGENT_HEADROOM_ENABLED, or set HEADROOM_IGNORE_MEMORY=1 to proceed." >&2
  exit 2
}

ensure_cc_bridge() {
  # OFF by default. Nothing this script runs reads :4000 -- the agent talks to
  # api.anthropic.com (or to zbridge), and the rubric judge shells out to the
  # codex CLI -- so starting it bought nothing, while costing 180s of dead
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

ensure_zbridge() {
  local port="${ZB_PORT:-8766}"
  # zbridge serves /healthz (bridge.py), NOT /health -- the adapter on :4001 and
  # cc-bridge on :4000 are the ones with /health. Probing /health here 404s on a
  # perfectly healthy bridge, so this guard never fired: every run fell through
  # to the start branch and spawned a second uvicorn, which could not bind the
  # port but did truncate the live bridge's log on its way out (the redirect
  # below is `>`), destroying its startup warnings.
  if curl -sf -m 2 "http://127.0.0.1:$port/healthz" >/dev/null 2>&1; then
    echo "[run_task] zbridge already running on :$port"
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
      ZB_PORT="$port" ZB_HOST="127.0.0.1" \
      ZB_THINKING_SIG_KEY="${ZB_THINKING_SIG_KEY:-yuji-harness-zbridge-v1}" \
      ZB_MODEL_ALIAS_JSON='{"claude-opus-5":"glm-5.3","claude-sonnet-5":"glm-5.3","claude-sonnet-4-6":"glm-5.3","claude-sonnet-4-5":"glm-5.3","claude-opus-4-8":"glm-5.3","claude-opus-4-7":"glm-5.3","claude-haiku-4-5-20251001":"glm-5.3","claude-haiku-4-5":"glm-5.3","claude-3-5-sonnet-latest":"glm-5.3","claude-3-opus-latest":"glm-5.3"}' \
      ZB_UPSTREAM_URL="${ZB_UPSTREAM_URL:-https://api.z.ai/api/coding/paas/v4/chat/completions}" \
      nohup uv run python -m zbridge --port "$port" --host 127.0.0.1 \
        >"$log_dir/zbridge.log" 2>&1 &)
    # Generous, because the first `uv run` on a machine resolves and builds the
    # bridge's venv before anything listens -- minutes, not seconds. The old 15s
    # was under that, so a cold start looked like a broken bridge.
    local i=0 wait_s="${ZB_START_TIMEOUT_SEC:-180}"
    while [ $i -lt "$wait_s" ]; do
      sleep 1; i=$((i+1))
      curl -sf -m 1 "http://127.0.0.1:$port/healthz" >/dev/null 2>&1 && {
        echo "[run_task] zbridge ready on :$port"; break
      }
    done
    # Pointing the agent at a port nothing listens on spends the whole trial on
    # connection errors, so this refuses instead of warning.
    curl -sf -m 1 "http://127.0.0.1:$port/healthz" >/dev/null 2>&1 || {
      echo "[run_task] ERROR: zbridge did not come up in ${wait_s}s on :$port" >&2
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
  local port="${ZB_PORT:-8766}" verdict
  verdict="$(python3 - "$port" <<'PYEOF' 2>/dev/null || true
import json, sys, urllib.error, urllib.request
body = json.dumps({"model": "claude-opus-5", "max_tokens": 1,
                   "messages": [{"role": "user", "content": "ping"}]}).encode()
req = urllib.request.Request(f"http://127.0.0.1:{sys.argv[1]}/v1/messages",
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

# Point the agent at the Headroom container, and tell the pieces around it what
# that implies: which upstream the proxy forwards to, and which squid config
# guards its egress. Call it directly, not in $(...), or the exports are lost.
#
#   Claude:  main -> headroom -> squid -> api.anthropic.com
#   GLM:     main -> headroom -> squid -> zbridge on the host -> z.ai
route_agent_through_headroom() {
  [ "${AGENT_HEADROOM_ENABLED:-false}" = "true" ] || return 0

  if [ "${CC_MODE:-}" = "zbridge" ]; then
    # zbridge runs on the host, so the proxy in front of headroom needs the
    # exemption main used to get. write_zbridge_squid_conf exports the generated
    # conf; stage_harbor calls it again for main's own squid, and it is
    # idempotent.
    write_zbridge_squid_conf
    export HEADROOM_UPSTREAM="http://host.docker.internal:${ZB_PORT:-8766}"
    export HEADROOM_SQUID_CONF="$EGRESS_SQUID_CONF"
  else
    export HEADROOM_UPSTREAM="https://api.anthropic.com"
    export HEADROOM_SQUID_CONF="$REPO/tools/network/egress-proxy/squid.conf"
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
  echo "[run_task]   alias (sonnet/opus/haiku/subagent) to $MODEL -- claude_code.py:1358."
}

# The overlays that add the headroom container to the project, one per line, or
# nothing when the flag is off. The isolated one carries the proxy's own squid;
# both must land after overlay.yaml, which the caller does by appending.
agent_headroom_overlays() {  # agent_headroom_overlays [isolated]
  [ "${AGENT_HEADROOM_ENABLED:-false}" = "true" ] || return 0
  echo "$REPO/tools/network/egress-proxy/overlay-headroom.yaml"
  [ -n "${1:-}" ] && echo "$REPO/tools/network/egress-proxy/overlay-headroom-isolated.yaml"
  return 0
}

# Echo the path of the network-isolation compose overlay, or nothing if the run
# should stay on the open network. Callers append it as --extra-docker-compose,
# which harbor lands AFTER the task's own compose (docker.py:277), so one file
# covers every bundle without editing any of them.
#
# See tools/network/egress-proxy/squid.conf for why the block lives in compose rather
# than in task.toml's network_mode.
network_isolation_overlay() {
  [ -z "${NETWORK_ISOLATION_OFF:-}" ] || { echo "[run_task] network isolation OFF (NETWORK_ISOLATION_OFF set)" >&2; return 0; }

  local overlay="$REPO/tools/network/egress-proxy/overlay.yaml"
  [ -f "$overlay" ] || {
    echo "[run_task] network isolation overlay missing at $overlay" >&2
    echo "[run_task]   refusing to run open-network by accident; set NETWORK_ISOLATION_OFF=1 to allow it" >&2
    exit 2
  }

  # This used to REFUSE a headroom run: the proxy was on the host, and main
  # cannot reach host.docker.internal once its network is internal. The proxy is
  # a container in the project now (overlay-headroom.yaml), so there is nothing
  # left to refuse -- and the refusal was what sent people to
  # NETWORK_ISOLATION_OFF=1 in .env, which turned the block off for every run.

  # stdout is the return channel here, and ensure_image narrates its build to
  # stdout. Without the redirect its progress lines end up inside the overlay
  # path and harbor is handed a -f that does not exist.
  ensure_image "egress-proxy:latest" >&2
  echo "[run_task] network isolation ON -- egress allowlist: api.anthropic.com" >&2
  echo "$overlay"
}

# GLM runs only. Writes squid-zbridge.conf with the real zbridge port and exports
# its path for overlay-zbridge.yaml. Call it directly, not in $(...), or the
# export is lost.
write_zbridge_squid_conf() {
  local port="${ZB_PORT:-8766}" dir="$OUTPUT_DIR/.egress-zbridge"
  case "$port" in
    ''|*[!0-9]*) echo "[run_task] ERROR: ZB_PORT must be a number, got '$port'" >&2; exit 2 ;;
  esac
  mkdir -p "$dir"
  dir="$(cd "$dir" && pwd)"
  sed "s/^acl zbridge_port port 8766\$/acl zbridge_port port $port/" \
    "$REPO/tools/network/egress-proxy/squid-zbridge.conf" > "$dir/squid.conf"
  grep -q "^acl zbridge_port port $port\$" "$dir/squid.conf" || {
    echo "[run_task] ERROR: could not set zbridge port in $dir/squid.conf" >&2
    exit 2
  }
  export EGRESS_SQUID_CONF="$dir/squid.conf"
  echo "[run_task] GLM run: squid also allows zbridge at host.docker.internal:$port" >&2
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
# WHY A HOOK WHEN THE ROUTER ALREADY BLOCKS
#
# It buys turns, not safety. Under isolation a `pip install` does not fail
# quickly -- it opens a connection to a gateway that is not there and sits until
# something times out, and the model then reasons about the error and tries
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
# Gated on isolation being ON, exactly like DISALLOWED_TOOLS above: an operator
# who asked for an open run must get one.
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

# --- judge container ----------------------------------------------------------
# A bundle may grade every channel in a `judge` service of its own
# (tools/judge/codexbridge.py, image codex-judge:latest): Channel A, the state
# dump, the rubric and the ledger, none of them in `main`, the container the
# agent had root in. The codex login is mounted there and nowhere else, and the
# judge runs /harness/scoring/tests/evaluate.sh over the trajectory main sends
# it. Detected per bundle, so one without a judge service grades as before.

bundle_has_judge() {
  grep -qE '^  judge:[[:space:]]*$' "$TASK/environment/docker-compose.yaml" 2>/dev/null
}

# Export what the bundle's compose file and task.toml read. Call it directly,
# not in $(...), or the exports are lost.
#
#   JUDGE_TOKEN      fresh per invocation, never taken from .env or the caller.
#                    Compose hands it to the judge; task.toml [verifier.env] hands
#                    it to test.sh. Harbor applies verifier env to the test script
#                    only, so the agent -- on the same network as the judge --
#                    never holds it.
#   CODEX_AUTH_FILE  the host's codex login, mounted read-only into the judge
#                    alone. Made absolute: compose resolves a relative bind source
#                    against the bundle's environment/ dir.
prepare_judge() {
  local auth="${CODEX_AUTH_FILE:-${CODEX_HOME:-$HOME/.codex}/auth.json}"
  if [ ! -s "$auth" ]; then
    echo "[run_task] ERROR: codex login not found at $auth -- the judge container cannot grade" >&2
    echo "[run_task]   Run: codex login   (or point CODEX_AUTH_FILE at an auth.json)" >&2
    exit 4
  fi
  auth="$(cd "$(dirname "$auth")" && pwd)/$(basename "$auth")"
  export CODEX_AUTH_FILE="$auth"
  JUDGE_TOKEN="$(python3 -c 'import secrets; print(secrets.token_hex(32))')"
  export JUDGE_TOKEN
  echo "[run_task] judge container ON -- every channel graded in-bundle (rubric by $JUDGE_MODEL); codex login mounted into the judge only" >&2
}

# Did this trial's own reward come from the judge container? Its evaluate.sh
# says so in verifier/reward_producer.json -- not in
# reward.json, which harbor validates as numbers only and fails the trial on a
# string. When it did, carry the producer into reward.json here, on the host,
# so harbor_to_output.py publishes the bundle's reward instead of calling it
# unscored.
adopt_container_reward() {  # adopt_container_reward <trial-dir>
  python3 - "$1/verifier" <<'PYEOF' 2>/dev/null
import json, os, sys
ver = sys.argv[1]
try:
    marker = json.load(open(os.path.join(ver, "reward_producer.json")))
    if marker.get("producer") != "judge_container":
        sys.exit(1)
    path = os.path.join(ver, "reward.json")
    doc = json.load(open(path))
    doc["producer"] = "judge_container"
    with open(path + ".tmp", "w") as fh:
        fh.write(json.dumps(doc, indent=2))
    os.replace(path + ".tmp", path)
except Exception:
    sys.exit(1)
PYEOF
}

# The trial dirs this invocation owns: the names stage_harbor recorded. Globbing
# the job dir instead also re-graded leftovers from earlier invocations,
# spending judge quota on runs this one never made. No `trials` key at all (a
# hand-driven --stage reshape over a job with no state) keeps the glob.
this_invocations_trials() {
  local d="$OUTPUT_DIR/$JOB" t
  if state_has trials; then
    for t in $(state_get trials | tr ',' ' '); do
      [ -d "$d/$t" ] && printf '%s\n' "$d/$t"
    done
    return 0
  fi
  find "$d" -maxdepth 1 -type d -name "*__*" 2>/dev/null | sort
}

# Grade the rubric channel on the host, between harbor and reshape.
#
# Bundles with a `judge` service mostly skip this now: their rubric is graded in
# the judge container and test.sh folds it into the reward. What stays here is
# the fallback -- a judge container that could not grade, a bundle whose own
# reward leaves the rubric out (its verdicts are reused, not re-bought), and
# bundles without a judge -- which is what the rest of this note describes.
#
# The in-container judge cannot do it on the pinned grader: gpt-5.6-sol runs
# over codex, and codex does not exist in python:3.12-slim. Putting one there
# would mean
# mounting this machine's ChatGPT credential into the container the agent just
# ran in under bypassPermissions, and Harbor cannot isolate the verifier from
# that container either -- environment_mode='separate' restarts light-servers
# clean and destroys the world the state channel reads. So the container grades
# everything that needs the live world, and the rubric is graded here.
#
# Runs before stage_reshape so harbor_to_output.py copies the corrected reward
# rather than the rubric-less one. Checkpointed, because a resume must not spend
# judge quota re-grading a trial it already graded.
stage_host_rubric() {
  [ "$(state_get host_rubric_done)" = "1" ] && { echo "[run_task] host rubric already graded; skipping"; return 0; }
  # Trial dirs are named after the TASK slug, not the job: JOB=Input_1_oracle
  # still produces Input_1__PMNhXaa. this_invocations_trials reads the names
  # stage_harbor recorded instead of globbing on "$JOB__*", which found nothing
  # whenever JOB was overridden. Harbor may produce >1 with --n-attempts N.
  local py; py="$REPO/.venv/bin/python"; [ -x "$py" ] || py=python3
  local trial any_graded=0 seen=0 hflags
  while IFS= read -r trial; do
    [ -n "$trial" ] || continue
    seen=$((seen+1))
    # The bundle's own reward already carries a rubric its judge container
    # graded (test.sh wrote reward_producer.json). Nothing to add but the label.
    if [ "${FORCE_HOST_RUBRIC:-0}" != "1" ] && adopt_container_reward "$trial"; then
      echo "[run_task] rubric graded in the judge container for $(basename "$trial"); host pass not needed"
      any_graded=1
      continue
    fi
    # Otherwise the host publishes the reward. host_rubric_pass.py reuses the
    # verdicts a judge container already wrote and calls the judge here only
    # when there are none. FORCE_HOST_RUBRIC=1 re-judges on the host regardless.
    hflags=()
    [ "${FORCE_HOST_RUBRIC:-0}" = "1" ] && hflags+=(--rejudge)
    if "$py" "$REPO/scripts/host_rubric_pass.py" --trial "$trial" --task "$TASK" ${hflags[@]+"${hflags[@]}"}; then
      any_graded=1
    else
      # A failed rubric is not a failed run: Channel A and the state channel are
      # already graded and still worth reporting. Leave the checkpoint unset so a
      # rerun retries this without repeating the agent phase.
      echo "[run_task] host rubric pass failed for $(basename "$trial"); rubric channel stays UNSCORED" >&2
    fi
  done < <(this_invocations_trials)
  if [ "$seen" -eq 0 ]; then
    # Two different conditions, said differently on purpose: a state-tracking
    # bug that grades nothing must not read as an empty job dir.
    if state_has trials && [ -n "$(find "$OUTPUT_DIR/$JOB" -maxdepth 1 -type d -name "*__*" 2>/dev/null)" ]; then
      echo "[run_task] host rubric: this invocation produced no trial dir of its own;" >&2
      echo "           the dirs under $OUTPUT_DIR/$JOB belong to earlier runs and are NOT graded." >&2
      echo "           To grade one deliberately: scripts/host_rubric_pass.py --trial <dir> --task \$TASK" >&2
    else
      echo "[run_task] no trial dir under $OUTPUT_DIR/$JOB; skipping host rubric" >&2
    fi
    return 0
  fi
  [ "$any_graded" = "1" ] && state_put host_rubric_done 1
}

# --- internet audit -----------------------------------------------------------
# These bundles are closed-world: every fact the agent needs is served by the
# light-servers sidecars or sits under the read-only /workspace/data mount. A run
# that answered from the open web did not solve the task -- but it grades exactly
# as though it had, because the reward is computed from the same tool calls
# either way and no channel looks at where a fact came from.
#
# Harbor cannot prevent this on the docker provider, and the two settings that
# look like they would both fail:
#
#   network_mode = "no-network"   detaches the compose bridge as well, so the MCP
#                                 sidecars go unreachable and the agent starts
#                                 with zero tools (grade 0).
#   network_mode = "allowlist"    the docker provider declares
#                                 network_allowlist=False; Harbor cannot express
#                                 a host allowlist here at all.
#
# Neither of those is where the block lives now. network_isolation_overlay()
# above passes tools/network/egress-proxy/overlay.yaml as --extra-docker-compose,
# which makes the project's default network `internal: true` and leaves one
# squid sidecar as the only route out, allowlisting api.anthropic.com. That is a
# Compose-level answer to a question Harbor's network_mode cannot express:
# network_mode says whether the container has a network, the overlay says where
# that network may go.
#
# This audit stays anyway, as a backstop. Prevention can regress silently -- a
# missing overlay, NETWORK_ISOLATION_OFF left set, an allowlist widened to
# unblock a run -- and the trajectory is the one place that shows what the model
# actually reached. It costs nothing on a clean run.
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
# audit says so in those words rather than crediting the proxy for it.
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
  # GLM runs reach zbridge on the host through squid. That is the model call, not a breach.
  [ "${CC_MODE:-}" = "zbridge" ] && flags+=(--proxy-allow host.docker.internal)

  local traj run_dir dirty=0 seen=0 empty=0
  # Iterate RUN DIRECTORIES, not trajectory files.
  #
  # This globbed [Rr]un_*/agent/trajectory.json and skipped anything that did not
  # match. Measured on an 11-run job: 3 runs carried a trajectory and were
  # audited, 8 did not and were passed over in silence -- and `seen` only warns
  # when it reaches ZERO, so the job reported three clean audits and said nothing
  # about the other eight. An unaudited run read exactly like a clean one.
  #
  # It also made the proxy log unreachable in the case it matters most.
  # detect_internet_use.py handles a missing trajectory beside a PRESENT access
  # log -- that is the run where the transcript was lost but squid still recorded
  # what was attempted -- and a loop over trajectories can never hand it that
  # pair. Enumerating run dirs and letting the auditor decide restores it.
  #
  # `[Rr]un_*` because the reshaper writes run_N while older stashed trees carry
  # Run_N; auditing only one casing would skip half a resumed task in silence.
  for run_dir in "$TRAJ_DIR"/[Rr]un_*; do
    [ -d "$run_dir" ] || continue
    traj="$run_dir/agent/trajectory.json"

    # A run with neither is an aborted trial that produced nothing -- counted and
    # reported below, not audited, because there is genuinely nothing to read.
    if [ ! -f "$traj" ] && [ ! -f "$run_dir/logs/egress-access.log" ]; then
      empty=$((empty+1))
      continue
    fi
    seen=$((seen+1))

    # squid's own record of this attempt, if the run made one. It is the only
    # ground truth about what actually reached the proxy; the trajectory scan
    # alone infers egress from shell verbs and cannot see, say, a requests.get
    # inside a python heredoc.
    #
    # Optional on purpose. NETWORK_ISOLATION_OFF=1 runs have no proxy, and trees
    # reshaped from before the capture landed have no file -- both audit fine on
    # the trajectory alone, and forcing the flag would turn a legitimate run
    # into a hard error.
    local _alog="$run_dir/logs/egress-access.log"
    local aflags=()
    [ -f "$_alog" ] && aflags+=(--access-log "$_alog")

    # ${flags[@]+...} is load-bearing under `set -u`: bash 3.2 on macOS treats a
    # bare "${flags[@]}" on an empty array as unbound and kills the script.
    python3 "$REPO/tools/network/detect_internet_use.py" "$traj" \
      --json "$run_dir/internet_audit.json" \
      ${flags[@]+"${flags[@]}"} ${aflags[@]+"${aflags[@]}"} || dirty=1
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
    echo "[run_task] internet audit: $_skipped run(s) had no trajectory and no proxy log -- not audited, NOT clean" >&2

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
    echo "    egress proxy in the path, so the run was not isolated and there is" >&2
    echo "    nothing to show the attempts failed." >&2
  fi
  echo "    This task is closed-world: the answer comes from the MCP sidecars" >&2
  echo "    and /workspace/data. Per-run detail: <run>/internet_audit.json" >&2
  echo "    Inspect without blocking: INTERNET_AUDIT_WARN=1 scripts/run_task.sh ..." >&2
  echo >&2
  exit 2
}


stage_reshape() {
  stage_host_rubric
  local offset; offset="$(state_get run_offset)"
  [ -n "$offset" ] || offset="$(resolve_run_offset)"

  # Restore stash BEFORE reshape so harbor_to_output.py reads the accumulated
  # summary.json and prior Run_N dirs. Harbor may have cleared the job dir,
  # which wipes summary.json; without it the reshaper sees n=1 on every run
  # and keeps overwriting pass@1.json instead of emitting pass@2.json, pass@3.json, ...
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

  local conv=(python3 tools/delivery/harbor_to_output.py "$OUTPUT_DIR/$JOB" \
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
  python3 tools/delivery/make_delivery.py --mask-only "$dir" \
    || echo "[mask] WARNING: path mask failed (non-fatal)" >&2
}

# finance API: report trajectory usage (after the runs are done).
# Skips itself when ODOO_URL is unset; never fails the task run.
# ODOO_URL normally lives in .env, which this script does not source, so read it
# from there when the environment doesn't already provide it.
stage_finance() {
  # The .env may live above the harness when it is vendored into a larger
  # workspace, so search upward the way finance_reporter.py does.
  if [ -z "${ODOO_URL:-}" ]; then
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
    echo "[finance] WARNING: ODOO_URL unset (no .env at or above $REPO) — usage NOT reported" >&2
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

# --- dispatch -----------------------------------------------------------------

# Only the stages that actually drive harbor. reshape/finance/mask move files
# and must not die because harbor drifted or is not installed.
case "$STAGE" in
  preflight|harbor|all) python3 "$REPO/scripts/patch_harbor.py" ;;
esac

trap on_interrupt INT
trap on_terminate TERM

refuse_uncmountable_paths
resolve_auth
check_credentials
check_finance_env

case "$STAGE" in
  preflight) stage_preflight ;;
  harbor)    stage_harbor ;;
  reshape)   stage_reshape ;;
  finance)   stage_finance ;;
  mask)      stage_mask ;;
  all)       stage_preflight; stage_harbor; stage_reshape; stage_finance
             echo "[run_task] done → $OUTPUT_DIR/$OUT_SLUG" ;;
esac
