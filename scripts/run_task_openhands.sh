#!/usr/bin/env bash
# Run one yuji-atlas task end-to-end via the OpenHands agent adapter.
#
# Bypasses Harbor's own --agent dispatch (which is claude-code/oracle/etc)
# and instead spins up the openhands-adapter container to produce the agent
# trajectory + report. Then hands off to the same test.sh + judge_client +
# grade + grade_coverage + combine_channels pipeline that claude-code uses.
#
# Usage:
#   scripts/run_task_openhands.sh /path/to/staging/<uuid>/
#
# Env overrides:
#   CLAUDE_CODE_OAUTH_TOKEN   (required) opus-5 auth via LiteLLM
#   MODEL                     default anthropic/claude-opus-5
#   MAX_ITERATIONS            default 100
#   OUTPUT_DIR                default <repo>/output
#   JOB                       default <task uuid>
#
# Same MCP servers, same trajectory format target (openhands.txt), same
# scoring pipeline. See services/openhands-adapter/README.md for details.
set -u

BUNDLE="${1:?usage: $0 /path/to/bundle}"
BUNDLE_UUID="$(basename "$BUNDLE")"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

OUTPUT_DIR="${OUTPUT_DIR:-$REPO/output}"
JOB="${JOB:-$BUNDLE_UUID}"
MODEL="${MODEL:-anthropic/claude-opus-5}"
MAX_ITERATIONS="${MAX_ITERATIONS:-100}"

if [ -z "${CLAUDE_CODE_OAUTH_TOKEN:-}" ]; then
    echo "[run_task_openhands] ERROR: CLAUDE_CODE_OAUTH_TOKEN not set" >&2
    exit 1
fi

if [ ! -d "$BUNDLE" ]; then
    echo "[run_task_openhands] ERROR: bundle dir not found: $BUNDLE" >&2
    exit 1
fi

RUN_DIR="$OUTPUT_DIR/$JOB/trajectory/run_1"
LOGS_DIR="$RUN_DIR/logs"
WORKSPACE_DIR="$RUN_DIR/workspace"
mkdir -p "$LOGS_DIR/agent" "$LOGS_DIR/verifier" "$WORKSPACE_DIR/out"

IMAGE="${OPENHANDS_ADAPTER_IMAGE:-harness/openhands-adapter:latest}"

echo "[run_task_openhands] bundle=$BUNDLE"
echo "[run_task_openhands] output_dir=$OUTPUT_DIR"
echo "[run_task_openhands] job=$JOB"
echo "[run_task_openhands] image=$IMAGE"
echo "[run_task_openhands] model=$MODEL"

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "[run_task_openhands] Building $IMAGE ..."
    docker build -t "$IMAGE" "$REPO/services/openhands-adapter/" || {
        echo "[run_task_openhands] ERROR: docker build failed" >&2
        exit 1
    }
fi

docker run --rm \
    -e CLAUDE_CODE_OAUTH_TOKEN="$CLAUDE_CODE_OAUTH_TOKEN" \
    -e MODEL="$MODEL" \
    -e MAX_ITERATIONS="$MAX_ITERATIONS" \
    -v "$BUNDLE/tests:/tests:ro" \
    -v "$BUNDLE/task.toml:/tests/../task.toml:ro" \
    -v "$WORKSPACE_DIR:/workspace" \
    -v "$LOGS_DIR:/logs" \
    "$IMAGE"

echo "[run_task_openhands] agent phase complete"
echo "[run_task_openhands] trajectory: $LOGS_DIR/agent/openhands.txt"
echo "[run_task_openhands] report: $WORKSPACE_DIR/out/report.md"

echo "[run_task_openhands] NOTE: verifier phase (test.sh + judge + grade) not"
echo "                     invoked by this wrapper. To score, either run the"
echo "                     full harness rollout after wiring AGENT=openhands"
echo "                     into run_task.sh, or invoke test.sh manually"
echo "                     against LOGS_DIR + WORKSPACE_DIR/out."
