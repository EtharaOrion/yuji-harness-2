#!/usr/bin/env bash
# Start zbridge (Anthropic-to-GLM proxy) on port 8766.
# Run from harness/ root: bash tools/bridges/start_zbridge.sh
# Requires ZB_ZAI_API_KEY and ZB_BRIDGE_SECRET to be set (or sourced from .env).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ZBRIDGE_DIR="$(cd "$SCRIPT_DIR/zbridge" && pwd)"

if [[ ! -f "$ZBRIDGE_DIR/pyproject.toml" ]]; then
    echo "ERROR: zbridge not found at $ZBRIDGE_DIR" >&2
    exit 1
fi

if [[ -z "${ZB_ZAI_API_KEY:-}" ]]; then
    if [[ -f "$SCRIPT_DIR/../../.env" ]]; then
        set -a; source "$SCRIPT_DIR/../../.env"; set +a
    fi
fi

if [[ -z "${ZB_ZAI_API_KEY:-}" ]]; then
    echo "ERROR: ZB_ZAI_API_KEY is not set" >&2
    exit 1
fi

if [[ -z "${ZB_BRIDGE_SECRET:-}" ]]; then
    echo "ERROR: ZB_BRIDGE_SECRET is not set" >&2
    exit 1
fi

cd "$ZBRIDGE_DIR"
uv sync --quiet 2>/dev/null || pip install -e . -q

exec env \
    ZB_ZAI_API_KEY="$ZB_ZAI_API_KEY" \
    ZB_BRIDGE_SECRET="$ZB_BRIDGE_SECRET" \
    ZB_PORT="${ZB_PORT:-8766}" \
    ZB_HOST="${ZB_HOST:-127.0.0.1}" \
    ZB_MODEL_ALIAS_JSON="${ZB_MODEL_ALIAS_JSON:-}" \
    ZB_UPSTREAM_URL="${ZB_UPSTREAM_URL:-https://api.z.ai/api/coding/paas/v4/chat/completions}" \
    python -m zbridge --port "${ZB_PORT:-8766}" --host "${ZB_HOST:-127.0.0.1}"
