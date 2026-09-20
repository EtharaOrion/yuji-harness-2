#!/usr/bin/env bash
# Local dev launcher. Requires ZB_ZAI_API_KEY + ZB_BRIDGE_SECRET in the env.
set -euo pipefail

cd "$(dirname "$0")/.."

: "${ZB_ZAI_API_KEY:?export ZB_ZAI_API_KEY=<your z.ai api key>}"
: "${ZB_BRIDGE_SECRET:?export ZB_BRIDGE_SECRET=<local shared secret>}"

export ZB_STREAM_LOG_PATH="${ZB_STREAM_LOG_PATH:-/tmp/zbridge-stream.jsonl}"

.venv/bin/python -m zbridge --host 127.0.0.1 --port 8766 --log-level info
