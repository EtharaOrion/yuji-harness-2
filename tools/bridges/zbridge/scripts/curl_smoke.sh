#!/usr/bin/env bash
# T20a smoke: exercise the running bridge with real GLM-5.3.
# Usage: bash scripts/curl_smoke.sh {text|stream-text|thinking|tool|multitool|health|probe-encoding}
set -euo pipefail

HOST="${ZB_HOST_URL:-http://127.0.0.1:8766}"
: "${ZB_BRIDGE_SECRET:?export ZB_BRIDGE_SECRET=<local shared secret>}"

HDRS=(
  -H "content-type: application/json"
  -H "x-zbridge-secret: ${ZB_BRIDGE_SECRET}"
  -H "anthropic-version: 2023-06-01"
)

case "${1:-text}" in
  health)
    curl -sS "${HOST}/healthz" | jq
    ;;

  text)
    curl -sS "${HDRS[@]}" "${HOST}/v1/messages" -d '{
      "model":"claude-3-5-sonnet-latest",
      "max_tokens":32,
      "messages":[{"role":"user","content":"Say hi in one word."}]
    }' | jq
    ;;

  stream-text)
    curl -N -sS "${HDRS[@]}" -H "accept: text/event-stream" "${HOST}/v1/messages" -d '{
      "model":"claude-3-5-sonnet-latest",
      "max_tokens":64,
      "stream":true,
      "messages":[{"role":"user","content":"Say hi in three words."}]
    }'
    ;;

  thinking)
    curl -N -sS "${HDRS[@]}" -H "accept: text/event-stream" "${HOST}/v1/messages" -d '{
      "model":"claude-3-5-sonnet-latest",
      "max_tokens":256,
      "stream":true,
      "thinking":{"type":"enabled","budget_tokens":2048},
      "messages":[{"role":"user","content":"Explain in one sentence why the sky is blue."}]
    }'
    ;;

  tool)
    curl -sS "${HDRS[@]}" "${HOST}/v1/messages" -d '{
      "model":"claude-3-5-sonnet-latest",
      "max_tokens":128,
      "tools":[{"name":"get_weather","description":"Get weather.",
                "input_schema":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}],
      "messages":[{"role":"user","content":"What is the weather in Tokyo?"}]
    }' | jq
    ;;

  multitool)
    curl -sS "${HDRS[@]}" "${HOST}/v1/messages" -d '{
      "model":"claude-3-5-sonnet-latest",
      "max_tokens":256,
      "tools":[
        {"name":"get_weather","description":"Get weather.",
         "input_schema":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}},
        {"name":"get_time","description":"Get local time.",
         "input_schema":{"type":"object","properties":{"tz":{"type":"string"}},"required":["tz"]}}
      ],
      "messages":[{"role":"user","content":"Get the weather and time in Paris. Use both tools."}]
    }' | jq
    ;;

  probe-encoding)
    echo "=== response Content-Encoding from upstream stream ==="
    curl -sS -N -D - "${HDRS[@]}" -H "accept: text/event-stream" "${HOST}/v1/messages" -d '{
      "model":"claude-3-5-sonnet-latest","max_tokens":8,"stream":true,
      "messages":[{"role":"user","content":"hi"}]
    }' -o /dev/null | grep -iE '^(content-encoding|transfer-encoding|zbridge-)' || echo "(none)"
    ;;

  *)
    echo "usage: $0 {text|stream-text|thinking|tool|multitool|health|probe-encoding}" >&2
    exit 2
    ;;
esac
