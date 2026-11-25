#!/usr/bin/env bash
set -euo pipefail
STATE="${PRIVATE_BRAIN_STATE_ROOT:-/workspace/private-ai-brain-state}"
PORT="${PRIVATE_BRAIN_API_PORT:-8877}"
TOKEN="$(cat "$STATE/owner_api.token")"
CODE="$(curl --max-time 3 -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/v1/brain/health")"
[ "$CODE" = 401 ] || { echo "FAIL: missing auth gave $CODE"; exit 1; }
CODE="$(curl --max-time 3 -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $TOKEN" "http://127.0.0.1:$PORT/v1/brain/health")"
[ "$CODE" = 200 ] || { echo "FAIL: authenticated health gave $CODE"; exit 2; }
CODE="$(curl --max-time 3 -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8765/health)"
[ "$CODE" = 200 ] || { echo "FAIL: Original Brain returned $CODE"; exit 3; }
echo 'PASS: isolated Brain API owner auth (401/200) + Original Brain health (200).'
