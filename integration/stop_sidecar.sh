#!/usr/bin/env bash
set -euo pipefail
STATE="${PRIVATE_BRAIN_STATE_ROOT:-/workspace/private-ai-brain-state}"
PIDFILE="$STATE/sidecar.pid"
if [ ! -f "$PIDFILE" ]; then echo 'No PID file; no action'; exit 0; fi
PID="$(cat "$PIDFILE")"
if [[ "$PID" =~ ^[0-9]+$ ]] && [ -d "/proc/$PID" ] && tr '\0' ' ' <"/proc/$PID/cmdline" | grep -q 'brain.api'; then
  kill -TERM "$PID"
  echo 'Stopped private brain sidecar; persistent state retained.'
else
  echo 'PID is stale or belongs to another process; left untouched.'
fi
rm -f "$PIDFILE"
