#!/usr/bin/env bash
set -euo pipefail
umask 077
SRC="$(cd "$(dirname "$0")/.." && pwd)"
STATE="${PRIVATE_BRAIN_STATE_ROOT:-/workspace/private-ai-brain-state}"
PORT="${PRIVATE_BRAIN_API_PORT:-8877}"
if [[ ! "$PORT" =~ ^[0-9]+$ ]] || ((PORT < 1024 || PORT > 65535)); then echo 'Invalid port' >&2; exit 2; fi
mkdir -p "$STATE"
chmod 700 "$STATE"
TOKENFILE="$STATE/owner_api.token"
if [ ! -f "$TOKENFILE" ]; then
  python3 - "$TOKENFILE" <<'PYI'
import os,secrets,sys
p=sys.argv[1]
try:
 fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
except FileExistsError:
 pass
else:
 with os.fdopen(fd,'w') as f:f.write(secrets.token_urlsafe(48))
PYI
fi
chmod 600 "$TOKENFILE"
PIDFILE="$STATE/sidecar.pid"
if [ -f "$PIDFILE" ]; then
  old="$(cat "$PIDFILE")"
  if [[ "$old" =~ ^[0-9]+$ ]] && [ -d "/proc/$old" ] && tr '\0' ' ' <"/proc/$old/cmdline" | grep -q 'brain.api'; then
    echo "Already running sidecar (pid=$old)"; exit 0
  fi
fi
if (echo >/dev/tcp/127.0.0.1/"$PORT") >/dev/null 2>&1; then echo "Port $PORT already in use; not starting" >&2; exit 3; fi
export PRIVATE_BRAIN_API_TOKEN="$(cat "$TOKENFILE")"
cd "$SRC"
nohup nice -n 15 "${PRIVATE_BRAIN_PYTHON:-python3}" -m brain.api --host 127.0.0.1 --port "$PORT" --state-root "$STATE" </dev/null >>"$STATE/api.log" 2>&1 &
echo $! >"$PIDFILE"
echo "Sidecar started on loopback port $PORT (pid=$(cat "$PIDFILE")). No training started."
