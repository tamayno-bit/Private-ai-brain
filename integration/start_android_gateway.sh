#!/usr/bin/env bash
set -euo pipefail
umask 077
ROOT=/workspace/services/private-ai-brain
STATE=/workspace/private-ai-brain-state
PORT=6080
mkdir -p "$STATE"; chmod 700 "$STATE"
if [ ! -f "$STATE/device_api.token" ]; then
 python3 - "$STATE/device_api.token" <<'PY'
import os,secrets,sys
fd=os.open(sys.argv[1],os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
with os.fdopen(fd,'w') as f: f.write(secrets.token_urlsafe(48))
PY
fi
chmod 600 "$STATE/device_api.token"
if [ ! -f "$STATE/owner_api.token" ]; then echo "Private Brain local token missing" >&2;exit 2;fi
if [ -f "$STATE/gateway.pid" ]; then
 old="$(cat "$STATE/gateway.pid")"
 if [[ "$old" =~ ^[0-9]+$ ]] && [ -d "/proc/$old" ] && tr '\0' ' ' </proc/"$old"/cmdline | grep -q 'brain_control_gateway.py';then
  echo "Gateway already running";exit 0
 fi
fi
if (echo >/dev/tcp/127.0.0.1/"$PORT") >/dev/null 2>&1;then
 echo "Port $PORT already used; refusing to replace existing service" >&2; exit 3
fi
cd "$ROOT"
nohup nice -n 10 python3 integration/brain_control_gateway.py --port "$PORT" --state-root "$STATE" </dev/null >>"$STATE/gateway.log" 2>&1 &
echo $! > "$STATE/gateway.pid"
echo "Brain Control ingress started on port $PORT; owner token stays private."
