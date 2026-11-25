"""Owner-only Android HTTPS ingress via RunPod's HTTPS reverse proxy.
Expose only allowlisted Brain endpoints, never the internal master token.
"""
import argparse, collections, hmac, json, os, threading, time, urllib.error, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

GET_PATHS = frozenset(("/v1/brain/health", "/v1/brain/status", "/v1/brain/reports", "/v1/brain/integration", "/v1/brain/md-plans", "/v1/brain/research/providers"))
POST_PATHS = frozenset(("/v1/brain/exam", "/v1/brain/md-plans", "/v1/brain/research", "/v1/brain/md-research/start"))
LOCK = threading.Lock()
CALLS = collections.deque()
SLOTS = threading.BoundedSemaphore(3)

class Gateway(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    def __init__(self, addr, device_key, owner_key):
        self.device_key, self.owner_key = device_key, owner_key
        super().__init__(addr, Handler)

class Handler(BaseHTTPRequestHandler):
    server_version = "BrainControlGateway/0.1"
    def log_message(self, fmt, *args):
        pass
    def reply(self, status, value):
        body = json.dumps(value).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Access-Control-Allow-Origin", "null")
        self.end_headers()
        self.wfile.write(body)
    def do_HEAD(self):
        self.reply(405, {"error": "Method not allowed"})
    def do_OPTIONS(self):
        self.reply(405, {"error": "Method not allowed"})
    def do_GET(self):
        self.dispatch("GET")
    def do_POST(self):
        self.dispatch("POST")
    def dispatch(self, method):
        if urlsplit(self.path).path != self.path:
            return self.reply(400, {"error": "Invalid path"})
        allowed = GET_PATHS if method == "GET" else POST_PATHS
        valid_document_path=(method=="GET" and self.path.startswith("/v1/brain/md-plans/")
            and len(self.path)==len("/v1/brain/md-plans/")+64
            and all(c in "0123456789abcdef" for c in self.path.rsplit("/",1)[-1]))
        parts=self.path.split("/")
        valid_model_review_job=(method=="GET" and len(parts)==6 and
             parts[:5]==["","v1","brain","md-model-review","jobs"] and
             len(parts[5])==32 and all(c in "0123456789abcdef" for c in parts[5]))
        valid_training_job=(method=="GET" and len(parts)==6 and
             parts[:5]==["","v1","brain","md-training","jobs"] and
             len(parts[5])==32 and all(c in "0123456789abcdef" for c in parts[5]))
        valid_dataset=(len(parts) in (6,7) and parts[:4]==["","v1","brain","md-plans"]
             and len(parts[4])==64 and all(c in "0123456789abcdef" for c in parts[4])
             and parts[5]=="dataset" and
             ((method=="GET" and len(parts)==6) or
              (method=="POST" and len(parts)==7 and parts[6] in ("propose","approve"))))
        valid_md_job=(method=="GET" and len(parts)==6 and
             parts[:5]==["","v1","brain","md-research","jobs"] and
             len(parts[5])==32 and all(c in "0123456789abcdef" for c in parts[5]))
        valid_plan_path=(len(parts)==6 and parts[:4]==["","v1","brain","md-plans"]
             and len(parts[4])==64 and all(c in "0123456789abcdef" for c in parts[4])
             and parts[5] in (("plan","evidence","report") if method=="GET" else ("analyze","approve","model-review")))
        if self.path not in allowed and not valid_document_path and not valid_plan_path and not valid_md_job and not valid_training_job and not valid_dataset and not valid_model_review_job:
            return self.reply(404, {"error": "Not found"})
        expected = "Bearer " + self.server.device_key
        provided = self.headers.get("Authorization", "")
        if not hmac.compare_digest(provided, expected):
            return self.reply(401, {"error": "Unauthorized"})
        with LOCK:
            now = time.monotonic()
            while CALLS and now - CALLS[0] > 60:
                CALLS.popleft()
            if len(CALLS) >= 60:
                return self.reply(429, {"error": "Rate limit"})
            CALLS.append(now)
        if not SLOTS.acquire(blocking=False):
            return self.reply(429, {"error": "Busy"})
        try:
            data = None
            if method == "POST":
                try:
                    n = int(self.headers.get("Content-Length", "-1"))
                except ValueError:
                    n = -1
                max_size=200000 if self.path=="/v1/brain/md-plans" else 12000
                if not (0 < n <= max_size):
                    return self.reply(413, {"error": "Invalid request size"})
                data = self.rfile.read(n)
            request = urllib.request.Request(
                "http://127.0.0.1:8877" + self.path,
                data=data,
                headers={"Authorization": "Bearer " + self.server.owner_key,
                         "Content-Type": "application/json"},
                method=method)
            try:
                with urllib.request.urlopen(request, timeout=15) as upstream:
                    status, raw = upstream.status, upstream.read(2_000_000)
            except urllib.error.HTTPError as err:
                status, raw = err.code, err.read(100_000)
            except (OSError, TimeoutError):
                return self.reply(502, {"error": "Private AI Brain unavailable"})
            try:
                parsed = json.loads(raw)
            except (ValueError, UnicodeError):
                return self.reply(502, {"error": "Invalid upstream response"})
            self.reply(status, parsed)
        finally:
            SLOTS.release()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=6080)
    p.add_argument("--state-root", required=True)
    args = p.parse_args()
    state = Path(args.state_root)
    assert state.is_dir() and 1024 <= args.port <= 65535
    device_file, owner_file = state / "device_api.token", state / "owner_api.token"
    device_key, owner_key = device_file.read_text().strip(), owner_file.read_text().strip()
    if len(device_key) < 40 or len(owner_key) < 40:
        raise SystemExit("Missing required bearer credentials")
    Gateway(("0.0.0.0", args.port), device_key, owner_key).serve_forever()

if __name__ == "__main__":
    main()
