"""Fail-closed GPU preflight: never rents GPU or launches work."""
import json,time,urllib.request
from pathlib import Path
from .gpu_admission import probe,decision

def admission(state, primary_root, approved=False, requested_mib=4096, reserve_mib=8192, now=None):
    now=time.time() if now is None else now
    p=probe()
    if not p["present"]: return {"admit":False,"reason":"no_gpu","probe":p}
    root=Path(primary_root)
    heartbeat=root/"live_queue"/"expert_alive.json"
    try:
        alive=json.loads(heartbeat.read_text())
        fresh=now-float(alive["ts"])<=8 and now>=float(alive["ts"])-2
        healthy=fresh and alive.get("ready") is True
    except (OSError,ValueError,TypeError,KeyError):
        healthy=False
    # Primary coding is always required. Optional Vision is either hot/idle,
    # or deliberately unloaded under a fresh on-demand supervisor heartbeat.
    # Unknown/failed Vision status without supervision is NOT safe for training.
    for port in (8770, 8772):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/slots", timeout=1.5) as response:
                slots=json.load(response)
            if not isinstance(slots,list) or not slots or any(
                not isinstance(slot,dict) or slot.get("is_processing") is not False for slot in slots):
                return {"admit":False,"reason":"primary_inference_busy"}
        except (OSError,ValueError,TypeError):
            if port==8770:
                return {"admit":False,"reason":"primary_slots_unavailable"}
            supervisor=root/"vision_queue"/"on_demand_status.json"
            try:
                status=json.loads(supervisor.read_text())
                age=now-float(status["timestamp"])
                inactive=(status.get("supervisor_online") is True and status.get("on_demand") is True
                    and status.get("vision_hot") is False and status.get("active_request") is False
                    and 0<=age<8)
            except (OSError,ValueError,TypeError,KeyError):
                inactive=False
            if not inactive:
                return {"admit":False,"reason":"primary_slots_unavailable"}
    # A coding request must always win. Stale responses are not requests.
    queues=[root/"live_queue"/"requests",root/"coding_queue"/"requests",root/"vision_queue"/"requests"]
    if any(q.is_dir() and any(q.glob("*.json")) for q in queues) or any(q.is_dir() and any(q.glob("*.working")) for q in queues):
        return {"admit":False,"reason":"primary_queue_pending"}
    if not healthy: return {"admit":False,"reason":"primary_unhealthy_or_stale"}
    free=min(g["free_mib"] for g in p["gpus"])
    total=min(g["total_mib"] for g in p["gpus"])
    d=decision(free,total,requested_mib,reserve_mib,healthy,approved)
    return dict(d,probe=p)

if __name__=="__main__":
    import argparse
    a=argparse.ArgumentParser()
    a.add_argument("--primary-root",default="/workspace/services/original-brain-v1")
    a.add_argument("--approve",action="store_true")
    opt=a.parse_args()
    print(json.dumps(admission("/workspace/private-ai-brain-state",opt.primary_root,opt.approve)))
