"""Read-only telemetry from a GPU Pod into shared persistent state."""
import json,os,subprocess,time,urllib.request
from pathlib import Path

STATE=Path("/workspace/private-ai-brain-state")
def gpu_info():
    proc=subprocess.run(["nvidia-smi","--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu","--format=csv,noheader,nounits"],
        capture_output=True,text=True,check=True,timeout=4)
    result=[]
    for line in proc.stdout.splitlines():
        bits=[x.strip() for x in line.split(",")]
        if len(bits)!=5:raise ValueError("Unexpected NVIDIA telemetry")
        result.append({"name":bits[0],"total_mib":int(bits[1]),"used_mib":int(bits[2]),"free_mib":int(bits[3]),"utilization_percent":int(bits[4])})
    if not result:raise ValueError("No GPU")
    return result

def primary_slots():
    detail={}
    for key,port in (("coding",8770),("vision",8772)):
        try:
            with urllib.request.urlopen("http://127.0.0.1:%s/slots"%port,timeout=1.0) as resp:
                slots=json.load(resp)
            detail[key]={"reachable":True,"busy":any(s.get("is_processing") is not False for s in slots),
                "slots":len(slots)}
        except (OSError,ValueError,TypeError,KeyError):
            detail[key]={"reachable":False,"busy":None}
    return detail

def snapshot():
    return {"updated_at":time.time(),"source":"gpu_pod_readonly_observer","gpu":gpu_info(),"primary":primary_slots(),"training_auto_started":False}

def write(value):
    STATE.mkdir(parents=True,exist_ok=True)
    target=STATE/"gpu_observer.json"
    temp=STATE/".gpu_observer.tmp"
    with temp.open("w",encoding="utf8") as f:
        json.dump(value,f);f.flush();os.fsync(f.fileno())
    os.replace(temp,target)

def main():
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--once",action="store_true")
    p.add_argument("--interval",type=float,default=10.0)
    args=p.parse_args()
    if not 5<=args.interval<=60:raise SystemExit("Invalid interval")
    if args.once:
        print(json.dumps(snapshot()));return
    while True:
        try:write(snapshot())
        except (OSError,ValueError,subprocess.SubprocessError):
            write({"updated_at":time.time(),"source":"gpu_pod_readonly_observer","gpu":[],"primary":{},"error":"telemetry_unavailable"})
        time.sleep(args.interval)
if __name__=="__main__":main()
