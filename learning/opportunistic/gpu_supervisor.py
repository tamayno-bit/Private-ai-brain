"""Optional supervised GPU training; never starts a Pod or runs without approval."""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path
from .primary_guard import admission

ROOT=Path("/workspace/services/private-ai-brain")
STATE=Path("/workspace/private-ai-brain-state")
PRIMARY=Path("/workspace/services/original-brain-v1")

def status_write(state, value):
    state.mkdir(parents=True,exist_ok=True)
    target=state/"gpu_supervisor_status.json"
    temp=state/("."+target.name+".tmp")
    temp.write_text(json.dumps({"updated_at":time.time(),**value},indent=2))
    os.replace(temp,target)

def run(args, gate=admission, popen=subprocess.Popen, clock=time.monotonic, sleep=time.sleep):
    if (args.max_seconds<5 or args.max_seconds>1800 or args.steps<1 or args.steps>500000
        or not (256<=args.requested_mib<=16384)
        or not (2048<=args.reserve_mib<=65536)
        or args.batch_size<1 or args.grad_accum<1):
        raise ValueError("Invalid resource limits")
    checkpoint=Path(args.output)/"latest.pt"
    check=lambda:gate(STATE,PRIMARY,True,args.requested_mib,args.reserve_mib)
    initial=check()
    if not args.approve or os.environ.get("PRIVATE_BRAIN_GPU_TRAINING_APPROVED")!="YES":
        return {"started":False,"reason":"explicit_approval_required","preflight":initial}
    if not initial["admit"]:
        return {"started":False,"reason":initial["reason"],"preflight":initial}
    import torch
    if not torch.cuda.is_available():
        return {"started":False,"reason":"cuda_runtime_unavailable"}
    # CUDA is required: a GPU job must never silently fall back to CPU.
    STATE.mkdir(parents=True,exist_ok=True)
    lock_path=STATE/"gpu_training_supervisor.lock"
    import fcntl
    with lock_path.open("a+") as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return {"started":False,"reason":"another_training_supervisor_active"}
        # Unique run output by default. Do not overwrite any older run.
        output=Path(args.output).resolve()
        output.mkdir(parents=True,exist_ok=True)
        train=[sys.executable,"-m","brain.train","--model-config",args.model_config,
            "--data",args.data,"--output",str(output),"--steps",str(args.steps),
            "--batch-size",str(args.batch_size),"--grad-accum",str(args.grad_accum),
            "--save-every","5","--log-every","1"]
        if checkpoint.is_file():train.extend(["--resume",str(checkpoint)])
        env=os.environ.copy()
        env.update({"OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1"})
        log_path=output/"supervisor.log"
        start=clock()
        status_write(STATE,{"state":"starting","checkpoint":str(checkpoint)})
        with log_path.open("a",encoding="utf8") as out:
            proc=popen(train,cwd=str(ROOT),env=env,stdout=out,stderr=subprocess.STDOUT)
            stopped=None
            status_write(STATE,{"state":"running","pid":proc.pid,"checkpoint":str(checkpoint)})
            while proc.poll() is None:
                elapsed=clock()-start
                if elapsed>=args.max_seconds:stopped="time_budget"
                elif not check()["admit"]:stopped="primary_preempted_or_vram_low"
                if stopped:
                    proc.terminate() # training handles SIGTERM and saves checkpoint
                    try:proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        proc.kill();proc.wait(timeout=5)
                    break
                sleep(2)
            result={"started":True,"state":"finished" if proc.returncode==0 else "failed",
                    "exit_code":proc.returncode,"stop_reason":stopped or "training_exited",
                    "checkpoint_available":checkpoint.is_file(),"log":str(log_path)}
            if checkpoint.is_file():
                try:
                    manifest=json.loads(Path(str(checkpoint)+".json").read_text())
                    result["checkpoint_step"]=manifest.get("step")
                except (OSError,ValueError):result["checkpoint_manifest_error"]=True
            status_write(STATE,result)
            return result

def main():
    p=argparse.ArgumentParser(description="Fail-closed opportunistic GPU worker")
    p.add_argument("--preflight",action="store_true")
    p.add_argument("--approve",action="store_true",help="Requires explicit approval and env var")
    p.add_argument("--model-config",default="config/model_tiny.json")
    p.add_argument("--data",default="config/dataset_example.jsonl")
    p.add_argument("--output",default="/workspace/private-ai-brain-state/training")
    p.add_argument("--steps",type=int,default=10)
    p.add_argument("--batch-size",type=int,default=1)
    p.add_argument("--grad-accum",type=int,default=1)
    p.add_argument("--requested-mib",type=int,default=4096)
    p.add_argument("--reserve-mib",type=int,default=8192)
    p.add_argument("--max-seconds",type=int,default=600)
    args=p.parse_args()
    if args.preflight:
        result=admission(STATE,PRIMARY,args.approve,args.requested_mib,args.reserve_mib)
        result["training_started"]=False
    else:result=run(args)
    print(json.dumps(result))
if __name__=="__main__":main()
