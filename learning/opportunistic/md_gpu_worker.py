"""Opt-in MD GPU training worker. No Pod creation, shutdown, model promotion.

Only owner-written explicitly approved datasets are accepted. All checkpoints are
isolated from live Original Brain and Private AI Brain chat inference.
"""
import fcntl,hashlib,json,math,os,time,traceback
from pathlib import Path
from types import SimpleNamespace
from brain.md_training import _load_manifest,TrainingPlanError,get_training_job
from learning.opportunistic.primary_guard import admission
from learning.opportunistic.gpu_supervisor import run as supervised_run
from learning.opportunistic.independent_eval import evaluate as loss_eval
from learning.opportunistic.md_functional_eval import evaluate as functional_eval

ROOT=Path("/workspace/private-ai-brain-state")
SOURCE=Path("/workspace/services/private-ai-brain")
PRIMARY=Path("/workspace/services/original-brain-v1")
MODEL={"vocab_size":260,"context":128,"dim":128,"layers":4,
       "heads":4,"kv_heads":2,"dropout":0.0,"rope_base":10000.0}
TERMINAL={"completed","failed","cancelled"}

def _atomic(path,entry):
    temporary=path.with_name("."+path.name+".part")
    with temporary.open("w",encoding="utf8") as f:
        json.dump(entry,f,ensure_ascii=False,indent=2);f.flush();os.fsync(f.fileno())
    os.replace(temporary,path)

def _update(jid,**fields):
    path=ROOT/"md_training_jobs"/(jid+".json")
    info=json.loads(path.read_text())
    if info["id"]!=jid:raise TrainingPlanError("Job identity mismatch")
    info.update(fields)
    info["updated_at"]=time.time()
    _atomic(path,info)
    return info

def _model_file():
    path=ROOT/"md_training_model_v1.json"
    if not path.exists():
        fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
        with os.fdopen(fd,"w") as out:json.dump(MODEL,out)
    if json.loads(path.read_text())!=MODEL:raise TrainingPlanError("Model config unexpectedly changed")
    return path

def _job_ready(info):
    if info.get("status") in TERMINAL:return False
    if info.get("owner_training_approved") is not True or info.get("rights_reviewed") is not True:
        return False
    if info.get("data_source")!="owner_written":return False
    if type(info.get("max_steps")) is not int or not 20<=info["max_steps"]<=500:
        return False
    return True

def process(jid):
    info=get_training_job(ROOT,jid)
    if not _job_ready(info):return False
    manifest=_load_manifest(ROOT,info["plan_id"],info["dataset_id"])
    if (manifest["train_sha256"]!=info["train_sha256"] or
        manifest["holdout_sha256"]!=info["holdout_sha256"]):raise TrainingPlanError("Dataset hash changed")
    out=ROOT/"md_training_runs"/jid
    out.mkdir(mode=0o700,parents=True,exist_ok=True)
    model_config=_model_file()
    preflight=admission(ROOT,PRIMARY,True,768,4096)
    if not preflight.get("admit"):
        _update(jid,status="waiting_gpu",reason=preflight.get("reason","blocked"))
        return False
    if info.get("baseline_evaluation") is None:
        benchmark=functional_eval(None,manifest["holdout_file"],limit=8)
        info=_update(jid,baseline_evaluation=benchmark,status="baseline_measured")
    checkpoint=out/"latest.pt"
    if checkpoint.exists():
        # Verify manifest and immutable dataset reference before any resume.
        meta=json.loads(Path(str(checkpoint)+".json").read_text())
        if meta.get("dataset_sha256")!=manifest["train_sha256"]:
            raise TrainingPlanError("Resume dataset mismatch")
        step=int(meta["step"])
    else:step=0
    if step>=info["max_steps"]:
        return _finalize(jid,manifest,out,checkpoint,step)
    target=min(step+20,info["max_steps"])
    args=SimpleNamespace(approve=True,requested_mib=768,reserve_mib=4096,
        output=str(out),model_config=str(model_config),data=manifest["train_file"],
        steps=target,batch_size=1,grad_accum=1,max_seconds=180)
    _update(jid,status="training",steps=step,target_step=target,reason=None)
    result=supervised_run(args)
    if not result.get("started"):
        _update(jid,status="waiting_gpu",reason=result.get("reason","supervisor_unavailable"))
        return False
    current=int(result.get("checkpoint_step") or 0)
    if current<1 or not checkpoint.exists():raise TrainingPlanError("No training checkpoint")
    if result.get("state")=="failed":
        _update(jid,status="waiting_gpu",steps=current,reason="training_interrupted_or_failed")
        return False
    # Independent holdout loss (not functional accuracy).
    metric=loss_eval(checkpoint,manifest["holdout_file"],window_limit=12)["independent_holdout_loss"]
    if not math.isfinite(metric):raise TrainingPlanError("Invalid holdout loss")
    ledger=json.loads((out/"best_metrics.json").read_text()) if (out/"best_metrics.json").exists() else {}
    better=metric<ledger.get("best_holdout_loss",float("inf"))-0.005
    if better:
        import shutil
        for src,dst in (("latest.pt","best.pt"),("latest.pt.json","best.pt.json")):
            temp=out/("."+dst+".part")
            shutil.copyfile(out/src,temp);os.replace(temp,out/dst)
    best=min(metric,ledger.get("best_holdout_loss",float("inf")))
    no_improve=0 if better else int(ledger.get("no_improvement_checks",0))+1
    _atomic(out/"best_metrics.json",{"best_holdout_loss":best,
        "current_holdout_loss":metric,"step":current,
        "no_improvement_checks":no_improve,"checkpoint_sha256":hashlib.sha256(checkpoint.read_bytes()).hexdigest()})
    _update(jid,status="checkpoint_saved",steps=current,holdout_loss=metric,
        best_holdout_loss=best,checkpoint=str(checkpoint),reason=result.get("stop_reason"))
    if current>=info["max_steps"] or no_improve>=4:
        return _finalize(jid,manifest,out,out/"best.pt" if (out/"best.pt").exists() else checkpoint,current)
    return True

def _finalize(jid,manifest,out,checkpoint,step):
    final=functional_eval(checkpoint,manifest["holdout_file"],limit=8)
    current=get_training_job(ROOT,jid)
    baseline=current.get("baseline_evaluation") or {}
    gain=final["accuracy"]>float(baseline.get("accuracy",1))
    _update(jid,status="completed",steps=step,
        final_evaluation=final,capability_gain_verified=gain,
        replaces_live_model=False,checkpoint=str(checkpoint),
        reason="Owner-defined dataset finished; model remains a candidate for review")
    return True

def scan_once():
    directory=ROOT/"md_training_jobs"
    if not directory.is_dir():return {"jobs":0,"worked":False}
    files=sorted(directory.glob("*.json"),key=lambda f:f.stat().st_mtime)
    for f in files[:200]:
        if f.is_symlink() or len(f.stem)!=32:continue
        try:
            info=get_training_job(ROOT,f.stem)
            if not _job_ready(info):continue
            worked=process(f.stem)
            return {"jobs":len(files),"worked":worked,"job_id":f.stem}
        except (ValueError,OSError,RuntimeError) as err:
            try:_update(f.stem,status="failed",reason=type(err).__name__+" (see private logs)")
            except Exception:pass
            return {"jobs":len(files),"worked":False,"job_id":f.stem,"error":type(err).__name__}
    return {"jobs":len(files),"worked":False}

def main():
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--once",action="store_true")
    args=p.parse_args()
    if os.environ.get("PRIVATE_BRAIN_GPU_TRAINING_APPROVED")!="YES":
        raise SystemExit("Global GPU training approval not set")
    import torch
    if not torch.cuda.is_available():raise SystemExit("CUDA unavailable")
    ROOT.mkdir(parents=True,exist_ok=True)
    lock=ROOT/"md_gpu_training_worker.lock"
    with lock.open("a+") as fd:
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            result=scan_once()
            print(json.dumps(result),flush=True)
            if args.once:return
            time.sleep(20 if result.get("worked") else 30)
if __name__=="__main__":main()
