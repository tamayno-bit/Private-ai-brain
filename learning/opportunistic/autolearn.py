"""Opt-in per-Pod continuous learning controller.

It resumes only validated, owned synthetic datasets and stops at saturation.
Separate candidate weights are never promoted to the live chat model.
"""
import argparse,fcntl,json,math,os,shutil,time,traceback
from pathlib import Path
from types import SimpleNamespace
from .gpu_supervisor import run as supervised_run
from .primary_guard import admission
from .synthetic_curriculum import create
from .disjoint_holdout import create_disjoint

ROOT=Path("/workspace/services/private-ai-brain")
STATE=Path("/workspace/private-ai-brain-state/autolearn")
LEARN=Path("/workspace/private-ai-brain-state")
DEFAULT_MODEL={"vocab_size":260,"context":64,"dim":128,"layers":4,"heads":4,"kv_heads":2,"dropout":0.0,"rope_base":10000.0}

def save_json(path,item):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name("."+path.name+".part")
    with temp.open("w") as f:
        json.dump(item,f,indent=2);f.flush();os.fsync(f.fileno())
    os.replace(temp,path)

def report(state,phase,**fields):
    status={"phase":phase,"updated_at":time.time(),"learning_enabled":True,
        "project":"Private AI Brain","training_isolation":"candidate_not_live",**fields}
    save_json(state/"auto_learning_status.json",status)
    return status

def config_init(state):
    state.mkdir(parents=True,exist_ok=True)
    model_path=state/"model_arithmetic_v1.json"
    if not model_path.exists():model_path.write_text(json.dumps(DEFAULT_MODEL))
    if json.loads(model_path.read_text())!=DEFAULT_MODEL:raise RuntimeError("Unexpected curriculum model config")
    return model_path

def approved(state,pod_id):
    flag=state/"approval.json"
    try:
        data=json.loads(flag.read_text())
        return data.get("approved") is True and data.get("gpu_pod_id")==pod_id and not (state/"PAUSE").exists()
    except (OSError,ValueError):return False

def prepared(state):
    model=config_init(state)
    tr,tr_meta=create(LEARN/"datasets",604)
    ho,ho_meta=create_disjoint(LEARN/"datasets",604,605)
    if tr_meta["sha256"]==ho_meta["sha256"] or ho_meta["removed_overlapping_rows"]<0:
        raise RuntimeError("Holdout invalid")
    return model,tr,ho,tr_meta["sha256"]

def best_update(run_dir,metric,step,label="legacy"):
    if label not in ("legacy","disjoint_v1"):raise ValueError("unknown benchmark label")
    suffix="" if label=="legacy" else "_"+label
    meta_file=run_dir/("learning_state"+suffix+".json")
    prior=json.loads(meta_file.read_text()) if meta_file.exists() else {}
    prev=float(prior.get("best_loss",float("inf")))
    best=math.isfinite(metric) and metric<prev-0.005
    patience=0 if best else int(prior.get("patience",0))+1
    if best:
        for source,target in (("latest.pt","best"+suffix+".pt"),("latest.pt.json","best"+suffix+".pt.json")):
            # All writes to best are atomic in the destination folder.
            sp=run_dir/source; temp=run_dir/("."+target+".part")
            shutil.copyfile(sp,temp);os.replace(temp,run_dir/target)
    result={"best_loss":metric if best else prev,"current_holdout_loss":metric,
        "best_step":step if best else prior.get("best_step"),"step":step,
        "patience":patience,"updated_at":time.time(),
        "candidate_promoted_to_live":False}
    save_json(meta_file,result)
    return result

def one(state,pod_id):
    if not approved(state,pod_id):return report(state,"paused_or_revoked")
    model,tr,ho,data_sha=prepared(state)
    run_dir=state/"runs"/("arithmetic-604-"+data_sha[:12])
    run_dir.mkdir(parents=True,exist_ok=True)
    ledger_file=run_dir/"learning_state_disjoint_v1.json"
    ledger=json.loads(ledger_file.read_text()) if ledger_file.exists() else {}
    step=int(ledger.get("step",0))
    if (run_dir/"latest.pt.json").exists():
        manifest=json.loads((run_dir/"latest.pt.json").read_text())
        if manifest.get("dataset_sha256")!=data_sha:raise RuntimeError("Candidate dataset changed")
        step=int(manifest["step"])
    # Establish an uncontaminated baseline before extending a legacy candidate.
    # Old overlapping holdout measurements remain archived and are not compared.
    if step and not ledger_file.exists() and (run_dir/"latest.pt").exists():
        from .independent_eval import evaluate
        initial_loss=evaluate(run_dir/"latest.pt",ho,window_limit=20)["independent_holdout_loss"]
        ledger=best_update(run_dir,initial_loss,step,"disjoint_v1")
    from brain.dataset import load_records,token_stream
    rows,_=load_records(tr)
    # Two approximate passes over this finite, verified original corpus.
    # This is a per-curriculum safety budget, not an overall project limit.
    tokens=len(token_stream(rows))
    step_budget=min(2000,max(100,(tokens*2+DEFAULT_MODEL["context"]*2-1)//(DEFAULT_MODEL["context"]*2)))
    if step>=step_budget or int(ledger.get("patience",0))>=4:
        return report(state,"idle_curriculum_saturated",step=step,
            reason="plateau_or_verified_corpus_exhausted",best_loss=ledger.get("best_loss"),
            curriculum_step_budget=step_budget)
    preflight=admission(LEARN,Path("/workspace/services/original-brain-v1"),True,768,4096)
    if not preflight["admit"]:
        return report(state,"waiting_for_gpu_headroom",reason=preflight["reason"],step=step)
    from .independent_eval import evaluate
    target=min(step+10,step_budget)
    report(state,"training",step=step,target_step=target,training_running=True,
        dataset_sha256=data_sha,gpu_pod_id=pod_id,curriculum_step_budget=step_budget)
    args=SimpleNamespace(approve=True,requested_mib=768,reserve_mib=4096,
        output=str(run_dir),model_config=str(model),data=str(tr),steps=target,
        batch_size=2,grad_accum=1,max_seconds=240)
    result=supervised_run(args)
    if not result.get("started"):
        return report(state,"waiting_for_gpu_headroom",reason=result.get("reason"),step=step)
    if not result.get("checkpoint_available"):
        return report(state,"training_error",reason=result.get("stop_reason","missing_checkpoint"))
    step=int(result.get("checkpoint_step") or 0)
    if step<1:raise RuntimeError("Checkpoint missing step")
    if result.get("state")=="failed":
        return report(state,"training_error",step=step,reason=result.get("stop_reason"))
    # Evaluate an independently seeded holdout; no promotion to production.
    metric=evaluate(run_dir/"latest.pt",ho,window_limit=20)["independent_holdout_loss"]
    if not math.isfinite(metric):raise RuntimeError("Invalid evaluation result")
    ledger=best_update(run_dir,metric,step,"disjoint_v1")
    return report(state,"trained_candidate" if result.get("state")=="finished" else "interrupted",
        step=step,best_step=ledger["best_step"],best_loss=ledger["best_loss"],
        independent_holdout_loss=metric,patience=ledger["patience"],
        stop_reason=result.get("stop_reason"),training_running=False,
        candidate_promoted_to_live=False)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--pod-id",required=True)
    p.add_argument("--once",action="store_true")
    p.add_argument("--poll-seconds",type=int,default=20)
    args=p.parse_args()
    if not 10<=args.poll_seconds<=300:raise SystemExit("poll interval must be 10..300 seconds")
    if not os.environ.get("PRIVATE_BRAIN_GPU_TRAINING_APPROVED")=="YES":
        raise SystemExit("GPU training approval environment not enabled")
    STATE.mkdir(parents=True,exist_ok=True)
    # Never allow two schedulers to train the same shared candidate dataset.
    with (STATE/"autolearn.lock").open("a+") as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit("An automatic learning controller already runs")
        while True:
            try:
                result=one(STATE,args.pod_id)
                print(json.dumps(result),flush=True)
            except Exception as exc:
                result=report(STATE,"error",reason=type(exc).__name__,message=str(exc)[:200])
                print(json.dumps(result),flush=True)
            if args.once:return
            time.sleep(max(args.poll_seconds,60) if result["phase"] in
                ("idle_until_new_verified_dataset","paused_or_revoked","error") else args.poll_seconds)

if __name__=="__main__":main()
