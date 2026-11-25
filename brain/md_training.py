"""Owner-reviewed MD dataset scheduling; NO training or GPU spending on API side."""
import hashlib,json,os,re,time,uuid
from pathlib import Path
from .learning_plans import read_plan
from .md_plans import ID_RE
from .md_dataset import propose,DatasetPlanError

JOBID=re.compile(r"^[a-f0-9]{32}$")
class TrainingPlanError(ValueError):pass
def _load_manifest(root,plan,dataset_id):
    if not ID_RE.fullmatch(plan) or not ID_RE.fullmatch(dataset_id):
        raise TrainingPlanError("Invalid training reference")
    root=Path(root)
    path=root/"md_datasets"/plan/(dataset_id+".manifest.json")
    if not path.is_file() or path.is_symlink():raise TrainingPlanError("Dataset not staged")
    info=json.loads(path.read_text())
    if info.get("dataset_id")!=dataset_id or info.get("document_id")!=plan:
        raise TrainingPlanError("Dataset identity mismatch")
    for kind in ("train","holdout"):
        fp=Path(info[kind+"_file"])
        if fp.is_symlink() or not fp.is_file() or not fp.resolve().is_relative_to((root/"md_datasets"/plan).resolve()):
            raise TrainingPlanError("Unsafe dataset path")
        if hashlib.sha256(fp.read_bytes()).hexdigest()!=info[kind+"_sha256"]:
            raise TrainingPlanError("Training dataset changed")
    return info

def approve_training(root,plan_id,dataset_id,confirmed=False,rights_reviewed=False,max_steps=100):
    plan=read_plan(root,plan_id)
    if plan.get("status")!="approved_for_review_and_research":
        raise TrainingPlanError("Plan must be approved")
    if confirmed is not True or rights_reviewed is not True:
        raise TrainingPlanError("Owner training and rights approval required")
    if type(max_steps) is not int or not 20<=max_steps<=500:
        raise TrainingPlanError("Per-dataset training step budget must be 20..500")
    manifest=_load_manifest(root,plan_id,dataset_id)
    if manifest.get("source_type")!="owner_written" or manifest.get("external_model_outputs_used") is not False:
        raise TrainingPlanError("Only reviewed owner-created examples are eligible")
    job_folder=Path(root)/"md_training_jobs"
    job_folder.mkdir(mode=0o700,parents=True,exist_ok=True)
    os.chmod(job_folder,0o700)
    for job_file in job_folder.glob("*.json"):
        if job_file.is_symlink():continue
        try:
            record=json.loads(job_file.read_text())
            if record.get("plan_id")==plan_id and record.get("dataset_id")==dataset_id:
                return record,False
        except (OSError,ValueError):continue
    jid=uuid.uuid4().hex
    entry={"id":jid,"plan_id":plan_id,"dataset_id":dataset_id,"status":"queued",
        "created_at":time.time(),"updated_at":time.time(),"owner_training_approved":True,
        "rights_reviewed":True,"data_source":"owner_written",
        "train_sha256":manifest["train_sha256"],"holdout_sha256":manifest["holdout_sha256"],
        "max_steps":max_steps,"steps":0,"checkpoint":None,
        "baseline_evaluation":None,"final_evaluation":None,"capability_gain_verified":False,
        "gpu_pod_created":False,"gpu_pod_stopped":False,"replaces_live_model":False,
        "reason":None}
    fp=job_folder/(jid+".json")
    fd=os.open(fp,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,"w") as f:json.dump(entry,f,indent=2);f.flush();os.fsync(f.fileno())
    return entry,True

def get_training_job(root,jid):
    if not isinstance(jid,str) or not JOBID.fullmatch(jid):
        raise TrainingPlanError("Invalid training job")
    path=Path(root)/"md_training_jobs"/(jid+".json")
    if not path.is_file() or path.is_symlink():raise TrainingPlanError("Training job not found")
    obj=json.loads(path.read_text())
    if obj.get("id")!=jid:raise TrainingPlanError("Invalid job reference")
    return obj

def plan_training_jobs(root,plan):
    directory=Path(root)/"md_training_jobs"
    if not directory.is_dir() or directory.is_symlink():return []
    rows=[]
    for f in directory.glob("*.json"):
        if f.is_symlink():continue
        try:
            item=json.loads(f.read_text())
            if item.get("plan_id")==plan and JOBID.fullmatch(str(item.get("id",""))):
                rows.append({k:item.get(k) for k in ("id","status","dataset_id","steps","max_steps",
                    "baseline_evaluation","final_evaluation","capability_gain_verified",
                    "created_at","updated_at","reason","replaces_live_model")})
        except (ValueError,OSError):continue
    return sorted(rows,key=lambda row:row.get("created_at") or 0,reverse=True)[:100]
