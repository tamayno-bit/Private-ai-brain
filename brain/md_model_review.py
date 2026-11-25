"""Private owner MD understanding through a dedicated local Tongyi CPU queue.

The model is hosted in the same account. MD content is never submitted to public
web search here. Its suggestions remain untrusted and require manual review.
"""
import hashlib,json,os,re,time,uuid
from pathlib import Path
from .md_plans import fetch,ID_RE
from .learning_plans import read_plan

ROOT=Path("/workspace/private-ai-brain-state")
FOLDER="md_local_model_review"
JOB_ID=re.compile(r"^[a-f0-9]{32}$")
class ModelReviewError(ValueError):pass

def _base(root):
    path=Path(root)/FOLDER
    if path.is_symlink():raise ModelReviewError("Unsafe review directory")
    for dir in (path,path/"requests",path/"responses"):
        dir.mkdir(mode=0o700,parents=True,exist_ok=True)
        os.chmod(dir,0o700)
    return path

def _atomic(path,obj):
    temporary=path.with_name("."+path.stem+"-"+uuid.uuid4().hex+".tmp")
    fd=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    try:
        with os.fdopen(fd,"w") as f:json.dump(obj,f,ensure_ascii=False);f.flush();os.fsync(f.fileno())
        os.replace(temporary,path)
    finally:temporary.unlink(missing_ok=True)

def submit(root,plan_id,approved=False,task_index=0):
    if approved is not True:raise ModelReviewError("Owner consent for local MD analysis required")
    plan=read_plan(root,plan_id)
    if plan.get("status")!="approved_for_review_and_research":
        raise ModelReviewError("Plan must be owner-approved first")
    tasks=plan.get("tasks",[])
    if type(task_index) is not int or not 0<=task_index<len(tasks):
        raise ModelReviewError("A single valid task must be selected")
    goal=tasks[task_index].get("goal","")
    if not isinstance(goal,str) or not 1<=len(goal)<=180:
        raise ModelReviewError("Task is invalid")
    base=_base(root)
    # Only one pending analysis per document; avoid duplicate model load.
    items=list(base.glob("*.json"))
    for file in items[:500]:
        if file.is_symlink():continue
        try:
            entry=json.loads(file.read_text())
            if entry.get("plan_id")==plan_id and entry.get("status")=="queued":
                # Settle any finished response before admitting the next task.
                entry=get(root,entry["id"])
            if entry.get("plan_id")==plan_id and entry.get("status")=="queued":
                if entry.get("task_index")==task_index:return entry,False
                raise ModelReviewError("Previous task still pending; one at a time")
            if entry.get("plan_id")==plan_id and entry.get("task_index")==task_index and entry.get("status")=="completed":
                return entry,False
        except (ValueError,OSError):continue
    job_id=uuid.uuid4().hex
    record={"id":job_id,"plan_id":plan_id,"task_index":task_index,"status":"queued",
        "created_at":time.time(),"updated_at":time.time(),
        "task_goal":goal,"task_characters":len(goal),
        "whole_document_sent":False,
        "private_local_inference_only":True,"external_web_search_used":False,
        "instructions_active":False,"model_output_trusted":False,
        "training_started":False,"answer":None,"error":None}
    _atomic(base/(job_id+".json"),record)
    # Never save untrusted document content to public research queues.
    payload={"id":job_id,"plan_id":plan_id,"task_index":task_index,
        "content":goal,
        "created_at":time.time(),"max_tokens":420}
    _atomic(base/"requests"/(job_id+".json"),payload)
    return record,True

def get(root,job_id):
    if not isinstance(job_id,str) or not JOB_ID.fullmatch(job_id):
        raise ModelReviewError("Invalid review id")
    base=_base(root)
    file=base/(job_id+".json")
    if not file.is_file() or file.is_symlink():raise ModelReviewError("Review not found")
    record=json.loads(file.read_text())
    if record.get("id")!=job_id:raise ModelReviewError("Review mismatch")
    result=base/"responses"/(job_id+".json")
    if record["status"]=="queued" and result.is_file():
        data=json.loads(result.read_text())
        if data.get("id")!=job_id:raise ModelReviewError("Response mismatch")
        record.update({"status":"completed" if data.get("model_inference_used") is True else "failed",
            "answer":data.get("answer") if data.get("model_inference_used") is True else None,
            "model_inference_used":data.get("model_inference_used") is True,
            "error":data.get("error"),"updated_at":time.time()})
        _atomic(file,record)
    if record["status"]=="queued" and time.time()-record["created_at"]>180:
        record["status"]="interrupted";record["error"]="Local model review was interrupted"
        record["updated_at"]=time.time();_atomic(file,record)
    return record

def list_for_plan(root,plan_id):
    base=_base(root)
    results=[]
    for path in base.glob("*.json"):
        try:
            record=json.loads(path.read_text())
            if record.get("plan_id")==plan_id and JOB_ID.fullmatch(str(record.get("id",""))):
                results.append(get(root,record["id"]))
        except (OSError,ValueError,ModelReviewError):continue
    return sorted(results,key=lambda x:x.get("created_at",0),reverse=True)[:10]
