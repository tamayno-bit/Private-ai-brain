"""Durable per-Markdown research orchestration using Original Brain's existing Tongyi backend.

A plan must be approved. Every query must be separately and explicitly approved as
PUBLIC; never derives private search queries from uploaded Markdown without review.
The module never runs scripts, installs packages, changes model weights or controls GPU.
"""
import hashlib,json,os,re,threading,time,urllib.error,urllib.request,uuid
from pathlib import Path
from .learning_plans import read_plan,append_research
from .research import validate_query,ResearchError

ROOT=Path("/workspace/services/original-brain-v1")
JOBS="md_research_jobs"
LOCK=threading.RLock()
ACTIVE=set()
STATES={"queued","searching","sources_ready","reading_sources","model_answering","ready","failed","interrupted"}

class PlanJobError(ValueError):pass

def _jobs_dir(state):
    path=Path(state)/JOBS
    if path.is_symlink():raise PlanJobError("Unsafe job storage")
    path.mkdir(mode=0o700,parents=True,exist_ok=True)
    os.chmod(path,0o700)
    return path

def _job_path(state,job_id):
    if not isinstance(job_id,str) or not re.fullmatch(r"[0-9a-f]{32}",job_id):
        raise PlanJobError("Invalid job identifier")
    return _jobs_dir(state)/(job_id+".json")

def _persist(path,data):
    tmp=path.with_name("."+path.stem+"-"+uuid.uuid4().hex+".tmp")
    fd=os.open(tmp,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    try:
        with os.fdopen(fd,"w",encoding="utf8") as f:
            json.dump(data,f,ensure_ascii=False);f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:tmp.unlink(missing_ok=True)

def _local_original(method,route,body=None,timeout=7):
    token_path=ROOT/"coding_api.token"
    if token_path.is_symlink() or not token_path.is_file():raise PlanJobError("Original Brain owner API unavailable")
    token=token_path.read_text().strip()
    if len(token)<40:raise PlanJobError("Owner API credential unavailable")
    raw=json.dumps(body,ensure_ascii=False).encode() if body is not None else None
    req=urllib.request.Request("http://127.0.0.1:8765"+route,data=raw,
        headers={"Authorization":"Bearer "+token,"Content-Type":"application/json"},
        method=method)
    try:
        with urllib.request.urlopen(req,timeout=timeout) as response:
            content=response.read(180000)
            if len(content)>=180000:raise PlanJobError("Research response too large")
            return response.status,json.loads(content)
    except urllib.error.HTTPError as e:
        raise PlanJobError("Original Brain research returned HTTP "+str(e.code)) from e
    except (OSError,ValueError,TimeoutError) as e:
        raise PlanJobError("Original Brain research temporarily unavailable") from e

def _update(state,job_id,**fields):
    path=_job_path(state,job_id)
    with LOCK:
        if not path.is_file():raise PlanJobError("Research job missing")
        data=json.loads(path.read_text())
        data.update(fields)
        data["updated_at"]=time.time()
        _persist(path,data)
        return data

def get(state,job_id):
    path=_job_path(state,job_id)
    if not path.exists() or path.is_symlink():raise PlanJobError("Job not found")
    data=json.loads(path.read_text())
    if data.get("id")!=job_id or data.get("phase") not in STATES:raise PlanJobError("Corrupt job")
    # Survive HTTP/container restarts without claiming a lost worker continues.
    if data["phase"] not in ("ready","failed","interrupted") and time.time()-data.get("updated_at",0)>180:
        with LOCK:
            if job_id not in ACTIVE:
                data=_update(state,job_id,phase="interrupted",error="Research worker interrupted; resubmit after review")
    return data

def start(state,plan_id,query,pages=1,tongyi=True,public_query_approved=False,task_index=None):
    plan=read_plan(state,plan_id)
    if plan.get("status")!="approved_for_review_and_research":
        raise PlanJobError("Owner must approve MD plan first")
    tasks=plan.get("tasks",[])
    if type(task_index) is not int or not 0<=task_index<len(tasks):
        raise PlanJobError("Select exactly one approved MD task")
    if public_query_approved is not True:raise PlanJobError("Explicit approval of public query required")
    try:q=validate_query(query)
    except ResearchError as e:raise PlanJobError("Unsafe public query") from e
    if len(q)>120 or type(pages) is not int or not 0<=pages<=2 or type(tongyi) is not bool:
        raise PlanJobError("Invalid research budget")
    # Avoid accidental private project data/exfiltration in public searches.
    private_signals=("/workspace/","/home/","localhost","127.0.0.1","private repo",
                     "ملفات خاصة","api token","secret key")
    if any(s in q.lower() for s in private_signals):
        raise PlanJobError("Private context in public research query")
    with LOCK:
        if ACTIVE:raise PlanJobError("Wait for current MD research step to finish")
        # Durable per-plan serial admission: no next step until prior step completes.
        for path in _jobs_dir(state).glob("*.json"):
            if path.is_symlink():continue
            try:
                other=json.loads(path.read_text())
            except (OSError,ValueError,TypeError):continue
            if other.get("plan_id")==plan_id and other.get("phase") not in ("ready","failed","interrupted"):
                if time.time()-float(other.get("updated_at",0))<180:
                    raise PlanJobError("A previous plan step is still running")
        jobs=_jobs_dir(state)
        # Avoid unbounded duplicate concurrency/history.
        if sum(1 for _ in jobs.glob("*.json"))>=5000:raise PlanJobError("Job journal requires archival")
        job_id=uuid.uuid4().hex
        item={"id":job_id,"plan_id":plan_id,"task_index":task_index,
            "task_goal":tasks[task_index]["goal"],"one_task_at_a_time":True,
            "whole_md_sent_to_tongyi":False,
            "query":q,"pages":pages,
            "requested_model":"Tongyi DeepResearch 30B-A3B" if tongyi else None,
            "phase":"queued","created_at":time.time(),"updated_at":time.time(),
            "public_query_approved":True,"source_count":0,"sources":[],
            "model_inference_used":False,"answer":None,"citations_valid_ids":False,
            "evidence_id":None,"training_started":False,"training_rights_verified":False,
            "checkpoint_changed":False,"error":None}
        _persist(_job_path(state,job_id),item)
        ACTIVE.add(job_id)
        try:
            thread=threading.Thread(target=_work,args=(str(state),job_id),
                name="md-research-"+job_id[:6],daemon=True)
            thread.start()
        except BaseException:
            ACTIVE.discard(job_id)
            _job_path(state,job_id).unlink(missing_ok=True)
            raise
    return {"job_id":job_id,"plan_id":plan_id,"phase":"queued",
        "status_url":"/v1/brain/md-research/jobs/"+job_id,
        "sources_first":True,"training_started":False}

def _work(state,job_id):
    try:
        obj=get(state,job_id)
        _update(state,job_id,phase="searching")
        status,started=_local_original("POST","/v1/research/fast-start",
            {"question":obj["query"],"approved":True,"pages":obj["pages"],
             "preview":obj["requested_model"] is not None})
        if status!=202 or not re.fullmatch(r"[0-9a-f]{32}",str(started.get("id",""))):
            raise PlanJobError("Original Brain research not accepted")
        original_id=started["id"]
        _update(state,job_id,upstream_job_id=original_id)
        deadline=time.monotonic()+120
        last=None
        while time.monotonic()<deadline:
            _,data=_local_original("GET","/v1/research/jobs/"+original_id,timeout=5)
            stage=data.get("stage")
            if stage not in ("queued","searching","sources_ready","reading_sources",
                             "model_answering","ready","failed"):
                raise PlanJobError("Invalid Original Brain research state")
            src=data.get("sources") or []
            source_rows=[]
            for item in src[:6]:
                if not isinstance(item,dict):continue
                source_rows.append({"title":str(item.get("title") or "")[:200],
                    "url":str(item.get("url") or "")[:1200],
                    "summary":str(item.get("snippet") or "")[:450],
                    "citation_id":item.get("id"),
                    "training_rights_verified":False,"untrusted":True})
            if stage!=last:
                _update(state,job_id,phase=stage,source_count=len(source_rows),
                    sources=source_rows)
                last=stage
            if stage=="ready":
                answer=data.get("answer")
                answer=answer[:7000] if isinstance(answer,str) else None
                evidence={"query":obj["query"],"provider":"original_brain_tongyi",
                    "results":source_rows,"retrieved_at":time.time(),
                    "original_job_id":original_id,"status":"ready",
                    "answer":answer,"model_inference_used":data.get("model_inference_used") is True,
                    "citations_valid_ids":data.get("citations_valid_ids") is True,
                    "external_data_for_training":False,"automatic_installation":False}
                record=append_research(state,obj["plan_id"],evidence)
                _update(state,job_id,phase="ready",source_count=len(source_rows),sources=source_rows,
                    answer=answer,model_inference_used=evidence["model_inference_used"],
                    citations_valid_ids=evidence["citations_valid_ids"],
                    evidence_id=record["evidence_id"])
                return
            if stage=="failed":raise PlanJobError("Original Brain public-source research failed")
            time.sleep(1)
        raise PlanJobError("Original Brain research timed out")
    except (PlanJobError,OSError,ValueError,KeyError) as error:
        try:_update(state,job_id,phase="failed",error=str(error)[:180])
        except (OSError,ValueError,PlanJobError):pass
    finally:
        with LOCK:ACTIVE.discard(job_id)
