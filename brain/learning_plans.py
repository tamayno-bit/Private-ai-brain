"""Markdown learning plan draft and owner approval; uploaded text never executes."""
import hashlib,json,os,re,time
from pathlib import Path
from .md_plans import fetch,InvalidMarkdown,ID_RE

KEYWORDS={
 "research":("بحث","مكتبة","مكتبات","مصادر","الإنترنت","search","library","libraries","research","documentation","docs","pypi"),
 "evaluation":("فحص","اختبار","تقييم","مقارنة","test","eval","benchmark","check","verify"),
 "dataset_review":("بيانات","ترخيص","حقوق","data","dataset","license","consent"),
 "training_proposal":("تدريب","تعلم","أوزان","train","fine-tun","weights","learn")
}
def _extract_tasks(md,digest):
    """Prefer actionable numbered tasks; never turn document headings into jobs."""
    numbered=[];bullets=[];headings=[];in_code=False
    for line in md.splitlines():
        stripped=line.strip()
        if stripped.startswith(chr(96)*3):
            in_code=not in_code
            continue
        if in_code or not stripped or stripped.startswith(">"):continue
        numbered_match=re.match(r"^([0-9]{1,3})[.)]\s+(.+)$",stripped)
        if numbered_match:
            title=numbered_match.group(2).strip()
            if title:numbered.append(title[:300])
            continue
        if stripped.startswith(("- ","* ","+ ")):
            title=stripped[2:].strip()
            if title:bullets.append(title[:300])
        elif stripped.startswith("#"):
            title=stripped.lstrip("#").strip()
            if title:headings.append(title[:300])
    # Numbered steps are a workflow. Never mix in section headings or rules.
    chosen=numbered if numbered else bullets if bullets else headings
    candidates=[];seen=set()
    for title in chosen:
        if title in seen:continue
        seen.add(title)
        lowered=title.lower()
        if lowered.startswith(("فحص","اختبر","تقييم","قارن","افحص","test ","verify ","evaluate ")):
            kind="evaluation"
        elif lowered.startswith(("تدريب","درّب","train ")):
            kind="training_proposal"
        elif lowered.startswith(("أنشئ بيانات","إعداد بيانات","جهز بيانات","create dataset")):
            kind="dataset_review"
        else:
            kind=next((k for k,keys in KEYWORDS.items() if any(x in lowered for x in keys)),"research")
        candidates.append({"goal":title,"kind":kind,"state":"proposed",
            "task_index":len(candidates),"source_document_sha256":digest,
            "requires_owner_approval":True,"code_execution_allowed":False,
            "automated_training_allowed":False})
        if len(candidates)>=100:break
    if not candidates:
        candidates=[{"goal":"Review this document and define measurable objectives",
            "kind":"evaluation","state":"proposed","task_index":0,
            "source_document_sha256":digest,"requires_owner_approval":True,
            "code_execution_allowed":False,"automated_training_allowed":False}]
    return candidates

def draft(root,digest):
    item=fetch(root,digest)
    candidates=_extract_tasks(item["content"],digest)
    return {"document_id":digest,"document_filename":item["document"]["filename"],
        "status":"draft_pending_approval",
        "extraction":"numbered_steps_first_deterministic_not_model_reasoning",
        "tasks":candidates,"automatic_execution":False,"automatic_training":False,
        "research_available":True,"created_at":time.time()}

def save_draft(root,digest):
    state=Path(root)/"learning_plans";state.mkdir(parents=True,exist_ok=True);os.chmod(state,0o700)
    target=state/(digest+".json")
    if target.is_file():
        result=json.loads(target.read_text())
        if result.get("document_id")!=digest:raise RuntimeError("Plan record mismatch")
        return result,False
    plan=draft(root,digest)
    fd=os.open(target,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,"w") as f:json.dump(plan,f,ensure_ascii=False,indent=2);f.flush();os.fsync(f.fileno())
    return plan,True

def read_plan(root,digest):
    if not ID_RE.fullmatch(digest):raise InvalidMarkdown("Invalid document ID")
    path=Path(root)/"learning_plans"/(digest+".json")
    if not path.is_file() or path.is_symlink():raise FileNotFoundError(digest)
    data=json.loads(path.read_text())
    if data.get("document_id")!=digest:raise RuntimeError("Plan mismatch")
    return data

def approve(root,digest,approved):
    if approved is not True:raise ValueError("Explicit approval required")
    plan=read_plan(root,digest)
    if plan["status"]=="approved_for_review_and_research":return plan
    if plan["status"]!="draft_pending_approval":raise ValueError("Plan not approvable")
    plan["status"]="approved_for_review_and_research"
    plan["approved_at"]=time.time()
    plan["automatic_training"]=False
    plan["automatic_execution"]=False
    target=Path(root)/"learning_plans"/(digest+".json")
    temp=target.with_suffix(".tmp")
    with temp.open("x") as f:json.dump(plan,f,ensure_ascii=False,indent=2);f.flush();os.fsync(f.fileno())
    os.replace(temp,target)
    return plan

def append_research(root,digest,result):
    plan=read_plan(root,digest)
    if plan["status"]!="approved_for_review_and_research":
        raise ValueError("Approve plan before attaching research")
    folder=Path(root)/"learning_plans"/digest
    folder.mkdir(mode=0o700,exist_ok=True)
    os.chmod(folder,0o700)
    raw=json.dumps(result,ensure_ascii=False,sort_keys=True).encode()
    h=hashlib.sha256(raw).hexdigest()
    target=folder/(h+".json")
    if not target.exists():
        fd=os.open(target,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,"wb") as f:f.write(raw);f.flush();os.fsync(f.fileno())
    return {"evidence_id":h,"plan_id":digest,"references":len(result.get("results",[])),
            "training_rights_verified":False,"dataset_ingested":False}


def list_evidence(root,digest):
    # For owner review only: read-only references to approved public-source queries.
    plan=read_plan(root,digest)
    if plan["status"]!="approved_for_review_and_research":
        return {"plan_id":digest,"evidence":[],"status":"plan_not_approved"}
    folder=Path(root)/"learning_plans"/digest
    results=[]
    if folder.is_dir() and not folder.is_symlink():
        paths=sorted(folder.glob("*.json"),key=lambda p:p.stat().st_mtime,reverse=True)
        for path in paths[:50]:
            if path.is_symlink() or not ID_RE.fullmatch(path.stem):continue
            try:
                raw=path.read_bytes()
                if hashlib.sha256(raw).hexdigest()!=path.stem:continue
                obj=json.loads(raw)
                if obj.get("external_data_for_training") is not False:continue
                results.append({"evidence_id":path.stem,"query":obj.get("query"),
                    "provider":obj.get("provider"),"retrieved_at":obj.get("retrieved_at"),
                    "results":obj.get("results",[])[:4],"dataset_ingested":False})
            except (OSError,ValueError,TypeError):continue
    return {"plan_id":digest,"evidence":results,"status":"read_only_research_evidence",
        "all_training_rights_unverified":True}
