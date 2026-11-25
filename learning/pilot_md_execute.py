"""Persistent, auditable owner-approved pilot workflow (no auto training/installation)."""
from pathlib import Path
import hashlib,json,os,time,urllib.request
from brain.md_plans import fetch
from brain.learning_plans import read_plan,append_research,list_evidence
from brain.research import search

PLAN="cd91980d44debbac64dbefbcf10b1a7d8ce7500184dfda92cd93e5349e3ee67b"
STATE=Path("/workspace/private-ai-brain-state")
DIR=STATE/"md_plan_executions"
FILE=DIR/(PLAN+".json")
DIR.mkdir(mode=0o700,parents=True,exist_ok=True)

def write(data):
    tmp=FILE.with_suffix(".tmp")
    with tmp.open("w",encoding="utf8") as handle:
        json.dump(data,handle,ensure_ascii=False,indent=2);handle.flush();os.fsync(handle.fileno())
    os.replace(tmp,FILE)
def read():
    if FILE.exists():return json.loads(FILE.read_text())
    return {"plan_id":PLAN,"plan_status":"approved_for_review_and_research",
        "scenario":"owner-requested stepwise pilot without training",
        "created_at":time.time(),"tasks":[],"training_started":False,
        "live_weights_modified":False,"all_web_sources_training_rights_verified":False}
def save_step(index,status,**kwargs):
    data=read();records=data["tasks"]
    entry={"task_index":index,"status":status,"updated_at":time.time(),**kwargs}
    for i,row in enumerate(records):
        if row["task_index"]==index:records[i]=entry;break
    else:records.append(entry)
    records.sort(key=lambda x:x["task_index"])
    data["updated_at"]=time.time()
    write(data)
    print("STEP",index+1,status,"DETAILS",str(kwargs.get("summary",""))[:140],flush=True)

plan=read_plan(STATE,PLAN)
doc=fetch(STATE,PLAN)
assert plan["status"]=="approved_for_review_and_research" and len(plan["tasks"])==8
assert doc["document"]["sha256"]==PLAN
for i,row in enumerate(plan["tasks"]):assert row["task_index"]==i

# STEP 1: existing real Original Brain + Tongyi research job, preserve the exact outcome.
jobs=list((STATE/"md_research_jobs").glob("*.json"))
upstream=[json.loads(p.read_text()) for p in jobs if p.is_file()]
matches=[x for x in upstream if x.get("plan_id")==PLAN and x.get("task_index")==0 and x.get("phase")=="ready"]
assert matches,"Python first-stage research must have completed"
j=sorted(matches,key=lambda x:x.get("created_at",0))[-1]
save_step(0,"completed_with_limitations",kind="original_brain_tongyi",
    sources=j["sources"],source_count=j["source_count"],job_id=j["id"],
    model_inference_used=j["model_inference_used"],citations_valid_ids=j["citations_valid_ids"],
    summary="Python documentation sources retrieved; Tongyi answer notes insufficient direct function/exception detail",
    training_started=False)

# STEP 2: exact PyPI metadata, never execute/install.
r=search("torch","pypi")
assert r["status"]=="ok" and len(r["results"])>=1
r["task_index"]=1
e=append_research(STATE,PLAN,r)
save_step(1,"completed",kind="pypi_metadata",sources=r["results"],source_count=len(r["results"]),
    evidence_id=e["evidence_id"],summary="Public PyPI package metadata verified, no installation",training_started=False)

# STEP 3: GitHub metadata; select project matching pytorch/pytorch, not an arbitrary star result.
r=search("pytorch","github")
verified=[x for x in r["results"] if x.get("url")=="https://github.com/pytorch/pytorch"]
assert verified,"Official PyTorch repository not in live GitHub search"
r["results"]=verified;r["task_index"]=2
e=append_research(STATE,PLAN,r)
save_step(2,"completed",kind="github_repository_metadata",sources=verified,
    evidence_id=e["evidence_id"],summary="Official PyTorch public repository metadata; no code cloned/executed",
    training_started=False)

# STEP 4: official arXiv landing page, distinguished from unrelated similarly titled Crossref items.
u="https://arxiv.org/abs/1706.03762"
req=urllib.request.Request(u,headers={"User-Agent":"PrivateAIBrainResearch/0.1"})
with urllib.request.urlopen(req,timeout=8) as response:
    if response.status!=200 or response.url!=u:raise ValueError("arXiv source verification failed")
    raw=response.read(240000)
    if len(raw)==240000 or b"Attention Is All You Need" not in raw:
        raise ValueError("arXiv paper identity not verified")
source={"title":"Attention Is All You Need (Vaswani et al., 2017)",
        "url":u,"source":"arXiv official abstract page","arxiv_id":"1706.03762",
        "doi":None,"doi_status":"not_verified_from_this_primary_source",
        "training_rights_verified":False,"role":"reference_only"}
r={"provider":"arxiv_landing_page","query":"Attention Is All You Need 1706.03762",
    "results":[source],"retrieved_at":time.time(),"status":"ok",
    "external_data_for_training":False,"automatic_installation":False,"task_index":3}
e=append_research(STATE,PLAN,r)
save_step(3,"completed_with_limitations",kind="primary_paper_metadata",
    sources=[source],evidence_id=e["evidence_id"],
    summary="Original arXiv paper identity verified; DOI not verified; no training-rights assumption",
    training_started=False)

# STEP 5: compare only source-supported fields and separately mark copyright rights unknown.
prior=read()["tasks"]
source_count=sum(x.get("source_count",len(x.get("sources",[]))) for x in prior[:4])
summary={"verified_public_reference_count":source_count,
    "python_functions_exceptions_detail":"not_directly_verified_from_collected_snippets",
    "pypi_version":prior[1]["sources"][0].get("version"),
    "pypi_license_declared":prior[1]["sources"][0].get("license_declared"),
    "github_official":prior[2]["sources"][0]["url"]=="https://github.com/pytorch/pytorch",
    "github_license_declared":prior[2]["sources"][0].get("license_declared"),
    "arxiv_paper_identifier":"1706.03762","doi_status":"not_confirmed",
    "training_rights_verified":False,"web_text_imported_to_training":False,
    "limitations":["PyPI and GitHub license metadata are declarations, not training permission",
        "arXiv paper metadata does not establish text reuse rights",
        "Python code skill not established by source discovery"]}
save_step(4,"completed",kind="documented_comparison",comparison=summary,
    summary="Verified source metadata compared; training licenses still unapproved",
    training_started=False)

# STEP 6: learning-data method, not a dataset approval.
spec={"source_policy":"owner-written examples only until licensing checked",
      "topics":["functions","parameters","return values","exceptions","bug fixing"],
      "splits":{"train":"owner-authored examples","holdout":"disjoint owner-authored test prompts"},
      "deduplication":"hash normalized questions and solutions; ensure disjoint holdout",
      "license_check":"every record requires provenance and owner confirmation",
      "acceptance":"user approval before converting proposed examples into a GPU training job",
      "dataset_created":False,"train_records":0,"holdout_records":0}
save_step(5,"proposal_ready_needs_owner_data",kind="dataset_design",
    dataset_proposal=spec,summary="Safe Python curriculum method drafted; no approved training examples in uploaded MD",
    training_started=False)

# STEP 7: real baseline from scratch checkpoint; no execution of untrusted generated code.
# This is a diagnostic only: an independent practical code benchmark is not yet implemented.
from brain.infer import generate
ckpt=STATE/"training"/"latest.pt"
prompts=["Write Python function add(a,b):","Fix the Python error: def f(x) print(x)"]
observations=[]
for q in prompts:
    result=generate(ckpt,q,max_new_tokens=40,device="cpu")
    answer=result["answer"]
    observations.append({"question":q,"generated_text":answer[:120],
        "looks_like_python_code":("def " in answer and ("return" in answer or "print(" in answer)),
        "tests_executed":False,"functional_score":None,
        "checkpoint_step":result["trained_step"]})
save_step(6,"partial_baseline_only",kind="scratch_checkpoint_baseline",
    observations=observations,summary="Checkpoint generated two diagnostic outputs; safe execution benchmark not yet established",
    training_started=False,capability_gain_verified=False)

# STEP 8: evidence-backed final assessment (no fake training success).
data=read()
done=[x for x in data["tasks"] if x["task_index"]<7]
complete=sum(x["status"]=="completed" for x in done)
limited=sum(x["status"].startswith("completed_with") for x in done)
result={"complete_steps_without_qualification":complete,
    "research_steps_completed_with_limitations":limited,
    "dataset_proposal_created":True,"dataset_approved":False,"actual_training_started":False,
    "functional_training_gain_proven":False,
    "next_work":"Obtain verified owner-written examples, independent holdout and explicit training approval"}
save_step(7,"completed",kind="final_evidence_report",conclusion=result,
    summary="Final report distinguishes research and proposals from unperformed training")
print("PILOT_FINAL",json.dumps({"states":[r["status"] for r in read()["tasks"]],
    "training_started":False,"checkpoint_unchanged":True},ensure_ascii=False),flush=True)
