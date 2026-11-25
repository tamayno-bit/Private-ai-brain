"""Opt-in supervised-training examples embedded in owner MD, not scraped web content.

Example format: fenced JSON object with type=private_brain_examples_v1,
train and holdout arrays. Every item has prompt, answer, license, consent.
No automatic licensing of internet text or model outputs.
"""
import hashlib,json,os,re,time
from pathlib import Path
from .md_plans import fetch,ID_RE

START=re.compile(r"^\s*"+chr(96)*3+r"(?:json|JSON)\s*$")
END=re.compile(r"^\s*"+chr(96)*3+r"\s*$")
ALLOWED_LICENSES={"own","cc0","public-domain","cc-by-4.0"}
SECRET=re.compile(r"(?i)(bearer\s+\S+|(?:api[_ -]?key|password|secret|token)\s*[:=]\s*\S+|-----BEGIN)")
class DatasetPlanError(ValueError):pass

def _examples(md):
    blocks=[];active=False;buffer=[]
    for line in md.splitlines():
        if not active and START.match(line):active=True;buffer=[];continue
        if active and END.match(line):
            blocks.append("\n".join(buffer));active=False;buffer=[];continue
        if active:buffer.append(line)
    for raw in blocks:
        try:data=json.loads(raw)
        except (ValueError,TypeError):continue
        if isinstance(data,dict) and data.get("type")=="private_brain_examples_v1":
            return data
    raise DatasetPlanError("MD needs fenced JSON private_brain_examples_v1 examples")

def _validate_rows(rows,tag):
    if not isinstance(rows,list) or not 5<=len(rows)<=500:
        raise DatasetPlanError("Insufficient or excessive "+tag+" records")
    seen=set();result=[]
    for item in rows:
        if not isinstance(item,dict) or set(item)!={"prompt","answer","license","consent","source_type"}:
            raise DatasetPlanError("Example must have exact documented fields")
        p=item.get("prompt");a=item.get("answer")
        if not isinstance(p,str) or not isinstance(a,str) or not p.strip() or not a.strip():
            raise DatasetPlanError("Empty example")
        if len(p.encode())>280 or len(a.encode())>280 or len(p.encode())+len(a.encode())>320:
            raise DatasetPlanError("Example too long for starter training context")
        if item.get("license") not in ALLOWED_LICENSES or item.get("consent") is not True:
            raise DatasetPlanError("Unverified provenance or permission")
        if item.get("source_type")!="owner_written":
            raise DatasetPlanError("External-model and copied web content require separate rights review")
        if SECRET.search(p+"\n"+a):raise DatasetPlanError("Possible credentials in examples")
        text="Question: "+p.strip()+"\nAnswer: "+a.strip()
        if text in seen:raise DatasetPlanError("Duplicate examples")
        seen.add(text)
        result.append({"text":text,"source":"owner-reviewed-md-"+tag,
            "license":item["license"],"consent":True})
    return result

def inspect(root,digest):
    content=fetch(root,digest)["content"]
    data=_examples(content)
    if set(data)!={"type","train","holdout","owner_rights_confirmed"}:
        raise DatasetPlanError("Unexpected training schema")
    if data.get("owner_rights_confirmed") is not True:
        raise DatasetPlanError("Owner provenance declaration required")
    train=_validate_rows(data["train"],"train")
    holdout=_validate_rows(data["holdout"],"holdout")
    if len(train)<12 or len(holdout)<8:raise DatasetPlanError("Need >=12 training and >=8 distinct holdout examples")
    if sum(len(x["text"].encode("utf8"))+2 for x in train)<1800:
        raise DatasetPlanError("Need more owner-written training material to form a disjoint internal validation split")
    if {x["text"] for x in train}&{x["text"] for x in holdout}:
        raise DatasetPlanError("Training examples leak into holdout")
    def payload(rows):
        return ("".join(json.dumps(row,ensure_ascii=False,separators=(",",":"))+"\n" for row in rows)).encode()
    tr=payload(train);ho=payload(holdout)
    return {"document_id":digest,"train_records":len(train),"holdout_records":len(holdout),
        "train_sha256":hashlib.sha256(tr).hexdigest(),"holdout_sha256":hashlib.sha256(ho).hexdigest(),
        "train_bytes":tr,"holdout_bytes":ho,
        "training_rights":"owner_written_and_declared","data_from_internet":False,
        "model_output_distilled":False,"heldout_exact_overlap":0}

def propose(root,digest):
    from .learning_plans import read_plan
    plan=read_plan(root,digest)
    if plan["status"]!="approved_for_review_and_research":
        raise DatasetPlanError("Approve plan before proposing dataset")
    result=inspect(root,digest)
    folder=Path(root)/"md_datasets"/digest
    folder.mkdir(mode=0o700,parents=True,exist_ok=True)
    os.chmod(folder,0o700)
    digest_pair=hashlib.sha256(result["train_bytes"]+result["holdout_bytes"]).hexdigest()
    files={"train":folder/(digest_pair+".train.jsonl"),"holdout":folder/(digest_pair+".holdout.jsonl")}
    for key,target in files.items():
        content=result[key+"_bytes"]
        if target.exists():
            if hashlib.sha256(target.read_bytes()).digest()!=hashlib.sha256(content).digest():
                raise DatasetPlanError("Immutable dataset collision")
        else:
            fd=os.open(target,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
            with os.fdopen(fd,"wb") as f:f.write(content);f.flush();os.fsync(f.fileno())
    manifest={"document_id":digest,"dataset_id":digest_pair,"train_records":result["train_records"],
        "holdout_records":result["holdout_records"],"train_sha256":result["train_sha256"],
        "holdout_sha256":result["holdout_sha256"],"train_file":str(files["train"]),
        "holdout_file":str(files["holdout"]),"rights":result["training_rights"],
        "source_type":"owner_written","web_content_used":False,
        "external_model_outputs_used":False,"status":"proposed_for_owner_review",
        "training_started":False,"created_at":time.time()}
    target=folder/(digest_pair+".manifest.json")
    if not target.exists():
        fd=os.open(target,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
        with os.fdopen(fd,"w") as f:json.dump(manifest,f,indent=2);f.flush();os.fsync(f.fileno())
    stored=json.loads(target.read_text())
    if stored["dataset_id"]!=digest_pair:raise DatasetPlanError("Manifest mismatch")
    return {k:v for k,v in stored.items() if not k.endswith("_file")}
