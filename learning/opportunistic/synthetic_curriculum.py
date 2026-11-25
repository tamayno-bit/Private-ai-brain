"""Versioned, original synthetic curriculum with independently verified labels.

This is a starter mathematics/code-format curriculum; it does not contain teacher
outputs or private chats and is NOT a substitute for licensed large-scale corpora.
"""
import hashlib,json,os,random
from pathlib import Path

SOURCE="original-synthetic-arithmetic-v1"
def examples(seed=604):
    r=random.Random(seed)
    seen=set()
    rows=[]
    attempts=0
    while len(rows)<3600 and attempts<120000:
        attempts+=1
        a=r.randrange(0,500);b=r.randrange(0,500);op=r.randrange(7)
        if op==0:
            statement=f"{a} + {b} = {a+b}."
        elif op==1:
            statement=f"{a} - {b} = {a-b}."
        elif op==2:
            a%=32;b%=32
            statement=f"{a} * {b} = {a*b}."
        elif op==3:
            b=1+b%40
            statement=f"{a} // {b} = {a//b}."
        elif op==4:
            statement=f"max({a}, {b}) = {max(a,b)}."
        elif op==5:
            statement=f"min({a}, {b}) = {min(a,b)}."
        else:
            statement=f"{a} > {b} is {str(a>b)}."
        if statement in seen:continue
        seen.add(statement)
        rows.append({"text":statement,"source":SOURCE,"license":"own","consent":True})
    if len(rows)<3600:raise RuntimeError("curriculum generation insufficient")
    r.shuffle(rows)
    return rows

def create(root,seed=604):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    path=root/("arithmetic-%s.jsonl"%seed)
    manifest=root/("arithmetic-%s.manifest.json"%seed)
    rows=examples(seed)
    content="".join(json.dumps(row,ensure_ascii=False,separators=(",",":"))+"\n" for row in rows).encode()
    checksum=hashlib.sha256(content).hexdigest()
    meta={"dataset":path.name,"sha256":checksum,"rows":len(rows),
          "generator":SOURCE,"seed":seed,"license":"own",
          "provenance":"Generated and independently computed by local deterministic Python arithmetic",
          "teacher_outputs":False,"private_user_data":False,
          "purpose":"Experimental scratch-model arithmetic pretraining only"}
    if path.exists() or manifest.exists():
        if not path.exists() or not manifest.exists():raise RuntimeError("incomplete immutable snapshot")
        if path.read_bytes()!=content or json.loads(manifest.read_text())!=meta:
            raise RuntimeError("immutable dataset mismatch; refusing overwrite")
        return path,meta
    fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    with os.fdopen(fd,"wb") as f:f.write(content);f.flush();os.fsync(f.fileno())
    fd=os.open(manifest,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    with os.fdopen(fd,"w") as f:json.dump(meta,f,indent=2);f.flush();os.fsync(f.fileno())
    return path,meta

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--root",default="/workspace/private-ai-brain-state/datasets")
    p.add_argument("--seed",type=int,default=604)
    a=p.parse_args()
    path,meta=create(a.root,a.seed)
    print(json.dumps({"file":str(path),**meta}))
