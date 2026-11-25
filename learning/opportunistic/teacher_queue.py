"""Quarantine external-model examples until rights, quality, and explicit approval."""
import json,hashlib,os,tempfile
from pathlib import Path
ALLOWED={'own','cc0','public-domain','cc-by-4.0','permitted-research'}
def stage(root,record):
    if not isinstance(record,dict) or any(not isinstance(record.get(k),str) or not record[k].strip() for k in ('source_model','prompt','response')):
        raise ValueError('source_model, prompt and response required')
    raw=json.dumps(record,sort_keys=True,ensure_ascii=False).encode()
    if len(raw)>150000:raise ValueError('example too large')
    digest=hashlib.sha256(raw).hexdigest()
    root=Path(root);root.mkdir(parents=True,exist_ok=True);os.chmod(root,0o700)
    target=root/(digest+'.json')
    if not target.exists():
        fd,tmp=tempfile.mkstemp(dir=root,prefix='.pending-')
        try:
            with os.fdopen(fd,'w') as f:json.dump({'id':digest,'status':'pending_review','record':record},f,ensure_ascii=False)
            os.replace(tmp,target)
        finally:
            if os.path.exists(tmp):os.unlink(tmp)
    os.chmod(target,0o600)
    return {'id':digest,'status':'pending_review'}
def export(root,ids,output):
    """No automatic approval: reviewed records require explicit status and rights."""
    from brain.dataset import validate_record
    rows=[]
    for key in ids:
        if not isinstance(key,str) or len(key)!=64 or any(ch not in '0123456789abcdef' for ch in key):raise ValueError('invalid ID')
        item=json.loads((Path(root)/(key+'.json')).read_text())
        r=item['record']
        if item['id']!=key or item['status']!='approved':raise ValueError('not approved')
        if r.get('license') not in ALLOWED or r.get('consent') is not True or r.get('quality_verified') is not True:raise ValueError('rights or quality unverified')
        if r.get('source_type')=='external_model' and r.get('training_permission_verified') is not True:raise ValueError('teacher output not licensed for training')
        rows.append(validate_record({'source':r['source_model'],'text':r['prompt']+'\n'+r['response'],'license':r['license'],'consent':True}))
    with Path(output).open('x',encoding='utf8') as out:
        for row in rows:out.write(json.dumps(row,ensure_ascii=False)+'\n')
    return len(rows)
