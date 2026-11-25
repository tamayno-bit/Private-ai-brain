"""Generate immutable evaluation snapshot with zero exact-text overlap.

Uses independently seeded original arithmetic examples, subtracting every
training example before writing holdout. Never includes teacher output.
"""
import hashlib,json,os
from pathlib import Path
from .synthetic_curriculum import create

def create_disjoint(root,train_seed=604,eval_seed=605):
    root=Path(root)
    train,_=create(root,train_seed)
    candidate,_=create(root,eval_seed)
    train_text={json.loads(line)['text'] for line in train.read_text().splitlines() if line.strip()}
    source=[json.loads(line) for line in candidate.read_text().splitlines() if line.strip()]
    included=[item for item in source if item['text'] not in train_text]
    if len(included)<2500:raise ValueError('Too few independent holdout examples')
    if any(item['text'] in train_text for item in included):
        raise RuntimeError('Detected train holdout leakage')
    content=''.join(json.dumps(item,ensure_ascii=False,separators=(',',':'))+'\n' for item in included).encode()
    hashval=hashlib.sha256(content).hexdigest()
    path=root/f'arithmetic-holdout-disjoint-{eval_seed}-from-{train_seed}.jsonl'
    manifest=path.with_suffix('.manifest.json')
    meta={'source_seed':eval_seed,'training_seed':train_seed,'rows':len(included),
        'removed_overlapping_rows':len(source)-len(included),'sha256':hashval,
        'provenance':'self-generated and exact text disjoint from training snapshot',
        'model_weights_not_promoted':True}
    if path.exists() or manifest.exists():
        if not path.exists() or not manifest.exists() or path.read_bytes()!=content or json.loads(manifest.read_text())!=meta:
            raise RuntimeError('immutable holdout mismatch')
    else:
        for dest,data in [(path,content),(manifest,json.dumps(meta,indent=2).encode())]:
            fd=os.open(dest,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
            with os.fdopen(fd,'wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    return path,meta

if __name__=='__main__':
    p,m=create_disjoint('/workspace/private-ai-brain-state/datasets')
    print(json.dumps({'holdout_path':str(p),**m}))
