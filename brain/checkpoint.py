"""Atomic full resumable training checkpoints, with checksum manifest."""
import hashlib,json,os,tempfile
from pathlib import Path
import torch

def save_checkpoint(path,model,optimizer,step,metadata,rng_state=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    payload={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'step':step,'metadata':metadata,
             'rng_cpu':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
    fd,tmp=tempfile.mkstemp(prefix='.checkpoint-',suffix='.part',dir=str(path.parent));os.close(fd)
    try:
        torch.save(payload,tmp)
        with open(tmp,'rb') as f:os.fsync(f.fileno())
        digest=hashlib.sha256(Path(tmp).read_bytes()).hexdigest()
        os.replace(tmp,path)
        manifest={'sha256':digest,'step':step,'model_config':metadata.get('model_config'),
                  'dataset_sha256':metadata.get('dataset_sha256'),'tokenizer':'utf8-byte-v1',
                  'checkpoint_type':'full_train_resume'}
        manifest_path=path.with_suffix(path.suffix+'.json')
        fd,mtmp=tempfile.mkstemp(prefix='.manifest-',dir=str(path.parent))
        try:
            with os.fdopen(fd,'w') as out:json.dump(manifest,out,indent=2);out.flush();os.fsync(out.fileno())
            os.replace(mtmp,manifest_path)
        finally:
            if os.path.exists(mtmp):os.unlink(mtmp)
        return manifest
    finally:
        if os.path.exists(tmp):os.unlink(tmp)

def load_checkpoint(path,model,optimizer=None,expected_dataset=None,restore_rng=True):
    path=Path(path);manifest_path=path.with_suffix(path.suffix+'.json')
    manifest=json.loads(manifest_path.read_text())
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if digest!=manifest['sha256']:raise ValueError('Checkpoint checksum mismatch')
    if expected_dataset is not None and expected_dataset!=manifest['dataset_sha256']:
        raise ValueError('Dataset changed since checkpoint; explicit new run required')
    # weights_only loads tensors/primitives, not arbitrary Python classes.
    payload=torch.load(path,map_location='cpu',weights_only=True)
    model.load_state_dict(payload['model'],strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(payload['optimizer'])
        # After CPU map_location, ensure optimizer tensors are on parameter devices.
        for group in optimizer.param_groups:
            for param in group['params']:
                for key,value in optimizer.state[param].items():
                    if isinstance(value,torch.Tensor):
                        optimizer.state[param][key]=value.to(param.device)
    if restore_rng:
        torch.set_rng_state(payload['rng_cpu'])
        if torch.cuda.is_available() and payload.get('rng_cuda'):
            if len(payload['rng_cuda'])==torch.cuda.device_count():torch.cuda.set_rng_state_all(payload['rng_cuda'])
    return payload
