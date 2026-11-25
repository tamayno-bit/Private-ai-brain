"""Independent holdout evaluation: never evaluates using training examples' split.

No model promotion or accuracy claims. This only measures next-token loss
on deterministic disjoint-seeded formula generation.
"""
import json,math
from pathlib import Path

def evaluate(checkpoint,holdout_path,window_limit=40):
    import torch
    from brain.infer import load_model
    from brain.dataset import load_records,token_stream
    model,manifest=load_model(checkpoint,device='cuda' if torch.cuda.is_available() else 'cpu')
    records,hold_sha=load_records(holdout_path)
    train_sha=manifest['dataset_sha256']
    if train_sha==hold_sha:raise ValueError('Holdout must differ from training dataset')
    ids=token_stream(records)
    c=model.config.context
    if len(ids)<c+2:raise ValueError('insufficient heldout examples')
    dev=next(model.parameters()).device
    losses=[]
    with torch.inference_mode():
        for start in range(0,len(ids)-c-1,c):
            if len(losses)>=window_limit:break
            x=torch.tensor([ids[start:start+c]],dtype=torch.long,device=dev)
            y=torch.tensor([ids[start+1:start+c+1]],dtype=torch.long,device=dev)
            _,loss=model(x,y)
            val=float(loss)
            if not math.isfinite(val):raise ValueError('non-finite holdout loss')
            losses.append(val)
    if not losses:raise ValueError('no holdout windows')
    return {'independent_holdout_loss':sum(losses)/len(losses),
        'checkpoint_step':manifest['step'],'holdout_sha256':hold_sha,
        'train_sha256':train_sha,'windows':len(losses),
        'description':'Next-byte loss on independently generated arithmetic examples; not measured reasoning accuracy'}

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--holdout',required=True)
    args=p.parse_args()
    print(json.dumps(evaluate(args.checkpoint,args.holdout),indent=2))
