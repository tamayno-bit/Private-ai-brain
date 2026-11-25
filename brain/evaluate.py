import argparse,json,math
from pathlib import Path
import torch
from .infer import load_model
from .dataset import load_records,token_stream,split_stream

def evaluate(checkpoint,dataset):
    model,manifest=load_model(checkpoint)
    records,sha=load_records(dataset)
    if sha!=manifest['dataset_sha256']:raise ValueError('Dataset mismatch; cannot claim matching split')
    _,ids=split_stream(token_stream(records),model.config.context)
    c=model.config.context; losses=[]
    with torch.no_grad():
        for start in range(0,len(ids)-c-1,c):
            x=torch.tensor([ids[start:start+c]]);y=torch.tensor([ids[start+1:start+c+1]])
            _,loss=model(x,y);losses.append(float(loss))
    if not losses:raise ValueError('Holdout too small')
    return {'validation_loss':sum(losses)/len(losses),'perplexity':math.exp(min(50,sum(losses)/len(losses))),
            'windows':len(losses),'checkpoint_step':manifest['step'],'benchmark':'held-out token prediction',
            'warning':'Perplexity on this split does not demonstrate useful chat, coding or reasoning.'}
def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--data',required=True);a=p.parse_args()
    print(json.dumps(evaluate(a.checkpoint,a.data),indent=2))
if __name__=='__main__':main()
