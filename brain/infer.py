import argparse,json
from pathlib import Path
import torch
from .model import ModelConfig,BrainTransformer
from .checkpoint import load_checkpoint
from .tokenizer import ByteTokenizer

def load_model(checkpoint,device='cpu'):
    # Separate model loading from training optimizer for standalone inference.
    manifest=json.loads(Path(str(checkpoint)+'.json').read_text())
    cfg=ModelConfig(**manifest['model_config']).validate()
    model=BrainTransformer(cfg)
    load_checkpoint(checkpoint,model,restore_rng=False)
    return model.to(device).eval(),manifest

def generate(checkpoint,prompt,max_new_tokens=60,device='cpu'):
    if not (1<=len(prompt)<=10000):raise ValueError('Prompt length must be 1..10000')
    model,manifest=load_model(checkpoint,device)
    tok=ByteTokenizer();prefix=tok.encode(prompt,bos=True)
    # Preserve user prompt tail when it exceeds context.
    ids=torch.tensor([prefix[-model.config.context:]],dtype=torch.long,device=device)
    result=model.generate(ids,max_new_tokens=max_new_tokens)
    generated=result[0,ids.size(1):].tolist()
    return {'answer':tok.decode(generated),'model':'scratch-transformer','trained_step':manifest['step'],
            'used_external_model':False,'note':'Experimental model; no claim of general reasoning ability'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--prompt',required=True)
    p.add_argument('--tokens',type=int,default=50);p.add_argument('--device',default='cpu');a=p.parse_args()
    print(json.dumps(generate(a.checkpoint,a.prompt,a.tokens,a.device),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
