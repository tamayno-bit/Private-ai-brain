"""Real next-token pretraining; CPU/GPU, optional torchrun DDP, resumable checkpoints.

Only runs when explicitly invoked. Never provisions GPU or modifies Original Brain.
"""
import argparse,json,os,random,signal,time
from dataclasses import asdict
from pathlib import Path
import torch
import torch.distributed as dist
from .model import BrainTransformer,ModelConfig
from .dataset import load_records,token_stream,split_stream
from .checkpoint import save_checkpoint,load_checkpoint

_STOP=False
def stop_signal(signum,frame):
    global _STOP;_STOP=True

def setup(seed):
    world=int(os.environ.get('WORLD_SIZE','1'))
    local=int(os.environ.get('LOCAL_RANK','0'))
    if world>1:
        if not torch.cuda.is_available():raise RuntimeError('Multi-process DDP requires GPUs in this starter; no CPU multi-node guarantee')
        torch.cuda.set_device(local);dist.init_process_group('nccl')
    torch.manual_seed(seed);random.seed(seed)
    return world,local,torch.device(f'cuda:{local}' if torch.cuda.is_available() else 'cpu')

def sample_batch(stream,c,batch_size,device,generator):
    starts=torch.randint(0,len(stream)-c-1,(batch_size,),generator=generator)
    x=torch.tensor([stream[i:i+c] for i in starts.tolist()],device=device,dtype=torch.long)
    y=torch.tensor([stream[i+1:i+c+1] for i in starts.tolist()],device=device,dtype=torch.long)
    return x,y

def run(args):
    global _STOP;_STOP=False
    if args.steps<=0 or args.batch_size<=0 or args.grad_accum<=0 or args.save_every<=0:raise ValueError('Training controls must be positive')
    world,rank,device=setup(args.seed)
    signal.signal(signal.SIGTERM,stop_signal);signal.signal(signal.SIGINT,stop_signal)
    try:
        c=ModelConfig(**json.loads(Path(args.model_config).read_text())).validate()
        records,dataset_sha=load_records(args.data)
        train_ids,valid_ids=split_stream(token_stream(records),c.context)
        model=BrainTransformer(c).to(device)
        base_model=model
        optim=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=0.1)
        start=0
        gen=torch.Generator().manual_seed(args.seed+rank)
        if args.resume:
            # All ranks load same model weights; optimizer and RNG fully restored when possible.
            payload=load_checkpoint(args.resume,base_model,optim,expected_dataset=dataset_sha)
            if payload['metadata'].get('world_size') != world:raise ValueError('DDP world_size change requires explicit migration')
            if payload['metadata'].get('model_config') != asdict(c):raise ValueError('Architecture mismatch')
            start=int(payload['step'])
            saved_states=payload['metadata'].get('sampler_states')
            if saved_states:
                gen.set_state(saved_states[rank])
        if world>1:
            model=torch.nn.parallel.DistributedDataParallel(base_model,device_ids=[rank])
        out=Path(args.output)
        if rank==0:out.mkdir(parents=True,exist_ok=True)
        if world>1:dist.barrier()
        history=[]
        for step in range(start+1,args.steps+1):
            model.train();optim.zero_grad(set_to_none=True)
            loss_total=0.0
            for i in range(args.grad_accum):
                # DDP no_sync avoids redundant gradient all-reduce on microbatches.
                ctx=model.no_sync() if world>1 and i<args.grad_accum-1 else __import__('contextlib').nullcontext()
                with ctx:
                    x,y=sample_batch(train_ids,c.context,args.batch_size,device,gen)
                    _,loss=model(x,y);(loss/args.grad_accum).backward()
                loss_total+=float(loss.detach())/args.grad_accum
            torch.nn.utils.clip_grad_norm_(base_model.parameters(),1.0);optim.step()
            if step==1 or step%args.log_every==0:
                if rank==0:
                    print(json.dumps({'step':step,'train_loss':round(loss_total,5),'device':str(device),'distributed_world_size':world}),flush=True)
                    history.append({'step':step,'train_loss':loss_total})
            if step%args.save_every==0 or step==args.steps or _STOP:
                sampler_states=[None]*world
                if world>1:dist.all_gather_object(sampler_states,gen.get_state())
                else:sampler_states=[gen.get_state()]
                if rank==0:
                    base_model.eval()
                    with torch.no_grad():
                        x,y=sample_batch(valid_ids,c.context,min(args.batch_size,2),device,torch.Generator().manual_seed(987))
                        _,val_loss=base_model(x,y)
                    meta={'model_config':asdict(c),'dataset_sha256':dataset_sha,'world_size':world,
                          'sampler_states':sampler_states,'validation_loss_sample':float(val_loss),
                          'source':'scratch_random_init','weights_provenance':'no pretrained weights',
                          'training_recipe':'causal_lm_pretraining'}
                    manifest=save_checkpoint(out/'latest.pt',base_model,optim,step,meta)
                    with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps({'step':step,'train_loss':loss_total,'val_loss_sample':float(val_loss),'time':time.time()})+'\n')
                    print(json.dumps({'checkpoint':str(out/'latest.pt'),'step':step,'validation_loss_sample':round(float(val_loss),5)}),flush=True)
                if world>1:dist.barrier()
            if _STOP:break
        if rank==0:print(json.dumps({'finished_step':step,'stopped_by_signal':_STOP,'checkpoint':str(out/'latest.pt')}))
    finally:
        if world>1:dist.destroy_process_group()

def main():
    p=argparse.ArgumentParser(description='Explicit, provenance-gated scratch pretraining')
    p.add_argument('--model-config',default='config/model_tiny.json');p.add_argument('--data',required=True)
    p.add_argument('--output',default='state/training/demo');p.add_argument('--resume')
    p.add_argument('--steps',type=int,default=20);p.add_argument('--batch-size',type=int,default=2)
    p.add_argument('--grad-accum',type=int,default=1);p.add_argument('--lr',type=float,default=3e-4)
    p.add_argument('--save-every',type=int,default=10);p.add_argument('--log-every',type=int,default=5)
    p.add_argument('--seed',type=int,default=42)
    run(p.parse_args())
if __name__=='__main__':main()
