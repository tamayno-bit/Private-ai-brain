"""Measure actual owner-defined holdout Q/A exact-match, not just language loss."""
import json,math,time
from pathlib import Path

def evaluate(checkpoint,holdout_path,limit=8,seed=42):
    import torch
    from brain.model import BrainTransformer,ModelConfig
    from brain.infer import load_model
    from brain.dataset import load_records
    from brain.tokenizer import ByteTokenizer
    rows,digest=load_records(holdout_path)
    if not 1<=limit<=32:raise ValueError("Invalid evaluation limit")
    if checkpoint is None:
        torch.manual_seed(seed)
        from learning.opportunistic.autolearn import DEFAULT_MODEL
        config=ModelConfig(**{**DEFAULT_MODEL,"context":128})
        model=BrainTransformer(config).to("cuda" if torch.cuda.is_available() else "cpu").eval()
        step=0
    else:
        model,manifest=load_model(checkpoint,"cuda" if torch.cuda.is_available() else "cpu")
        step=manifest["step"]
        if digest==manifest.get("dataset_sha256"):raise ValueError("Evaluation leak from training set")
    tok=ByteTokenizer()
    dev=next(model.parameters()).device
    successes=0;trials=0
    t0=time.monotonic()
    with torch.inference_mode():
        for row in rows[:limit]:
            text=row["text"]
            if not text.startswith("Question: ") or "\nAnswer: " not in text:raise ValueError("Unsupported evaluation record")
            question,answer=text.split("\nAnswer: ",1)
            prompt=question+"\nAnswer: "
            prefix=tok.encode(prompt,bos=True)[-model.config.context:]
            ids=torch.tensor([prefix],dtype=torch.long,device=dev)
            max_tokens=min(24,max(4,len(answer.encode())+1))
            generated=[]
            for _ in range(max_tokens):
                logits,_=model(ids[:,-model.config.context:])
                nxt=int(logits[0,-1].float().argmax())
                if nxt==257:break
                generated.append(nxt)
                ids=torch.cat([ids,torch.tensor([[nxt]],dtype=torch.long,device=dev)],dim=1)
            predicted=tok.decode(generated)
            if predicted.strip()==answer.strip():successes+=1
            trials+=1
    return {"metric":"owner_holdout_exact_answer_match","accuracy":successes/trials,
        "correct":successes,"total":trials,"checkpoint_step":step,
        "holdout_sha256":digest,"evaluated_at":time.time(),
        "elapsed_seconds":round(time.monotonic()-t0,3),
        "no_training_examples_used":True,
        "note":"Exact answer match on held-out owner examples. Low performance is expected for a tiny from-scratch model."}
