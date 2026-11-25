"""Original, randomly initialized decoder-only Transformer. No pretrained weights."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import math
import torch
from torch import nn
from torch.nn import functional as F

@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 260  # byte tokens [0..255], BOS=256, EOS=257, PAD=258, UNK=259
    context: int = 256
    dim: int = 256
    layers: int = 4
    heads: int = 8
    kv_heads: int = 4
    dropout: float = 0.0
    rope_base: float = 10000.0

    def validate(self):
        if not (260 <= self.vocab_size <= 2000000 and 8 <= self.context <= 131072):
            raise ValueError('Invalid vocabulary/context')
        if not (self.dim > 0 and self.layers > 0 and self.heads > 0 and self.kv_heads > 0):
            raise ValueError('Model dimensions must be positive')
        if self.dim % self.heads or self.heads % self.kv_heads or (self.dim // self.heads) % 2:
            raise ValueError('dim/heads, heads/kv_heads, and RoPE head dimension must be compatible')
        if not (0 <= self.dropout < 1): raise ValueError('Invalid dropout')
        return self

class RMSNorm(nn.Module):
    def __init__(self, n: int):
        super().__init__(); self.weight = nn.Parameter(torch.ones(n)); self.eps = 1e-6
    def forward(self, x):
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype) * self.weight

def rotary(q, k, base: float):
    # q (B, H, T, D), k (B, kvH, T, D)
    length, d = q.size(-2), q.size(-1)
    inv = 1.0 / (base ** (torch.arange(0, d, 2, device=q.device, dtype=torch.float32) / d))
    angle = torch.outer(torch.arange(length, device=q.device, dtype=torch.float32), inv)
    cos, sin = angle.cos()[None,None,:,:], angle.sin()[None,None,:,:]
    def apply(x):
        x1, x2 = x[..., ::2].float(), x[..., 1::2].float()
        return torch.stack((x1*cos-x2*sin, x1*sin+x2*cos),dim=-1).flatten(-2).to(x.dtype)
    return apply(q),apply(k)

class Attention(nn.Module):
    def __init__(self, c: ModelConfig):
        super().__init__(); self.c=c; self.hdim=c.dim//c.heads
        self.q=nn.Linear(c.dim,c.dim,bias=False)
        self.k=nn.Linear(c.dim,c.kv_heads*self.hdim,bias=False)
        self.v=nn.Linear(c.dim,c.kv_heads*self.hdim,bias=False)
        self.o=nn.Linear(c.dim,c.dim,bias=False)
    def forward(self,x):
        b,t,_=x.shape
        q=self.q(x).view(b,t,self.c.heads,self.hdim).transpose(1,2)
        k=self.k(x).view(b,t,self.c.kv_heads,self.hdim).transpose(1,2)
        v=self.v(x).view(b,t,self.c.kv_heads,self.hdim).transpose(1,2)
        q,k=rotary(q,k,self.c.rope_base)
        # PyTorch SDPA supports GQA; explicit repeat for CPU and older runtime compatibility.
        if self.c.heads!=self.c.kv_heads:
            factor=self.c.heads//self.c.kv_heads
            k=k.repeat_interleave(factor,dim=1);v=v.repeat_interleave(factor,dim=1)
        out=F.scaled_dot_product_attention(q,k,v,is_causal=True,dropout_p=self.c.dropout if self.training else 0.0)
        return self.o(out.transpose(1,2).contiguous().view(b,t,self.c.dim))

class Block(nn.Module):
    def __init__(self,c:ModelConfig):
        super().__init__();self.a_norm=RMSNorm(c.dim);self.attn=Attention(c);self.m_norm=RMSNorm(c.dim)
        hidden=int((8*c.dim/3+255)//256*256)
        self.gate=nn.Linear(c.dim,hidden,bias=False)
        self.up=nn.Linear(c.dim,hidden,bias=False)
        self.down=nn.Linear(hidden,c.dim,bias=False)
    def forward(self,x):
        x=x+self.attn(self.a_norm(x));y=self.m_norm(x)
        return x+self.down(F.silu(self.gate(y))*self.up(y))

class BrainTransformer(nn.Module):
    def __init__(self,c:ModelConfig):
        super().__init__(); self.config=c.validate(); self.emb=nn.Embedding(c.vocab_size,c.dim)
        self.blocks=nn.ModuleList(Block(c) for _ in range(c.layers));self.norm=RMSNorm(c.dim)
        self.lm_head=nn.Linear(c.dim,c.vocab_size,bias=False)
        self.lm_head.weight=self.emb.weight
        self.apply(self._init)
    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):nn.init.normal_(m.weight,mean=0,std=0.02)
        elif isinstance(m,nn.Embedding):nn.init.normal_(m.weight,mean=0,std=0.02)
    def forward(self,ids,targets=None):
        if ids.ndim!=2 or ids.size(1)>self.config.context:raise ValueError('Context limit exceeded')
        x=self.emb(ids)
        for layer in self.blocks:x=layer(x)
        logits=self.lm_head(self.norm(x))
        loss=F.cross_entropy(logits.reshape(-1,logits.size(-1)).float(),targets.reshape(-1),ignore_index=258) if targets is not None else None
        return logits,loss
    @torch.inference_mode()
    def generate(self,ids,max_new_tokens=32,temperature=0.8,top_k=40):
        self.eval()
        if temperature<=0 or max_new_tokens<0:raise ValueError('Invalid sampling parameters')
        for _ in range(max_new_tokens):
            logits,_=self(ids[:,-self.config.context:]);logits=logits[:,-1,:].float()/temperature
            if top_k:
                values,_=torch.topk(logits,k=min(int(top_k),logits.size(-1)))
                logits=logits.masked_fill(logits<values[:,-1,None],float('-inf'))
            nxt=torch.multinomial(torch.softmax(logits,dim=-1),1);ids=torch.cat((ids,nxt),dim=1)
            if bool((nxt==257).all()):break
        return ids
    def metadata(self):return {'config':asdict(self.config),'parameters':sum(p.numel() for p in self.parameters()),'source':'random_initialization','external_base_model':False}
