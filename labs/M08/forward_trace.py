"""Full pre-norm block trace with numerical invariants and an independent formula."""
from pathlib import Path
import json,math
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[2]

def manual_norm(x,eps=1e-5):
    return (x-x.mean(-1,keepdim=True))/torch.sqrt(x.var(-1,unbiased=False,keepdim=True)+eps)

def main():
    torch.set_num_threads(2);torch.manual_seed(29)
    x=torch.tensor([[[1.,3.,5.,7.],[2.,0.,4.,6.],[3.,1.,2.,8.]]])
    norm=nn.LayerNorm(4);n=norm(x)
    assert torch.allclose(n,manual_norm(x),atol=1e-6)
    # One attention head, identity Q/K/V projections to isolate the block operations.
    q,k,v=n,n,n;scores=q@k.transpose(-1,-2)/math.sqrt(4)
    mask=torch.triu(torch.ones(3,3,dtype=torch.bool),1)
    a=scores.masked_fill(mask,float('-inf')).softmax(-1)
    attention=a@v;residual=x+attention
    ff=nn.Sequential(nn.Linear(4,6),nn.GELU(),nn.Linear(6,4))
    normalized2=norm(residual);mlp=ff(normalized2);output=residual+mlp
    assert torch.allclose(a.sum(-1),torch.ones(1,3)) and torch.equal(a.masked_select(mask.unsqueeze(0)),torch.zeros(3))
    # MLP transforms each position independently; replacing only position 3 preserves outputs 1/2.
    perturbed=normalized2.detach().clone();perturbed[:,2]+=5
    assert torch.allclose(ff(perturbed)[:,:2],mlp[:,:2])
    tensors={'X':x,'LayerNorm1':n,'Q':q,'K':k,'V':v,'scores':scores,'attention_weights':a,
             'attention_output':attention,'residual1':residual,'LayerNorm2':normalized2,'MLP':mlp,'output':output}
    report={'steps':{name:{'shape':list(t.shape),'values':t.detach().tolist()} for name,t in tensors.items()},
            'normalization_axis':'last feature dimension, not sequence','mlp_hidden_width':6,
            'note':'One head and identity projections for inspection; learned projections/multiple heads follow the same shape rules.'}
    out=ROOT/'artifacts/M08';out.mkdir(parents=True,exist_ok=True)
    (out/'forward-trace.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({name:list(t.shape) for name,t in tensors.items()}))

if __name__=='__main__':main()
