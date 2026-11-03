"""Tiny, real causal Transformer on explicitly synthetic grammar. CPU only."""
from pathlib import Path
import itertools, json, math
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[2]

class TinyLM(nn.Module):
    def __init__(self,vocab,max_length,dim=16):
        super().__init__()
        self.token=nn.Embedding(vocab,dim);self.position=nn.Embedding(max_length,dim)
        self.qkv=nn.Linear(dim,dim*3);self.out=nn.Linear(dim,dim)
        self.norm1=nn.LayerNorm(dim);self.norm2=nn.LayerNorm(dim)
        self.ff=nn.Sequential(nn.Linear(dim,dim*2),nn.GELU(),nn.Linear(dim*2,dim))
        self.head=nn.Linear(dim,vocab);self.dim=dim
    def forward(self,ids):
        t=ids.shape[1];h=self.token(ids)+self.position(torch.arange(t))
        q,k,v=self.qkv(self.norm1(h)).chunk(3,dim=-1)
        scores=q@k.transpose(-2,-1)/math.sqrt(self.dim)
        blocked=torch.triu(torch.ones(t,t,dtype=torch.bool),diagonal=1)
        weights=scores.masked_fill(blocked,float('-inf')).softmax(-1)
        h=h+self.out(weights@v);h=h+self.ff(self.norm2(h))
        return self.head(h)

def main():
    torch.set_num_threads(1);torch.manual_seed(19)
    sentences=[f'{a} {b} {c}.' for a,b,c in itertools.product(['red','blue','green'],['cat','dog','fox'],['runs','sits','jumps'])]
    # Complete sentences are split BEFORE token arrays: no overlapping train/valid windows.
    train=[s for i,s in enumerate(sentences) if i%5!=0]
    valid=[s for i,s in enumerate(sentences) if i%5==0]
    chars=['<PAD>']+sorted(set(''.join(train)+'\n'))
    assert set(''.join(valid))<=set(chars)
    stoi={s:i for i,s in enumerate(chars)};max_len=max(map(len,sentences))+1
    def encode(rows):
        full=torch.zeros(len(rows),max_len,dtype=torch.long)
        for i,s in enumerate(rows):full[i,:len(s)+1]=torch.tensor([stoi[c] for c in s+'\n'])
        return full[:,:-1],full[:,1:]
    x,y=encode(train);vx,vy=encode(valid)
    model=TinyLM(len(chars),max_len);opt=torch.optim.AdamW(model.parameters(),lr=.008,weight_decay=.01)
    criterion=nn.CrossEntropyLoss(ignore_index=0)
    def loss_on(a,b):return criterion(model(a).reshape(-1,len(chars)),b.reshape(-1))
    model.eval()
    before=x[:2].clone();after=before.clone();after[:,-1]=(after[:,-1]+1)%len(chars)
    with torch.inference_mode():
        prefix_error=float((model(before)[:,:-1]-model(after)[:,:-1]).abs().max())
        initial=float(loss_on(vx,vy))
    assert prefix_error<1e-6
    counts=torch.bincount(y[y!=0],minlength=len(chars)).float()+1
    counts[0]=0;probs=counts/counts.sum()
    unigram=float(-probs[vy[vy!=0]].log().mean())
    history=[]
    for step in range(300):
        model.train();opt.zero_grad(set_to_none=True);loss=loss_on(x,y)
        assert torch.isfinite(loss);loss.backward();opt.step()
        if step%20==0:
            model.eval()
            with torch.inference_mode():vl=float(loss_on(vx,vy))
            history.append({'step':step,'train_loss':float(loss.detach()),'validation_loss':vl})
    model.eval()
    with torch.inference_mode():final=float(loss_on(vx,vy));train_loss=float(loss_on(x,y))
    assert final<initial and final<unigram
    samples=[]
    for prompt in ['red ','blue ','green ']:
        ids=torch.tensor([[stoi[c] for c in prompt]])
        for _ in range(max_len-len(prompt)-1):
            with torch.inference_mode():nxt=int(model(ids)[0,-1].argmax())
            if nxt==0 or chars[nxt]=='\n':break
            ids=torch.cat([ids,torch.tensor([[nxt]])],1)
        samples.append(''.join(chars[int(i)] for i in ids[0]))
    out=ROOT/'artifacts/M08';out.mkdir(parents=True,exist_ok=True)
    report={'train_sentences':len(train),'validation_sentences':len(valid),'vocab_size':len(chars),
            'initial_validation_nll':initial,'final_validation_nll':final,'train_nll':train_loss,
            'unigram_validation_nll':unigram,'validation_perplexity':math.exp(final),
            'causal_prefix_max_abs_error':prefix_error,'generations':samples,
            'scope':'synthetic combinations only; not general language or interview quality'}
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (out/'history.json').write_text(json.dumps(history),encoding='utf-8')
    (out/'corpus.json').write_text(json.dumps({'train':train,'validation':valid,'vocabulary':chars}),encoding='utf-8')
    torch.save(model.state_dict(),out/'model.pt')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
