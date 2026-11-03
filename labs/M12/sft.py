"""Actual assistant-only SFT of adapters in a pinned pretrained causal Transformer.

Small English synthetic tutor-format task. No claim of Russian interview quality.
The 24 training steps are fixed before inspecting the independent check split.
"""
from pathlib import Path
import hashlib, importlib.util, json, time
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[2]
SYSTEM='You are an interview tutor. Answer with one short instruction. If the candidate is vague, ask them to name their own action. If an action and measurement are present, ask about the tradeoff.'

TRAIN=[
 ('I helped the team.','CLARIFY: Name your own action.'),
 ('We made things better.','CLARIFY: Name your own action.'),
 ('The product was successful.','CLARIFY: Name your own action.'),
 ('I worked on backend.','CLARIFY: Name your own action.'),
 ('We improved performance.','CLARIFY: Name your own action.'),
 ('The service is fast now.','CLARIFY: Name your own action.'),
 ('I added a cache and p95 fell from 200 to 90 ms.','NEXT: Explain one tradeoff.'),
 ('I added an index and measured reads at 20 ms instead of 80 ms.','NEXT: Explain one tradeoff.'),
 ('I batched requests and latency dropped from 90 to 40 ms.','NEXT: Explain one tradeoff.'),
 ('I replaced polling and CPU use dropped from 70 to 30 percent.','NEXT: Explain one tradeoff.'),
 ('I reduced allocations and memory fell from 500 to 300 MB.','NEXT: Explain one tradeoff.'),
 ('I moved work to a queue and request p95 fell by 40 percent.','NEXT: Explain one tradeoff.'),
]
CHECK=[
 ('My project went very well.','CLARIFY: Name your own action.'),
 ('Everyone liked our improvements.','CLARIFY: Name your own action.'),
 ('I compressed response bodies and transferred bytes fell from 900 to 200 KB.','NEXT: Explain one tradeoff.'),
 ('I pooled connections and measured p95 as 35 ms instead of 120 ms.','NEXT: Explain one tradeoff.'),
]
OLD=[('What is 2 plus 2?','4.'),('Name a programming language.','Python.')]

class LowRankLinear(nn.Module):
    def __init__(self,base,rank=4,alpha=8):
        super().__init__();self.base=base;self.rank=rank;self.scale=alpha/rank
        self.a=nn.Parameter(torch.empty(rank,base.in_features));nn.init.normal_(self.a,std=.02)
        self.b=nn.Parameter(torch.zeros(base.out_features,rank));self.base.requires_grad_(False)
    def forward(self,x):
        return self.base(x)+(x@self.a.T@self.b.T)*self.scale

def dialogue(tokenizer,prompt,answer=None,system=None):
    if system is None:system=SYSTEM
    messages=[{'role':'system','content':system},{'role':'user','content':prompt}]
    if answer is not None:messages.append({'role':'assistant','content':answer})
    return tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=answer is None)

def masked_example(tokenizer,prompt,answer,max_length=192,system=None):
    if system is None:system=SYSTEM
    prefix=dialogue(tokenizer,prompt,system=system)
    full=dialogue(tokenizer,prompt,answer,system=system)
    if full[:len(prefix)]!=prefix:raise ValueError('Template prefix changed; inspect token boundaries before masking')
    if len(full)>max_length:raise ValueError('Example too long; do not silently truncate the supervised response')
    labels=[-100]*len(prefix)+full[len(prefix):]
    assert any(x!=-100 for x in labels)
    return {'input_ids':torch.tensor([full]),'attention_mask':torch.ones(1,len(full),dtype=torch.long),
            'labels':torch.tensor([labels]),'prefix_length':len(prefix)}

def attach(model):
    model.requires_grad_(False);names=[]
    # Last four blocks: earlier frozen blocks need no gradient storage for their parameters.
    for index in range(len(model.model.layers)-4,len(model.model.layers)):
        layer=model.model.layers[index]
        for name in ['q_proj','v_proj']:
            original=getattr(layer.self_attn,name)
            setattr(layer.self_attn,name,LowRankLinear(original));names.append(f'model.layers.{index}.self_attn.{name}')
    return names

def fingerprint(model):
    digest=hashlib.sha256()
    for name,p in model.named_parameters():
        if name.endswith(('.a','.b')):continue
        digest.update(name.encode());digest.update(p.detach().contiguous().numpy().tobytes())
    return digest.hexdigest()

def mean_loss(model,examples):
    model.eval()
    with torch.inference_mode():
        return sum(float(model(**{k:v for k,v in row.items() if k!='prefix_length'},use_cache=False).loss) for row in examples)/len(examples)

def generate_rows(tokenizer,model,cases,system=None):
    if system is None:system=SYSTEM
    result=[];model.eval()
    for prompt,reference in cases:
        ids=torch.tensor([dialogue(tokenizer,prompt,system=system)])
        with torch.inference_mode():
            out=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),max_new_tokens=18,do_sample=False,pad_token_id=tokenizer.eos_token_id)
        result.append({'prompt':prompt,'reference':reference,'actual':tokenizer.decode(out[0,ids.shape[1]:],skip_special_tokens=True)})
    return result

def task_review(rows):
    """Narrow synthetic task rubric, not a generic judge: expected command is known."""
    wrong=[i for i,row in enumerate(rows) if row['actual'].strip().split(':',1)[0]!=row['reference'].split(':',1)[0]]
    exact=sum(row['actual'].strip()==row['reference'] for row in rows)
    return {'exact_match':exact/len(rows),'critical_wrong_command_indices':wrong,
            'scope':'Exact expected commands only; does not establish semantic quality on real interview answers.'}

def main(output_dir=None,steps=24,comparison_adapter=None):
    torch.set_num_threads(2);torch.manual_seed(41)
    spec=importlib.util.spec_from_file_location('sft_m09',ROOT/'labs/M09/demo.py')
    d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
    tokenizer,model,load_seconds=d.load_local();modules=attach(model)
    train=[masked_example(tokenizer,q,a) for q,a in TRAIN]
    check=[masked_example(tokenizer,q,a) for q,a in CHECK]
    old=[masked_example(tokenizer,q,a,system='You are a concise assistant.') for q,a in OLD]
    assert not set(q for q,_ in TRAIN)&set(q for q,_ in CHECK)
    model.eval()
    with torch.inference_mode():
        sample=train[0]
        automatic=model(**{k:v for k,v in sample.items() if k!='prefix_length'},use_cache=False)
        manual=torch.nn.functional.cross_entropy(automatic.logits[:,:-1].reshape(-1,automatic.logits.shape[-1]),sample['labels'][:,1:].reshape(-1),ignore_index=-100)
        shift_error=float((automatic.loss-manual).abs())
    assert shift_error<1e-6,'Labels/model next-token shift disagree'
    frozen_hash=fingerprint(model)
    before={'train_nll':mean_loss(model,train),'check_nll':mean_loss(model,check),'old_nll':mean_loss(model,old),
            'check_generations':generate_rows(tokenizer,model,CHECK),'old_generations':generate_rows(tokenizer,model,OLD,'You are a concise assistant.')}
    comparison=None
    if comparison_adapter:
        zero={name:p.detach().clone() for name,p in model.named_parameters() if p.requires_grad}
        saved=torch.load(comparison_adapter,weights_only=True)
        with torch.no_grad():
            for name,p in model.named_parameters():
                if name in saved:p.copy_(saved[name])
        comparison={'check_nll':mean_loss(model,check),'old_nll':mean_loss(model,old),
                    'check_generations':generate_rows(tokenizer,model,CHECK),'old_generations':generate_rows(tokenizer,model,OLD,'You are a concise assistant.')}
        with torch.no_grad():
            for name,p in model.named_parameters():
                if name in zero:p.copy_(zero[name])
    parameters=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(parameters,lr=.006,weight_decay=0.0);history=[];start=time.perf_counter()
    for step in range(steps):
        model.train();row=train[step%len(train)];opt.zero_grad(set_to_none=True)
        loss=model(**{k:v for k,v in row.items() if k!='prefix_length'},use_cache=False).loss
        assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(parameters,1.0);opt.step()
        history.append(float(loss.detach()))
    training_seconds=time.perf_counter()-start
    after={'train_nll':mean_loss(model,train),'check_nll':mean_loss(model,check),'old_nll':mean_loss(model,old),
           'check_generations':generate_rows(tokenizer,model,CHECK),'old_generations':generate_rows(tokenizer,model,OLD,'You are a concise assistant.')}
    assert fingerprint(model)==frozen_hash,'Frozen pretrained weights changed'
    assert after['train_nll']<before['train_nll'],'Training objective did not improve; investigate before declaring training verified'
    out=Path(output_dir) if output_dir else ROOT/'artifacts/M12/sft';out.mkdir(parents=True,exist_ok=True)
    adapters={name:p.detach().clone() for name,p in model.named_parameters() if p.requires_grad}
    torch.save(adapters,out/'adapter.pt')
    probe=check[0]['input_ids'];model.eval()
    with torch.inference_mode():expected=model(probe).logits.detach().clone()
    # Erase and restore only adapter tensors, proving that the saved artifact actually controls inference.
    with torch.no_grad():
        for p in parameters:p.zero_()
        erased=model(probe).logits.detach().clone()
    saved=torch.load(out/'adapter.pt',weights_only=True)
    with torch.no_grad():
        for name,p in model.named_parameters():
            if name in saved:p.copy_(saved[name])
        restored=model(probe).logits.detach().clone()
    restore_error=float((expected-restored).abs().max());erased_difference=float((expected-erased).abs().max())
    assert restore_error<1e-6 and erased_difference>1e-6
    # A conservative predeclared loss guard; a failed release is a valid experimental outcome.
    allowed=after['check_nll']<before['check_nll'] and after['old_nll']<=before['old_nll']*1.2
    for stage in [before,after]+([comparison] if comparison else []):stage['task_review']=task_review(stage['check_generations'])
    semantic_allowed=not after['task_review']['critical_wrong_command_indices']
    inspected=train[0];mask_tokens=[{'id':int(i),'token':tokenizer.convert_ids_to_tokens(int(i)),'label':int(y)} for i,y in zip(inspected['input_ids'][0],inspected['labels'][0])]
    report={'model':d.REPO,'revision':d.REVISION,'seed':41,'fixed_steps':steps,'rank':4,'alpha':8,'modules':modules,
            'trainable_parameters':sum(p.numel() for p in parameters),'total_parameters':sum(p.numel() for p in model.parameters()),
            'train_examples':len(train),'check_examples':len(check),'old_examples':len(old),'before':before,'after':after,
            'load_seconds':load_seconds,'training_seconds':training_seconds,'history':history,'base_sha256':frozen_hash,
            'base_unchanged':True,'restore_max_abs_error':restore_error,'erase_max_abs_difference':erased_difference,
            'release_allowed_by_loss_guard':allowed,'release_rule':'check NLL improves AND old NLL <= 1.2*before; human semantics review also required',
            'release_allowed':allowed and semantic_allowed,
            'semantic_gate':'No wrong CLARIFY/NEXT/LIMIT command on the fixed check; read content separately even if command passes.',
            'mask_inspection':mask_tokens,'manual_loss_shift_abs_error':shift_error,'comparison_adapter':comparison,
            'limits':'English synthetic format adaptation; no claim of Russian HR quality or statistically reliable generalization.'}
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'config.json').write_text(json.dumps({k:report[k] for k in ['model','revision','modules','rank','alpha']},indent=2),encoding='utf-8')
    (out/'data.json').write_text(json.dumps({'origin':'course-authored synthetic examples, no personal data','train':TRAIN,'check':CHECK,'old':OLD},indent=2),encoding='utf-8')
    torch.save({'ids':probe,'expected_logits':expected},out/'restore-probe.pt')
    print(json.dumps({k:report[k] for k in ['trainable_parameters','training_seconds','base_unchanged','restore_max_abs_error','release_allowed_by_loss_guard']}))

if __name__=='__main__':main()
