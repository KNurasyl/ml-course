"""Real CPU PyTorch MLP, honest splits and fresh-process checkpoint verification."""
from pathlib import Path
import argparse, copy, json, subprocess, sys
import numpy as np
import torch
from torch import nn
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/M07'

def network():return nn.Sequential(nn.Linear(2,16),nn.ReLU(),nn.Linear(16,2))

def verify():
    cfg=json.loads((OUT/'release.json').read_text(encoding='utf-8'))
    model=network()
    model.load_state_dict(torch.load(OUT/'model.pt',map_location='cpu',weights_only=True))
    model.eval()
    x=(np.array(cfg['control_inputs'])-np.array(cfg['mean']))/np.array(cfg['scale'])
    with torch.inference_mode():actual=model(torch.tensor(x,dtype=torch.float32)).numpy()
    error=float(np.abs(actual-np.array(cfg['control_logits'])).max())
    assert error<1e-6,(error,'checkpoint or preprocessing differs')
    print(json.dumps({'fresh_process_max_abs_error':error}))
    return error

def main():
    global OUT
    parser=argparse.ArgumentParser();parser.add_argument('--verify',action='store_true')
    parser.add_argument('--seed',type=int,default=17);parser.add_argument('--noise',type=float,default=.20)
    parser.add_argument('--output-dir',type=Path,default=OUT)
    args=parser.parse_args()
    OUT=args.output_dir.resolve()
    torch.set_num_threads(1)
    if args.verify:verify();return
    torch.manual_seed(args.seed)
    x,y=make_moons(n_samples=1000,noise=args.noise,random_state=args.seed)
    indices=np.arange(len(y))
    train,rest=train_test_split(indices,test_size=.4,random_state=args.seed,stratify=y)
    valid,test=train_test_split(rest,test_size=.5,random_state=args.seed,stratify=y[rest])
    assert not set(train)&set(valid) and not set(train)&set(test) and not set(valid)&set(test)
    scaler=StandardScaler().fit(x[train])
    tx=torch.tensor(scaler.transform(x),dtype=torch.float32);ty=torch.tensor(y,dtype=torch.long)
    model=network();opt=torch.optim.Adam(model.parameters(),lr=.02);criterion=nn.CrossEntropyLoss()
    history=[];best=float('inf');checkpoint=None
    for epoch in range(240):
        model.train();opt.zero_grad(set_to_none=True)
        loss=criterion(model(tx[train]),ty[train]);assert torch.isfinite(loss)
        loss.backward();opt.step();model.eval()
        with torch.inference_mode():vl=float(criterion(model(tx[valid]),ty[valid]))
        history.append({'epoch':epoch+1,'train_loss':float(loss.detach()),'val_loss':vl})
        if vl<best:best=vl;checkpoint=copy.deepcopy(model.state_dict());best_epoch=epoch+1
    model.load_state_dict(checkpoint);model.eval()
    with torch.inference_mode():
        logits=model(tx[test]);pred=logits.argmax(dim=1);accuracy=float((pred==ty[test]).float().mean())
        controls=model(tx[test[:5]]).tolist()
    baseline=LogisticRegression().fit(tx[train].numpy(),y[train])
    base=float(baseline.score(tx[test].numpy(),y[test]))
    assert history[-1]['train_loss']<history[0]['train_loss']*.5
    assert accuracy>.8
    w=torch.tensor(1.,requires_grad=True);b=torch.tensor(1.,requires_grad=True)
    ((w*2+b-5)**2).backward();assert w.grad.item()==-8 and b.grad.item()==-4
    OUT.mkdir(parents=True,exist_ok=True)
    torch.save(model.state_dict(),OUT/'model.pt')
    cfg={'architecture':[2,16,2],'activation':'relu','seed':args.seed,'noise':args.noise,
         'mean':scaler.mean_.tolist(),'scale':scaler.scale_.tolist(),'control_inputs':x[test[:5]].tolist(),
         'control_logits':controls,'torch_version':torch.__version__}
    (OUT/'release.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    (OUT/'split.json').write_text(json.dumps({'train':train.tolist(),'validation':valid.tolist(),'test':test.tolist()}),encoding='utf-8')
    (OUT/'history.json').write_text(json.dumps(history),encoding='utf-8')
    check=subprocess.run([sys.executable,str(Path(__file__)), '--verify', '--output-dir',str(OUT)],capture_output=True,text=True,check=True)
    report={'train_n':len(train),'validation_n':len(valid),'test_n':len(test),'best_epoch':best_epoch,
            'mlp_test_accuracy':accuracy,'logistic_test_accuracy':base,
            'first_train_loss':history[0]['train_loss'],'last_train_loss':history[-1]['train_loss'],
            'fresh_process':json.loads(check.stdout),'gradient_w':w.grad.item(),'gradient_b':b.grad.item()}
    (OUT/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
