"""Fresh-process adapter restore: validates base revision and target names before inference."""
from pathlib import Path
import argparse,importlib.util,json,torch

def main():
    p=argparse.ArgumentParser();p.add_argument('checkpoint',type=Path,nargs='?',default=Path(__file__).resolve().parents[2]/'artifacts/M12/sft');args=p.parse_args()
    root=Path(__file__).resolve().parents[2]
    def load(name,path):
        spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
    sft=load('restore_sft_impl',Path(__file__).with_name('sft.py'));d=load('restore_m09',root/'labs/M09/demo.py')
    config=json.loads((args.checkpoint/'config.json').read_text(encoding='utf-8'))
    if config['model']!=d.REPO or config['revision']!=d.REVISION or config['rank']!=4 or config['alpha']!=8:raise ValueError('Incompatible adapter/base configuration')
    tok,model,_=d.load_local();names=sft.attach(model)
    if names!=config['modules']:raise ValueError('Target module mismatch')
    saved=torch.load(args.checkpoint/'adapter.pt',weights_only=True)
    expected_names={name for name,p in model.named_parameters() if p.requires_grad}
    if set(saved)!=expected_names:raise ValueError('Missing/extra adapter keys')
    with torch.no_grad():
        for name,param in model.named_parameters():
            if name in saved:
                if param.shape!=saved[name].shape:raise ValueError('Shape mismatch')
                param.copy_(saved[name])
    model.eval();probe=torch.load(args.checkpoint/'restore-probe.pt',weights_only=True)
    with torch.inference_mode():actual=model(probe['ids']).logits
    error=float((actual-probe['expected_logits']).abs().max());assert error<1e-6
    result={'fresh_process':True,'max_abs_error':error,'keys_checked':len(saved),'revision':d.REVISION}
    (args.checkpoint/'fresh-restore.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))

if __name__=='__main__':main()
