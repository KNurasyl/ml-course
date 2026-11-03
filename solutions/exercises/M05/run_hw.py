"""Full answer to independent M05 variants; run only after your attempt."""
from pathlib import Path
import argparse,json,sys,hashlib,subprocess
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import importlib.util
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss,confusion_matrix,balanced_accuracy_score
from labs.M05.pipeline import check_fit_rows,FEATURES
spec=importlib.util.spec_from_file_location('reference_m05',Path(__file__).with_name('solution.py'))
solution=importlib.util.module_from_spec(spec);spec.loader.exec_module(solution)


def run(variant='hw'):
    path=ROOT/f'exercises/M05/data/{variant}.csv'
    df=pd.read_csv(path,index_col='row_id')
    boundary=140 if variant=='retry' else 144
    fn_cost=4 if variant=='retry' else 2
    parts=solution.split_records(df,train_before=boundary)
    train,val,test=[df.loc[parts[k]] for k in ['train','validation','test']]
    candidates=[];models={}
    for C in [.1,1.,10.]:
        model=solution.build_pipeline(C).fit(train,train.is_ready)
        check_fit_rows(model,train.index)
        p=model.predict_proba(val)[:,1]
        threshold=solution.choose_threshold(val.is_ready,p,fn_cost=fn_cost)
        pred=p>=threshold;y=val.is_ready.to_numpy()
        cost=float((fn_cost*((y==1)&~pred).sum()+((y==0)&pred).sum())/len(y))
        candidates.append({'C':C,'threshold':threshold,'cost':cost,'log_loss':float(log_loss(y,p))})
        models[C]=model
    chosen=min(candidates,key=lambda r:(r['cost'],r['log_loss'],r['C'],r['threshold']))
    out=ROOT/f'artifacts/foundation-hw/M05/{variant}';out.mkdir(parents=True,exist_ok=True)
    frozen={'variant':variant,'seed_not_tuned':True,'fn_cost':fn_cost,'split':parts,
            'selected':chosen,'candidate_validation':candidates,'selection_data':'validation only',
            'dataset_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    (out/'selection-before-test.json').write_text(json.dumps(frozen,indent=2),encoding='utf-8')
    model=models[chosen['C']];p=model.predict_proba(test)[:,1];y=test.is_ready.to_numpy();pred=p>=chosen['threshold']
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    prior=float(train.is_ready.mean())
    slices={}
    for label,mask in [('known_track',test.track!='rust'),('unknown_track',test.track=='rust'),('experience_missing',test.experience_years.isna())]:
        mask=mask.to_numpy()
        slices[label]={'n':int(mask.sum()),'log_loss':float(log_loss(y[mask],p[mask],labels=[0,1])) if mask.any() else None}
    artifact={'pipeline':model,'threshold':chosen['threshold'],'features':FEATURES,'version':'1.1.0'}
    joblib.dump(artifact,out/'classifier.joblib')
    probe=test.iloc[:3].copy();probe.iloc[0,probe.columns.get_loc('track')]='new-language'
    probe.iloc[0,probe.columns.get_loc('experience_years')]=np.nan
    probe['review_score_after']=10000
    assert np.isfinite(model.predict_proba(probe)).all()
    changed=test.copy();changed.review_score_after=1-changed.review_score_after
    assert np.allclose(p,model.predict_proba(changed)[:,1])
    report={**frozen,'split_sizes':{k:len(v) for k,v in parts.items()},
            'test':{'log_loss':float(log_loss(y,p)),'baseline_log_loss':float(log_loss(y,np.full(len(y),prior))),
              'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
              'tn_fp_fn_tp':[int(tn),int(fp),int(fn),int(tp)],'cost':float((fn_cost*fn+fp)/len(y))},
            'slices':slices,'future_column_invariance':True,'unknown_missing_probe':True,
            'conclusion':'Synthetic results only; report both gains and costs; new policy needs a fresh check set.'}
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    test.to_csv(out/'test-input.csv')
    np.save(out/'probabilities.npy',p)
    restore_code="""from pathlib import Path
import sys,joblib,numpy as np,pandas as pd
p=Path(sys.argv[1]);saved=joblib.load(p/'classifier.joblib')
x=pd.read_csv(p/'test-input.csv',index_col='row_id')
assert np.allclose(saved['pipeline'].predict_proba(x)[:,1],np.load(p/'probabilities.npy'),atol=1e-12,rtol=0)
print('fresh_process_restore_passed')
"""
    restored=subprocess.run([sys.executable,'-c',restore_code,str(out)],cwd=ROOT,capture_output=True,text=True,timeout=60)
    if restored.returncode:raise AssertionError(restored.stderr)
    report['fresh_process_restore']=True
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='split'},indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--variant',choices=['hw','retry','delayed'],default='hw')
    run(parser.parse_args().variant)
