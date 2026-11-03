from pathlib import Path
import sys,json,importlib.util
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location("reference",Path(__file__).with_name("solution.py"))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def experiment(xs,ys,candidates,new_x):
    rows=[]
    for w,b in candidates:
        predictions=[m.predict(x,w,b) for x in xs]
        mse=sum((p-y)**2 for p,y in zip(predictions,ys))/len(ys)
        rows.append({'w':w,'b':b,'predictions':predictions,'mse':mse})
    chosen=min(rows,key=lambda r:r['mse'])
    return {'candidates':rows,'model':{'w':chosen['w'],'b':chosen['b'],'version':'1.1.0-A','input_unit':'minutes'},'new_prediction':m.predict(new_x,chosen['w'],chosen['b'])}
report={'A':experiment([-2,0,3],[5,1,-5],[(-2,1),(-1,0)],4),
        'B':experiment([-4,0,2],[-4,-2,-1],[(.5,-2),(1,-1)],4),
        'C':{'physical_input_minutes':2,'same_input_seconds':120,'prediction_old':m.predict(2,-2,1),'prediction_new_preprocessed':m.predict(120/60,-2,1),'prediction_new_weight':m.predict(120,-2/60,1)}}
assert report['A']['new_prediction']==-7
assert report['A']['candidates'][1]['mse']==14/3
assert report['B']['candidates'][1]['mse']==2
assert len(set(report['C'][k] for k in ['prediction_old','prediction_new_preprocessed','prediction_new_weight']))==1

out=ROOT/"artifacts/foundation-hw"/Path(__file__).parent.name
out.mkdir(exist_ok=True,parents=True)
(out/"report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps(report,indent=2))
