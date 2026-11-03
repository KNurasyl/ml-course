from pathlib import Path
import sys,json,importlib.util
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location("reference",Path(__file__).with_name("solution.py"))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
import numpy as np
def train(xs,ys,penalty):
    w=b=0.;history=[]
    for step in range(400):
        dw,db=m.gradient(xs,ys,w,b);dw+=2*penalty*w
        w,b=w-.08*dw,b-.08*db
        mse=sum((w*x+b-y)**2 for x,y in zip(xs,ys))/len(xs)
        history.append({'step':step+1,'w':w,'b':b,'mse':mse,'penalty':penalty*w*w,'objective':mse+penalty*w*w})
    return {'w':w,'b':b,'final':history[-1],'history':history}
x=[-1,0,2];y=[4,1,-5]
report={'A_no_penalty':train(x,y,0),'A_penalty':train(x,y,.5),'B':train(x,[2*v-1 for v in x],1)}
def mse(w,b):return sum((w*value+b-target)**2 for value,target in zip(x,y))/len(x)
w0,b0,h=.7,-.2,1e-5
analytic=m.gradient(x,y,w0,b0)
numeric=((mse(w0+h,b0)-mse(w0-h,b0))/(2*h),(mse(w0,b0+h)-mse(w0,b0-h))/(2*h))
assert np.allclose(analytic,numeric,atol=1e-7,rtol=0)
report['gradient_check']={'point':[w0,b0],'analytic':list(analytic),'central_difference':list(numeric),'h':h}
assert np.allclose([report['A_no_penalty']['w'],report['A_no_penalty']['b']],[-3,1],atol=1e-8)
assert np.allclose([report['A_penalty']['w'],report['A_penalty']['b']],[-84/37,28/37],atol=1e-8)
assert np.allclose([report['B']['w'],report['B']['b']],[28/23,-17/23],atol=1e-8)
assert np.allclose(m.gradient(x,y,.7,-.2),m.gradient(x+x,y+y,.7,-.2))
report['C_duplicate_gradient_unchanged']=True

out=ROOT/"artifacts/foundation-hw"/Path(__file__).parent.name
out.mkdir(exist_ok=True,parents=True)
(out/"report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps(report,indent=2))
