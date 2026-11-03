from pathlib import Path
import sys,json,importlib.util
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location("reference",Path(__file__).with_name("solution.py"))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
import numpy as np
from scipy.stats import binomtest
from labs.M04.bridges import paired_plan
def compare(a,b,seed):
    d=np.asarray(b)-np.asarray(a);rng=np.random.default_rng(seed)
    means=d[rng.integers(0,len(d),size=(20000,len(d)))].mean(1)
    wins=int((d>0).sum());losses=int((d<0).sum())
    return {'a_mean':float(np.mean(a)),'b_mean':float(np.mean(b)),'differences':d.tolist(),'effect':float(d.mean()),'wins':wins,'losses':losses,'sign_two_sided_p':float(binomtest(wins,wins+losses).pvalue) if wins+losses else 1.,'bootstrap_95':np.quantile(means,[.025,.975]).tolist(),'seed':seed,'resamples_are_not_new_people':True}
report={'A':compare([1,0,1,0,1,0,0,1,0,1],[1,1,1,0,0,1,1,1,0,1],101),
        'B':compare([0,0,1,1,0,1],[1,0,0,1,1,1],102),
        'C':{'effect':2/12,'sign_two_sided_p':float(binomtest(4,6).pvalue)},
        'planning':paired_plan(improve=.15,regress=.05,seed=73)}
assert report['A']['effect']==.2 and report['A']['sign_two_sided_p']==.625
assert report['B']['effect']==1/6 and report['B']['sign_two_sided_p']==1
assert report['C']['sign_two_sided_p']==.6875

out=ROOT/"artifacts/foundation-hw"/Path(__file__).parent.name
out.mkdir(exist_ok=True,parents=True)
(out/"report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps(report,indent=2))
