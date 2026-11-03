"""M04: paired bootstrap, Wilson intervals and sign-test assumptions."""
from pathlib import Path
from math import sqrt,comb
import json
import numpy as np
from scipy.stats import binomtest

def wilson(k,n,z=1.959963984540054):
    if type(n) is not int or n<=0 or type(k) is not int or not 0<=k<=n:
        raise ValueError('valid integer counts required')
    p=k/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den
    half=z*sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return center-half,center+half

def paired_bootstrap(a,b,seed=42,repeats=20000):
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
    if a.ndim!=1 or a.shape!=b.shape or len(a)<2:
        raise ValueError('at least two aligned pairs required')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('finite observations required')
    d=b-a;rng=np.random.default_rng(seed)
    distribution=d[rng.integers(0,len(d),size=(repeats,len(d)))].mean(1)
    return float(d.mean()),np.quantile(distribution,[.025,.975]),distribution

def run():
    A=np.array([1,0,1,0,1,0,1,0]);B=np.array([1,1,1,0,1,1,1,0])
    effect,interval,distribution=paired_bootstrap(A,B)
    differences=B-A;nonzero=differences[differences!=0]
    wins=int((nonzero>0).sum());n=len(nonzero)
    p=float(binomtest(wins,n,p=.5).pvalue)
    assert effect==.25 and np.allclose(interval,[0,.625]) and p==.5
    assert np.array_equal(distribution,paired_bootstrap(A,B)[2])
    intervals={}
    for k,n in [(9,10),(900,1000),(18,20),(180,200),(20,20),(0,10)]:
        actual=wilson(k,n);reference=binomtest(k,n).proportion_ci(method='wilson')
        assert np.allclose(actual,[reference.low,reference.high])
        intervals[f'{k}/{n}']=actual
    assert binomtest(9,10).pvalue==22/1024
    assert binomtest(7,8).pvalue==18/256
    result={'A_mean':float(A.mean()),'B_mean':float(B.mean()),'paired_effect':effect,'bootstrap_95_percentile':interval.tolist(),'independent_sessions':len(A),'bootstrap_repeats':20000,'non_ties':len(nonzero),'sign_test_p':p,'wilson_intervals':intervals}
    out=Path(__file__).parent/'output';out.mkdir(exist_ok=True)
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
    return result

if __name__=='__main__':run()
