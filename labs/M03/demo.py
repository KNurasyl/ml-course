"""M03: analytic gradients, independent finite differences, and L2 optimum."""
from pathlib import Path
import json
import numpy as np

def objective(theta,x,y,lam=0.):
    w,b=theta
    return float(np.mean((w*x+b-y)**2)+lam*w*w)

def gradient(theta,x,y,lam=0.):
    w,b=theta;e=w*x+b-y
    return np.array([2*np.mean(e*x)+2*lam*w,2*e.mean()])

def fit(x,y,lam=0.,steps=200,eta=.1):
    theta=np.zeros(2);history=[]
    for _ in range(steps):
        history.append(objective(theta,x,y,lam))
        theta-=eta*gradient(theta,x,y,lam)
        assert np.isfinite(theta).all()
    history.append(objective(theta,x,y,lam))
    return theta,np.array(history)

def run():
    x=np.arange(-2,3,dtype=float);y=3*x+1
    point=np.array([.7,-.3]);h=1e-5;numeric=[]
    for direction in np.eye(2):
        numeric.append((objective(point+h*direction,x,y,.5)-objective(point-h*direction,x,y,.5))/(2*h))
    analytic=gradient(point,x,y,.5)
    assert np.allclose(analytic,numeric,atol=1e-8)
    assert np.allclose(gradient(point,np.r_[x,x],np.r_[y,y]),gradient(point,x,y))
    results={};out=Path(__file__).parent/'output';out.mkdir(exist_ok=True)
    for name,lam,expected in [('plain',0.,[3,1]),('l2',.5,[2.4,1])]:
        theta,history=fit(x,y,lam)
        assert np.allclose(theta,expected,atol=1e-8)
        assert np.all(np.diff(history)<=1e-10)
        mse=objective(theta,x,y);penalty=lam*theta[0]**2
        results[name]={'weights':theta.tolist(),'mse':mse,'penalty':float(penalty),'objective':mse+penalty}
        np.savetxt(out/f'{name}_history.csv',np.c_[np.arange(len(history)),history],delimiter=',',header='step,objective',comments='')
    assert np.isclose(results['l2']['mse'],.72)
    result={'gradient_analytic':analytic.tolist(),'gradient_numeric':numeric,'runs':results,'steps':200,'eta':.1}
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
    return result

if __name__=='__main__':run()
