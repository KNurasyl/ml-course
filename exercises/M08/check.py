from pathlib import Path
import argparse,importlib.util,json,numpy as np

def run_checks(module):
    q=np.array([[[1.,0.],[0.,1.],[1.,1.]],[[0.,1.],[1.,0.],[1.,1.]]]);k=q.copy()
    v=np.array([[[1.],[3.],[9.]],[[2.],[4.],[100.]]]);valid=np.array([[1,1,1],[1,1,0]],bool)
    out,a=module.causal_attention(q,k,v,valid)
    assert out.shape==(2,3,1) and a.shape==(2,3,3),'Wrong shapes'
    assert np.allclose(a.sum(-1),1),'Normalize along key axis'
    assert np.all(a[:,np.triu_indices(3,1)[0],np.triu_indices(3,1)[1]]==0),'Future key leaked'
    assert np.all(a[1,:,2]==0),'Padding key leaked'
    assert np.allclose(out[:,0,0],[1,2]),'First causal query sees only first key'
    changed=v.copy();changed[:,2]+=1000
    later,_=module.causal_attention(q,k,changed,valid)
    assert np.allclose(out[:,:2],later[:,:2]) and np.allclose(out[1],later[1]),'Future/padding values changed protected outputs'
    assert not np.allclose(out[0,2],later[0,2]),'Real available value has no effect'
    try:module.causal_attention(q,k,v,np.zeros((2,3),bool))
    except ValueError:pass
    else:raise AssertionError('All-masked row must be rejected, not NaN or uniform')
    return {'checks':8,'passed':True,'mastery':'explain each mask and new sequence'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('submission',type=Path);a=p.parse_args()
    s=importlib.util.spec_from_file_location('student',a.submission);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    print(json.dumps(run_checks(m)))
