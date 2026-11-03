import numpy as np
def causal_attention(q,k,v,valid_keys):
    q,k,v=np.asarray(q,float),np.asarray(k,float),np.asarray(v,float)
    valid_keys=np.asarray(valid_keys,bool)
    if q.ndim!=3 or k.shape!=q.shape or v.shape[:2]!=q.shape[:2] or valid_keys.shape!=q.shape[:2]:raise ValueError('shape mismatch')
    if q.shape[-1]==0:raise ValueError('empty feature dimension')
    t=q.shape[1];allowed=np.tril(np.ones((t,t),bool))[None]&valid_keys[:,None,:]
    if not allowed.any(axis=-1).all():raise ValueError('query has no available key')
    scores=q@k.swapaxes(-1,-2)/np.sqrt(q.shape[-1]);scores=np.where(allowed,scores,-np.inf)
    weights=np.exp(scores-scores.max(-1,keepdims=True));weights/=weights.sum(-1,keepdims=True)
    return weights@v,weights
