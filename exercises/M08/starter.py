"""Numpy arrays: Q/K [B,T,D], V [B,T,Dv], valid_keys [B,T] bool."""
def causal_attention(q,k,v,valid_keys):
    """Return (output, weights). Reject any row with no available key."""
    raise NotImplementedError('Implement scale, causal/padding masks, row softmax and AV')
