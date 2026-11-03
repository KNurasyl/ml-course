"""M02: exact cosine search with explicit contracts and relevance metrics."""
from pathlib import Path
import json
import numpy as np

def search(ids,documents,query,k):
    D=np.asarray(documents,dtype=float);q=np.asarray(query,dtype=float)
    if D.ndim!=2 or q.ndim!=1 or D.shape[1]!=q.shape[0]:
        raise ValueError('document/query dimension mismatch')
    if len(ids)!=len(D) or len(set(ids))!=len(ids):
        raise ValueError('unique IDs required for all documents')
    if type(k) is not int or not 1<=k<=len(D):
        raise ValueError('k outside corpus range')
    if not np.isfinite(D).all() or not np.isfinite(q).all():
        raise ValueError('nonfinite representation')
    norms=np.linalg.norm(D,axis=1);qn=np.linalg.norm(q)
    if (norms==0).any() or qn==0:
        raise ValueError('zero vector has no cosine direction')
    scores=(D/norms[:,None])@(q/qn)
    # Explicit tie rule: stable document identifier, independent of row order.
    order=sorted(range(len(D)),key=lambda i:(-scores[i],ids[i]))
    return [(ids[i],float(scores[i])) for i in order[:k]]

def recall(found,relevant):
    if not relevant:raise ValueError('recall undefined without relevant documents')
    return len(set(found)&set(relevant))/len(set(relevant))
