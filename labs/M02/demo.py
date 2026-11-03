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

def run():
    ids=['A','B','C','D'];D=[[1,0],[.8,.6],[0,1],[-1,0]]
    rows=[]
    for name,q,relevant in [('q1',[1,0],{'A'}),('q2',[0,1],{'B','C'})]:
        ranked=search(ids,D,q,2);found=[r[0] for r in ranked]
        rows.append({'query':name,'top2':ranked,'recall_at_1':recall(found[:1],relevant),'recall_at_2':recall(found,relevant)})
    assert [r[0] for r in rows[0]['top2']]==['A','B']
    assert [r[0] for r in rows[1]['top2']]==['C','B']
    assert [r['recall_at_1'] for r in rows]==[1,.5]
    invalids=[([0,0],2),([1,0,0],2),([1,0],0),([1,0],5),([float('nan'),0],1)]
    for q,k in invalids:
        try:search(ids,D,q,k)
        except ValueError:pass
        else:raise AssertionError('invalid search input accepted')
    tied=search(['z','a'],[[1,0],[1,0]],[1,0],2)
    assert [r[0] for r in tied]==['a','z']
    result={'queries':rows,'mean_recall_at_1':.75,'mean_recall_at_2':1.,'invalid_queries_rejected':len(invalids),'tie_rule':'identifier ascending'}
    out=Path(__file__).parent/'output';out.mkdir(exist_ok=True)
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
    return result

if __name__=='__main__':run()
