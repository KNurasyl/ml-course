import math
def prepare(train,test):
    def valid(v):return v is None or (type(v) in (int,float) and math.isfinite(v))
    if not all(valid(v) for v in [*train,*test]):raise ValueError('bad numeric value')
    known=[v for v in train if v is not None]
    if not known:raise ValueError('no observed train values')
    mean=sum(known)/len(known)
    return {'mean':mean,'train':[mean if v is None else v for v in train],
            'test':[mean if v is None else v for v in test]}
def deduplicate(rows):
    unique={}
    for row in rows:
        sid=row['session_id']
        if sid in unique and unique[sid]!=row:raise ValueError('conflicting duplicate')
        unique[sid]=dict(row)
    return list(unique.values())
