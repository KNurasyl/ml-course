import math
def gradient(xs,ys,w,b):
    if not xs or len(xs)!=len(ys):raise ValueError('nonempty equal lengths required')
    if any(type(v) not in (float,int) or not math.isfinite(v) for v in [*xs,*ys,w,b]):
        raise ValueError('finite numeric values required')
    residual=[w*x+b-y for x,y in zip(xs,ys)]
    return 2*sum(e*x for e,x in zip(residual,xs))/len(xs),2*sum(residual)/len(xs)
