import math
def predict(x,w,b):
    if any(type(v) not in (int,float) or not math.isfinite(v) for v in (x,w,b)):
        raise ValueError('finite numbers required')
    return w*x+b
