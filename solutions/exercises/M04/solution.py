from math import comb,isfinite
def binomial_tail(n,minimum,p=.5):
    if type(n) is not int or type(minimum) is not int or n<0 or not isfinite(p) or not 0<=p<=1:
        raise ValueError('invalid binomial parameters')
    return sum(comb(n,k)*p**k*(1-p)**(n-k) for k in range(max(0,minimum),n+1))
def choose_size(rows,required_power=.8):
    if not 0<required_power<=1:raise ValueError('invalid power target')
    candidates=[r['n'] for r in rows if r['exact_power']>=required_power]
    return min(candidates) if candidates else None
