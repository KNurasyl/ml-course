"""Small exact probability and Monte Carlo planning experiments (CPU)."""
from pathlib import Path
import argparse
import json
import math
import numpy as np
from scipy.stats import binom


def binomial_tail(n: int, minimum: int, p: float = .5) -> float:
    if type(n) is not int or type(minimum) is not int or n < 0 or not 0 <= p <= 1:
        raise ValueError("n/minimum must be integers; n >= 0; 0 <= p <= 1")
    return sum(math.comb(n, k) * p**k * (1-p)**(n-k)
               for k in range(max(0, minimum), n+1))


def rejection_threshold(n: int, p0: float, alpha: float) -> int:
    """Smallest count whose one-sided exact null tail is <= alpha."""
    if type(n) is not int or n <= 0 or not 0 < p0 < 1 or not 0 < alpha < 1:
        raise ValueError("invalid planning assumptions")
    return next((k for k in range(n+1) if binom.sf(k-1, n, p0) <= alpha), n+1)


def plan(p0=.6, p1=.7, sizes=(50, 100, 200, 400), alpha=.05,
         simulations=20000, seed=71):
    if not 0 <= p1 <= 1 or simulations < 1000:
        raise ValueError("invalid alternative or too few planning simulations")
    rng = np.random.default_rng(seed)
    rows = []
    for n in sizes:
        k = rejection_threshold(n, p0, alpha)
        counts = rng.binomial(n, p1, size=simulations)
        power = float(np.mean(counts >= k))
        exact_power = float(binom.sf(k-1, n, p1))
        rows.append({"n": int(n), "reject_at": k, "simulated_power": power,
                     "monte_carlo_se": math.sqrt(power*(1-power)/simulations),
                     "exact_power": exact_power,
                     "actual_alpha": float(binom.sf(k-1, n, p0))})
    return rows


def paired_plan(improve=.15, regress=.05, sizes=(50,100,200,400),
                alpha=.05, simulations=20000, seed=73):
    """Plan a one-sided paired sign test under explicit transition assumptions."""
    if min(improve,regress)<0 or improve+regress>1 or not 0<alpha<1:
        raise ValueError('invalid transition probabilities')
    rng=np.random.default_rng(seed);rows=[]
    for n in sizes:
        counts=rng.multinomial(n,[improve,regress,1-improve-regress],size=simulations)
        wins,losses=counts[:,0],counts[:,1];discordant=wins+losses
        tails=np.where(discordant>0,binom.sf(wins-1,discordant,.5),1.)
        power=float(np.mean(tails<=alpha))
        m=np.arange(n+1)
        thresholds=binom.isf(alpha,m,.5).astype(int)+1
        discordance=improve+regress
        exact=float(np.sum(binom.pmf(m,n,discordance)*
            binom.sf(thresholds-1,m,improve/discordance))) if discordance else 0.
        rows.append({'n':n,'simulated_power':power,
            'exact_power':exact,
            'monte_carlo_se':math.sqrt(power*(1-power)/simulations),
            'mean_discordant':float(discordant.mean()),
            'expected_effect':improve-regress})
    return rows


def run():
    rows = plan()
    assert binomial_tail(4, 3) == 5/16
    assert binomial_tail(5, 4) == 6/32
    assert all(r["actual_alpha"] <= .05 + 1e-12 for r in rows)
    assert all(abs(r["simulated_power"]-r["exact_power"]) < .02 for r in rows)
    result = {"assumptions": {"p0": .6, "p1": .7, "alpha": .05,
              "independent_binary_cases": True, "fixed_n": True,
              "baseline_probability_known": True}, "rows": rows,
              "chosen_n_for_power_0_8": next(r["n"] for r in rows if r["exact_power"] >= .8),
              "smaller_effect": plan(p1=.65, sizes=(100,200,400,800), seed=72),
              "paired_comparison":paired_plan(),
              "paired_null_check":paired_plan(improve=.1,regress=.1,seed=74),
              "not_measured_product_quality": True}
    out=Path(__file__).parent/'output';out.mkdir(exist_ok=True)
    (out/'planning.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
    return result


if __name__ == '__main__':
    run()
