"""Check M04-ex02 (sample summary, Wilson interval): fixed, hidden, random and property cases."""

import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from exercises.checklib import close, hidden_cases, main, rejected, run_property  # noqa: E402


def scipy_wilson(k, n):
    from scipy.stats import binomtest

    ci = binomtest(k, n).proportion_ci(0.95, method="wilson")
    return ci.low, ci.high


def check(m, seed):
    s = math.sqrt(5 / 3)
    close(m.describe([1, 2, 3, 4]), (2.5, 5 / 3, s, s / 2), "описание выборки: среднее, s², s, SE", [1, 2, 3, 4])
    close(m.wilson_interval(9, 10), scipy_wilson(9, 10), "интервал Уилсона для 9/10", (9, 10))
    close(m.wilson_interval(900, 1000), scipy_wilson(900, 1000), "интервал Уилсона для 900/1000", (900, 1000))
    close(m.wilson_interval(0, 25), scipy_wilson(0, 25), "интервал Уилсона у границы 0/25", (0, 25), atol=1e-12)
    for case in hidden_cases("M04", "ex02"):
        close(m.describe(case["values"]), case["summary"], "описание выборки, скрытый случай", "(скрыт)")
        close(
            m.wilson_interval(case["k"], case["n"], case["z"]),
            case["interval"],
            "интервал Уилсона, скрытый случай",
            "(скрыт)",
            atol=1e-12,
        )

    rng = np.random.default_rng(seed)
    for _ in range(20):
        x = rng.normal(rng.uniform(-20, 20), rng.uniform(0.5, 10), size=int(rng.integers(2, 30))).round(3)
        expected = (x.mean(), x.var(ddof=1), x.std(ddof=1), x.std(ddof=1) / np.sqrt(len(x)))
        close(m.describe(x.tolist()), expected, "описание выборки против NumPy (ddof=1)", x.tolist())
        n = int(rng.integers(1, 500))
        k = int(rng.integers(0, n + 1))
        close(m.wilson_interval(k, n), scipy_wilson(k, n), "интервал Уилсона против SciPy", (k, n), atol=1e-12)

    from hypothesis import given, strategies as st

    number = st.floats(-100, 100, allow_nan=False, allow_infinity=False)

    @given(st.lists(number, min_size=2, max_size=30), number, st.floats(-10, 10))
    def shift_and_scale(values, shift, scale):
        mean, s2, std, se = m.describe(values)
        close(se, std / math.sqrt(len(values)), "SE равна s/√n", values, atol=1e-12)
        shifted = m.describe([v + shift for v in values])
        close(shifted[0], mean + shift, "сдвиг данных сдвигает среднее", (values, shift), rtol=1e-6, atol=1e-6)
        close(shifted[1], s2, "сдвиг данных не меняет дисперсию", (values, shift), rtol=1e-6, atol=1e-6)
        scaled = m.describe([scale * v for v in values])[2]
        close(scaled, abs(scale) * std, "s умножается на |c|", (values, scale), rtol=1e-6, atol=1e-6)

    @given(st.integers(1, 2000).flatmap(lambda n: st.tuples(st.just(n), st.integers(0, n))))
    def wilson_bounds(case):
        n, k = case
        low, high = m.wilson_interval(k, n)
        if not (0 <= low <= k / n + 1e-12 and k / n - 1e-12 <= high <= 1):
            raise AssertionError(
                f"должно быть 0 ≤ low ≤ k/n ≤ high ≤ 1: вход k={k}, n={n}; ожидалось выполнение; получено ({low}, {high})"
            )
        mirror = m.wilson_interval(n - k, n)[1]
        close(low, 1 - mirror, "интервал симметричен: low(k) = 1 − high(n − k)", (k, n), atol=1e-9)
        low4, high4 = m.wilson_interval(4 * k, 4 * n)
        if not high4 - low4 < high - low:
            raise AssertionError(
                f"при вчетверо большем n интервал уже: вход k={k}, n={n}; "
                f"ожидалась ширина меньше {high - low}; получено {high4 - low4}"
            )
        if (k == 0 and not high > 0) or (k == n and not low < 1):
            raise AssertionError(
                f"у границы интервал не вырождается: вход k={k}, n={n}; "
                f"ожидалась ненулевая ширина; получено ({low}, {high})"
            )

    run_property(shift_and_scale, seed)
    run_property(wilson_bounds, seed)

    rejected(lambda: m.describe([3.0]), "describe с одним значением", "[3.0]")
    rejected(lambda: m.describe([1.0, float("nan")]), "describe с NaN", "[1.0, nan]")
    rejected(lambda: m.wilson_interval(11, 10), "wilson_interval с k > n", "k = 11, n = 10")
    rejected(lambda: m.wilson_interval(-1, 10), "wilson_interval с k < 0", "k = −1")
    rejected(lambda: m.wilson_interval(3, 0), "wilson_interval с n = 0", "n = 0")
    rejected(lambda: m.wilson_interval(2.5, 10), "wilson_interval с дробным k", "k = 2.5")
    rejected(lambda: m.wilson_interval(3, 10, 0), "wilson_interval с z = 0", "z = 0")
    return [
        "fixed_summary",
        "fixed_wilson",
        "hidden_cases",
        "random_vs_numpy_scipy",
        "shift_scale_property",
        "wilson_bounds_property",
        "invalid_input",
    ]


if __name__ == "__main__":
    raise SystemExit(main(check))
