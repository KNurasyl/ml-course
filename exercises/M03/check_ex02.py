"""Check M03-ex02 (finite differences, chain rule): fixed, hidden, random and property cases."""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from exercises.checklib import close, hidden_cases, main, rejected, run_property  # noqa: E402


def polynomial(coefficients):
    return lambda w: sum(c * w**k for k, c in enumerate(coefficients))


def numeric_gradient(x, y, w, b, h=1e-6):
    loss = lambda w_, b_: (w_ * x + b_ - y) ** 2
    return ((loss(w + h, b) - loss(w - h, b)) / (2 * h), (loss(w, b + h) - loss(w, b - h)) / (2 * h))


def check(m, seed):
    square = lambda w: w * w
    close(m.forward_difference(square, 3, 0.1), 6.1, "прямая разность для w² при w = 3, h = 0.1", (3, 0.1))
    close(m.central_difference(square, 3, 0.1), 6.0, "центральная разность для w² при w = 3, h = 0.1", (3, 0.1))
    close(m.single_gradient(2, 1, 0.5, 1), (4.0, 2.0), "градиент одного примера", "x=2, y=1, w=0.5, b=1")
    for case in hidden_cases("M03", "ex02"):
        f = polynomial(case["coefficients"])
        close(
            m.central_difference(f, case["w"], case["h"]),
            case["central"],
            "центральная разность, скрытый случай",
            "(скрыт)",
            rtol=1e-8,
        )
        close(m.single_gradient(*case["xywb"]), case["gradient"], "градиент, скрытый случай", "(скрыт)")

    rng = np.random.default_rng(seed)
    for _ in range(20):
        x, y, w, b = (float(v) for v in rng.normal(0, 3, size=4).round(3))
        close(
            m.single_gradient(x, y, w, b),
            numeric_gradient(x, y, w, b),
            "градиент против конечной разности",
            (x, y, w, b),
            rtol=1e-5,
            atol=1e-5,
        )

    from hypothesis import given, strategies as st

    number = st.floats(-50, 50, allow_nan=False, allow_infinity=False)

    @given(number, number, number, number, st.floats(1e-3, 1.0))
    def exact_for_quadratic(a, b, c, w, h):
        f = lambda t: a * t * t + b * t + c
        close(
            m.central_difference(f, w, h),
            2 * a * w + b,
            "центральная разность точна для квадратичной функции",
            (a, b, c, w, h),
            rtol=1e-6,
            atol=1e-6,
        )

    @given(number, number, number)
    def zero_input_freezes_weight(y, w, b):
        close(m.single_gradient(0.0, y, w, b)[0], 0.0, "при x = 0 производная по w равна нулю", (0.0, y, w, b))

    run_property(exact_for_quadratic, seed)
    run_property(zero_input_freezes_weight, seed)

    rejected(lambda: m.central_difference(square, 1.0, 0.0), "central_difference с h = 0", "h = 0")
    rejected(lambda: m.forward_difference(square, 1.0, -0.1), "forward_difference с h < 0", "h = −0.1")
    rejected(lambda: m.central_difference(square, float("nan"), 0.1), "central_difference с w = NaN", "w = nan")
    rejected(lambda: m.single_gradient(1.0, float("inf"), 0.0, 0.0), "single_gradient с бесконечностью", "y = inf")
    return [
        "fixed_differences",
        "chain_rule_example",
        "hidden_cases",
        "random_vs_numeric",
        "quadratic_exactness_property",
        "zero_input_property",
        "invalid_input",
    ]


if __name__ == "__main__":
    raise SystemExit(main(check))
