"""Check M03-ex03 (descent, stable step, L2 optimum): fixed, hidden, random and property cases."""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from exercises.checklib import close, hidden_cases, main, rejected, run_property  # noqa: E402


def quadratic(c, m):
    return lambda w: 2 * c * (w - m)


def check(m, seed):
    close(m.descend(quadratic(1, 3), 0, 0.25, 2), [0, 1.5, 2.25], "спуск по (w − 3)², η = 0.25, два шага", (0, 0.25, 2))
    close(m.descend(quadratic(1, 3), 0, 0.25, 0), [0], "ноль шагов возвращает только старт", (0, 0.25, 0))
    close(m.stable_step(1), 1.0, "граница шага для c = 1", 1)
    close(m.ridge_optimum(10, 4), 2.0, "оптимум (w − 10)² + 4w²", (10, 4))
    for case in hidden_cases("M03", "ex03"):
        c, center, w0, eta, steps = case["descend"]
        close(m.descend(quadratic(c, center), w0, eta, steps), case["path"], "траектория, скрытый случай", "(скрыт)")
        close(m.stable_step(case["c"]), case["bound"], "граница шага, скрытый случай", "(скрыт)")
        close(m.ridge_optimum(*case["ridge"]), case["optimum"], "оптимум с L2, скрытый случай", "(скрыт)")

    rng = np.random.default_rng(seed)
    for _ in range(20):
        c, a, lam = float(rng.uniform(0.1, 10)), float(rng.normal(0, 20)), float(rng.uniform(0, 5))
        close(m.stable_step(c), 1 / c, "граница шага, случайное c", c)
        close(m.ridge_optimum(a, lam), a / (1 + lam), "оптимум с L2, случайный случай", (a, lam))

    from hypothesis import given, strategies as st

    @given(st.floats(0.1, 10), st.floats(-20, 20), st.floats(-20, 20), st.floats(0.05, 0.95), st.floats(1.1, 3.0))
    def step_decides_convergence(c, center, w0, below, above):
        distance = lambda path: [abs(w - center) for w in path]
        if abs(w0 - center) < 1e-3:
            return
        stable = distance(m.descend(quadratic(c, center), w0, below / c, 6))
        unstable = distance(m.descend(quadratic(c, center), w0, above / c, 6))
        if not all(b < a or a == 0 for a, b in zip(stable, stable[1:])):  # η = 0.5/c reaches the minimum in one step
            raise AssertionError(
                f"при η ниже 1/c расстояние до минимума должно убывать: вход c={c}, η={below / c}; "
                f"ожидалось убывание; получено {stable}"
            )
        if not all(b > a for a, b in zip(unstable, unstable[1:])):
            raise AssertionError(
                f"при η выше 1/c расстояние до минимума должно расти: вход c={c}, η={above / c}; "
                f"ожидался рост; получено {unstable}"
            )

    @given(st.floats(-100, 100), st.floats(0, 50))
    def optimum_has_zero_gradient(a, lam):
        w = m.ridge_optimum(a, lam)
        close(2 * (w - a) + 2 * lam * w, 0.0, "в оптимуме градиент цели равен нулю", (a, lam), atol=1e-7)
        if abs(w) > abs(a) + 1e-9:
            raise AssertionError(f"штраф тянет вес к нулю: вход a={a}, λ={lam}; ожидалось |w| ≤ |a|; получено w={w}")

    run_property(step_decides_convergence, seed)
    run_property(optimum_has_zero_gradient, seed)

    grad = quadratic(1, 0)
    rejected(lambda: m.descend(grad, 1.0, 0.0, 3), "descend с η = 0", "η = 0")
    rejected(lambda: m.descend(grad, 1.0, -0.1, 3), "descend с η < 0", "η = −0.1")
    rejected(lambda: m.descend(grad, 1.0, 0.1, -1), "descend с отрицательным числом шагов", "steps = −1")
    rejected(lambda: m.descend(grad, 1.0, 0.1, 1.5), "descend с дробным числом шагов", "steps = 1.5")
    rejected(lambda: m.descend(grad, float("nan"), 0.1, 2), "descend со стартом NaN", "w0 = nan")
    rejected(lambda: m.stable_step(0), "stable_step с c = 0", "c = 0")
    rejected(lambda: m.ridge_optimum(1.0, -0.5), "ridge_optimum с λ < 0", "λ = −0.5")
    return [
        "fixed_trajectory",
        "fixed_bound_and_optimum",
        "hidden_cases",
        "random_vs_formula",
        "step_bound_property",
        "optimum_property",
        "invalid_input",
    ]


if __name__ == "__main__":
    raise SystemExit(main(check))
