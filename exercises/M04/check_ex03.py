"""Check M04-ex03 (sign test, binary log loss): fixed, hidden, random and property cases."""

import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from exercises.checklib import close, hidden_cases, main, rejected, run_property  # noqa: E402


def check(m, seed):
    from scipy.stats import binomtest
    from sklearn.metrics import log_loss

    close(m.sign_test_p(9, 1), 22 / 1024, "двусторонний знаковый тест: 9 побед, 1 поражение", (9, 1))
    close(m.sign_test_p(8, 2), 112 / 1024, "двусторонний знаковый тест: 8 побед, 2 поражения", (8, 2))
    close(m.sign_test_p(9, 1, 5), 22 / 1024, "ничьи не учитываются: 9 побед, 1 поражение, 5 ничьих", (9, 1, 5))
    close(m.sign_test_p(5, 5), 1.0, "знаковый тест при равенстве: p-value не больше 1", (5, 5))
    lesson = -(math.log(0.8) + math.log(0.7)) / 2
    close(m.binary_log_loss([1, 0], [0.8, 0.3]), lesson, "средняя log loss для примера урока", "y=[1, 0], p=[0.8, 0.3]")
    close(m.binary_log_loss([1, 0], [1.0, 0.0]), 0.0, "log loss уверенных верных ответов", "y=[1, 0], p=[1.0, 0.0]")
    for case in hidden_cases("M04", "ex03"):
        close(m.sign_test_p(*case["counts"]), case["p_value"], "знаковый тест, скрытый случай", "(скрыт)")
        close(m.binary_log_loss(case["y"], case["p"]), case["loss"], "log loss, скрытый случай", "(скрыт)")

    rng = np.random.default_rng(seed)
    for _ in range(20):
        wins, losses, ties = (int(v) for v in rng.integers(0, 60, size=3))
        if wins + losses == 0:
            losses = 1
        expected = binomtest(wins, wins + losses, 0.5).pvalue
        close(m.sign_test_p(wins, losses, ties), expected, "знаковый тест против SciPy", (wins, losses, ties))
        size = int(rng.integers(1, 40))
        y, p = rng.integers(0, 2, size=size).tolist(), rng.uniform(0.01, 0.99, size=size).round(4).tolist()
        close(m.binary_log_loss(y, p), log_loss(y, p, labels=[0, 1]), "log loss против scikit-learn", f"y={y}, p={p}")

    from hypothesis import given, strategies as st

    counts = st.integers(0, 150)

    @given(counts, counts, counts)
    def symmetric_and_ignores_ties(wins, losses, ties):
        if wins + losses == 0:
            return
        value = m.sign_test_p(wins, losses, ties)
        close(m.sign_test_p(losses, wins), value, "знаковый тест симметричен по сторонам", (wins, losses, ties))
        close(m.sign_test_p(wins, losses), value, "ничьи не меняют p-value", (wins, losses, ties))
        if not 0 < value <= 1:
            raise AssertionError(
                f"p-value лежит в (0, 1]: вход {(wins, losses, ties)}; ожидалось 0 < p ≤ 1; получено {value}"
            )

    @given(st.integers(2, 150).flatmap(lambda n: st.tuples(st.just(n), st.integers((n + 1) // 2, n - 1))))
    def more_wins_smaller_p(case):
        n, wins = case
        before, after = m.sign_test_p(wins, n - wins), m.sign_test_p(wins + 1, n - wins - 1)
        if not after <= before + 1e-15:
            raise AssertionError(
                f"ещё одна победа не увеличивает p-value: вход n={n}, побед {wins}; "
                f"ожидалось не больше {before}; получено {after}"
            )

    probability = st.floats(0.001, 0.999)

    @given(st.lists(st.tuples(st.integers(0, 1), probability), min_size=1, max_size=30), probability, probability)
    def loss_properties(pairs, a, b):
        y, p = [label for label, _ in pairs], [prob for _, prob in pairs]
        value = m.binary_log_loss(y, p)
        swapped = m.binary_log_loss([1 - v for v in y], [1 - v for v in p])
        close(swapped, value, "замена классов местами не меняет потерю", f"y={y}, p={p}", rtol=1e-6)
        close(m.binary_log_loss(y, [0.5] * len(y)), math.log(2), "при p = 0.5 потеря равна ln 2", f"y={y}")
        low, high = sorted((a, b))
        # A gap of 1e-6 keeps float rounding from making two nearly equal losses look reversed.
        if high - low > 1e-6 and not m.binary_log_loss([1], [low]) > m.binary_log_loss([1], [high]):
            raise AssertionError(
                f"для y = 1 меньшая p даёт большую потерю: вход p={low} и p={high}; "
                "ожидалось убывание потери; получено обратное"
            )

    run_property(symmetric_and_ignores_ties, seed)
    run_property(more_wins_smaller_p, seed)
    run_property(loss_properties, seed)

    rejected(lambda: m.sign_test_p(0, 0, 4), "sign_test_p без нетождественных пар", "0 побед, 0 поражений")
    rejected(lambda: m.sign_test_p(-1, 3), "sign_test_p с отрицательным числом побед", "wins = −1")
    rejected(lambda: m.sign_test_p(2.5, 3), "sign_test_p с дробным числом побед", "wins = 2.5")
    rejected(lambda: m.sign_test_p(3, 2, -1), "sign_test_p с отрицательным числом ничьих", "ties = −1")
    rejected(lambda: m.binary_log_loss([], []), "binary_log_loss с пустыми списками", "y=[], p=[]")
    rejected(lambda: m.binary_log_loss([1, 0], [0.5]), "binary_log_loss со списками разной длины", "y=[1, 0], p=[0.5]")
    rejected(lambda: m.binary_log_loss([2], [0.5]), "binary_log_loss с меткой 2", "y=[2]")
    rejected(lambda: m.binary_log_loss([1], [1.5]), "binary_log_loss с вероятностью 1.5", "p=[1.5]")
    rejected(lambda: m.binary_log_loss([1], [0.0]), "ln(0) нельзя считать нулём: y = 1 при p = 0", "y=[1], p=[0.0]")
    rejected(lambda: m.binary_log_loss([0], [1.0]), "ln(0) нельзя считать нулём: y = 0 при p = 1", "y=[0], p=[1.0]")
    return [
        "fixed_sign_test",
        "fixed_log_loss",
        "hidden_cases",
        "random_vs_scipy_sklearn",
        "sign_test_symmetry_property",
        "sign_test_monotone_property",
        "log_loss_property",
        "invalid_input",
    ]


if __name__ == "__main__":
    raise SystemExit(main(check))
