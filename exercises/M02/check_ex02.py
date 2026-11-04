"""Check M02-ex02 (matrix product, order of transformations): fixed, hidden, random and property cases."""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from exercises.checklib import close, equal, hidden_cases, main, rejected, run_property  # noqa: E402

ROTATE = [[0, -1], [1, 0]]  # 90° counter-clockwise
STRETCH = [[2, 0], [0, 1]]  # x twice as long


def check(m, seed):
    a, b = [[1, 2, 0], [0, 1, 3]], [[1, 0], [2, 1], [0, 4]]
    close(m.matmul(a, b), [[5, 2], [2, 13]], "matmul 2×3 на 3×2", (a, b))
    for case in hidden_cases("M02", "ex02"):
        close(m.matmul(case["a"], case["b"]), case["expected"], "matmul, скрытый случай", "(скрыт)")

    rng = np.random.default_rng(seed)
    for _ in range(20):
        n, k, p = (int(v) for v in rng.integers(1, 5, size=3))
        x, y = rng.integers(-5, 6, size=(n, k)).tolist(), rng.integers(-5, 6, size=(k, p)).tolist()
        close(m.matmul(x, y), (np.array(x) @ np.array(y)).tolist(), f"matmul {n}×{k} на {k}×{p}", (x, y))

    from hypothesis import given, strategies as st

    small = st.integers(-4, 4)

    @given(st.integers(1, 3), st.integers(1, 3), st.integers(1, 3), st.integers(1, 3), st.data())
    def associative(n, k, p, q, data):
        draw = lambda rows, cols: data.draw(
            st.lists(st.lists(small, min_size=cols, max_size=cols), min_size=rows, max_size=rows)
        )
        x, y, z = draw(n, k), draw(k, p), draw(p, q)
        close(m.matmul(m.matmul(x, y), z), m.matmul(x, m.matmul(y, z)), "свойство (XY)Z = X(YZ)", (x, y, z))

    run_property(associative, seed)

    x = [1, 0]
    equal(m.apply_in_order([ROTATE, STRETCH], x), [0, 1], "сначала поворот, затем растяжение", x)
    equal(m.apply_in_order([STRETCH, ROTATE], x), [0, 2], "сначала растяжение, затем поворот", x)
    close(m.apply_in_order([ROTATE, STRETCH], [3, 4]), [-8, 3], "порядок преобразований", [3, 4])

    rejected(lambda: m.matmul([[1, 2]], [[1, 2]]), "matmul с несовместимыми размерами", "1×2 на 1×2")
    rejected(lambda: m.matmul([[1, 2], [3]], [[1], [2]]), "matmul с рваными строками", "[[1, 2], [3]]")
    rejected(lambda: m.matmul([], [[1]]), "matmul с пустой матрицей", "[]")
    rejected(lambda: m.apply_in_order([], [1, 2]), "apply_in_order без матриц", "[]")
    rejected(lambda: m.apply_in_order([ROTATE], [1, 2, 3]), "apply_in_order с вектором другой длины", "[1, 2, 3]")
    return [
        "fixed_product",
        "hidden_cases",
        "random_vs_numpy",
        "associativity_property",
        "transform_order",
        "invalid_input",
    ]


if __name__ == "__main__":
    raise SystemExit(main(check))
