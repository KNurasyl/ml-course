"""Check M02-ex03 (standardisation and nearest neighbour): fixed, hidden, random and property cases."""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from exercises.checklib import close, equal, hidden_cases, main, rejected, run_property  # noqa: E402


def scaled_nearest(m, query, rows):
    means, stds = m.fit_scale(rows)
    return m.nearest(m.apply_scale([query], means, stds)[0], m.apply_scale(rows, means, stds))


def check(m, seed):
    rows = [[30, 1], [5, 4], [10, 2]]
    means, stds = m.fit_scale(rows)
    close(means, [15, 7 / 3], "средние столбцов", rows)
    close(stds, np.std(np.array(rows, dtype=float), axis=0).tolist(), "стандартные отклонения (делим на n)", rows)
    equal(m.nearest([0, 0], rows), 1, "ближайший без масштабирования", ([0, 0], rows))
    equal(m.nearest([0, 0], [[1, 0], [0, 1]]), 0, "равенство расстояний: меньший индекс", [[1, 0], [0, 1]])
    for case in hidden_cases("M02", "ex03"):
        close(m.fit_scale(case["rows"])[1], case["stds"], "стандартные отклонения, скрытый случай", "(скрыт)")
        equal(
            scaled_nearest(m, case["query"], case["rows"]),
            case["nearest"],
            "ближайший после масштабирования",
            "(скрыт)",
        )

    rng = np.random.default_rng(seed)
    for _ in range(20):
        table = rng.normal(0, rng.uniform(0.5, 50), size=(int(rng.integers(3, 8)), 3)).round(3)
        expected_means, expected_stds = table.mean(axis=0), table.std(axis=0)
        means, stds = m.fit_scale(table.tolist())
        close(means, expected_means.tolist(), "средние, случайная таблица", table.tolist(), rtol=1e-7)
        close(stds, expected_stds.tolist(), "стандартные отклонения, случайная таблица", table.tolist(), rtol=1e-7)

    from hypothesis import assume, given, strategies as st

    value = st.floats(-1e3, 1e3, allow_nan=False, allow_infinity=False)

    @given(st.lists(st.lists(value, min_size=2, max_size=2), min_size=3, max_size=8))
    def standardized(table):
        assume(np.std(np.array(table), axis=0).min() > 1e-2)
        means, stds = m.fit_scale(table)
        result = np.array(m.apply_scale(table, means, stds))
        close(result.mean(axis=0).tolist(), [0, 0], "после масштабирования среднее столбца 0", table, atol=1e-7)
        close(result.std(axis=0).tolist(), [1, 1], "после масштабирования разброс столбца 1", table, atol=1e-7)

    @given(st.lists(st.lists(value, min_size=2, max_size=2), min_size=3, max_size=6), st.floats(0.01, 1000))
    def unit_free(table, factor):
        assume(np.std(np.array(table), axis=0).min() > 1e-1)
        query = [float(np.mean([r[0] for r in table])), float(np.median([r[1] for r in table]))]
        distances = np.linalg.norm((np.array(table) - np.array(query)) / np.std(np.array(table), axis=0), axis=1)
        assume(np.sort(distances)[1] - np.sort(distances)[0] > 1e-6)  # no near-tie that rounding could flip
        stretched = [[r[0] * factor, r[1]] for r in table]
        equal(
            scaled_nearest(m, [query[0] * factor, query[1]], stretched),
            scaled_nearest(m, query, table),
            "после масштабирования ближайший не зависит от единиц признака",
            (table, factor),
        )

    run_property(standardized, seed)
    run_property(unit_free, seed)

    rejected(lambda: m.fit_scale([[1, 2], [1, 3]]), "fit_scale с постоянным столбцом", "[[1, 2], [1, 3]]")
    rejected(lambda: m.fit_scale([[1, 2], [3]]), "fit_scale с рваными строками", "[[1, 2], [3]]")
    rejected(lambda: m.fit_scale([[1, float("nan")], [2, 3]]), "fit_scale с NaN", "[[1, nan], [2, 3]]")
    rejected(lambda: m.fit_scale([]), "fit_scale с пустой таблицей", "[]")
    rejected(lambda: m.nearest([1, 2, 3], [[1, 2]]), "nearest с запросом другой длины", "[1, 2, 3]")
    rejected(lambda: m.nearest([1, 2], []), "nearest без строк", "[]")
    return [
        "fixed_means_and_stds",
        "tie_rule",
        "hidden_cases",
        "random_vs_numpy",
        "standardized_property",
        "unit_free_nearest_property",
        "invalid_input",
    ]


if __name__ == "__main__":
    raise SystemExit(main(check))
