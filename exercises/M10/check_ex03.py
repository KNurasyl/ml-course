"""Check M10-ex03 (word windows, context selection): fixed, hidden, random and property cases."""

import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from exercises.checklib import equal, hidden_cases, main, rejected, run_property  # noqa: E402

REASONS = {"obsolete", "duplicate", "budget"}


def windows(m, n, size, overlap):
    return [tuple(w) for w in m.window_chunks(n, size, overlap)]


def expected_windows(n, size, overlap):
    """Closed form: 1 + ceil((n - size) / step) windows, each starting at a multiple of the step."""
    if n == 0:
        return []
    step = size - overlap
    return [(i * step, min(i * step + size, n)) for i in range(1 + math.ceil(max(0, n - size) / step))]


def selection(m, ranked, active, budget):
    result = m.select_context(ranked, active, budget)
    return list(result["selected"]), [tuple(item) for item in result["excluded"]], result["used"]


def check_selection(m, ranked, active, budget, what):
    """Every rule of the greedy pass, checked chunk by chunk, so a failure names the chunk and the rule."""
    given = f"ranked={[c['id'] for c in ranked]}, active={active}, budget={budget}"
    selected, excluded, used = selection(m, ranked, active, budget)
    reasons = dict(excluded)
    order = [c["id"] for c in ranked]
    equal(sorted(selected + list(reasons)), sorted(order), f"{what}: каждый chunk ровно один раз", given)
    equal(len(excluded), len(reasons), f"{what}: chunk исключён один раз", given)
    equal(selected, [cid for cid in order if cid in selected], f"{what}: выбранные идут в порядке выдачи", given)
    by_id = {c["id"]: c for c in ranked}
    equal(used, sum(by_id[cid]["tokens"] for cid in selected), f"{what}: used — сумма tokens выбранных", given)
    taken, texts = 0, set()
    for chunk in ranked:
        cid = chunk["id"]
        obsolete = chunk["document_id"] not in active or active[chunk["document_id"]] != chunk["version"]
        if obsolete:
            reason = "obsolete"
        elif chunk["text"] in texts:
            reason = "duplicate"
        elif taken + chunk["tokens"] > budget:
            reason = "budget"
        else:
            reason = None
            taken += chunk["tokens"]
            texts.add(chunk["text"])
        got = reasons.get(cid) if cid not in selected else None
        if got not in REASONS | {None}:
            raise AssertionError(f"{what}: вход {given}; ожидалась причина из {sorted(REASONS)}; получено {got!r}")
        equal(got, reason, f"{what}: chunk {cid} (None — выбран)", given)
    if used > budget:
        raise AssertionError(f"{what}: вход {given}; ожидалось used ≤ {budget}; получено {used}")


def chunk(cid, document, version, tokens, text):
    return {"id": cid, "document_id": document, "version": version, "tokens": tokens, "text": text}


def check(m, seed):
    equal(
        windows(m, 500, 200, 50), [(0, 200), (150, 350), (300, 500)], "окна с overlap: пример карточки", (500, 200, 50)
    )
    equal(windows(m, 500, 200, 0), [(0, 200), (200, 400), (400, 500)], "окна без overlap", (500, 200, 0))
    equal(windows(m, 120, 200, 50), [(0, 120)], "документ короче окна — одно окно", (120, 200, 50))
    equal(windows(m, 0, 10, 3), [], "пустой документ — нет окон", (0, 10, 3))
    step_one = [(i, i + 4) for i in range(7)]
    equal(windows(m, 10, 4, 3), step_one, "окна с шагом 1", (10, 4, 3))

    ranked = [
        chunk("idx:v1:1", "idx", 1, 30, "Индекс всегда ускоряет чтение"),
        chunk("idx:v2:1", "idx", 2, 40, "Индекс может ускорить чтение"),
        chunk("idx:v2:1-copy", "idx", 2, 40, "Индекс может ускорить чтение"),
        chunk("http:v1:1", "http", 1, 10, "GET обычно не меняет ресурс"),
        chunk("sql:v1:2", "sql", 1, 50, "Транзакция фиксирует операции вместе"),
        chunk("sql:v1:3", "sql", 1, 25, "Откат отменяет все операции"),
    ]
    active = {"idx": 2, "sql": 1}
    expected = (
        ["idx:v2:1", "sql:v1:3"],
        [("idx:v1:1", "obsolete"), ("idx:v2:1-copy", "duplicate"), ("http:v1:1", "obsolete"), ("sql:v1:2", "budget")],
        65,
    )
    for budget, what in (
        (80, "отбор контекста: устаревшая версия, дубликат, бюджет"),
        (65, "отбор контекста: ровно по бюджету"),
    ):
        equal(
            selection(m, ranked, active, budget), expected, what, f"ranked={[c['id'] for c in ranked]}, budget={budget}"
        )
    check_selection(m, ranked, active, 0, "отбор контекста при бюджете 0")

    for case in hidden_cases("M10", "ex03"):
        equal(windows(m, *case["window"]), [tuple(w) for w in case["windows"]], "окна, скрытый случай", "(скрыт)")
        got = selection(m, case["ranked"], case["active"], case["budget"])
        want = (case["selected"], [tuple(item) for item in case["excluded"]], case["used"])
        equal(got, want, "отбор контекста, скрытый случай", "(скрыт)")

    rng = np.random.default_rng(seed)
    for _ in range(20):
        size = int(rng.integers(1, 80))
        overlap, n = int(rng.integers(0, size)), int(rng.integers(0, 600))
        equal(
            windows(m, n, size, overlap), expected_windows(n, size, overlap), "окна против формулы", (n, size, overlap)
        )

    from hypothesis import given, strategies as st

    @given(st.integers(1, 300).flatmap(lambda s: st.tuples(st.integers(0, 2000), st.just(s), st.integers(0, s - 1))))
    def window_properties(case):
        n, size, overlap = case
        got = windows(m, n, size, overlap)
        given_ = (n, size, overlap)
        if n == 0:
            equal(got, [], "пустой документ — нет окон", given_)
            return
        equal((got[0][0], got[-1][1]), (0, n), "окна начинаются с 0 и заканчиваются концом документа", given_)
        for (s1, e1), (s2, e2) in zip(got, got[1:]):
            equal(e1 - s1, size, "окно перед последним имеет полную длину", given_)
            equal(e1 - s2, overlap, "соседние окна перекрываются ровно на overlap слов", given_)
            if not e2 > e1:
                raise AssertionError(
                    f"каждое окно добавляет новые слова: вход {given_}; ожидалось {e2} > {e1}; получено обратное"
                )
        equal(len(got), len(expected_windows(n, size, overlap)), "число окон", given_)

    texts = st.sampled_from(["a", "b", "c", "d"])
    chunks = st.lists(
        st.tuples(st.sampled_from(["idx", "sql", "http"]), st.integers(1, 2), st.integers(1, 40), texts), max_size=10
    )

    @given(chunks, st.dictionaries(st.sampled_from(["idx", "sql", "http"]), st.integers(1, 2)), st.integers(0, 150))
    def selection_properties(rows, active_, budget):
        ranked_ = [chunk(f"c{i}", d, v, t, x) for i, (d, v, t, x) in enumerate(rows)]
        check_selection(m, ranked_, active_, budget, "отбор контекста, свойство")

    run_property(window_properties, seed)
    run_property(selection_properties, seed)

    rejected(lambda: m.window_chunks(100, 0, 0), "window_chunks с size = 0", "size = 0")
    rejected(lambda: m.window_chunks(100, 10, 10), "window_chunks с overlap = size (шаг 0)", "size = 10, overlap = 10")
    rejected(lambda: m.window_chunks(100, 10, -1), "window_chunks с отрицательным overlap", "overlap = −1")
    rejected(lambda: m.window_chunks(-1, 10, 2), "window_chunks с отрицательной длиной документа", "n_words = −1")
    rejected(lambda: m.window_chunks(100, 2.5, 0), "window_chunks с дробным size", "size = 2.5")
    rejected(lambda: m.window_chunks(100, True, 0), "window_chunks с size = True", "size = True")
    one = [chunk("a", "idx", 1, 5, "x")]
    rejected(lambda: m.select_context(one, {"idx": 1}, -1), "select_context с отрицательным бюджетом", "budget = −1")
    rejected(lambda: m.select_context(one, {"idx": 1}, True), "select_context с бюджетом True", "budget = True")
    bad = [chunk("a", "idx", 1, 0, "x")]
    rejected(lambda: m.select_context(bad, {"idx": 1}, 10), "select_context с tokens = 0", "tokens = 0")
    twice = one + [chunk("a", "idx", 1, 3, "y")]
    rejected(lambda: m.select_context(twice, {"idx": 1}, 10), "select_context с повтором ID", "id 'a' дважды")
    return [
        "fixed_windows",
        "fixed_selection",
        "hidden_cases",
        "random_windows_vs_formula",
        "window_property",
        "selection_property",
        "invalid_input",
    ]


if __name__ == "__main__":
    raise SystemExit(main(check))
