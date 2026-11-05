"""Check M10-ex02 (RRF, candidate recall, MRR): fixed, hidden, random and property cases."""

from fractions import Fraction
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from exercises.checklib import close, equal, hidden_cases, main, rejected, run_property  # noqa: E402

TIE = 1e-12


def exact_rrf(rankings, c):
    """Exact RRF scores; the checker needs them to tell a true tie from a real difference."""
    scores = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, 1):
            scores[doc] = scores.get(doc, Fraction(0)) + Fraction(1) / (Fraction(c) + rank)
    return scores


def check_rrf(m, rankings, c, what):
    given = f"rankings={rankings}, c={c}"
    expected = exact_rrf(rankings, c)
    got = [(doc, score) for doc, score in m.rrf(rankings, c)]
    equal(sorted(doc for doc, _ in got), sorted(expected), f"{what}: каждый документ ровно один раз", given)
    for doc, score in got:
        close(score, float(expected[doc]), f"{what}: score документа {doc}", given)
    for (a, _), (b, _) in zip(got, got[1:]):
        diff = expected[a] - expected[b]
        if diff == 0 and not a < b:
            raise AssertionError(
                f"{what}, ничья по ID: вход {given}; у {a} и {b} равные суммы {float(diff + expected[b])}; "
                f"ожидалось {min(a, b)} раньше {max(a, b)}; получено {a} раньше {b} "
                "(суммы одних дробей во float могут отличаться в последнем знаке: сравнивай с допуском 1e-12)"
            )
        if diff < -TIE:
            raise AssertionError(
                f"{what}, порядок по убыванию score: вход {given}; ожидалось {b} ({float(expected[b])}) "
                f"раньше {a} ({float(expected[a])}); получено {a} раньше {b}"
            )
    return got


def check(m, seed):
    check_rrf(m, [["A", "B", "C"], ["B", "D", "A"]], 10, "RRF: пример урока")
    check_rrf(m, [["P", "Q", "R"], ["R", "S", "P"]], 1, "RRF: две ничьи")
    check_rrf(m, [["A"], ["B"]], 1, "RRF: отсутствие в списке не даёт вклада")
    tie = [["A", "C", "D", "B"], ["C", "D", "E", "A", "F", "B"], ["B", "C", "D", "E", "F", "A"]]
    check_rrf(m, tie, 1, "RRF: точная ничья A и B (1/2 + 1/5 + 1/7)")
    equal(list(m.rrf([], 10)), [], "RRF без списков", "rankings=[]")
    one = ["E", "B", "D", "A"]
    equal([doc for doc, _ in m.rrf([one], 10)], one, "RRF одного списка сохраняет его порядок", f"rankings=[{one}]")

    pool = ["B", "A", "D", "C"]
    close(m.candidate_recall(pool, ["C"], 2), 0.0, "candidate Recall@2: C вне пула", (pool, ["C"], 2))
    close(m.candidate_recall(pool, ["C"], 4), 1.0, "candidate Recall@4: C в пуле", (pool, ["C"], 4))
    close(
        m.candidate_recall(pool, ["A", "C"], 3),
        0.5,
        "candidate Recall@3 делится на число релевантных",
        (pool, ["A", "C"], 3),
    )
    equal(m.candidate_recall(pool, [], 3), None, "candidate recall без релевантных не определён", (pool, [], 3))

    rankings, gold = [["X", "A"], ["B"], ["Y", "Z", "W", "C"]], [["A"], ["B"], ["C"]]
    close(m.mean_reciprocal_rank(rankings, gold), 1.75 / 3, "MRR: места 2, 1 и 4", (rankings, gold))
    rankings_q, gold_q = rankings + [["Q"]], gold + [[]]
    close(
        m.mean_reciprocal_rank(rankings_q, gold_q),
        1.75 / 3,
        "MRR: вопрос без ответа не входит в среднее",
        (rankings_q, gold_q),
    )
    close(m.mean_reciprocal_rank([["X"]], [["A"]]), 0.0, "MRR: релевантный не найден", ([["X"]], [["A"]]))
    equal(m.mean_reciprocal_rank([["X"]], [[]]), None, "MRR без вопросов с ответом не определён", ([["X"]], [[]]))

    for case in hidden_cases("M10", "ex02"):
        got = list(m.rrf(case["rankings"], case["c"]))
        equal([doc for doc, _ in got], case["order"], "RRF, скрытый случай: порядок", "(скрыт)")
        close([score for _, score in got], case["scores"], "RRF, скрытый случай: scores", "(скрыт)")
        result = m.candidate_recall(case["candidates"], case["relevant"], case["n"])
        if case["candidate_recall"] is None:
            equal(result, None, "candidate recall, скрытый случай", "(скрыт)")
        else:
            close(result, case["candidate_recall"], "candidate recall, скрытый случай", "(скрыт)")
        result = m.mean_reciprocal_rank(case["mrr_rankings"], case["mrr_relevant"])
        if case["mrr"] is None:
            equal(result, None, "MRR, скрытый случай", "(скрыт)")
        else:
            close(result, case["mrr"], "MRR, скрытый случай", "(скрыт)")

    rng = np.random.default_rng(seed)
    letters = list("ABCDEFGHIJKL")
    for _ in range(20):
        rankings = [
            [str(v) for v in rng.permutation(letters)[: int(rng.integers(0, 9))]]
            for _ in range(int(rng.integers(1, 5)))
        ]
        c = int(rng.integers(1, 101)) if rng.random() < 0.7 else round(float(rng.uniform(0.5, 80)), 3)
        check_rrf(m, rankings, c, "RRF против точных дробей")

    from hypothesis import given, strategies as st

    ranking = st.lists(st.sampled_from(letters), unique=True, max_size=8)

    @given(st.lists(ranking, min_size=1, max_size=4), ranking, st.integers(1, 100))
    def fusion_properties(rankings, extra, c):
        got = dict(check_rrf(m, rankings, c, "RRF, свойство"))
        bound = len(rankings) / (c + 1)
        for doc, score in got.items():
            if not 0 < score <= bound + 1e-12:
                raise AssertionError(
                    f"score RRF лежит в (0, m/(c + 1)]: вход rankings={rankings}, c={c}; "
                    f"ожидалось 0 < score ≤ {bound}; получено {doc}: {score}"
                )
        more = dict(m.rrf(rankings + [extra], c))
        for doc, score in got.items():
            if more[doc] < score - 1e-12:
                raise AssertionError(
                    f"новый список не уменьшает score: вход rankings={rankings} + [{extra}], c={c}; "
                    f"ожидалось для {doc} не меньше {score}; получено {more[doc]}"
                )

    @given(ranking, st.lists(st.sampled_from(letters), max_size=4), st.integers(1, 12))
    def pool_grows_with_n(candidates, relevant, n):
        small, large = m.candidate_recall(candidates, relevant, n), m.candidate_recall(candidates, relevant, n + 1)
        if not relevant:
            equal(small, None, "candidate recall без релевантных не определён", (candidates, relevant, n))
            return
        if not 0 <= small <= large <= 1:
            raise AssertionError(
                f"candidate recall не убывает с ростом N и лежит в [0, 1]: вход candidates={candidates}, "
                f"relevant={relevant}, N={n} и {n + 1}; ожидалось 0 ≤ {small} ≤ {large} ≤ 1; получено обратное"
            )

    @given(ranking.filter(lambda r: len(r) >= 2), st.integers(0, 7))
    def earlier_hit_higher_rr(ranking_, position):
        position = min(position, len(ranking_) - 1)
        target = ranking_[position]
        moved = [target] + [doc for doc in ranking_ if doc != target]
        before, after = m.mean_reciprocal_rank([ranking_], [[target]]), m.mean_reciprocal_rank([moved], [[target]])
        close(before, 1 / (position + 1), "reciprocal rank единственного релевантного", (ranking_, [target]))
        if not before <= after == 1.0:
            raise AssertionError(
                f"релевантный на первом месте даёт MRR = 1: вход {moved}, gold {[target]}; "
                f"ожидалось 1.0 и не меньше {before}; получено {after}"
            )

    run_property(fusion_properties, seed)
    run_property(pool_grows_with_n, seed)
    run_property(earlier_hit_higher_rr, seed)

    rejected(lambda: m.rrf([["A", "B"]], 0), "rrf с c = 0", "c = 0")
    rejected(lambda: m.rrf([["A", "B"]], -1), "rrf с отрицательным c", "c = −1")
    rejected(lambda: m.rrf([["A", "B"]], True), "rrf с c = True", "c = True")
    rejected(lambda: m.rrf([["A", "B", "A"]], 10), "rrf со списком, где ID повторяется", "rankings=[['A', 'B', 'A']]")
    rejected(lambda: m.candidate_recall(["A"], ["A"], 0), "candidate_recall с N = 0", "n = 0")
    rejected(lambda: m.candidate_recall(["A"], ["A"], True), "candidate_recall с N = True", "n = True")
    rejected(lambda: m.candidate_recall(["A"], ["A"], 2.5), "candidate_recall с дробным N", "n = 2.5")
    rejected(
        lambda: m.candidate_recall(["A", "A"], ["A"], 2), "candidate_recall с повтором ID", "candidates=['A', 'A']"
    )
    rejected(
        lambda: m.mean_reciprocal_rank([["A"]], [["A"], ["B"]]), "MRR со списками разной длины", "1 выдача, 2 разметки"
    )
    rejected(lambda: m.mean_reciprocal_rank([["A", "A"]], [["A"]]), "MRR с повтором ID в выдаче", "ranking=['A', 'A']")
    return [
        "fixed_rrf",
        "fixed_candidate_recall",
        "fixed_mrr",
        "hidden_cases",
        "random_vs_exact_fractions",
        "rrf_property",
        "candidate_recall_property",
        "mrr_property",
        "invalid_input",
    ]


if __name__ == "__main__":
    raise SystemExit(main(check))
