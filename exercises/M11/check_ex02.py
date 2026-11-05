"""Check M11-ex02 (answer rates, judge agreement, Cohen's kappa): fixed, hidden, random and property cases."""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from exercises.checklib import close, equal, hidden_cases, main, rejected, run_property  # noqa: E402

RATES = ("coverage", "accuracy_answered", "correct_total", "appropriate_refusals", "success")
AGREEMENT = ("coverage", "agreement_valid", "correct_over_all")


def records(answered_correct, answered_wrong, refused_unanswerable, refused_answerable):
    """A run built from four counts; answerable answers keep the example readable."""
    rows = [{"answered": True, "correct": True, "answerable": True}] * answered_correct
    rows += [{"answered": True, "correct": False, "answerable": True}] * answered_wrong
    rows += [{"answered": False, "correct": False, "answerable": False}] * refused_unanswerable
    rows += [{"answered": False, "correct": False, "answerable": True}] * refused_answerable
    return [dict(row) for row in rows]


def check_rates(m, rows, expected, what):
    got = m.answer_rates(rows)
    given = f"{len(rows)} записей: ответов {sum(r['answered'] for r in rows)}, верных {sum(r['correct'] for r in rows)}"
    for key, value in zip(RATES, expected):
        close(got[key], value, f"{what}: {key}", given)


def check(m, seed):
    from sklearn.metrics import accuracy_score, cohen_kappa_score

    check_rates(m, records(12, 3, 3, 2), (0.75, 0.8, 0.6, 0.6, 0.75), "доли ответов: пример урока, 20 запросов")
    check_rates(m, records(30, 10, 7, 3), (0.8, 0.75, 0.6, 0.7, 0.74), "доли ответов: пример карточки, 50 запросов")
    unanswered = [{"answered": True, "correct": False, "answerable": False}] + records(1, 0, 0, 1)
    check_rates(m, unanswered, (2 / 3, 0.5, 1 / 3, 0.0, 1 / 3), "ответ на вопрос без ответа в базе — не отказ")

    human, judge = [1, 1, 0, 0], [1, 0, 1, 0]
    got = m.judge_agreement(human, judge)
    for key, value in zip(AGREEMENT, (1.0, 0.5, 0.5)):
        close(got[key], value, f"judge: пример урока, {key}", f"human={human}, judge={judge}")
    equal(list(got["false_approvals"]), [3], "judge: ложное одобрение", f"human={human}, judge={judge}")
    human, judge = [1, 0, 1, 0, 1], [1, 1, None, 0, 0]
    got = m.judge_agreement(human, judge)
    for key, value in zip(AGREEMENT, (0.8, 0.5, 0.4)):
        close(got[key], value, f"judge: invalid остаётся в знаменателе, {key}", f"human={human}, judge={judge}")
    equal(list(got["false_approvals"]), [2], "judge: ложное одобрение среди валидных", f"human={human}, judge={judge}")

    close(m.cohen_kappa([1, 1, 0, 0], [1, 0, 0, 0]), 0.5, "kappa: пример урока", "a=[1, 1, 0, 0], b=[1, 0, 0, 0]")
    a, b = [0] * 9 + [1], [0] * 8 + [1, 0]
    close(m.cohen_kappa(a, b), -1 / 9, "kappa: совпадений 0.8, но ниже случайного", f"a={a}, b={b}")
    close(m.cohen_kappa([0, 1, 2, 2], [0, 2, 2, 1]), 0.2, "kappa с тремя метками", "a=[0, 1, 2, 2], b=[0, 2, 2, 1]")
    words = (["yes", "no", "yes"], ["yes", "yes", "yes"])
    close(m.cohen_kappa(*words), 0.0, "kappa со строковыми метками", f"a={words[0]}, b={words[1]}")

    for case in hidden_cases("M11", "ex02"):
        got = m.answer_rates(case["records"])
        for key in RATES:
            close(got[key], case["rates"][key], f"доли ответов, скрытый случай: {key}", "(скрыт)")
        got = m.judge_agreement(case["human"], case["judge"])
        for key in AGREEMENT:
            close(got[key], case["agreement"][key], f"judge, скрытый случай: {key}", "(скрыт)")
        equal(list(got["false_approvals"]), case["agreement"]["false_approvals"], "judge, скрытый случай", "(скрыт)")
        close(m.cohen_kappa(case["a"], case["b"]), case["kappa"], "kappa, скрытый случай", "(скрыт)")

    rng = np.random.default_rng(seed)
    for _ in range(20):
        n = int(rng.integers(2, 30))
        labels = list(range(int(rng.integers(2, 4))))
        a, b = rng.choice(labels, size=n).tolist(), rng.choice(labels, size=n).tolist()
        if len(set(a)) == len(set(b)) == 1 and a[0] == b[0]:
            b[0] = labels[(labels.index(b[0]) + 1) % len(labels)]  # p_e = 1 is the variant C question, not a check
        close(m.cohen_kappa(a, b), cohen_kappa_score(a, b), "kappa против scikit-learn", f"a={a}, b={b}")
        human = rng.integers(0, 2, size=n).tolist()
        judge = [None if rng.random() < 0.2 else int(v) for v in rng.integers(0, 2, size=n)]
        if all(v is None for v in judge):
            judge[0] = human[0]  # a run without any valid judge label is not checked
        valid = [i for i, v in enumerate(judge) if v is not None]
        got = m.judge_agreement(human, judge)
        expected = accuracy_score([human[i] for i in valid], [judge[i] for i in valid])
        close(
            got["agreement_valid"],
            expected,
            "agreement среди валидных против scikit-learn",
            f"human={human}, judge={judge}",
        )

    from hypothesis import given, strategies as st

    @given(st.integers(0, 15), st.integers(0, 15), st.integers(0, 15), st.integers(0, 15))
    def rates_fit_together(right, wrong, refused_ok, refused_bad):
        if right + wrong == 0 or refused_ok + refused_bad == 0:
            return
        got = m.answer_rates(records(right, wrong, refused_ok, refused_bad))
        given_ = f"верных {right}, неверных {wrong}, уместных отказов {refused_ok}, неуместных {refused_bad}"
        close(
            got["correct_total"],
            got["coverage"] * got["accuracy_answered"],
            "correct_total = coverage · accuracy",
            given_,
        )
        both = got["correct_total"] + (1 - got["coverage"]) * got["appropriate_refusals"]
        close(got["success"], both, "success = верные ответы + уместные отказы", given_)
        for key in RATES:
            if not 0 <= got[key] <= 1:
                raise AssertionError(f"доля {key} лежит в [0, 1]: вход {given_}; получено {got[key]}")

    @given(st.lists(st.tuples(st.integers(0, 1), st.sampled_from([0, 1, None])), min_size=1, max_size=25))
    def judge_properties(pairs):
        human, judge = [h for h, _ in pairs], [j for _, j in pairs]
        if all(j is None for j in judge):
            return
        got = m.judge_agreement(human, judge)
        given_ = f"human={human}, judge={judge}"
        close(
            got["correct_over_all"],
            got["coverage"] * got["agreement_valid"],
            "correct_over_all = coverage · agreement",
            given_,
        )
        fixed = [h if j is None else j for h, j in pairs]  # replace every invalid output with the human label
        close(
            m.judge_agreement(human, fixed)["coverage"],
            1.0,
            "без invalid coverage = 1",
            f"human={human}, judge={fixed}",
        )
        expected = [i + 1 for i, (h, j) in enumerate(pairs) if j == 1 and h == 0]
        equal(list(got["false_approvals"]), expected, "ложные одобрения: judge 1 при ручной 0", given_)

    @given(st.lists(st.tuples(st.sampled_from("xyz"), st.sampled_from("xyz")), min_size=2, max_size=25))
    def kappa_properties(pairs):
        a, b = [x for x, _ in pairs], [y for _, y in pairs]
        if len(set(a)) == len(set(b)) == 1 and a[0] == b[0]:
            return
        value = m.cohen_kappa(a, b)
        close(m.cohen_kappa(b, a), value, "kappa симметрична", f"a={a}, b={b}")
        rename = {"x": "q", "y": "x", "z": "y"}
        close(
            m.cohen_kappa([rename[v] for v in a], [rename[v] for v in b]),
            value,
            "kappa не зависит от имён меток",
            f"a={a}, b={b}",
        )
        if not value <= 1 + 1e-12:
            raise AssertionError(f"kappa не больше 1: вход a={a}, b={b}; получено {value}")
        if len(set(a)) > 1:
            close(m.cohen_kappa(a, a), 1.0, "kappa полного согласия равна 1", f"a=b={a}")

    run_property(rates_fit_together, seed)
    run_property(judge_properties, seed)
    run_property(kappa_properties, seed)

    rejected(lambda: m.answer_rates([]), "answer_rates без записей", "records=[]")
    bad = {"answered": 1, "correct": False, "answerable": True}
    rejected(lambda: m.answer_rates([bad]), "answer_rates с answered = 1 вместо bool", bad)
    contradiction = {"answered": False, "correct": True, "answerable": True}
    rejected(lambda: m.answer_rates([contradiction]), "answer_rates: отказ не может быть верным ответом", contradiction)
    rejected(lambda: m.judge_agreement([1, 0], [1]), "judge_agreement со списками разной длины", "2 и 1 метка")
    rejected(lambda: m.judge_agreement([], []), "judge_agreement без случаев", "human=[], judge=[]")
    rejected(lambda: m.judge_agreement([1, 2], [1, 0]), "judge_agreement с меткой 2", "human=[1, 2]")
    rejected(lambda: m.judge_agreement([1, 0], [True, 0]), "judge_agreement с меткой True", "judge=[True, 0]")
    rejected(
        lambda: m.judge_agreement([1, None], [1, 0]), "judge_agreement с None в ручной разметке", "human=[1, None]"
    )
    rejected(lambda: m.cohen_kappa([], []), "cohen_kappa без случаев", "a=[], b=[]")
    rejected(lambda: m.cohen_kappa([1, 0], [1]), "cohen_kappa со списками разной длины", "2 и 1 метка")
    return [
        "fixed_answer_rates",
        "fixed_judge_agreement",
        "fixed_kappa",
        "hidden_cases",
        "random_vs_sklearn",
        "rates_property",
        "judge_property",
        "kappa_property",
        "invalid_input",
    ]


if __name__ == "__main__":
    raise SystemExit(main(check))
