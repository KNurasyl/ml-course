"""Check M11-ex03 (scenario bootstrap, release gate): fixed, hidden, random and property cases."""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from exercises.checklib import close, equal, hidden_cases, main, rejected, run_property  # noqa: E402

BOOT = ("mean_delta", "case_delta", "low", "high")
BLOCKERS = (
    "candidate_has_critical_errors",
    "insufficient_grounding",
    "insufficient_usefulness",
    "model_observation_coverage_below_half",
)


def check_boot(m, baseline, candidate, scenarios, resamples, level, expected, what):
    given = f"baseline={baseline}, candidate={candidate}, scenarios={scenarios}, resamples={resamples}, level={level}"
    got = m.scenario_bootstrap(baseline, candidate, scenarios, resamples, level)
    for key, value in zip(BOOT, expected):
        close(got[key], value, f"{what}: {key}", given)
    return got


def expected_boot(baseline, candidate, scenarios, resamples, level):
    """Loop form of the contract; the checker uses it for random runs."""
    order, changes = [], {}
    for s, x, y in zip(scenarios, baseline, candidate):
        if s not in changes:
            order.append(s)
            changes[s] = []
        changes[s].append(y - x)
    means = [sum(changes[s]) / len(changes[s]) for s in order]
    values = [sum(means[g] for g in row) / len(row) for row in resamples]
    low, high = np.quantile(values, [(1 - level) / 2, (1 + level) / 2])
    case = sum(y - x for x, y in zip(baseline, candidate)) / len(baseline)
    return sum(means) / len(means), case, float(low), float(high)


def gate_rows(grounded, useful, critical=(), fallback=()):
    return [
        {
            "case_id": f"c{i + 1}",
            "grounded": g,
            "useful": u,
            "critical": i + 1 in critical,
            "fallback": i + 1 in fallback,
        }
        for i, (g, u) in enumerate(zip(grounded, useful))
    ]


def check_gate(m, rows, expected_blockers, what, **policy):
    given = f"rows={[(r['case_id'], r['grounded'], r['useful'], r['critical'], r['fallback']) for r in rows]}, policy={policy}"
    got = m.release_gate(rows, **policy)
    equal(list(got["blockers"]), list(expected_blockers), f"{what}: blockers", given)
    equal(got["release_allowed"], not expected_blockers, f"{what}: release_allowed", given)
    n = len(rows)
    close(got["grounded_mean"], sum(r["grounded"] for r in rows) / n, f"{what}: grounded_mean", given)
    close(got["useful_share"], sum(r["useful"] >= 1 for r in rows) / n, f"{what}: useful_share (useful ≥ 1)", given)
    close(got["fallback_share"], sum(r["fallback"] for r in rows) / n, f"{what}: fallback_share", given)
    equal(
        list(got["critical_cases"]),
        sorted(r["case_id"] for r in rows if r["critical"]),
        f"{what}: critical_cases",
        given,
    )
    return got


def check(m, seed):
    run = (
        [0, 1, 0, 1, 1],
        [1, 1, 1, 0, 1],
        ["s1", "s1", "s2", "s3", "s3"],
        [[0, 1, 2], [0, 0, 0], [2, 2, 1], [1, 1, 1]],
    )
    check_boot(m, *run, 0.95, (1 / 3, 0.2, 0.025, 0.9625), "bootstrap по сценариям: s1 с двумя перефразированиями")
    check_boot(m, *run, 0.5, (1 / 3, 0.2, 0.25, 0.625), "bootstrap по сценариям, level 0.5")
    first = ([0, 0, 1], [1, 0, 1], ["b", "a", "b"], [[0, 0], [0, 0], [1, 1]])
    check_boot(m, *first, 0.5, (0.25, 1 / 3, 0.25, 0.5), "номера сценариев — по первому появлению, а не по алфавиту")

    good = gate_rows([2, 2, 1, 1], [2, 1, 0, 1], fallback={2})
    got = check_gate(m, good, [], "gate: все пороги ровно выполнены")
    close(
        (got["grounded_mean"], got["useful_share"]),
        (1.5, 0.75),
        "gate: граница не блокирует",
        "grounded 1.5, useful 0.75",
    )
    check_gate(
        m, gate_rows([2] * 4, [2] * 4, critical={3}), BLOCKERS[:1], "gate: критическая ошибка при максимальных баллах"
    )
    check_gate(m, gate_rows([1, 1, 2, 1], [2] * 4), BLOCKERS[1:2], "gate: средний grounded 1.25")
    check_gate(m, gate_rows([2] * 4, [0, 0, 1, 2]), BLOCKERS[2:3], "gate: useful ≥ 1 только у половины")
    check_gate(m, gate_rows([2] * 4, [2] * 4, fallback={1, 2}), [], "gate: fallback ровно у половины")
    check_gate(m, gate_rows([2] * 4, [2] * 4, fallback={1, 2, 3}), BLOCKERS[3:], "gate: fallback у трёх из четырёх")
    worst = gate_rows([0, 0, 0, 1], [0, 0, 0, 1], critical={4}, fallback={1, 2, 3})
    check_gate(m, worst, BLOCKERS, "gate: все четыре причины в заданном порядке")
    check_gate(m, gate_rows([1, 1, 2, 1], [2] * 4), [], "gate со своей политикой min_grounded=1.0", min_grounded=1.0)

    for case in hidden_cases("M11", "ex03"):
        got = m.scenario_bootstrap(
            case["baseline"], case["candidate"], case["scenarios"], case["resamples"], case["level"]
        )
        for key in BOOT:
            close(got[key], case["bootstrap"][key], f"bootstrap, скрытый случай: {key}", "(скрыт)")
        equal(list(m.release_gate(case["rows"])["blockers"]), case["blockers"], "gate, скрытый случай", "(скрыт)")

    rng = np.random.default_rng(seed)
    for _ in range(20):
        n = int(rng.integers(1, 15))
        scenarios = [str(s) for s in rng.choice(list("abcde"), size=n)]
        baseline, candidate = rng.integers(0, 3, size=n).tolist(), rng.integers(0, 3, size=n).tolist()
        groups = len(set(scenarios))
        resamples = rng.integers(0, groups, size=(int(rng.integers(1, 40)), groups)).tolist()
        level = float(rng.choice([0.5, 0.8, 0.9, 0.95]))
        expected = expected_boot(baseline, candidate, scenarios, resamples, level)
        check_boot(m, baseline, candidate, scenarios, resamples, level, expected, "bootstrap против расчёта в цикле")

    from hypothesis import given, strategies as st

    cases = st.lists(st.tuples(st.sampled_from("abcd"), st.integers(0, 2), st.integers(0, 2)), min_size=1, max_size=12)

    @given(cases, st.integers(0, 2**32 - 1), st.integers(1, 30), st.sampled_from([0.5, 0.8, 0.9, 0.95]))
    def bootstrap_properties(rows, draw, size, level):
        scenarios, baseline, candidate = [s for s, _, _ in rows], [x for _, x, _ in rows], [y for _, _, y in rows]
        groups = len(set(scenarios))
        resamples = np.random.default_rng(draw).integers(0, groups, size=(size, groups)).tolist()
        given_ = (
            f"scenarios={scenarios}, baseline={baseline}, candidate={candidate}, resamples={resamples}, level={level}"
        )
        got = m.scenario_bootstrap(baseline, candidate, scenarios, resamples, level)
        back = m.scenario_bootstrap(candidate, baseline, scenarios, resamples, level)
        close(
            (back["mean_delta"], back["low"], back["high"]),
            (-got["mean_delta"], -got["high"], -got["low"]),
            "обмен версий меняет знак разницы и переворачивает интервал",
            given_,
        )
        # Doubling every case of one scenario changes the case average but not the scenario weights.
        same = [i for i, s in enumerate(scenarios) if s == scenarios[0]]
        twin = m.scenario_bootstrap(
            baseline + [baseline[i] for i in same],
            candidate + [candidate[i] for i in same],
            scenarios + [scenarios[i] for i in same],
            resamples,
            level,
        )
        close(
            (twin["mean_delta"], twin["low"], twin["high"]),
            (got["mean_delta"], got["low"], got["high"]),
            "удвоение перефразирований одного сценария не меняет его вес",
            given_,
        )
        means = {}
        for s, x, y in rows:
            means.setdefault(s, []).append(y - x)
        lo, hi = min(sum(v) / len(v) for v in means.values()), max(sum(v) / len(v) for v in means.values())
        if not lo - 1e-12 <= got["low"] <= got["high"] <= hi + 1e-12:
            raise AssertionError(
                f"интервал лежит между средними сценариев: вход {given_}; ожидалось {lo} ≤ low ≤ high ≤ {hi}; "
                f"получено low={got['low']}, high={got['high']}"
            )

    score = st.integers(0, 2)
    rows_strategy = st.lists(st.tuples(score, score, st.booleans(), st.booleans()), min_size=1, max_size=12)

    @given(
        rows_strategy, st.sampled_from([1.0, 1.5, 2.0]), st.sampled_from([0.5, 0.75, 1.0]), st.sampled_from([0.25, 0.5])
    )
    def gate_follows_rules(raw, min_grounded, min_useful, max_fallback):
        rows = [
            {"case_id": f"c{i}", "grounded": g, "useful": u, "critical": c, "fallback": f}
            for i, (g, u, c, f) in enumerate(raw)
        ]
        n = len(rows)
        expected = [
            name
            for name, broken in zip(
                BLOCKERS,
                (
                    any(r["critical"] for r in rows),
                    sum(r["grounded"] for r in rows) / n < min_grounded,
                    sum(r["useful"] >= 1 for r in rows) / n < min_useful,
                    sum(r["fallback"] for r in rows) / n > max_fallback,
                ),
            )
            if broken
        ]
        policy = {"min_grounded": min_grounded, "min_useful_share": min_useful, "max_fallback_share": max_fallback}
        check_gate(m, rows, expected, "gate, свойство", **policy)

    run_property(bootstrap_properties, seed)
    run_property(gate_follows_rules, seed)

    rejected(lambda: m.scenario_bootstrap([], [], [], [[0]]), "scenario_bootstrap без случаев", "пустой прогон")
    rejected(
        lambda: m.scenario_bootstrap([1, 0], [1], ["a", "a"], [[0]]), "scenario_bootstrap с разной длиной", "2 и 1 балл"
    )
    rejected(lambda: m.scenario_bootstrap([1], [0], ["a"], [[0]], 1.0), "scenario_bootstrap с level = 1", "level = 1")
    rejected(lambda: m.scenario_bootstrap([1], [0], ["a"], []), "scenario_bootstrap без пересэмплов", "resamples=[]")
    two = ([1, 0], [0, 1], ["a", "b"])
    rejected(
        lambda: m.scenario_bootstrap(*two, [[0]]), "scenario_bootstrap: пересэмпл не длины G", "G = 2, resamples=[[0]]"
    )
    rejected(
        lambda: m.scenario_bootstrap(*two, [[0, 2]]),
        "scenario_bootstrap: номер вне 0..G-1",
        "G = 2, resamples=[[0, 2]]",
    )
    rejected(lambda: m.release_gate([]), "release_gate без строк", "rows=[]")
    row = gate_rows([2], [2])[0]
    rejected(lambda: m.release_gate([row, dict(row)]), "release_gate с повтором case_id", "c1 дважды")
    rejected(lambda: m.release_gate([dict(row, case_id="")]), "release_gate с пустым case_id", "case_id=''")
    rejected(
        lambda: m.release_gate([dict(row, grounded=None)]),
        "release_gate: пропущенная оценка — не ноль",
        "grounded=None",
    )
    rejected(lambda: m.release_gate([dict(row, grounded=3)]), "release_gate с оценкой 3", "grounded=3")
    rejected(lambda: m.release_gate([dict(row, useful=True)]), "release_gate с оценкой True", "useful=True")
    rejected(lambda: m.release_gate([dict(row, critical="no")]), "release_gate с critical не bool", "critical='no'")
    rejected(lambda: m.release_gate([dict(row, fallback=1)]), "release_gate с fallback не bool", "fallback=1")
    return [
        "fixed_bootstrap",
        "fixed_gate",
        "hidden_cases",
        "random_bootstrap_vs_loop",
        "bootstrap_property",
        "gate_property",
        "invalid_input",
    ]


if __name__ == "__main__":
    raise SystemExit(main(check))
