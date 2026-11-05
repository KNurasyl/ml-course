"""Reference for M11-ex03: paired bootstrap over scenarios and the M11-L06 release gate."""

import numpy as np


def scenario_bootstrap(baseline, candidate, scenarios, resamples, level=0.95):
    baseline, candidate, scenarios = list(baseline), list(candidate), list(scenarios)
    if not baseline or not len(baseline) == len(candidate) == len(scenarios):
        raise ValueError("need non-empty paired scores and one scenario per case")
    if isinstance(level, bool) or not isinstance(level, (int, float)) or not 0 < level < 1:
        raise ValueError("level must lie in (0, 1)")
    groups = {}
    for scenario, before, after in zip(scenarios, baseline, candidate):
        groups.setdefault(scenario, []).append(after - before)  # the pair of one case stays together
    means = np.array([sum(changes) / len(changes) for changes in groups.values()])
    picks = np.asarray(resamples)
    if picks.ndim != 2 or picks.shape[0] == 0 or picks.shape[1] != len(means) or picks.dtype.kind not in "iu":
        raise ValueError("each resample must list one scenario number per scenario")
    if picks.min() < 0 or picks.max() >= len(means):
        raise ValueError("scenario numbers must lie in 0..G-1")
    values = means[picks].mean(axis=1)  # each scenario weighs the same, however many paraphrases it has
    low, high = np.quantile(values, [(1 - level) / 2, (1 + level) / 2])
    deltas = [after - before for before, after in zip(baseline, candidate)]
    return {
        "mean_delta": float(means.mean()),
        "case_delta": sum(deltas) / len(deltas),
        "low": float(low),
        "high": float(high),
    }


def release_gate(rows, min_grounded=1.5, min_useful_share=0.75, max_fallback_share=0.5):
    rows = list(rows)
    if not rows:
        raise ValueError("need at least one annotated row")
    seen = set()
    for row in rows:
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or not case_id or case_id in seen:
            raise ValueError("case_id must be a unique non-empty string")
        seen.add(case_id)
        if any(type(row.get(key)) is not int or not 0 <= row[key] <= 2 for key in ("grounded", "useful")):
            raise ValueError("ratings must be explicit integers 0..2; a missing rating is not zero")
        if any(type(row.get(key)) is not bool for key in ("critical", "fallback")):
            raise ValueError("critical and fallback must be explicit booleans")
    n = len(rows)
    grounded_mean = sum(row["grounded"] for row in rows) / n
    useful_share = sum(row["useful"] >= 1 for row in rows) / n  # at least partly useful
    fallback_share = sum(row["fallback"] for row in rows) / n
    critical = sorted(row["case_id"] for row in rows if row["critical"])
    blockers = []
    if critical:
        blockers.append("candidate_has_critical_errors")  # no average compensates a critical error
    if grounded_mean < min_grounded:
        blockers.append("insufficient_grounding")
    if useful_share < min_useful_share:
        blockers.append("insufficient_usefulness")
    if fallback_share > max_fallback_share:
        blockers.append("model_observation_coverage_below_half")
    return {
        "release_allowed": not blockers,
        "blockers": blockers,
        "grounded_mean": grounded_mean,
        "useful_share": useful_share,
        "fallback_share": fallback_share,
        "critical_cases": critical,
    }
