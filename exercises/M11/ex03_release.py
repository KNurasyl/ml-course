def scenario_bootstrap(baseline, candidate, scenarios, resamples, level=0.95):
    """Paired bootstrap over scenarios; the caller supplies the resampling indices.

    baseline, candidate: scores of the same cases in the same order; scenarios: scenario_id of each case.
    Average candidate - baseline inside each scenario first; scenarios are numbered 0..G-1 in the order
    of their first appearance. Each resample is a list of G scenario numbers drawn with replacement
    (for example np.random.default_rng(seed).integers(0, G, size=(B, G))); its value is the mean of the
    drawn scenario means. Return {"mean_delta": mean of the scenario means, "case_delta": mean over cases,
    "low", "high"}, where low and high are the (1 - level) / 2 and (1 + level) / 2 quantiles of the
    resample values (numpy.quantile with its default linear method).
    Reject an empty run, unequal lengths, level outside (0, 1) and resamples that are empty, not of
    length G or hold numbers outside 0..G-1.
    """
    raise NotImplementedError("Implement scenario_bootstrap")


def release_gate(rows, min_grounded=1.5, min_useful_share=0.75, max_fallback_share=0.5):
    """Apply the M11-L06 release policy, fixed before the run, to the candidate's annotated rows.

    rows: list of {"case_id": str, "grounded": int 0..2, "useful": int 0..2, "critical": bool, "fallback": bool}.
    Blockers, in this order: "candidate_has_critical_errors" (any critical row), "insufficient_grounding"
    (mean grounded below min_grounded), "insufficient_usefulness" (share of rows with useful >= 1 below
    min_useful_share), "model_observation_coverage_below_half" (share of fallback rows above max_fallback_share).
    Return {"release_allowed": bool, "blockers": [...], "grounded_mean", "useful_share", "fallback_share",
    "critical_cases": sorted case_ids}.
    Reject an empty list, an empty or repeated case_id, ratings that are not int 0..2 (bool included;
    a missing rating is not zero) and critical or fallback that are not bool.
    """
    raise NotImplementedError("Implement release_gate")
