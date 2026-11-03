"""M11-HW01 computational part. Implement before opening the separate solution."""


def compare_pairs(baseline, candidate):
    """Join by case_id, never by row order.

    Each input is a list of {case_id: str, score: int 0..2, critical: bool}.
    Return {improved: sorted IDs, regressed: sorted IDs,
            critical_regressions: sorted IDs, mean_delta: float}.
    Reject empty input, missing/duplicate IDs, unequal sets, bool scores and
    missing ratings with ValueError. A new critical error is a blocker even
    when its numerical score was increased.
    """
    raise NotImplementedError("Implement the contract, then run check.py against this file")
