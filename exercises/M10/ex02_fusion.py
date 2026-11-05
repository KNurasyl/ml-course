def rrf(rankings, c=60):
    """Reciprocal Rank Fusion: score(d) = sum over the rankings of 1 / (c + rank of d), ranks start at 1.

    rankings is a list of ranked lists of unique document IDs. A document missing from a ranking gets
    nothing from it. Return [(doc_id, score), ...] by score descending; scores that differ by at most
    1e-12 are a tie, ordered by doc_id ascending (float sums of the same fractions can differ in the last
    digit). Reject c <= 0, a bool c and a ranking that repeats an ID.
    """
    raise NotImplementedError("Implement rrf")


def candidate_recall(candidates, relevant, n):
    """Share of the relevant IDs among the first n candidates: the pool a reranker can still reorder.

    Return None when relevant is empty: recall is undefined there, not zero.
    Reject n that is not a positive int (bool included) and candidates that repeat an ID.
    """
    raise NotImplementedError("Implement candidate_recall")


def mean_reciprocal_rank(rankings, relevant_sets):
    """Mean over answerable queries of 1 / rank of the first relevant ID, 0 when none is found.

    Queries with an empty relevant set are left out of the mean; return None when no query is answerable.
    Reject lists of different lengths and a ranking that repeats an ID.
    """
    raise NotImplementedError("Implement mean_reciprocal_rank")
