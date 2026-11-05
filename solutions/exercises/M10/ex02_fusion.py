"""Reference for M10-ex02: reciprocal rank fusion, the candidate pool and MRR over answerable queries."""

from fractions import Fraction


def _unique(ranking, name):
    ranking = list(ranking)
    if len(ranking) != len(set(ranking)):
        raise ValueError(f"{name} repeats an ID: collapse chunks to documents first")
    return ranking


def rrf(rankings, c=60):
    if isinstance(c, bool) or not isinstance(c, (int, float)) or not c > 0:
        raise ValueError("c must be a positive number")
    scores = {}
    for ranking in rankings:
        for rank, doc in enumerate(_unique(ranking, "ranking"), 1):  # the leader of a list gets 1 / (c + 1)
            # Exact fractions: equal sums stay equal, so a tie is always broken by the ID, not by float noise.
            scores[doc] = scores.get(doc, Fraction(0)) + 1 / (Fraction(c) + rank)
    return [(doc, float(score)) for doc, score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))]


def candidate_recall(candidates, relevant, n):
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    candidates, gold = _unique(candidates, "candidates"), set(relevant)
    if not gold:
        return None  # nothing to find: recall is undefined, not zero
    return len(gold & set(candidates[:n])) / len(gold)


def mean_reciprocal_rank(rankings, relevant_sets):
    rankings, relevant_sets = list(rankings), list(relevant_sets)
    if len(rankings) != len(relevant_sets):
        raise ValueError("need one relevant set per ranking")
    values = []
    for ranking, relevant in zip(rankings, relevant_sets):
        ranking, gold = _unique(ranking, "ranking"), set(relevant)
        if not gold:
            continue  # unanswerable: left out of the mean and reported separately
        rank = next((i for i, doc in enumerate(ranking, 1) if doc in gold), None)
        values.append(0.0 if rank is None else 1 / rank)
    return sum(values) / len(values) if values else None
