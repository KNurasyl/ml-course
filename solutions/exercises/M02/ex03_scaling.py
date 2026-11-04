"""Reference for M02-ex03: standardisation fitted on a table and the nearest neighbour after it."""

import math


def _table(rows):
    if not rows or not rows[0]:
        raise ValueError("rows must be a non-empty table")
    width = len(rows[0])
    for row in rows:
        if len(row) != width:
            raise ValueError("rows have different length")
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in row):
            raise ValueError("rows must contain finite numbers")
    return width


def fit_scale(rows):
    width = _table(rows)
    n = len(rows)
    means = [sum(row[j] for row in rows) / n for j in range(width)]
    stds = [math.sqrt(sum((row[j] - means[j]) ** 2 for row in rows) / n) for j in range(width)]
    if any(s == 0 for s in stds):
        raise ValueError("a constant column cannot be scaled")
    return means, stds


def apply_scale(rows, means, stds):
    for row in rows:
        if len(row) != len(means):
            raise ValueError("row length differs from the fitted columns")
    return [[(v - m) / s for v, m, s in zip(row, means, stds)] for row in rows]


def nearest(query, rows):
    if not rows:
        raise ValueError("rows must not be empty")
    if any(len(row) != len(query) for row in rows):
        raise ValueError("query length differs from the rows")
    distances = [math.dist(query, row) for row in rows]
    return distances.index(min(distances))  # index() returns the first minimum: smallest index wins a tie
