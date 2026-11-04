def fit_scale(rows):
    """Return (means, stds): per-column mean and population standard deviation (divide by n).

    Reject (ValueError) an empty table, ragged rows, non-finite values and a constant column,
    whose standard deviation is zero: scaling it is undefined.
    """
    raise NotImplementedError("Implement fit_scale")


def apply_scale(rows, means, stds):
    """Return the table with every value replaced by (x - mean) / std of its column.

    Reject a row whose length differs from len(means).
    """
    raise NotImplementedError("Implement apply_scale")


def nearest(query, rows):
    """Index of the row with the smallest Euclidean distance to query; on a tie the smallest index.

    Reject an empty table and a query of a different length.
    """
    raise NotImplementedError("Implement nearest")
