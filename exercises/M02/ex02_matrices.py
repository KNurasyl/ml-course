def matmul(a, b):
    """Return the product a·b of two matrices given as lists of rows, without NumPy.

    Reject (ValueError) an empty matrix, ragged rows and incompatible shapes: the number
    of columns of a must equal the number of rows of b. The result has len(a) rows and
    len(b[0]) columns.
    """
    raise NotImplementedError("Implement matmul")


def apply_in_order(matrices, x):
    """Apply square or rectangular matrices to the vector x in the listed order.

    matrices[0] acts first, then matrices[1], and so on; return the final vector as a list.
    Reject an empty list of matrices and any size mismatch along the way.
    """
    raise NotImplementedError("Implement apply_in_order")
