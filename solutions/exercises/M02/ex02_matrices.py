"""Reference for M02-ex02: matrix product and the order of linear transformations."""


def _shape(m, name):
    if not m or not all(isinstance(row, (list, tuple)) for row in m) or not m[0]:
        raise ValueError(f"{name} must be a non-empty list of rows")
    width = len(m[0])
    if any(len(row) != width for row in m):
        raise ValueError(f"{name} has rows of different length")
    return len(m), width


def matmul(a, b):
    rows, inner = _shape(a, "a")
    inner_b, cols = _shape(b, "b")
    if inner != inner_b:
        raise ValueError(f"cannot multiply {rows}x{inner} by {inner_b}x{cols}")
    return [[sum(a[i][k] * b[k][j] for k in range(inner)) for j in range(cols)] for i in range(rows)]


def apply_in_order(matrices, x):
    if not matrices:
        raise ValueError("at least one matrix is required")
    vector = [[value] for value in x]
    for matrix in matrices:
        vector = matmul(matrix, vector)
    return [row[0] for row in vector]
