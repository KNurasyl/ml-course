"""Reference for M04-ex02: sample summary with the standard error and the Wilson interval for a proportion."""

import math


def describe(values):
    values = list(values)
    if len(values) < 2:
        raise ValueError("need at least two values")
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in values):
        raise ValueError("values must be finite numbers")
    n = len(values)
    mean = sum(values) / n
    s2 = sum((v - mean) ** 2 for v in values) / (n - 1)  # n - 1: the mean is estimated from the same data
    s = math.sqrt(s2)
    return mean, s2, s, s / math.sqrt(n)


def wilson_interval(k, n, z=1.959963984540054):
    if isinstance(k, bool) or isinstance(n, bool) or not isinstance(k, int) or not isinstance(n, int):
        raise TypeError("k and n must be integers")
    if n <= 0 or not 0 <= k <= n:
        raise ValueError("need n > 0 and 0 <= k <= n")
    if not (isinstance(z, (int, float)) and math.isfinite(z) and z > 0):
        raise ValueError("z must be positive")
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den  # pulled towards 0.5, so the interval stays inside [0, 1]
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, center - half), min(1.0, center + half)
