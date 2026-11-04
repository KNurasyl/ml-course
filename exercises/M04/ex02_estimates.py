def describe(values):
    """Return (mean, s2, s, se) for a sample: s2 divides by n - 1, s = sqrt(s2), se = s / sqrt(n).

    Reject fewer than two values and non-finite values.
    """
    raise NotImplementedError("Implement describe")


def wilson_interval(k, n, z=1.959963984540054):
    """Return (low, high) of the Wilson interval for k successes out of n independent trials; both lie in [0, 1].

    Reject non-integer k or n, n <= 0, k outside 0..n and z <= 0.
    """
    raise NotImplementedError("Implement wilson_interval")
