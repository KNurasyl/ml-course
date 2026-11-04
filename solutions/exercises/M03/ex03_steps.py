"""Reference for M03-ex03: gradient descent trajectory, stable step bound and the L2 optimum."""

import math


def descend(grad, w0, eta, steps):
    if not (isinstance(eta, (int, float)) and math.isfinite(eta) and eta > 0):
        raise ValueError("eta must be a positive finite number")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
        raise ValueError("steps must be a non-negative integer")
    if not (isinstance(w0, (int, float)) and math.isfinite(w0)):
        raise ValueError("w0 must be finite")
    path = [w0]
    for _ in range(steps):
        path.append(path[-1] - eta * grad(path[-1]))
    return path


def stable_step(c):
    if not (isinstance(c, (int, float)) and math.isfinite(c) and c > 0):
        raise ValueError("c must be positive")
    return 1 / c  # delta_next = (1 - 2*c*eta) * delta shrinks only while |1 - 2*c*eta| < 1


def ridge_optimum(a, lam):
    if lam < 0:
        raise ValueError("lam must be non-negative")
    return a / (1 + lam)
