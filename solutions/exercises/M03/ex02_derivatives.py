"""Reference for M03-ex02: finite differences and the chain rule for one example."""

import math


def _finite(*values):
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in values):
        raise ValueError("inputs must be finite numbers")


def _step(w, h):
    _finite(w, h)
    if h <= 0:
        raise ValueError("step h must be positive")


def forward_difference(f, w, h):
    _step(w, h)
    return (f(w + h) - f(w)) / h


def central_difference(f, w, h):
    _step(w, h)
    return (f(w + h) - f(w - h)) / (2 * h)


def single_gradient(x, y, w, b):
    _finite(x, y, w, b)
    e = w * x + b - y  # residual: dL/de = 2e, de/dw = x, de/db = 1
    return 2 * e * x, 2 * e
