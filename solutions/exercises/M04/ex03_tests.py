"""Reference for M04-ex03: exact two-sided sign test and the binary log loss."""

import math


def _count(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def sign_test_p(wins, losses, ties=0):
    for value, name in ((wins, "wins"), (losses, "losses"), (ties, "ties")):
        _count(value, name)
    n = wins + losses  # ties say nothing about which side is better
    if n == 0:
        raise ValueError("need at least one non-tied pair")
    tail = sum(math.comb(n, k) for k in range(max(wins, losses), n + 1)) / 2**n
    return min(1.0, 2 * tail)


def binary_log_loss(y_true, p):
    y_true, p = list(y_true), list(p)
    if not y_true or len(y_true) != len(p):
        raise ValueError("need two non-empty lists of equal length")
    total = 0.0
    for label, prob in zip(y_true, p):
        if isinstance(label, bool) or label not in (0, 1):
            raise ValueError("labels must be 0 or 1")
        if not (isinstance(prob, (int, float)) and 0 <= prob <= 1):
            raise ValueError("probabilities must lie in [0, 1]")
        q = prob if label == 1 else 1 - prob  # probability the model gave to what actually happened
        if q == 0:
            raise ValueError("the observed class has zero probability: the loss is infinite")
        total -= math.log(q)
    return total / len(y_true)
