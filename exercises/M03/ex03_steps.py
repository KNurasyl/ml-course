def descend(grad, w0, eta, steps):
    """Return [w0, w1, ..., w_steps] for w_next = w - eta * grad(w).

    Reject eta <= 0, a negative or non-integer number of steps and a non-finite w0.
    """
    raise NotImplementedError("Implement descend")


def stable_step(c):
    """For L(w) = c * (w - m) ** 2 with c > 0 return the bound: descent converges for 0 < eta < bound. Reject c <= 0."""
    raise NotImplementedError("Implement stable_step")


def ridge_optimum(a, lam):
    """Return the minimiser of J(w) = (w - a) ** 2 + lam * w ** 2. Reject lam < 0."""
    raise NotImplementedError("Implement ridge_optimum")
