def forward_difference(f, w, h):
    """Return (f(w + h) - f(w)) / h. Reject h <= 0 and non-finite w or h."""
    raise NotImplementedError("Implement forward_difference")


def central_difference(f, w, h):
    """Return (f(w + h) - f(w - h)) / (2 * h). Reject h <= 0 and non-finite w or h."""
    raise NotImplementedError("Implement central_difference")


def single_gradient(x, y, w, b):
    """For L = (w * x + b - y) ** 2 return (dL/dw, dL/db) by the chain rule. Reject non-finite inputs."""
    raise NotImplementedError("Implement single_gradient")
