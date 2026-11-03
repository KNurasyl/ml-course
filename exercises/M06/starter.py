"""M06-HW01: implement a versioned, idempotent numerical request."""


def execute(request, release, cache):
    """request={session:str, turn:nonnegative int, x:finite real}; no extra keys.

    release={id:str, mean:finite real, scale:positive finite real,
             weight:finite real, bias:finite real} is one immutable bundle.
    Return {prediction:float, release_id:str}. Cache key is (session, turn).
    Same key+same x returns the ORIGINAL response, even after a new release.
    Same key+different x is a conflict (ValueError). Invalid input must not
    change cache. Reject bool as a number. Never refit preprocessing here.
    """
    raise NotImplementedError("Implement this contract before opening the solution")
