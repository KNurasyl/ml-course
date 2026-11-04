"""Check a learner implementation; does not grade explanations or independence."""

from pathlib import Path
import importlib.util
import json
import sys


def checks(fn):
    a = [
        {"case_id": "x", "score": 1, "critical": False},
        {"case_id": "y", "score": 2, "critical": False},
        {"case_id": "z", "score": 0, "critical": False},
    ]
    b = [
        {"case_id": "z", "score": 2, "critical": False},
        {"case_id": "x", "score": 2, "critical": True},
        {"case_id": "y", "score": 1, "critical": False},
    ]
    result = fn(a, b)
    assert result["improved"] == ["x", "z"] and result["regressed"] == ["y"], "join by ID, not row index"
    assert result["critical_regressions"] == ["x"], "critical change is independent of numeric score"
    assert abs(result["mean_delta"] - 2 / 3) < 1e-10
    assert fn(a, a)["mean_delta"] == 0
    for aa, bb in [
        (a, b[:-1]),
        (a + [a[0]], b),
        ([], []),
        (a, [dict(r, score=None) for r in b]),
        (a, [dict(r, score=True) for r in b]),
    ]:
        try:
            fn(aa, bb)
        except ValueError:
            continue
        raise AssertionError("invalid or missing rating must be rejected, not silently repaired")
    return {"behavior_checks": 9, "passed": True, "mastery": "explanation and new attempt required"}


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python exercises/M11/check.py path/to/your.py")
    spec = importlib.util.spec_from_file_location("learner_m11", Path(sys.argv[1]))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    print(json.dumps(checks(module.compare_pairs)))
