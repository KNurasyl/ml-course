from pathlib import Path
import importlib.util
import sys
import json

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load(path):
    spec = importlib.util.spec_from_file_location("learner_submission", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rejected(call):
    try:
        call()
    except (ValueError, TypeError):
        return
    raise AssertionError("invalid input was accepted")


def check(m):
    assert m.binomial_tail(5, 4) == 6 / 32
    assert abs(m.binomial_tail(3, 2, 0.6) - 0.648) < 1e-12
    assert m.binomial_tail(0, 0) == 1
    assert m.binomial_tail(4, 5) == 0
    assert m.binomial_tail(4, 4, 1) == 1
    rejected(lambda: m.binomial_tail(-1, 0))
    rejected(lambda: m.binomial_tail(2, 1, 1.1))
    rows = [{"n": 800, "exact_power": 0.89}, {"n": 200, "exact_power": 0.42}, {"n": 400, "exact_power": 0.65}]
    assert m.choose_size(rows) == 800 and m.choose_size(rows, 0.95) is None
    return ["inclusive_tail", "unequal_probability", "boundary_cases", "power_not_effect"]


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: check.py path/to/your_solution.py")
    try:
        checks = check(load(sys.argv[1]))
    except Exception as error:
        print(json.dumps({"passed": False, "error": type(error).__name__, "message": str(error)}, ensure_ascii=True))
        raise SystemExit(1)
    print(
        json.dumps(
            {"passed": True, "checks": checks, "understanding": "requires your explanation and a new case"},
            ensure_ascii=True,
        )
    )
