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
    assert m.predict(-3, 2, 5) == -1
    assert m.predict(4, 0, -2) == -2
    assert m.predict(0.5, -4, 3) == 1
    for x in [True, float("nan"), float("inf"), "3"]:
        rejected(lambda x=x: m.predict(x, 2, 1))
    rejected(lambda: m.predict(1, float("inf"), 0))
    return ["negative_input", "zero_weight", "fraction", "finite_contract"]


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
