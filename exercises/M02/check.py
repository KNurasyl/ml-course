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


import math


def check(m):
    ids = ["V", "U", "W", "X", "Y"]
    D = [[2, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0]]
    rows = m.search(ids, D, [1, 1, 0], 3)
    assert [r[0] for r in rows] == ["U", "V", "W"]
    assert all(abs(r[1] - 2**-0.5) < 1e-12 for r in rows)
    assert m.recall(["U", "U", "V"], ["U", "W"]) == 0.5
    assert [r[0] for r in m.search(ids, D, [10, 10, 0], 2)] == ["U", "V"]
    rejected(lambda: m.search(ids, D, [0, 0, 0], 2))
    rejected(lambda: m.search(ids, D, [1, 0], 2))
    rejected(lambda: m.search(ids, D, [1, 0, 0], True))
    rejected(lambda: m.search(["a", "a"], [[1, 0], [0, 1]], [1, 0], 1))
    rejected(lambda: m.recall(["U"], []))
    return ["3D_cosine", "ID_tie_rule", "query_scale", "set_recall", "invalid_input"]


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
