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
    for xs, ys, w, b in [([-2, 0, 3], [3, 1, 4], 0.7, -0.2), ([0, 0], [1, 3], 2, 0), ([1], [4], 0, 0)]:

        def loss(w, b):
            return sum((w * x + b - y) ** 2 for x, y in zip(xs, ys)) / len(xs)

        dw, db = m.gradient(xs, ys, w, b)
        h = 1e-5
        assert abs(dw - (loss(w + h, b) - loss(w - h, b)) / (2 * h)) < 1e-7
        assert abs(db - (loss(w, b + h) - loss(w, b - h)) / (2 * h)) < 1e-7
    rejected(lambda: m.gradient([], [], 0, 0))
    rejected(lambda: m.gradient([1], [1, 2], 0, 0))
    rejected(lambda: m.gradient([float("nan")], [1], 0, 0))
    return ["finite_difference_negative_zero_inputs", "bias_gradient", "mean_factor", "contract"]


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
