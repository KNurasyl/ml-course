"""Check numerical behavior in a learner-owned Python file, not understanding.

The supplied file is imported and executed locally. Use only a file you trust.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--task", choices=["mse", "gradient", "predict"], required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("learner_submission", args.path.resolve())
    if spec is None or spec.loader is None:
        raise SystemExit("Could not load the Python file")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    checks = []

    def check(name, condition):
        checks.append({"name": name, "passed": bool(condition)})

    try:
        if args.task == "mse":
            function = module.mse
            check("known_squared_error", math.isclose(function([3, 5, 7], [2, 3, 4]), 14 / 3, rel_tol=1e-9))
            check("zero_error", math.isclose(function([1, 2], [1, 2]), 0, abs_tol=1e-12))
            for name, a, b in [("empty", [], []), ("unequal_length", [1], [1, 2]), ("nonfinite", [float("nan")], [1])]:
                try:
                    function(a, b)
                except ValueError:
                    check(name, True)
                else:
                    check(name, False)
        elif args.task == "gradient":
            dw, db = module.linear_gradient([0, 1, 2], [1, 3, 5], 0, 0)
            check("weight_derivative", math.isclose(dw, -26 / 3, rel_tol=1e-8))
            check("bias_derivative", math.isclose(db, -6, rel_tol=1e-8))
            dw, db = module.linear_gradient([0, 1], [2, 5], 0, 0)
            check("new_example", math.isclose(dw, -5) and math.isclose(db, -7))
        else:
            check("new_input", module.predict_payload({"x": 4}) == {"prediction": 9.0, "model_version": "v1"})
            for request in ({}, {"x": "4"}, {"x": float("nan")}):
                try:
                    module.predict_payload(request)
                except ValueError:
                    check("reject_invalid_request", True)
                else:
                    check("reject_invalid_request", False)
    except Exception as error:
        checks.append({"name": "execution", "passed": False, "error": f"{type(error).__name__}: {error}"})
    report = {"task": args.task, "checks": checks, "behavior_passed": all(row["passed"] for row in checks), "learner_mastery": "not_assessed", "next": "Explain your reasoning to the mentor and solve a new example."}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["behavior_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
