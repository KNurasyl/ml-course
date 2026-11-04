"""Validate the student's behavior, not filenames. No model or internet required."""

import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


def check(module):
    a = {"version": "A", "schema": 1}
    b = {"version": "B", "schema": 1}
    assert module.choose_release(a, b, {"schema": True, "canary": True, "regressions": True}) == b
    assert module.choose_release(a, b, {"schema": True, "canary": False, "regressions": True}) == a
    assert module.choose_release(a, dict(b, schema=2), {"schema": True, "canary": True, "regressions": True}) == a
    assert module.choose_release(a, b, {"schema": True, "canary": True}) == a
    assert module.choose_release(a, b, {"schema": True, "canary": 1, "regressions": True}) == a
    return True


def load(path):
    spec = importlib.util.spec_from_file_location("student_exercise", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("submission", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    check(load(args.submission))
    if args.self_test:
        bad = SimpleNamespace(choose_release=lambda previous, candidate, checks: candidate)
        try:
            check(bad)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError("checker failed to detect deliberately wrong implementation")
    print(json.dumps({"behavior_passed": True, "mutant_rejected": args.self_test, "mastery": "not_assessed"}))
