"""Shared helpers for the extra exercise checkers (improvement plan, section 3.4).

A checker loads the learner's file, compares numbers with a tolerance, tests
invariants with hypothesis, reads hidden cases from tests/hidden/ and draws
random inputs from --seed. Messages are in Russian and always name the input,
the expected value and the actual one.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEED = 2026


def load(path: str | Path):
    spec = importlib.util.spec_from_file_location("learner_submission", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def hidden_cases(module_id: str, exercise: str) -> list[dict]:
    """Cases kept out of the README; the learner does not open them under the self-check rules."""
    path = ROOT / "tests" / "hidden" / f"{module_id}_{exercise}.json"
    return json.loads(path.read_text(encoding="utf-8"))["cases"]


def close(actual, expected, what: str, given, rtol: float = 1e-9, atol: float = 1e-9) -> None:
    try:
        ok = np.shape(actual) == np.shape(expected) and np.allclose(
            np.asarray(actual, dtype=float), np.asarray(expected, dtype=float), rtol=rtol, atol=atol
        )
    except (TypeError, ValueError):
        ok = False
    if not ok:
        raise AssertionError(f"{what}: вход {given}; ожидалось {expected}; получено {actual}")


def equal(actual, expected, what: str, given) -> None:
    if actual != expected:
        raise AssertionError(f"{what}: вход {given}; ожидалось {expected!r}; получено {actual!r}")


def rejected(call, what: str, given) -> None:
    try:
        call()
    except (ValueError, TypeError):
        return
    raise AssertionError(f"{what}: вход {given}; ожидалась ошибка ValueError или TypeError; функция приняла вход")


def finite(value: float) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def run_property(test, seed: int) -> None:
    """Run a @given-decorated test with a fixed seed and no example database, so a failure reproduces."""
    from hypothesis import HealthCheck, seed as fix_seed, settings

    configured = settings(
        max_examples=120,
        deadline=None,
        database=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
    )
    configured(fix_seed(seed)(test))()


def main(check, argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check your implementation of the exercise")
    parser.add_argument("path", help="path to your solution file")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="seed for random and property-based cases")
    args = parser.parse_args(argv)
    try:
        checks = check(load(args.path), args.seed)
    except Exception as error:  # Report any learner failure as data, not as a crash of the checker.
        notes = "\n".join(getattr(error, "__notes__", []))
        message = f"{error}\n{notes}".strip()
        print(json.dumps({"passed": False, "error": type(error).__name__, "message": message}, ensure_ascii=False))
        return 1
    print(
        json.dumps(
            {
                "passed": True,
                "seed": args.seed,
                "checks": checks,
                "understanding": "requires your explanation and a new case",
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit("checklib.py is a helper; run exercises/Mxx/check_exNN.py instead")
