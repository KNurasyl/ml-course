"""Validate the student's behavior, not filenames. No model or internet required."""

import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


def check(module):
    good = '{"tool":"read_document","args":{"document_id":"manual-v2"}}'
    assert module.parse_action(good, {"manual-v2"})["args"]["document_id"] == "manual-v2"
    invalid = [
        "x",
        "[]",
        '{"tool":"delete","args":{}}',
        good.replace("manual-v2", "private"),
        '{"tool":"read_document","args":{"document_id":true}}',
        '{"tool":"read_document","args":{"document_id":"manual-v2","session":"admin"}}',
    ]
    for raw in invalid:
        try:
            module.parse_action(raw, {"manual-v2"})
        except ValueError:
            pass
        else:
            raise AssertionError("unsafe proposal accepted: " + raw)
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
        bad = SimpleNamespace(
            parse_action=lambda raw, allowed_ids: {"tool": "read_document", "args": {"document_id": "manual-v2"}}
        )
        try:
            check(bad)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError("checker failed to detect deliberately wrong implementation")
    print(json.dumps({"behavior_passed": True, "mutant_rejected": args.self_test, "mastery": "not_assessed"}))
