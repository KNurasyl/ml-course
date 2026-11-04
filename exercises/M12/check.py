from pathlib import Path
import argparse, importlib.util, json


def run_checks(module):
    r = module.training_example([4, 5], [6, 7, 2], 7, 0)
    assert r == {
        "input_ids": [4, 5, 6, 7, 2, 0, 0],
        "attention_mask": [1, 1, 1, 1, 1, 0, 0],
        "labels": [-100, -100, 6, 7, 2, -100, -100],
    }, "Wrong assistant boundary, padding or extra shift"
    pairs = [(r["input_ids"][t], r["labels"][t + 1]) for t in range(6) if r["labels"][t + 1] != -100]
    assert pairs == [(5, 6), (6, 7), (7, 2)], "First assistant token or EOS lost"
    r2 = module.training_example([9, 8, 7], [2], 4, 99)
    assert r2["labels"] == [-100, -100, -100, 2] and sum(r2["attention_mask"]) == 4
    for prefix, answer, limit in [([1], [2, 3], 2), ([1], [], 4), ([], [2], 4)]:
        try:
            module.training_example(prefix, answer, limit)
        except ValueError:
            pass
        else:
            raise AssertionError("Overflow or empty supervision must be rejected")
    return {"checks": 6, "passed": True, "mastery": "explain why tokenizer template prefix must match full IDs"}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("submission", type=Path)
    a = p.parse_args()
    s = importlib.util.spec_from_file_location("student", a.submission)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    print(json.dumps(run_checks(m)))
