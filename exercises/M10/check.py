from pathlib import Path
import argparse, importlib.util, json


def run_checks(module):
    r = module.retrieval_scores(["b", "a", "c"], ["a", "d"], 2, False)
    assert r == {"answerable": True, "recall": 0.5, "mrr": 0.5, "correct_abstention": None}, (
        "Recall denominator or reciprocal rank incorrect"
    )
    assert module.retrieval_scores([], ["a"], 3, True)["recall"] == 0
    for refused in [False, True]:
        r = module.retrieval_scores(["x"], [], 3, refused)
        assert r == {"answerable": False, "recall": None, "mrr": None, "correct_abstention": refused}, (
            "Undefined recall is not zero"
        )
    for ranking, k in [(["a", "a"], 2), (["a"], 0), (["a"], True)]:
        try:
            module.retrieval_scores(ranking, ["a"], k, False)
        except ValueError:
            pass
        else:
            raise AssertionError("Duplicate ranking or invalid k accepted")
    return {"checks": 7, "passed": True, "mastery": "also explain document vs chunk vs claim"}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("submission", type=Path)
    a = p.parse_args()
    s = importlib.util.spec_from_file_location("student", a.submission)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    print(json.dumps(run_checks(m)))
