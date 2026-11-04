from pathlib import Path
import argparse, importlib.util, json


def run_checks(module):
    valid = module.respond('{"feedback":"Explain the tradeoff","score":2}')
    assert valid["status"] == "ok" and valid["score"] == 2 and not valid["fallback_used"]
    cases = [
        "",
        "[]",
        '{"score":true,"feedback":"x"}',
        '{"score":3,"feedback":"x"}',
        '{"score":1,"feedback":" "}',
        '{"score":1,"feedback":"x","extra":0}',
        '{"score":',
        'prefix {"score":1,"feedback":"x"}',
        None,
    ]
    for case in cases:
        r = module.respond(case)
        assert r["status"] == "needs_review" and r["score"] is None and r["fallback_used"], (
            f"Invented valid output for {case!r}"
        )
        assert isinstance(r["feedback"], str) and r["feedback"].strip(), (
            "Fallback must actually return a useful response"
        )
    return {"checks": 10, "passed": True, "mastery": "format validity does not establish factual correctness"}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("submission", type=Path)
    a = p.parse_args()
    s = importlib.util.spec_from_file_location("student", a.submission)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    print(json.dumps(run_checks(m)))
