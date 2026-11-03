"""Evaluation cannot silently discard failed pairs or reuse stale judgement."""
import copy
import importlib.util
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    spec = importlib.util.spec_from_file_location("eval_" + Path(path).stem, ROOT/path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


ev = load("labs/M11/evaluate.py")


def fixture():
    cases = [{"case_id": "a", "track": "hr", "split": "check", "scenario_id": "s", "question": "Q", "answer": "A", "origin": "test fixture, not model run"}]
    rows=[]; annotations=[]
    for config in ev.CONFIGS:
        result={"display_text":"Observed text", "fallback_used":False}
        row = {"case_id":"a","track":"hr","split":"check","scenario_id":"s", "sources":[],
            "config":config,"result":result,"output_sha256":ev.digest(result),"seconds":1,
            "case_sha256":ev.digest(cases[0])}
        row["record_sha256"] = ev.digest(row)
        rows.append(row)
        annotations.append({"case_id":"a","config":config,"output_sha256":ev.digest(result),"rubric_version":ev.RUBRIC,
            "grounded":2,"useful":2,"critical_error":False,"evidence":"Observed","reason":"fixture only","rater":"synthetic-test"})
    return cases,rows,annotations


def test_paired_score_preserves_tracks_and_critical_gate():
    cases,rows,a=fixture(); a[1]["critical_error"]=True
    report=ev.evaluate_annotations(cases,rows,a)["tracks"]["hr"]["check"]
    assert report["critical_regressions"]==["a"] and not report["release_allowed"]


@pytest.mark.parametrize("mutation", ["missing","duplicate","stale_hash","missing_rating","invented_quote","changed_output","rubric"])
def test_invalid_judgement_is_not_a_score(mutation):
    cases,rows,a=fixture()
    if mutation=="missing":a.pop()
    elif mutation=="duplicate":a.append(a[0])
    elif mutation=="stale_hash":a[0]["output_sha256"]="old"
    elif mutation=="missing_rating":a[0]["grounded"]=None
    elif mutation=="invented_quote":a[0]["evidence"]="Not observed"
    elif mutation=="changed_output":rows[0]["result"]["display_text"]="New response"
    else:a[0]["rubric_version"]="another"
    with pytest.raises(ValueError):ev.evaluate_annotations(cases,rows,a)


def test_rerunning_collection_cannot_overwrite_outputs(tmp_path):
    (tmp_path/"outputs.jsonl").write_text("original")
    with pytest.raises(FileExistsError):ev.collect(fixture()[0],object(),tmp_path)
    assert (tmp_path/"outputs.jsonl").read_text()=="original"


@pytest.mark.parametrize("field,value", [("track", "backend"), ("scenario_id", "other"),
    ("sources", [{"text": "replaced evidence"}]), ("seconds", 999)])
def test_provenance_mutation_is_rejected(field, value):
    cases, rows, annotations = fixture()
    rows[0][field] = value
    with pytest.raises(ValueError):
        ev.evaluate_annotations(cases, rows, annotations)


def test_new_scenario_cannot_leak_across_split():
    cases=fixture()[0];cases.append(dict(cases[0],case_id="b",split="development"))
    with pytest.raises(ValueError,match="scenario leakage"):ev.validate_cases(cases)


def test_student_checker_accepts_reference_and_rejects_zip_or_missing_critical():
    checker=load("exercises/M11/check.py")
    good=load("solutions/exercises/M11/solution.py").compare_pairs
    assert checker.checks(good)["passed"]
    def bad(a,b):
        result=good(a,b);result["critical_regressions"]=[];return result
    with pytest.raises(AssertionError):checker.checks(bad)
