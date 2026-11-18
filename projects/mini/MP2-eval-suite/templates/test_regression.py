"""Regression tests over the MP2 set. Template: replace run_operator with your operator (MP3) or saved answers."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CASES = [json.loads(line) for line in (HERE / "cases.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]


def test_every_unanswerable_case_has_no_sources():
    assert all(not c["expected_sources"] for c in CASES if not c["answerable"])

# TODO: a test that runs the operator (or reads its saved answers) and fails when the share of correct refusals
# or the share of answers citing an expected source drops below your threshold.
