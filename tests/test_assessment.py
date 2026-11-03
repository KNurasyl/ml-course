import json
from pathlib import Path

import pytest

from mlcourse.assessment import assess_evidence

CASES = json.loads((Path(__file__).resolve().parents[1] / "data/assessment/rubric-scenarios.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
def test_evidence_gate(case):
    fields = {k: v for k, v in case.items() if k not in {"id", "description", "expected"}}
    assert assess_evidence(**fields).state == case["expected"]


def test_delayed_supported_result_is_not_mastery():
    result = assess_evidence(criteria={"correct": True}, critical_errors=[],
                             assistance="worked_solution", delayed=True, explanation_checked=True)
    assert result.state == "with_support"


def test_gate_does_not_accept_truthy_strings():
    with pytest.raises(ValueError):
        assess_evidence(criteria={"correct": "false"}, critical_errors=[], assistance="none")
