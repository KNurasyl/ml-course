import json
from pathlib import Path
import pytest
from mlcourse.assessment import assess_evidence

CASES=json.loads((Path(__file__).resolve().parents[1]/"metadata/assessment-cases.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("case",CASES,ids=lambda c:c["id"])
def test_required_mastery_scenarios(case):
    payload={k:v for k,v in case.items() if k not in ("id","expected")}
    assert assess_evidence(**payload).state==case["expected"]


def test_four_strengths_cannot_compensate_one_failed_condition():
    result=assess_evidence(criteria={"mechanism":True,"code":True,"evidence":True,"presentation":True,"independent_transfer":False},
        critical_errors=[],assistance="none",explanation_checked=True)
    assert result.state=="needs_review"
