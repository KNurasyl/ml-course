"""Acceptance evidence checks; sample scores are labelled synthetic, never a human pilot."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUBRICS = {"hr": ["clarity", "specificity", "reasoning"],
           "technical": ["correctness", "tradeoffs", "verification"]}


def validate_case(case: dict) -> None:
    if case["track"] not in RUBRICS:
        raise ValueError("unknown interview track")
    expected = set(RUBRICS[case["track"]])
    if set(case["scores"]) != expected:
        raise ValueError("rubric does not match the interview track")
    if any(type(value) is not int or not 0 <= value <= 2 for value in case["scores"].values()):
        raise ValueError("scores must be integers 0, 1, 2")
    if not case["evidence"] or not case["next_step"]:
        raise ValueError("feedback needs evidence and an actionable next step")


def run() -> dict:
    cases = [
        {"id": "synthetic-hr-01", "track": "hr", "scores": {"clarity": 2, "specificity": 1, "reasoning": 1}, "evidence": "Mentions a team result but no personal action.", "next_step": "State which change you personally implemented."},
        {"id": "synthetic-tech-01", "track": "technical", "scores": {"correctness": 2, "tradeoffs": 1, "verification": 0}, "evidence": "Correctly names an index, but gives no EXPLAIN or timing comparison.", "next_step": "Compare query plans before and after adding the index."},
    ]
    for case in cases:
        validate_case(case)
    wrong = dict(cases[0], track="technical")
    try:
        validate_case(wrong)
    except ValueError:
        mixed_rubric_rejected = True
    else:
        raise AssertionError("Mixed HR and technical rubrics must be rejected")
    evidence = {}
    for module in ("M12", "M13", "M14", "M15"):
        report = ROOT / "artifacts" / module / ("metrics.json" if module == "M12" else "report.json")
        evidence[module] = {"available": report.exists(), "path": str(report.relative_to(ROOT))}
    result = {"fixture_cases_valid": len(cases), "mixed_rubric_rejected": mixed_rubric_rejected,
              "evidence_files": evidence, "participant_pilot": "not performed by this lab",
              "integration_required": "Run projects/interview_trainer and independently test both interview tracks, audio, cancellation and failures"}
    out = ROOT / "artifacts" / "M16"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (out / "rubric-examples.json").write_text(json.dumps(cases, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
