"""Deterministic routing of explicit rubric evidence, not automated essay grading."""
from dataclasses import dataclass


@dataclass(frozen=True)
class GateResult:
    state: str
    next_action: str


def assess_evidence(*, criteria: dict[str, bool], critical_errors: list[str],
                    assistance: str, delayed: bool = False,
                    explanation_checked: bool = False) -> GateResult:
    if assistance not in {"none", "hint", "worked_solution"}:
        raise ValueError("Unknown assistance level")
    if not criteria or any(type(value) is not bool for value in criteria.values()):
        raise ValueError("Explicit boolean evidence is required for each criterion")
    if critical_errors or not all(criteria.values()):
        return GateResult("needs_review", "Explain the failed criterion and assign a new attempt")
    if not explanation_checked:
        return GateResult("not_assessed", "Check reasoning; execution alone is insufficient")
    if assistance != "none":
        return GateResult("with_support", "Assign a new independent variation")
    if delayed:
        return GateResult("retained", "Schedule transfer to another context")
    return GateResult("independent", "Return to the skill after an interval")
