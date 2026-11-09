"""0.5B against 4B (T51): the inputs of the comparison, the agreement arithmetic, the annotation contract and the frozen
report, which keeps the negative results (judges no better than chance, no profile passing the M11 gate)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from projects.interview_trainer import compare_profiles as cp
from projects.interview_trainer import compare_report as cr

REPORT = cr.build()


def test_the_comparison_uses_the_m11_cases_and_annotations_made_before_this_task():
    cases = cp.cases()
    assert len(cases) == 16 and {c["track"] for c in cases} == {"hr", "backend"}
    judged = cp.judged_outputs()
    assert len(judged) == 32 and sum(cr.passes(r["label"]) for r in judged) == 18
    assert len(cp.JUDGE_CHOICES) == 6 and all(json.loads(c).keys() == {"useful", "critical_error"} for c in cp.JUDGE_CHOICES)


def test_kappa_matches_the_m11_lesson_example():
    labels, judge = [1, 1, 0, 0], [1, 0, 1, 0]  # M11-L04: accuracy 0.5 and chance-level agreement
    assert cr.kappa(list(zip(labels, judge))) == 0.0
    assert cr.kappa([(1, 1), (0, 0), (1, 1), (0, 0)]) == 1.0


def test_annotations_are_tied_to_the_shown_text():
    collected, annotations = cr.load()
    assert cr.check_annotations(collected, annotations) and len(annotations) == 32
    broken = [dict(annotations[0], evidence="текст, которого не было")] + annotations[1:]
    with pytest.raises(ValueError, match="evidence"):
        cr.check_annotations(collected, broken)
    assert {a["rater"] for a in annotations} == {"course-author AI-assisted review; not independent human validation"}


def test_rubric_pass_and_no_profile_passes_the_m11_gate():
    rubric = REPORT["rubric"]
    assert (rubric["tiny"]["pass"], rubric["strong"]["pass"]) == (8, 10)
    assert rubric["tiny"]["critical"] == rubric["strong"]["critical"] == 0  # the bounded output never asserts a claim
    for profile in ("tiny", "strong"):  # M11 gate: useful >= 1 in at least 75% of answers
        assert rubric[profile]["useful_at_least_1"] / rubric[profile]["n"] < 0.75


def test_both_judges_disagree_with_the_annotations():
    judge = REPORT["judge"]
    assert judge["tiny"]["kappa"] < 0 < judge["strong"]["kappa"] < 0.4
    assert judge["tiny"]["critical_found"] <= 1 and judge["strong"]["critical_found"] == 0  # of 6 labelled critical
    assert judge["strong"]["false_rejections"] > judge["strong"]["false_approvals"]  # the 4B judge is strict, not lenient


def test_json_validity_needs_constraints_for_the_small_model():
    js = REPORT["json"]
    assert js["tiny"]["free_text"]["schema_valid"] == js["tiny"]["json_mode"]["schema_valid"] == 0
    assert js["tiny"]["bounded_choice"]["schema_valid"] == js["strong"]["bounded_choice"]["schema_valid"] == 16
    assert js["strong"]["free_text"]["schema_valid"] == 16


def test_costs_are_recorded_with_the_ram_cache_off():
    cost = REPORT["cost"]
    assert cost["strong"]["peak_mb_end"]["llama_server"] < 6000  # was 9020 MB with the default 8 GiB prompt cache
    for profile in ("tiny", "strong"):
        voice = cost[profile]["voice"]
        assert voice["vosk_sapi"]["p50"] <= voice["vosk_sapi"]["p95"] and voice["whisper_piper"]["p50"] <= voice["whisper_piper"]["p95"]
        assert cost[profile]["dialogue"]["hr"]["turns"] == 4
    assert cost["tiny"]["speed"]["decode_tps"] > cost["strong"]["speed"]["decode_tps"]
