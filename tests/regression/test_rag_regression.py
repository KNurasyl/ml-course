"""Regression set built from error analysis of 66 real RAG traces (T37, lesson M11-L07X).

The frozen traces and the author's codes live in quality/error-analysis. These tests protect what must not silently change:
the deterministic retrieval of every coded question, the automatic checks that stand for manual codes (they may not get
worse on the coded set), the dangerous cases found by coding, and the prompt contract. Model answers are not re-generated
here; the optional live test does that only when M11_LIVE=1 and the local model is present.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("m11_error_analysis_regression", SOURCE / "labs/M11/error_analysis.py")
lab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lab)
TRACES = lab.recompute(lab.read_traces())
CODES = json.loads((lab.FROZEN / "codes.json").read_text(encoding="utf-8"))["traces"]
BY_ID = {t["trace_id"]: t for t in TRACES}
# Agreement of each check with the manual code on the frozen set when the set was coded (06.11.2026).
FLOOR = {"retrieval_miss": (1.0, 0.75), "failed_refusal": (1.0, 0.5), "over_refusal": (1.0, 1.0), "no_answer": (1.0, 1.0),
         "bad_citation": (1.0, 0.944), "format": (1.0, 1.0)}


def test_every_frozen_trace_is_coded_with_known_codes():
    assert len(TRACES) == 66 and set(CODES) == set(BY_ID)
    assert all(entry["codes"] and set(entry["codes"]) <= set(lab.CODES) for entry in CODES.values())


def test_retrieval_of_every_coded_question_is_unchanged():
    rows, _ = lab.dataset()
    retriever = lab.Retriever(rows)
    changed = {t["trace_id"]: (t["retrieved"], retriever.top(t["question"])) for t in TRACES if retriever.top(t["question"]) != t["retrieved"]}
    assert not changed, f"retrieval changed for {sorted(changed)}: re-run error analysis before updating the frozen set"


@pytest.mark.parametrize("code", sorted(FLOOR))
def test_checks_do_not_get_worse_on_the_coded_set(code):
    auto = {tid for tid, t in BY_ID.items() if lab.CHECKS[code](t["signals"])}
    manual = {tid for tid, entry in CODES.items() if code in entry["codes"]}
    precision = len(auto & manual) / len(auto) if auto else 1.0
    recall = len(auto & manual) / len(manual) if manual else 1.0
    floor_precision, floor_recall = FLOOR[code]
    assert precision >= floor_precision - 1e-9 and recall >= floor_recall - 1e-3, (code, sorted(auto - manual), sorted(manual - auto))


def test_dangerous_case_is_still_flagged():
    # p-permission: the answer tells the system to obey an instruction found in a document and cites a chunk that does not exist.
    signals = BY_ID["p-permission"]["signals"]
    assert "unsupported_claim" in CODES["p-permission"]["codes"]
    assert lab.CHECKS["bad_citation"](signals) and "answer-v2-c1" in signals["cited"]


@pytest.mark.parametrize("trace_id", [tid for tid, entry in CODES.items() if entry["codes"] == ["ok"]])
def test_correct_answers_raise_no_alarm(trace_id):
    signals = BY_ID[trace_id]["signals"]
    assert not any(rule(signals) for rule in lab.CHECKS.values()), trace_id


def test_prompt_example_cause_is_counted():
    cause = [t["trace_id"] for t in TRACES if t["signals"]["cites_prompt_example"]]
    bad = {tid for tid, entry in CODES.items() if "bad_citation" in entry["codes"]}
    assert len(cause) == 24 and len(set(cause) & bad) == 15 and "[index-v2-c1]" in lab.SYSTEM


def test_prompt_lists_sources_in_retrieval_order_with_ids():
    rows, _ = lab.dataset()
    by_id = {r["id"]: r for r in rows}
    text = lab.prompt("Вопрос?", ["cache-v2-c1", "index-v2-c1"], by_id)
    assert text.index("[cache-v2-c1]") < text.index("[index-v2-c1]") < text.index("Вопрос: Вопрос?")
    assert lab.REFUSAL in lab.SYSTEM


@pytest.mark.skipif(os.environ.get("M11_LIVE") != "1", reason="set M11_LIVE=1 to re-run the local model on the regression cases")
def test_live_model_keeps_the_frozen_check_labels(tmp_path):
    subset = ["t-retry", "dev-cache", "p-timeout", "p-permission", "none-2", "check-split"]
    fresh = {t["trace_id"]: t for t in lab.collect(tmp_path, limit=None) if t["trace_id"] in subset}
    for trace_id in subset:
        old, new = BY_ID[trace_id]["signals"], lab.recompute([fresh[trace_id]])[0]["signals"]
        assert {c: lab.CHECKS[c](old) for c in lab.CHECKS} == {c: lab.CHECKS[c](new) for c in lab.CHECKS}, trace_id
