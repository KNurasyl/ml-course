"""Error analysis lab (T37): signals, first failure, Wilson interval and the report built from frozen traces and codes."""
from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("m11_error_analysis", SOURCE / "labs/M11/error_analysis.py")
lab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lab)
ROWS, _ = lab.dataset()
BY_ID = {r["id"]: r for r in ROWS}


def test_signals_separate_citations_refusal_and_content():
    retrieved = ["cache-v2-c1", "index-v2-c1", "rag-v2-c1"]
    only_ids = lab.signals("[cache-v2-c1] [index-v2-c1]", retrieved, ["cache"], BY_ID)
    assert only_ids["content_letters"] == 0 and only_ids["cyrillic_share"] is None and lab.CHECKS["no_answer"](only_ids)
    assert only_ids["cites_prompt_example"] and not only_ids["citation_outside_context"]
    refusal = lab.signals("[cache-v2-c1] В источниках нет ответа.", retrieved, ["cache"], BY_ID)
    assert refusal["refused"] and lab.CHECKS["over_refusal"](refusal) and not lab.CHECKS["no_answer"](refusal)
    outside = lab.signals("[answer-v2-c1] Да.", retrieved, ["cache"], BY_ID)
    assert outside["citation_outside_context"] and not outside["cited_relevant"]
    echo = lab.signals("Вопрос: Что изменить в кэше?", retrieved, ["cache"], BY_ID, question="Что изменить в кэше?")
    assert echo["echoes_question"] and lab.CHECKS["no_answer"](echo)
    cut = lab.signals("[cache-v2-c1] Инвалидировать записи и", retrieved, ["cache"], BY_ID, finish_reason="length")
    assert cut["cut_by_limit"] and lab.CHECKS["format"](cut)


def test_unanswerable_question_without_refusal_is_flagged():
    s = lab.signals("[index-v2-c1]", ["index-v2-c1"], [], BY_ID)
    assert s["answerable"] is False and s["retrieval_hit"] is None and lab.CHECKS["failed_refusal"](s)
    assert not lab.CHECKS["retrieval_miss"](s) and not lab.CHECKS["over_refusal"](s)


def test_first_failure_follows_the_pipeline_order():
    assert lab.first_failure(["bad_citation", "retrieval_miss", "no_answer"]) == "retrieval_miss"
    assert lab.first_failure(["format", "unsupported_claim"]) == "unsupported_claim"
    assert lab.FIRST[0] == "retrieval_miss" and lab.FIRST[-1] == "ok"


def test_wilson_interval_matches_the_lesson():
    assert lab.wilson(43, 66) == [0.531, 0.755] and lab.wilson(2, 66) == [0.008, 0.104] and lab.wilson(0, 0) is None
    low, high = lab.wilson(0, 66)
    assert low == 0 and 0 < high < 0.1


def test_report_needs_every_trace_coded(tmp_path):
    folder = tmp_path / "ea"
    shutil.copytree(lab.FROZEN, folder)
    report = lab.report(folder)
    assert report["traces"] == 66 and sum(v["first_failure"] for v in report["taxonomy"].values()) == 66
    assert report["taxonomy"]["no_answer"]["traces"] == 43 and report["checks"]["retrieval_miss"]["manual_only"] == ["dev-multi", "p-norm"]
    assert set(report["manual_only_codes"]) == {"off_question", "unsupported_claim"}
    codes = json.loads((folder / "codes.json").read_text(encoding="utf-8"))
    del codes["traces"]["t-retry"]
    (folder / "codes.json").write_text(json.dumps(codes, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(AssertionError, match="coded exactly once"):
        lab.report(folder)
