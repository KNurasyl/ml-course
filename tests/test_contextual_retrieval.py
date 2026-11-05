"""Contextual retrieval lab (T36): prompt order, representations, the faithfulness signal and a run without models."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("m10_contextual", SOURCE / "labs/M10/contextual_retrieval.py")
lab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lab)
bm = lab.load("bm25_hybrid")


def test_prompt_puts_the_document_before_the_chunk_so_chunks_share_a_prefix():
    first, second = lab.prompt_for("ДОК", "абзац один"), lab.prompt_for("ДОК", "абзац два")
    assert first.index("ДОК") < first.index("абзац один")
    shared = len(next((i for i, (a, b) in enumerate(zip(first, second)) if a != b), len(first)) * "x")
    assert first[:shared].endswith("<фрагмент>\nабзац ") and "ДОК" in first[:shared]


def test_novel_word_share_counts_context_words_missing_from_the_document():
    document = "Индекс ускоряет чтение, но замедляет запись."
    assert lab.novel_word_share("Индекс ускоряет чтение", document, bm.tokenize) == 0.0
    assert lab.novel_word_share("Индекс хранится в облаке", document, bm.tokenize) == 0.75
    assert lab.novel_word_share("", document, bm.tokenize) == 0.0


def test_representations_add_context_before_the_paragraph_and_keep_order():
    rows = [{"id": "a-c0", "title": "Индекс", "text": "Абзац."}, {"id": "a-c1", "title": "Индекс", "text": "Второй."}]
    contexts = {"4b": [{"id": "a-c1", "context": "Ктx2."}, {"id": "a-c0", "context": "Ктx1."}]}
    reps = lab.representations(rows, contexts)
    assert list(reps) == ["text", "title", "ctx_4b", "title_ctx_4b"]
    assert reps["ctx_4b"] == ["Ктx1. Абзац.", "Ктx2. Второй."]
    assert reps["title_ctx_4b"][0] == "Индекс Ктx1. Абзац." and reps["title"][1] == "Индекс Второй."
    assert list(lab.representations(rows, {})) == ["text", "title"]


def test_document_metrics_and_recall_at_20():
    rows = [{"document_id": d} for d in ["a", "a", "b", "c"]]
    assert lab.doc_metrics([0, 1, 3, 2], rows, ["b"]) == {"recall_at_1": 0.0, "recall_at_3": 1.0, "recall_at_20": 1.0, "mrr": 1 / 3,
                                                         "first_rank": 3}
    assert lab.doc_metrics([0], rows, ["c"])["recall_at_20"] == 0.0


def test_corpus_hash_changes_with_text():
    rows = [{"id": "x", "text": "один"}]
    assert lab.corpus_hash(rows) == lab.corpus_hash([dict(rows[0])]) != lab.corpus_hash([{"id": "x", "text": "два"}])


def test_run_without_generators_scores_text_and_title(tmp_path, monkeypatch):
    monkeypatch.setattr(lab, "generate_contexts", lambda name, rows, documents, threads=4: {"status": "not_downloaded"})
    monkeypatch.setattr(lab, "encoder", lambda reps, texts: (None, None))
    report = lab.run(out_dir=tmp_path)
    assert report["generation"] == {"0.5b": {"status": "not_downloaded"}, "4b": {"status": "not_downloaded"}}
    assert list(report["results"]) == ["text", "title"] and list(report["results"]["text"]) == ["bm25"]
    for group in lab.GROUPS:
        row = report["results"]["title"]["bm25"][group]
        assert row["n"] in (24, 12) and 0 <= row["recall_at_1"] <= row["recall_at_3"] <= row["recall_at_20"] <= 1
    assert json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))["task"] == "T36"
