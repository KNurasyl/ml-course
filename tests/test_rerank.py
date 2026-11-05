"""Reranking lab of labs/M10/rerank.py (T35): second stages reorder only the pool; metrics and a run without the model."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

SOURCE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("m10_rerank", SOURCE / "labs/M10/rerank.py")
lab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lab)
bm = lab.load("bm25_hybrid")


def test_heuristic_counts_query_words_ignores_stop_words_and_keeps_order_on_ties():
    texts = ["индекс замедляет запись", "кэш и индекс", "ёлка", "индекс запись таблица"]
    pool = [1, 0, 2, 3]
    # 'как' and 'на' are stop words; 'елка' matches 'ёлка' after ё → е
    assert lab.heuristic("Как индекс влияет на запись?", pool, texts, bm.tokenize) == [0, 3, 1, 2]
    assert lab.heuristic("елка", pool, texts, bm.tokenize)[0] == 2
    assert lab.heuristic("как на", pool, texts, bm.tokenize) == pool  # no words left: first-stage order


def test_by_scores_reorders_only_the_pool_and_keeps_order_on_ties():
    scores = np.array([9.0, 1.0, 5.0, 5.0, 100.0])
    assert lab.by_scores([1, 2, 3], scores) == [2, 3, 1]
    assert lab.by_scores([3, 2, 1], scores) == [3, 2, 1]
    assert 4 not in lab.by_scores([0, 1], scores)  # the best chunk outside the pool cannot appear


def test_document_metrics_collapse_chunks_and_mark_misses():
    rows = [{"document_id": d} for d in ["a", "a", "b", "c"]]
    assert lab.documents_of([1, 0, 3, 2], rows) == ["a", "c", "b"]
    hit = lab.doc_metrics([0, 1, 3, 2], rows, ["b"])
    assert hit["first_rank"] == 3 and hit["mrr"] == 1 / 3 and hit["recall_at_1"] == 0 and hit["recall_at_3"] == 1
    miss = lab.doc_metrics([0, 1], rows, ["c"])
    assert miss == {"recall_at_1": 0.0, "recall_at_3": 0.0, "mrr": 0.0, "in_pool": 0.0, "first_rank": None}


def test_stop_words_are_lowercase_and_cover_question_words():
    assert all(word == word.lower() for word in lab.STOP)
    assert {"как", "почему", "что", "сколько"} <= lab.STOP and "индекс" not in lab.STOP


def test_run_without_reranker_model_reports_first_stages_and_heuristic(tmp_path, monkeypatch):
    registry = json.loads((SOURCE / "metadata/reference-models.json").read_text(encoding="utf-8"))
    assert [e["name"] for e in registry["rerankers"]] == ["bge-reranker-v2-m3"]
    fake_root = tmp_path / "root"
    (fake_root / "metadata").mkdir(parents=True)
    (fake_root / "metadata/reference-models.json").write_text(json.dumps(registry), encoding="utf-8")
    monkeypatch.setattr(lab, "ROOT", fake_root)  # no model files under the fake root
    report = lab.run(use_minilm=False, use_dense=True, use_reranker=True, out_dir=tmp_path / "out")
    assert report["reranker"]["status"] == "not_downloaded"
    assert set(report["results"]) == {"lexical_rrf"} and set(report["results"]["lexical_rrf"]) == {"none", "heuristic"}
    for table in report["results"]["lexical_rrf"].values():
        assert [table[g]["n"] for g in lab.GROUPS] == [24, 24, 12]
        assert all(0 <= table[g]["recall_at_1"] <= table[g]["recall_at_3"] <= table[g]["in_pool"] <= 1 for g in lab.GROUPS)
    none, heuristic = (report["per_query"]["lexical_rrf"][k] for k in ("none", "heuristic"))
    assert [r["in_pool"] for r in none if "mrr" in r] == [r["in_pool"] for r in heuristic if "mrr" in r]  # same pool
    assert (tmp_path / "out/report.json").is_file() and "| lexical_rrf | heuristic |" in lab.table_text(report)


def test_sigmoid_maps_logits_to_probability_scale():
    assert np.allclose(lab.sigmoid([0, 100, -100]), [0.5, 1, 0], atol=1e-12)
