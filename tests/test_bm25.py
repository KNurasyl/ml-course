"""BM25 retriever of labs/M10/bm25_hybrid.py (T33): the formula by hand, its parameters and the fused report."""
from __future__ import annotations
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest


SOURCE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("m10_bm25", SOURCE / "labs/M10/bm25_hybrid.py")
lab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lab)
DOCS = ["индекс ускоряет чтение", "индекс замедляет запись и занимает место", "транзакция фиксирует изменения"]


def hand_bm25(docs, query, k1=1.5, b=0.75):
    docs = [d.split() for d in docs]
    n, avgdl = len(docs), sum(map(len, docs)) / len(docs)
    total = []
    for d in docs:
        s = 0.0
        for t in query.split():
            df = sum(t in doc for doc in docs)
            idf = math.log((n - df + 0.5) / (df + 0.5) + 1)
            tf = d.count(t)
            s += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(d) / avgdl))
        total.append(s)
    return total


def test_scores_match_the_formula_by_hand():
    scores = lab.BM25(DOCS).scores("индекс запись")
    assert np.allclose(scores, hand_bm25(DOCS, "индекс запись"))
    assert np.allclose(scores, [0.5296, 1.1844, 0.0], atol=1e-4)  # the lesson example: the rare word lifts the long d2
    assert np.argsort(-scores).tolist() == [1, 0, 2]


def test_unknown_words_add_nothing_and_idf_stays_positive():
    bm25 = lab.BM25(DOCS)
    assert np.allclose(bm25.scores("индекс кеш"), bm25.scores("индекс"))
    assert np.allclose(bm25.scores("кеш"), 0)
    every = lab.BM25(["a b", "a c", "a d"])
    assert (every.idf > 0).all() and every.idf[every.vocab["a"]] < every.idf[every.vocab["b"]]


def test_b_controls_length_and_k1_controls_repetition():
    short, long_ = "a b", "a b c d e f"
    assert lab.BM25([short, long_], b=0).scores("a")[0] == pytest.approx(lab.BM25([short, long_], b=0).scores("a")[1])
    assert lab.BM25([short, long_]).scores("a")[0] > lab.BM25([short, long_]).scores("a")[1]
    once, thrice = lab.BM25(["a x y", "a a a"], k1=0, b=0).scores("a")
    assert once == pytest.approx(thrice)  # k1 = 0: only presence counts
    once, thrice = lab.BM25(["a x y", "a a a"], b=0).scores("a")
    assert once < thrice < 2.5 * once  # repetition helps, but saturates below (k1 + 1) times


def test_tokens_parameters_and_empty_documents():
    assert lab.tokenize("Ёжик, SQL-запрос! 42") == ["ежик", "sql", "запрос", "42"]
    for k1, b in [(-1, 0.5), (1.5, 1.5), (1.5, -0.1)]:
        with pytest.raises(ValueError):
            lab.BM25(DOCS, k1=k1, b=b)
    for docs in [[], ["индекс", "!!!"]]:
        with pytest.raises(ValueError):
            lab.BM25(docs)


def test_lexical_run_reports_three_methods_and_keeps_old_reports(tmp_path):
    before = sorted((SOURCE / "artifacts/M10").glob("ru/*.json")) if (SOURCE / "artifacts/M10/ru").exists() else []
    stamps = {p: p.stat().st_mtime for p in before}
    report = lab.run(use_encoder=False, out_dir=tmp_path)
    assert (tmp_path / "report.json").is_file()
    methods = {key.split("/")[1] for key in report["summary"]}
    assert methods == {"lexical_tfidf", "bm25", "rrf_tfidf_bm25"}
    assert report["baseline"] == "lexical_tfidf" and report["candidate"] == "rrf_tfidf_bm25"
    for values in report["summary"].values():
        assert 0 <= values["recall_at_3"] <= 1 and 0 <= values["mrr"] <= 1 and values["answerable_n"] == 6
    assert {p: p.stat().st_mtime for p in before} == stamps
