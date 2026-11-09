"""Search over the interview question base (T52): the query set, the MCP baseline it replaces, RRF, the lexical mode and
the /questions endpoint without models, and the frozen before/after report with its negative results."""
from __future__ import annotations

import http.client
import importlib.util
import json
import sys
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from projects.interview_trainer import question_rag as rag

REPORT = json.loads((ROOT / "quality/question-search/report.json").read_text(encoding="utf-8"))
QUERIES = json.loads(rag.QUERIES.read_text(encoding="utf-8"))


def test_query_set_covers_every_question_twice_and_off_bank_topics():
    ids = {q["id"] for q in rag.bank()}
    answerable = QUERIES["answerable"]
    assert len(ids) == 20 and len(answerable) == 40 and len(QUERIES["unanswerable"]) == 8
    assert {g for q in answerable for g in q["gold"]} == ids
    for qid in ids:  # one term query and one paraphrase per bank question
        assert sorted(q["style"] for q in answerable if q["gold"] == [qid]) == ["paraphrase", "term"]


def test_stem6_reproduces_the_t42_and_t43_misses():
    docs = rag.bank()
    assert rag.stem6_ranking("просит помощи", docs) == []                     # «просит» is not «просишь», «помощи» not «помощь»
    assert docs[rag.stem6_ranking("сложная задача", docs)[0]]["id"] == "HR04"  # «задача» matches «задачах», not HR01


def test_stem6_is_the_mcp_server_search():
    pytest.importorskip("mcp")
    spec = importlib.util.spec_from_file_location("t52_mcp_server", ROOT / "labs/M14/mcp_server.py")
    server = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(server)
    for text in ("Как ты просишь помощь, когда застрял?", "параметризованные SQL-запросы", "идемпотентность"):
        assert rag.stem6(text) == server.words(text)
    assert [q["id"] for q in server.questions()] == [d["id"] for d in rag.bank()]


def test_rrf_rewards_agreement_between_rankings():
    assert rag.rrf([[0, 1, 2], [1, 0, 2]]) == [0, 1, 2]  # 0 and 1 tie on score, the smaller index wins
    assert rag.rrf([[2, 0, 1], [2, 1, 0]])[0] == 2


def test_lexical_mode_needs_no_model_and_does_not_claim_confidence():
    search = rag.QuestionSearch("lexical")
    result = search.search("как кандидат просит помощи", k=20)
    assert result["mode"] == "lexical" and result["confident"] is None and result["results"][0]["id"] == "HR07"
    assert len(result["results"]) == 10  # k is capped
    with pytest.raises(ValueError, match="от 2 до 500"):
        search.search(" ")


def test_the_trainer_serves_question_search():
    from projects.interview_trainer import app
    app.QUESTIONS.update(mode="lexical", search=None)
    server = ThreadingHTTPServer(("127.0.0.1", 0), app.Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        connection = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=30)
        connection.request("GET", "/questions?q=%D0%B8%D0%B4%D0%B5%D0%BC%D0%BF%D0%BE%D1%82%D0%B5%D0%BD%D1%82%D0%BD%D0%BE%D1%81%D1%82%D1%8C&k=1")
        body = json.loads(connection.getresponse().read())
        assert body["results"][0]["id"] == "TECH04" and len(body["results"]) == 1
        connection.request("GET", "/questions?q=x")
        assert connection.getresponse().status == 400
    finally:
        server.shutdown()
        server.server_close()
        app.QUESTIONS.update(mode="auto", search=None)


def test_frozen_report_before_and_after():
    table = REPORT["table"]
    assert (table["stem6"]["recall@1"], table["tfidf_char"]["recall@1"]) == (0.65, 0.65)   # before
    assert (table["rrf_rerank"]["recall@1"], table["rrf_rerank"]["mrr"]) == (0.9, 0.933)    # after
    assert table["rrf"]["recall@1"] < table["dense"]["recall@1"]  # fusing with plain BM25 hurt the dense ranking
    assert REPORT["recall@1_by_style"]["paraphrase"]["stem6"] == 0.4 and REPORT["recall@1_by_style"]["paraphrase"]["rrf_rerank"] == 0.8
    rows = {r["id"]: r for r in REPORT["rows"]}
    assert rows["q37"]["top"]["rrf_rerank"][0] == "HR07" and rows["q37"]["stem6_empty"]   # «просит помощи»
    assert rows["q25"]["top"]["rrf_rerank"][0] == "HR01" and rows["q25"]["top"]["stem6"][0] == "HR04"
    assert REPORT["vector_parity_with_t32"] < 1e-5 and REPORT["store"]["kind"] == "sqlite-vec"


def test_the_threshold_did_not_separate_off_bank_queries_and_stays_reported():
    rule = REPORT["threshold_rule"]
    assert REPORT["threshold"] == rag.THRESHOLD == 0.1
    assert (rule["unanswerable_rejected"], rule["answerable_kept"]) == (8, 22)  # 18 of 40 answerable rejected too
    answerable = [r["best_probability"] for r in REPORT["rows"] if r["gold"]]
    off_bank = [r["best_probability"] for r in REPORT["rows"] if not r["gold"]]
    assert min(answerable) <= max(off_bank)  # the distributions overlap: no threshold separates them on this bank
