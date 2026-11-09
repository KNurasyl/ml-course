"""Russian RAG over the interview question base (extension 1.2.0, T52).

The bank is data/interview: 12 backend and 8 HR questions with what to look for in an answer. An interviewer types a
topic in their own words and gets the bank question, or an explicit "not in the bank". Every method searches the same
text per question, the question plus its look-for items, which is what the MCP search of T42 compares.

Before: the 6-letter stem overlap of the MCP server (it failed in T42 and T43) and the char TF-IDF of core.retrieve.
After: BM25 over words (T33), USER-bge-m3 vectors kept in a sqlite-vec file (T32, T34), their RRF, and the
bge-reranker-v2-m3 cross-encoder over the top POOL candidates (T35). POOL, THRESHOLD and the query set
(data/interview/search-queries.json) were fixed before the first run; nothing is trained or tuned.

The first run showed that the reranker probability does not separate "not in the bank" from a paraphrase on this bank:
THRESHOLD rejected all 8 off-bank queries and 18 of 40 answerable ones. So search() always returns the nearest
questions and marks them confident only above THRESHOLD; the interviewer decides, the bank never pretends to cover a topic.

    python -m projects.interview_trainer.question_rag search "как кандидат просит помощи"
    python -m projects.interview_trainer.question_rag evaluate     # needs the T32 and T35 models and sqlite-vec
    python -m projects.interview_trainer.question_rag report       # frozen run in quality/question-search
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
QUERIES = ROOT / "data/interview/search-queries.json"
FROZEN = ROOT / "quality/question-search"
OUT = ROOT / "artifacts/interview_trainer/question-search"
STORE = ROOT / "artifacts/interview_trainer/questions.sqlite"
ENCODER, RERANKER = "USER-bge-m3", "bge-reranker-v2-m3"
POOL, THRESHOLD, RRF_C = 10, 0.1, 10  # fixed before the first run


def load_lab(rel):
    spec = importlib.util.spec_from_file_location(f"t52_{Path(rel).stem}", ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bank() -> list[dict]:
    out = []
    for track, name in (("backend", "backend-development.json"), ("hr", "hr-development.json")):
        data = json.loads((ROOT / "data/interview" / name).read_text(encoding="utf-8"))
        for case in data["cases"]:
            look_for = case.get("expected_concepts") or case.get("evidence_to_seek") or []
            out.append({"id": case["id"], "track": track, "question": case["question"], "look_for": look_for,
                        "followups": case.get("followups", []), "text": case["question"] + " " + " ".join(look_for)})
    return out


def stem6(text):
    """The MCP server's word key of T42: lower-case words longer than two letters cut to six letters."""
    return {w[:6] for w in re.findall(r"[a-zа-яё0-9]+", text.lower()) if len(w) > 2}


def stem6_ranking(query, docs):
    """The MCP search_questions order: overlap of stems, ties by id, questions without overlap are not returned."""
    wanted = stem6(query)
    scored = [(len(wanted & stem6(d["text"])), i) for i, d in enumerate(docs)]
    return [i for s, i in sorted(scored, key=lambda x: (-x[0], docs[x[1]]["id"])) if s > 0]


def order(scores):
    return np.argsort(-np.asarray(scores), kind="stable").tolist()


def rrf(rankings, c=RRF_C):
    scores = {}
    for ranking in rankings:
        for rank, index in enumerate(ranking, 1):
            scores[index] = scores.get(index, 0.0) + 1 / (c + rank)
    return sorted(scores, key=lambda i: (-scores[i], i))


def registry(section, name):
    entries = json.loads((ROOT / "metadata/reference-models.json").read_text(encoding="utf-8"))[section]
    return next(e for e in entries if e["name"] == name)


def full_mode_ready() -> list[str]:
    """What the full pipeline still needs on this machine; empty means ready."""
    missing = []
    for section, name in (("embedders", ENCODER), ("rerankers", RERANKER)):
        if not (ROOT / registry(section, name)["path"] / "config.json").exists():
            missing.append(f"{name}: python tools/download_models.py --{section}")
    if importlib.util.find_spec("sqlite_vec") is None:
        missing.append("sqlite-vec: pip install -r requirements-optional.in")
    return missing


class Lexical:
    """No model: BM25 over words (T33) and char 3-5 TF-IDF (core.retrieve), fused by RRF."""

    def __init__(self, docs):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.bm = load_lab("labs/M10/bm25_hybrid.py")
        self.bm25 = self.bm.BM25([d["text"] for d in docs])
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform([d["text"] for d in docs])

    def bm25_scores(self, query):
        return self.bm25.scores(query)

    def tfidf_scores(self, query):
        return (self.matrix @ self.vectorizer.transform([query]).T).toarray().ravel()


class Encoder:
    """USER-bge-m3 kept in memory; the pooling and prompts are read the same way as ru_embedders.embed of T32."""

    def __init__(self):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.emb = load_lab("labs/M10/ru_embedders.py")
        entry = registry("embedders", ENCODER)
        self.path, self.name = ROOT / entry["path"], f"{entry['repo_id']}@{entry['revision']}"
        self.config = self.emb.encoder_config(self.path)
        started = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(self.path, local_files_only=True)
        self.model = AutoModel.from_pretrained(self.path, local_files_only=True, dtype=torch.float32).eval()
        self.load_seconds = time.perf_counter() - started

    def vectors(self, texts, kind="query", batch=8):
        import torch
        prefix = self.config["query_prompt" if kind == "query" else "document_prompt"]
        out = []
        for i in range(0, len(texts), batch):
            encoded = self.tokenizer([prefix + t for t in texts[i:i + batch]], padding=True, truncation=True, max_length=512, return_tensors="pt")
            with torch.inference_mode():
                hidden = self.model(**encoded).last_hidden_state
                pooled = self.emb.pool(hidden, encoded["attention_mask"], self.config["pooling"])
                out.append(torch.nn.functional.normalize(pooled, p=2, dim=1).numpy())
        return np.concatenate(out).astype(np.float32)


class VectorIndex:
    """The question vectors in one sqlite-vec file (T34). The store remembers the encoder and a hash of the bank texts,
    so a changed bank or another encoder rebuilds the file instead of answering with stale vectors."""

    def __init__(self, encoder, docs, path=STORE):
        store = load_lab("labs/M10/vector_store.py")
        digest = hashlib.sha256("\n".join(d["id"] + "\t" + d["text"] for d in docs).encode("utf-8")).hexdigest()[:16]
        self.key, self.count = f"{encoder.name}#bank-{digest}", len(docs)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        dimension = encoder.vectors(["пробный текст"], "document").shape[1]
        try:
            self.store, self.rebuilt = store.SqliteVecStore(path, self.key, dimension), False
        except store.EncoderMismatch:
            path.unlink()
            self.store, self.rebuilt = store.SqliteVecStore(path, self.key, dimension), True
        if self.store.db.execute("select count(*) from items").fetchone()[0] != len(docs):
            self.store.db.execute("delete from items")
            self.store.add(encoder.vectors([d["text"] for d in docs], "document"), [d["track"] for d in docs])
            self.rebuilt = True

    def ranking(self, vector):
        return [i for i, _ in self.store.search(vector, self.count)]


class QuestionSearch:
    """search(query) -> the bank questions in order, or found = False; mode 'full' with models, else 'lexical'."""

    def __init__(self, mode="auto", store_path=STORE):
        self.docs = bank()
        self.lexical = Lexical(self.docs)
        self.mode = ("full" if not full_mode_ready() else "lexical") if mode == "auto" else mode
        self.lock = threading.Lock()
        if self.mode == "full":
            rerank = load_lab("labs/M10/rerank.py")
            self.encoder = Encoder()
            self.index = VectorIndex(self.encoder, self.docs, store_path)
            self.reranker = rerank.CrossEncoder(ROOT / registry("rerankers", RERANKER)["path"])

    def rankings(self, query, vector=None):
        stages = {"bm25": order(self.lexical.bm25_scores(query)), "tfidf_char": order(self.lexical.tfidf_scores(query))}
        if self.mode == "full":
            vector = self.encoder.vectors([query])[0] if vector is None else vector
            stages["dense"] = self.index.ranking(vector)
            stages["rrf"] = rrf([stages["bm25"], stages["dense"]])
        else:
            stages["rrf"] = rrf([stages["bm25"], stages["tfidf_char"]])
        return stages

    def search(self, query: str, k: int = 3) -> dict:
        if not isinstance(query, str) or not 2 <= len(query.strip()) <= 500:
            raise ValueError("Запрос должен содержать от 2 до 500 символов")
        k = max(1, min(int(k), 10))
        with self.lock:
            stages = self.rankings(query)
            if self.mode == "full":
                pool = stages["rrf"][:POOL]
                logits = self.reranker.score(query, [self.docs[i]["text"] for i in pool])
                ranked = [(pool[j], float(logits[j])) for j in order(logits)]
                best = 1 / (1 + np.exp(-ranked[0][1]))
                confident = bool(best >= THRESHOLD)
                results = [{**self._public(i), "probability": round(float(1 / (1 + np.exp(-s))), 4)} for i, s in ranked[:k]]
            else:
                scores = self.lexical.bm25_scores(query)
                confident, best = None, None  # no model judges relevance in the lexical mode
                results = [{**self._public(i), "bm25": round(float(scores[i]), 3)} for i in stages["rrf"][:k]]
        note = ("Уверенное совпадение." if confident else
                "Нет уверенного совпадения: такой темы может не быть в базе. Проверь ближайшие вопросы сам." if confident is False else
                "Лексический режим без моделей: порядок по общим словам и буквосочетаниям, уверенность не оценивается.")
        return {"mode": self.mode, "confident": confident, "note": note, "threshold": THRESHOLD if self.mode == "full" else None,
                "best_probability": None if best is None else round(float(best), 4), "results": results}

    def _public(self, i):
        d = self.docs[i]
        return {"id": d["id"], "track": d["track"], "question": d["question"], "look_for": d["look_for"], "followups": d["followups"]}


def metrics(ranking, gold, docs):
    ids = [docs[i]["id"] for i in ranking]
    rank = next((n for n, x in enumerate(ids, 1) if x in gold), None)
    return {"rank": rank, "r1": rank == 1, "r3": bool(rank and rank <= 3), "r5": bool(rank and rank <= 5), "rr": 1 / rank if rank else 0.0}


def summarize(rows, method):
    picked = [r["methods"][method] for r in rows]
    n = len(picked)
    return {"recall@1": round(sum(m["r1"] for m in picked) / n, 3), "recall@3": round(sum(m["r3"] for m in picked) / n, 3),
            "recall@5": round(sum(m["r5"] for m in picked) / n, 3), "mrr": round(sum(m["rr"] for m in picked) / n, 3),
            "missed": [r["id"] for r in rows if not r["methods"][method]["rank"]]}


def evaluate(out_dir=OUT, store_path=None):
    """Every method on every query of search-queries.json; also latency, vector parity with T32 and the threshold rule."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = json.loads(QUERIES.read_text(encoding="utf-8"))
    started = time.perf_counter()
    search = QuestionSearch("full", store_path or out / "questions.sqlite")
    loaded = time.perf_counter() - started
    docs = search.docs
    queries = data["answerable"] + data["unanswerable"]
    t0 = time.perf_counter()
    vectors = search.encoder.vectors([q["text"] for q in queries])
    encode_ms = (time.perf_counter() - t0) * 1000 / len(queries)
    reference_docs, reference_queries, _ = load_lab("labs/M10/ru_embedders.py").embed(search.encoder.path, [d["text"] for d in docs], [q["text"] for q in queries[:4]])
    parity = float(max(np.abs(reference_queries - vectors[:4]).max(),
                       np.abs(reference_docs - search.encoder.vectors([d["text"] for d in docs], "document")).max()))
    rows, rerank_ms = [], []
    for q, vector in zip(queries, vectors):
        stages = search.rankings(q["text"], vector)
        stages["stem6"] = stem6_ranking(q["text"], docs)
        t0 = time.perf_counter()
        pool = stages["rrf"][:POOL]
        logits = search.reranker.score(q["text"], [docs[i]["text"] for i in pool])
        rerank_ms.append((time.perf_counter() - t0) * 1000)
        stages["rrf_rerank"] = [pool[j] for j in order(logits)] + stages["rrf"][POOL:]
        all_logits = search.reranker.score(q["text"], [d["text"] for d in docs])
        stages["rerank_all"] = order(all_logits)
        best = float(1 / (1 + np.exp(-max(logits))))
        row = {"id": q["id"], "text": q["text"], "gold": q.get("gold", []), "style": q.get("style", "unanswerable"),
               "best_probability": round(best, 4), "top": {m: [docs[i]["id"] for i in r[:3]] for m, r in stages.items()},
               "stem6_empty": not stages["stem6"]}
        if q.get("gold"):
            row["methods"] = {m: metrics(r, set(q["gold"]), docs) for m, r in stages.items()}
        rows.append(row)
    answerable = [r for r in rows if r["gold"]]
    unanswerable = [r for r in rows if not r["gold"]]
    methods = ["stem6", "tfidf_char", "bm25", "dense", "rrf", "rrf_rerank", "rerank_all"]
    table = {m: summarize(answerable, m) for m in methods}
    by_style = {s: {m: summarize([r for r in answerable if r["style"] == s], m)["recall@1"] for m in methods} for s in ("term", "paraphrase")}
    sweep = [{"threshold": t, "answerable_kept": sum(r["best_probability"] >= t for r in answerable),
              "answerable_kept_and_right": sum(r["best_probability"] >= t and r["methods"]["rrf_rerank"]["r1"] for r in answerable),
              "unanswerable_rejected": sum(r["best_probability"] < t for r in unanswerable)} for t in (0.01, 0.05, 0.1, 0.3, 0.5)]
    report = {"task": "T52", "bank": len(docs), "answerable": len(answerable), "unanswerable": len(unanswerable),
              "pool": POOL, "threshold": THRESHOLD, "rrf_c": RRF_C, "encoder": search.encoder.name,
              "reranker": f"{registry('rerankers', RERANKER)['repo_id']}@{registry('rerankers', RERANKER)['revision']}",
              "store": {"kind": "sqlite-vec", "rebuilt": search.index.rebuilt, "key": search.index.key},
              "vector_parity_with_t32": parity, "table": table, "recall@1_by_style": by_style,
              "stem6_empty": {"answerable": sum(r["stem6_empty"] for r in answerable), "unanswerable": sum(r["stem6_empty"] for r in unanswerable)},
              "threshold_rule": {"answerable_kept": sweep[2]["answerable_kept"], "answerable_kept_and_right": sweep[2]["answerable_kept_and_right"],
                                 "unanswerable_rejected": sweep[2]["unanswerable_rejected"]},
              "threshold_sweep": sweep,
              "latency": {"load_seconds": round(loaded, 1), "encode_ms_per_query": round(encode_ms, 1),
                          "rerank_ms_per_query_pool": round(float(np.median(rerank_ms)), 1)},
              "rows": rows, "collected": time.strftime("%Y-%m-%d"),
              "scope": "Author-written queries on a 20-question synthetic bank; a check of the pipeline, not a benchmark."}
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return report


def table_text(report):
    names = {"stem6": "6 букв (MCP, T42)", "tfidf_char": "символьный TF-IDF", "bm25": "BM25", "dense": "USER-bge-m3",
             "rrf": "RRF BM25 + плотный", "rrf_rerank": "RRF + рерэнкер", "rerank_all": "рерэнкер по всем 20"}
    lines = ["| Метод | Recall@1 | Recall@3 | MRR |", "| --- | --- | --- | --- |"]
    for m, t in report["table"].items():
        lines.append(f"| {names[m]} | {t['recall@1']:.3f} | {t['recall@3']:.3f} | {t['mrr']:.3f} |")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["search", "evaluate", "report"])
    parser.add_argument("query", nargs="?")
    parser.add_argument("--mode", choices=["auto", "full", "lexical"], default="auto")
    args = parser.parse_args()
    if args.command == "search":
        print(json.dumps(QuestionSearch(args.mode).search(args.query or ""), ensure_ascii=False, indent=2))
    else:
        report = evaluate() if args.command == "evaluate" else json.loads((FROZEN / "report.json").read_text(encoding="utf-8"))
        print(table_text(report))
        print(json.dumps({k: report[k] for k in ("threshold_rule", "stem6_empty", "latency")}, ensure_ascii=False))
