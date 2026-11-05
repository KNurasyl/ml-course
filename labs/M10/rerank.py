"""Cross-encoder reranking over first-stage candidates (extension 1.2.0, T35, lesson M10-L09X).

Corpus, chunks and queries are those of ru_embedders.py (T32): 72 paragraph chunks of 24 current documents,
48 ru-embed-1 queries with an answer and 4 without, plus the 12 answerable diagnostic queries of 1.1.0.
First stages: RRF of char TF-IDF and BM25 (no model), English MiniLM, USER-bge-m3 if downloaded. Each gives a pool of
the top N = 20 chunks, and second stages reorder only that pool:
- none: the first-stage order;
- heuristic: number of query words found in the chunk, the rule of demo.py moved to Russian words, ties keep the order;
- cross-encoder: bge-reranker-v2-m3 reads every (query, chunk) pair together and returns one relevance logit.
Metrics are document level after collapsing chunks, as in T32. Candidate Recall@N bounds any reranker.
Latency: milliseconds per query for N = 5, 10, 20, 40 pairs on CPU. Pool size, stop words, max_length 512,
the first stages and the query set were fixed before the reranker ran; nothing is trained or tuned.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/M10/rerank"
POOL = 20
LATENCY_N = [5, 10, 20, 40]
LATENCY_QUERIES = 16
GROUPS = ["ru-embed-1/paraphrase", "ru-embed-1/term", "diagnostic-1.1.0/diagnostic"]
STOP_WORDS = ("и в во на не что как ли а но или по с со к ко из за от до для о об при это то же бы его её ее их мы вы он она "
              "они я ты у так там где когда почему зачем какой какая какие каких каким чем чему кто все всё ещё еще только "
              "уже можно нужно есть был была было быть ли сколько чтобы если")
STOP = set(STOP_WORDS.split())


def load(name):
    spec = importlib.util.spec_from_file_location(f"m10_{name}", Path(__file__).with_name(f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def query_words(text, tokenize):
    return set(tokenize(text)) - STOP


def heuristic(query, pool, texts, tokenize):
    """Second stage of 1.1.0: more query words in the chunk first; ties keep the first-stage order."""
    terms = query_words(query, tokenize)
    position = {index: rank for rank, index in enumerate(pool)}
    return sorted(pool, key=lambda i: (-len(terms & set(tokenize(texts[i]))), position[i]))


def by_scores(pool, scores):
    """Reorder the pool by a relevance score per chunk; ties keep the first-stage order. Nothing outside the pool."""
    position = {index: rank for rank, index in enumerate(pool)}
    return sorted(pool, key=lambda i: (-scores[i], position[i]))


def documents_of(ranking, rows):
    return list(dict.fromkeys(rows[i]["document_id"] for i in ranking))


def doc_metrics(ranking, rows, relevant):
    docs = documents_of(ranking, rows)
    first = next((rank for rank, doc in enumerate(docs, 1) if doc in relevant), None)
    hit = lambda n: float(bool(set(docs[:n]) & set(relevant)))  # one relevant document per query here
    return {"recall_at_1": hit(1), "recall_at_3": hit(3), "mrr": 0.0 if first is None else 1 / first,
            "in_pool": float(first is not None), "first_rank": first}


def summarize(records):
    table = {}
    for group in GROUPS:
        part = [r for r in records if f"{r['set']}/{r['style']}" == group]
        table[group] = {"n": len(part), **{k: round(float(np.mean([r[k] for r in part])), 4)
                                           for k in ["recall_at_1", "recall_at_3", "mrr", "in_pool"]}}
    return table


class CrossEncoder:
    """bge-reranker-v2-m3: query and chunk enter the model together; the output is one relevance logit."""

    def __init__(self, path, max_length=512, batch=16):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        started = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True, dtype=torch.float32)
        self.model.eval()
        self.max_length, self.batch = max_length, batch
        self.load_seconds = time.perf_counter() - started

    def score(self, query, passages):
        out = []
        for i in range(0, len(passages), self.batch):
            part = passages[i:i + self.batch]
            encoded = self.tokenizer([query] * len(part), part, padding=True, truncation=True,
                                     max_length=self.max_length, return_tensors="pt")
            with torch.inference_mode():
                out.append(self.model(**encoded).logits.view(-1).float().numpy())
        return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def sigmoid(x):
    return 1 / (1 + np.exp(-np.asarray(x, dtype=np.float64)))


def registry_entry(section, name):
    data = json.loads((ROOT / "metadata/reference-models.json").read_text(encoding="utf-8"))
    return next(e for e in data.get(section, []) if e["name"] == name)


def model_present(path):
    return any(Path(path).glob("*.safetensors")) or (Path(path) / "pytorch_model.bin").exists()


def first_stages(documents, texts, use_minilm=True, use_dense=True):
    bm, demo, emb = load("bm25_hybrid"), load("demo"), load("ru_embedders")
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
    matrix = vec.fit_transform(documents)
    tfidf = (vec.transform(texts) @ matrix.T).toarray()
    index = bm.BM25(documents)
    bm25 = np.array([index.scores(t) for t in texts])
    order = lambda scores: np.argsort(-scores, kind="stable").tolist()
    stages, info = {"lexical_rrf": [demo.rrf([order(tfidf[j]), order(bm25[j])]) for j in range(len(texts))]}, {}
    info["lexical_rrf"] = "RRF (c = 10) of char 3-5 TF-IDF and BM25 k1 = 1.5 b = 0.75, full rankings"
    if use_minilm:
        vectors = demo.pretrained_vectors(documents + texts)
        scores = vectors[len(documents):] @ vectors[:len(documents)].T
        stages["minilm"], info["minilm"] = [order(scores[j]) for j in range(len(texts))], f"{demo.ENCODER_ID}@{demo.ENCODER_REVISION}"
    entry = registry_entry("embedders", "USER-bge-m3")
    if use_dense and model_present(ROOT / entry["path"]):
        doc_vectors, query_vectors, _ = emb.embed(ROOT / entry["path"], documents, texts)
        scores = query_vectors @ doc_vectors.T
        stages["user_bge_m3"], info["user_bge_m3"] = [order(scores[j]) for j in range(len(texts))], f"{entry['repo_id']}@{entry['revision']}"
    return stages, info


def latency(encoder, queries, documents, pools):
    """Milliseconds per query when the reranker reads N pairs in one call; the same 16 queries for every N."""
    out = {}
    encoder.score(queries[0], documents[:4])  # warm-up: first call allocates buffers
    for n in LATENCY_N:
        timings = []
        for j in range(LATENCY_QUERIES):
            passages = [documents[i] for i in pools[j][:n]]
            started = time.perf_counter()
            encoder.score(queries[j], passages)
            timings.append((time.perf_counter() - started) * 1000)
        median = float(np.median(timings))
        out[f"N={n}"] = {"median_ms": round(median, 1), "pairs_per_second": round(n / median * 1000, 1)}
    return out


def run(use_minilm=True, use_dense=True, use_reranker=True, threads=4, out_dir=None):
    diag, emb, bm = load("diagnostic_ru"), load("ru_embedders"), load("bm25_hybrid")
    torch.set_num_threads(threads)
    rows = diag.chunks("paragraph")
    documents = [r["title"] + " " + r["text"] for r in rows]
    queries = [{**q, "set": "ru-embed-1"} for q in emb.ru_embed_queries()]
    queries += [{"id": q["id"], "set": "diagnostic-1.1.0", "style": "diagnostic", "text": q["text"], "relevant": q["relevant"]}
                for q in diag.QUERIES if q["relevant"]]
    texts = [q["text"] for q in queries]
    stages, stage_info = first_stages(documents, texts, use_minilm, use_dense)
    entry = registry_entry("rerankers", "bge-reranker-v2-m3")
    path = ROOT / entry["path"]
    status, encoder, all_scores = "disabled", None, None
    if use_reranker:
        status = "not_downloaded"
        if model_present(path):
            encoder = CrossEncoder(path)
            started = time.perf_counter()
            # Every (query, chunk) pair once: pools are subsets, and full-corpus reranking is the upper bound.
            all_scores = np.stack([encoder.score(t, documents) for t in texts])
            score_seconds = time.perf_counter() - started
            status = "measured"
    results, per_query, changes = {}, {}, {}
    for stage, rankings in stages.items():
        variants = {"none": [], "heuristic": []}
        if all_scores is not None:
            variants["cross_encoder"] = []
        for j, q in enumerate(queries):
            pool = rankings[j][:POOL]
            orders = {"none": pool, "heuristic": heuristic(q["text"], pool, documents, bm.tokenize)}
            if all_scores is not None:
                orders["cross_encoder"] = by_scores(pool, all_scores[j])
            for name, ranking in orders.items():
                record = {"id": q["id"], "set": q["set"], "style": q["style"], "top_chunks": [rows[i]["id"] for i in ranking[:3]]}
                if q["relevant"]:
                    record.update(doc_metrics(ranking, rows, q["relevant"]))
                variants[name].append(record)
        results[stage] = {name: summarize([r for r in recs if "mrr" in r]) for name, recs in variants.items()}
        per_query[stage] = variants
        for name in [n for n in variants if n != "none"]:
            moved = [(a["first_rank"], b["first_rank"]) for a, b in zip(variants["none"], variants[name]) if "mrr" in a]
            changes[f"{stage}/{name}"] = {"up": sum(b is not None and a is not None and b < a for a, b in moved),
                                          "down": sum(a is not None and b is not None and b > a for a, b in moved)}
    report = {"task": "T35", "corpus": "ru-diagnostic-1, 72 paragraph chunks of 24 current documents",
              "pool": POOL, "first_stages": stage_info, "threads": threads,
              "fixed_before_running": "query sets of T32, first stages, pool N = 20, stop words, max_length 512, batch 16; no tuning",
              "reranker": {"name": entry["name"], "repo_id": entry["repo_id"], "revision": entry["revision"],
                           "license": entry["license"], "status": status},
              "results": results, "rank_changes": changes, "per_query": per_query,
              "limits": "Synthetic Russian teaching corpus and authored queries; one relevant document per query. "
                        "A comparison on this pipeline, not a benchmark of the model."}
    if all_scores is not None:
        full = [{"id": q["id"], "set": q["set"], "style": q["style"],
                 **doc_metrics(by_scores(list(range(len(documents))), all_scores[j]), rows, q["relevant"])}
                for j, q in enumerate(queries) if q["relevant"]]
        top = sigmoid(all_scores.max(axis=1))
        answerable = [top[j] for j, q in enumerate(queries) if q["relevant"]]
        unanswerable = [top[j] for j, q in enumerate(queries) if not q["relevant"]]
        report["reranker"].update(
            load_seconds=round(encoder.load_seconds, 1), all_pairs=int(all_scores.size), all_pairs_seconds=round(score_seconds, 1),
            full_corpus=summarize(full), latency=latency(encoder, texts, documents, stages["lexical_rrf"]),
            top1_sigmoid={"answerable": len(answerable),
                          **{f"answerable_{name}": round(float(np.quantile(answerable, p)), 4)
                             for name, p in [("min", 0), ("p10", 0.1), ("p25", 0.25), ("median", 0.5)]},
                          **{f"answerable_below_{str(t).replace('.', '_')}": int(sum(x < t for x in answerable)) for t in [0.5, 0.1, 0.01]},
                          "unanswerable": [round(float(x), 4) for x in unanswerable]},
            # Logits of every (query, chunk) pair: other pool sizes can be scored offline without rerunning the model.
            pair_logits={"query_ids": [q["id"] for q in queries], "chunk_ids": [r["id"] for r in rows],
                         "values": [[round(float(x), 4) for x in row] for row in all_scores]})
    out = Path(out_dir) if out_dir else OUT
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def table_text(report):
    lines = ["| Первая стадия | Второй этап | Перефраз. R@1 / MRR | С термином R@1 / MRR | 1.1.0 R@1 / MRR | В пуле, перефраз. |",
             "| --- | --- | --- | --- | --- | --- |"]
    for stage, variants in report["results"].items():
        for name, table in variants.items():
            p, t, d = (table[g] for g in GROUPS)
            lines.append(f"| {stage} | {name} | {p['recall_at_1']:.2f} / {p['mrr']:.2f} | {t['recall_at_1']:.2f} / {t['mrr']:.2f} | "
                         f"{d['recall_at_1']:.2f} / {d['mrr']:.2f} | {p['in_pool']:.2f} |")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-minilm", action="store_true")
    parser.add_argument("--no-dense", action="store_true", help="skip USER-bge-m3 as a first stage")
    parser.add_argument("--no-reranker", action="store_true", help="only first stages and the heuristic")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads for PyTorch; the author run used 4")
    args = parser.parse_args()
    report = run(not args.no_minilm, not args.no_dense, not args.no_reranker, args.threads)
    print(table_text(report))
    print(json.dumps({k: v for k, v in report["reranker"].items() if k not in ("full_corpus", "pair_logits")}, ensure_ascii=False))
