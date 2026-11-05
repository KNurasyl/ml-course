"""Contextual retrieval: a local model writes a short context for every chunk before indexing (extension 1.2.0, T36).

Corpus, chunks and queries are those of ru_embedders.py (T32): 72 paragraph chunks of 24 current documents, 24 paraphrase
and 24 term queries of ru-embed-1 and the 12 answerable diagnostic queries of 1.1.0. Chunk representations, fixed before
running:
- text: the paragraph alone;
- title: title + paragraph, the course default since 1.1.0;
- ctx_0.5b, ctx_4b: a context written by Qwen2.5-0.5B or Qwen3.5-4B (Q4_K_M in llama-server) + paragraph;
- title_ctx_4b: title + the 4B context + paragraph.
The prompt follows Anthropic's contextual retrieval: the whole document first, then the chunk, then a request for one or two
sentences that place the chunk in the document; greedy, at most 80 new tokens, Qwen3.5 thinking off. The document comes first,
so llama-server reuses its KV cache for the next chunks of the same document (cache_prompt on); computed prompt tokens are
recorded. Retrievers: BM25, USER-bge-m3 (MiniLM if it is not downloaded) and their RRF. Metrics are document level:
Recall@1, @3, @20 and MRR per query style. Contexts are saved, and --reuse-contexts reruns retrieval without generating.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import statistics
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/M10/contextual"
MAX_TOKENS, THREADS = 80, 4
GROUPS = ["ru-embed-1/paraphrase", "ru-embed-1/term", "diagnostic-1.1.0/diagnostic"]
GENERATORS = {"0.5b": "qwen2.5-0.5b-instruct-q4_k_m", "4b": "qwen3.5-4b-q4_k_m"}
REPRESENTATIONS = ["text", "title", "ctx_0.5b", "ctx_4b", "title_ctx_4b"]
PROMPT = ("<документ>\n{document}\n</документ>\nВот фрагмент из этого документа:\n<фрагмент>\n{chunk}\n</фрагмент>\n"
          "Напиши одно-два коротких предложения, которые помещают этот фрагмент в контекст всего документа, чтобы его было "
          "легче найти поиском. Ответь только этим контекстом.")


def load(name):
    spec = importlib.util.spec_from_file_location(f"m10_{Path(name).name}", Path(__file__).parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def corpus_hash(rows):
    return hashlib.sha256(json.dumps([[r["id"], r["text"]] for r in rows], ensure_ascii=False).encode("utf-8")).hexdigest()[:16]


def prompt_for(document_text, chunk_text):
    """The document goes first: chunks of one document share this prefix, and the server reuses its KV cache."""
    return PROMPT.format(document=document_text, chunk=chunk_text)


def novel_word_share(context, document_text, tokenize, stop=frozenset()):
    """Crude faithfulness signal: the share of context words that do not occur in the document at all."""
    words = [w for w in tokenize(context) if w not in stop]
    known = set(tokenize(document_text))
    return round(sum(w not in known for w in words) / len(words), 3) if words else 0.0


def generate_contexts(name, rows, documents, threads=THREADS):
    """One context per chunk from llama-server; returns per-chunk records and a status."""
    local, serving = load("../M09/llama_local"), load("../M09/llama_serving")
    entry = serving.gguf_entry(GENERATORS[name])
    model = ROOT / entry["path"] / entry["file"]
    if not model.exists():
        return {"status": "not_downloaded"}
    try:
        local.server_exe()
    except FileNotFoundError:
        return {"status": "llama_cpp_missing"}
    extra = {"chat_template_kwargs": {"enable_thinking": False}} if name == "4b" else {}
    records, started = [], time.perf_counter()
    with local.Server(model, threads=threads, ctx=4096, alias=entry["name"]):
        for row in rows:
            payload = {"model": entry["name"], "max_tokens": MAX_TOKENS, "temperature": 0, "seed": 17, "cache_prompt": True,
                       "messages": [{"role": "user", "content": prompt_for(documents[row["document_id"]], row["text"])}], **extra}
            begin = time.perf_counter()
            with local.request("/v1/chat/completions", payload) as response:
                data = json.load(response)
            timings = data.get("timings") or {}
            records.append({"id": row["id"], "context": (data["choices"][0]["message"].get("content") or "").strip(),
                            "prompt_tokens_total": data.get("usage", {}).get("prompt_tokens"),
                            "prompt_tokens_computed": timings.get("prompt_n"), "generated_tokens": timings.get("predicted_n"),
                            "ms": round((time.perf_counter() - begin) * 1000)})
    return {"status": "generated", "model": entry["name"], "repo_id": entry["repo_id"], "revision": entry["revision"],
            "seconds": round(time.perf_counter() - started, 1), "records": records}


def representations(rows, contexts):
    texts = {"text": [r["text"] for r in rows], "title": [r["title"] + " " + r["text"] for r in rows]}
    for name, key in [("0.5b", "ctx_0.5b"), ("4b", "ctx_4b")]:
        if name in contexts:
            by_id = {c["id"]: c["context"] for c in contexts[name]}
            texts[key] = [by_id[r["id"]] + " " + r["text"] for r in rows]
    if "4b" in contexts:
        by_id = {c["id"]: c["context"] for c in contexts["4b"]}
        texts["title_ctx_4b"] = [r["title"] + " " + by_id[r["id"]] + " " + r["text"] for r in rows]
    return {k: texts[k] for k in REPRESENTATIONS if k in texts}


def doc_metrics(ranking, rows, relevant):
    docs = list(dict.fromkeys(rows[i]["document_id"] for i in ranking))
    first = next((rank for rank, doc in enumerate(docs, 1) if doc in relevant), None)
    hit = lambda n: float(first is not None and first <= n)  # one relevant document per query here
    return {"recall_at_1": hit(1), "recall_at_3": hit(3), "recall_at_20": hit(20), "mrr": 0.0 if first is None else 1 / first,
            "first_rank": first}


def summarize(records):
    out = {}
    for group in GROUPS:
        part = [r for r in records if f"{r['set']}/{r['style']}" == group]
        out[group] = {"n": len(part), **{k: round(float(np.mean([r[k] for r in part])), 4)
                                         for k in ["recall_at_1", "recall_at_3", "recall_at_20", "mrr"]}}
    return out


def encoder(documents_by_rep, texts):
    """Dense vectors for every representation and the queries in one model load; USER-bge-m3, else MiniLM."""
    emb, demo = load("ru_embedders"), load("demo")
    registry = json.loads((ROOT / "metadata/reference-models.json").read_text(encoding="utf-8"))
    entry = next(e for e in registry["embedders"] if e["name"] == "USER-bge-m3")
    path = ROOT / entry["path"]
    names = list(documents_by_rep)
    flat = [t for name in names for t in documents_by_rep[name]]
    if any(path.glob("*.safetensors")):
        doc_vectors, query_vectors, _ = emb.embed(path, flat, texts)
        label = f"{entry['repo_id']}@{entry['revision']}"
    elif (demo.ENCODER / "model.safetensors").exists():
        vectors = demo.pretrained_vectors(flat + texts)
        doc_vectors, query_vectors, label = vectors[:len(flat)], vectors[len(flat):], f"{demo.ENCODER_ID}@{demo.ENCODER_REVISION}"
    else:
        return None, None
    size = len(next(iter(documents_by_rep.values())))
    return {name: query_vectors @ doc_vectors[i * size:(i + 1) * size].T for i, name in enumerate(names)}, label


def run(generators=("0.5b", "4b"), reuse=False, threads=THREADS, out_dir=None):
    diag, emb, bm, demo = load("diagnostic_ru"), load("ru_embedders"), load("bm25_hybrid"), load("demo")
    rows = diag.chunks("paragraph")
    documents = {d["document_id"]: d["title"] + "\n\n" + d["text"] for d in diag.documents() if d["version"] == 2}
    queries = [{**q, "set": "ru-embed-1"} for q in emb.ru_embed_queries()]
    queries += [{"id": q["id"], "set": "diagnostic-1.1.0", "style": "diagnostic", "text": q["text"], "relevant": q["relevant"]}
                for q in diag.QUERIES if q["relevant"]]
    texts = [q["text"] for q in queries]
    out = Path(out_dir) if out_dir else OUT
    out.mkdir(parents=True, exist_ok=True)
    fingerprint, contexts, generation = corpus_hash(rows), {}, {}
    for name in generators:
        cache = out / f"contexts-{name}.json"
        saved = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else None
        if reuse and saved and saved.get("corpus") == fingerprint:
            result = saved["generation"]
        else:
            result = generate_contexts(name, rows, documents, threads)
            if result["status"] == "generated":
                cache.write_text(json.dumps({"corpus": fingerprint, "generation": result}, ensure_ascii=False, indent=2), encoding="utf-8")
        if result["status"] == "generated":
            contexts[name] = result["records"]
            records = result["records"]
            computed = sum(r["prompt_tokens_computed"] or 0 for r in records)
            total = sum(r["prompt_tokens_total"] or 0 for r in records)
            novel = [novel_word_share(r["context"], documents[row["document_id"]], bm.tokenize) for r, row in zip(records, rows)]
            generation[name] = {**{k: result[k] for k in ("status", "model", "repo_id", "revision", "seconds")},
                                "prompt_tokens_total": total, "prompt_tokens_computed": computed,
                                "generated_tokens": sum(r["generated_tokens"] or 0 for r in records),
                                "ms_median": statistics.median(r["ms"] for r in records),
                                "context_words_median": statistics.median(len(r["context"].split()) for r in records),
                                "novel_word_share_median": statistics.median(novel), "novel_word_share_max": max(novel),
                                "empty_contexts": sum(not r["context"] for r in records)}
        else:
            generation[name] = {"status": result["status"]}
    reps = representations(rows, contexts)
    order = lambda scores: np.argsort(-scores, kind="stable").tolist()
    dense, dense_label = encoder(reps, texts)
    results, first_ranks = {}, {}
    for rep, docs in reps.items():
        index = bm.BM25(docs)
        bm25 = [order(index.scores(t)) for t in texts]
        rankings = {"bm25": bm25}
        if dense is not None:
            rankings["dense"] = [order(dense[rep][j]) for j in range(len(texts))]
            rankings["rrf"] = [demo.rrf([bm25[j], rankings["dense"][j]]) for j in range(len(texts))]
        results[rep] = {}
        for retriever, ranked in rankings.items():
            records = [{"id": q["id"], "set": q["set"], "style": q["style"], **doc_metrics(ranked[j], rows, q["relevant"])}
                       for j, q in enumerate(queries)]
            results[rep][retriever] = summarize(records)
            first_ranks[f"{rep}/{retriever}"] = {r["id"]: r["first_rank"] for r in records}
    examples = [{"id": row["id"], "title": row["title"], **{name: next(c["context"] for c in contexts[name] if c["id"] == row["id"])
                                                           for name in contexts}} for row in rows[1:72:12]]
    report = {"task": "T36", "corpus": "ru-diagnostic-1, 72 paragraph chunks of 24 current documents", "corpus_hash": fingerprint,
              "prompt": PROMPT, "max_tokens": MAX_TOKENS, "dense_encoder": dense_label, "generation": generation,
              "fixed_before_running": "representations, prompt, greedy decoding, 80 new tokens, retrievers, query sets of T32",
              "results": results, "first_ranks": first_ranks, "examples": examples,
              "limits": "Short synthetic Russian documents whose paragraphs already carry the topic word in the title; "
                        "Recall@20 saturates on 24 documents. Shows the mechanism, not the gain on a large corpus."}
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def table_text(report, retriever="dense"):
    lines = [f"| Представление ({retriever}) | Перефраз. R@1 / R@3 / MRR | С термином R@1 / MRR | 1.1.0 R@1 / MRR | R@20, перефраз. |",
             "| --- | --- | --- | --- | --- |"]
    for rep, by_retriever in report["results"].items():
        if retriever not in by_retriever:
            continue
        p, t, d = (by_retriever[retriever][g] for g in GROUPS)
        lines.append(f"| {rep} | {p['recall_at_1']:.2f} / {p['recall_at_3']:.2f} / {p['mrr']:.2f} | {t['recall_at_1']:.2f} / {t['mrr']:.2f} | "
                     f"{d['recall_at_1']:.2f} / {d['mrr']:.2f} | {p['recall_at_20']:.2f} |")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse-contexts", action="store_true", help="use saved contexts of the same corpus instead of generating")
    parser.add_argument("--no-4b", action="store_true", help="generate contexts with Qwen2.5-0.5B only")
    parser.add_argument("--threads", type=int, default=THREADS)
    args = parser.parse_args()
    report = run(("0.5b",) if args.no_4b else ("0.5b", "4b"), args.reuse_contexts, args.threads)
    for retriever in ["bm25", "dense", "rrf"]:
        print(table_text(report, retriever))
    print(json.dumps(report["generation"], ensure_ascii=False))
