"""Error analysis as a process (extension 1.2.0, T37, lesson M11-L07X).

collect: a small RAG pipeline answers 66 questions over the Russian course corpus of M10: the 52 questions of ru-embed-1
(4 without an answer) and the 14 diagnostic questions of 1.1.0 (2 without an answer). Retrieval is lexical and deterministic,
RRF of char TF-IDF and BM25 with the top 3 paragraph chunks, so the regression set runs without models. Qwen2.5-0.5B Q4_K_M
answers through llama-server: greedy, at most 200 new tokens, numbered sources, a fixed refusal phrase. Each trace keeps the
retrieved ids, the answer and automatic signals.
report: reads the author's open codes, counts the taxonomy and checks how well each automatic signal agrees with the manual
code it stands for. The pipeline, the prompt and the signal rules were fixed before the first answer was read.
"""
from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import math
import re
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / "quality/error-analysis"
TOP, MAX_TOKENS, THREADS = 3, 200, 4
MODEL = "qwen2.5-0.5b-instruct-q4_k_m"
REFUSAL = "В источниках нет ответа."
SYSTEM = ("Ты отвечаешь на вопросы по технической базе знаний. Используй только приведённые источники. После каждого "
          "утверждения укажи id источника в квадратных скобках, например [index-v2-c1]. Если в источниках нет ответа, "
          f"напиши ровно: «{REFUSAL}» Ответ — одно-три предложения на русском.")
CITATION = re.compile(r"\[([a-z]+(?:-[a-z0-9]+)+)\]")
CODES = {  # the taxonomy that came out of open coding of the 66 traces; FIRST orders it along the pipeline
    "retrieval_miss": "в контексте нет фрагмента с ответом",
    "failed_refusal": "в контексте нет ответа, а модель не отказалась",
    "over_refusal": "отказ, хотя контекст содержит ответ",
    "no_answer": "нет содержательного ответа: только ссылки, повтор вопроса или повтор одного токена",
    "off_question": "текст о другом: переписан чужой или шаблонный абзац вместо ответа",
    "unsupported_claim": "утверждение не следует из источников или противоречит им",
    "bad_citation": "ссылка вне контекста, несуществующая или на абзац, где утверждения нет",
    "format": "ответ оборван лимитом токенов, не по-русски или длиннее трёх предложений",
    "ok": "ответ по существу, опирается на источник и ссылается на него, либо верный отказ",
}
FIRST = list(CODES)


def load(path):
    name = Path(path).stem
    spec = importlib.util.spec_from_file_location(f"m11_{name}", ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dataset():
    diag, emb = load("labs/M10/diagnostic_ru.py"), load("labs/M10/ru_embedders.py")
    rows = diag.chunks("paragraph")
    questions = [{"id": q["id"], "set": "ru-embed-1", "style": q["style"], "text": q["text"], "relevant": q["relevant"]}
                 for q in emb.ru_embed_queries()]
    questions += [{"id": q["id"], "set": "diagnostic-1.1.0", "style": "diagnostic", "text": q["text"], "relevant": q["relevant"]}
                  for q in diag.QUERIES]
    return rows, questions


class Retriever:
    """RRF (c = 10) of char 3-5 TF-IDF and BM25 over title + paragraph; deterministic and model-free."""

    def __init__(self, rows):
        bm, demo = load("labs/M10/bm25_hybrid.py"), load("labs/M10/demo.py")
        self.rows, self.rrf = rows, demo.rrf
        self.documents = [r["title"] + " " + r["text"] for r in rows]
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
        self.matrix = self.vectorizer.fit_transform(self.documents)
        self.bm25 = bm.BM25(self.documents)

    def top(self, text, k=TOP):
        order = lambda s: np.argsort(-np.asarray(s), kind="stable").tolist()
        tfidf = (self.vectorizer.transform([text]) @ self.matrix.T).toarray()[0]
        return [self.rows[i]["id"] for i in self.rrf([order(tfidf), order(self.bm25.scores(text))])[:k]]


def prompt(question, retrieved, rows_by_id):
    sources = "\n".join(f"[{cid}] {rows_by_id[cid]['title']}: {rows_by_id[cid]['text']}" for cid in retrieved)
    return f"Источники:\n{sources}\n\nВопрос: {question}"


def signals(answer, retrieved, relevant, rows_by_id, finish_reason="stop", question=""):
    """Automatic checks, computed the same way for live answers and for the frozen regression cases."""
    cited = CITATION.findall(answer)
    text = CITATION.sub(" ", answer)
    letters = [ch for ch in text if ch.isalpha()]
    retrieved_docs = {rows_by_id[c]["document_id"] for c in retrieved}
    stripped = text.replace(question, " ") if question else text
    return {"refused": REFUSAL.lower().rstrip(".") in answer.lower(),
            "cited": cited, "has_citation": bool(cited),
            "citation_outside_context": any(c not in retrieved for c in cited),
            "cited_relevant": any(c in rows_by_id and rows_by_id[c]["document_id"] in relevant for c in cited),
            "cites_prompt_example": any(c.startswith("index-v2-") for c in cited) and "index" not in relevant,
            "retrieval_hit": bool(set(relevant) & retrieved_docs) if relevant else None,
            "answerable": bool(relevant),
            "letters": len(letters), "echoes_question": bool(question) and question in answer,
            "content_letters": sum(ch.isalpha() for ch in stripped),
            "cut_by_limit": finish_reason == "length",
            "cyrillic_share": round(sum("а" <= ch.lower() <= "я" or ch.lower() == "ё" for ch in letters) / len(letters), 3) if letters else None,
            "sentences": len([s for s in re.split(r"[.!?]+(?:\s|$)", text) if s.strip()])}


def collect(out_dir, threads=THREADS, limit=None):
    local, serving = load("labs/M09/llama_local.py"), load("labs/M09/llama_serving.py")
    rows, questions = dataset()
    rows_by_id = {r["id"]: r for r in rows}
    retriever = Retriever(rows)
    entry = serving.gguf_entry(MODEL)
    model = ROOT / entry["path"] / entry["file"]
    if not model.exists():
        raise SystemExit("Run: .\\.venv\\Scripts\\python.exe tools\\download_models.py --gguf")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    traces = []
    with local.Server(model, threads=threads, ctx=4096, alias=MODEL):
        for q in questions[:limit]:
            retrieved = retriever.top(q["text"])
            payload = {"model": MODEL, "max_tokens": MAX_TOKENS, "temperature": 0, "seed": 17, "cache_prompt": False,
                       "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt(q["text"], retrieved, rows_by_id)}]}
            started = time.perf_counter()
            with local.request("/v1/chat/completions", payload) as response:
                data = json.load(response)
            answer = (data["choices"][0]["message"].get("content") or "").strip()
            traces.append({"trace_id": q["id"], "set": q["set"], "style": q["style"], "question": q["text"], "relevant": q["relevant"],
                           "retrieved": retrieved, "answer": answer, "finish_reason": data["choices"][0].get("finish_reason"),
                           "ms": round((time.perf_counter() - started) * 1000),
                           "signals": signals(answer, retrieved, q["relevant"], rows_by_id, data["choices"][0].get("finish_reason"), q["text"])})
    meta = {"task": "T37", "model": MODEL, "repo_id": entry["repo_id"], "revision": entry["revision"],
            "llama_cpp": local.RELEASE["tag"], "threads": threads, "top": TOP, "max_tokens": MAX_TOKENS, "system": SYSTEM,
            "retriever": "RRF (c = 10) of char 3-5 TF-IDF and BM25 over title + paragraph, top 3",
            "decoding": "greedy, seed 17, cache_prompt off", "collected": time.strftime("%Y-%m-%d")}
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    with (out / "traces.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for trace in traces:
            handle.write(json.dumps(trace, ensure_ascii=False) + "\n")
    return traces


def read_traces(folder=FROZEN):
    return [json.loads(line) for line in (Path(folder) / "traces.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]


def first_failure(codes):
    return min(codes, key=FIRST.index)


def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    centre, half = (p + z * z / (2 * n)) / (1 + z * z / n), z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(centre - half, 3), round(centre + half, 3)]


# Automatic signal → the manual code it is meant to detect; unsupported_claim and off_question have no cheap check.
CHECKS = {
    "retrieval_miss": lambda s: bool(s["answerable"] and not s["retrieval_hit"]),
    "failed_refusal": lambda s: not s["answerable"] and not s["refused"],
    "over_refusal": lambda s: bool(s["answerable"] and s["retrieval_hit"] and s["refused"]),
    "no_answer": lambda s: not s["refused"] and (s["content_letters"] < 15 or s["echoes_question"]),
    "bad_citation": lambda s: s["citation_outside_context"],
    "format": lambda s: s["cut_by_limit"] or s["sentences"] > 3 or (s["cyrillic_share"] is not None and s["cyrillic_share"] < 0.6),
}


def recompute(traces):
    """Signals from the frozen answers with the current rules; answers themselves never change."""
    rows, _ = dataset()
    rows_by_id = {r["id"]: r for r in rows}
    for t in traces:
        t["signals"] = signals(t["answer"], t["retrieved"], t["relevant"], rows_by_id, t["finish_reason"], t["question"])
    return traces


def report(folder=FROZEN):
    traces = recompute(read_traces(folder))
    codes = json.loads((Path(folder) / "codes.json").read_text(encoding="utf-8"))
    by_id = {t["trace_id"]: t for t in traces}
    assert set(codes["traces"]) == set(by_id), "every trace must be coded exactly once"
    n = len(traces)
    any_code = collections.Counter(c for entry in codes["traces"].values() for c in entry["codes"])
    first = collections.Counter(first_failure(entry["codes"]) for entry in codes["traces"].values())
    taxonomy = {code: {"description": CODES[code], "traces": any_code.get(code, 0), "share": round(any_code.get(code, 0) / n, 3),
                       "ci95": wilson(any_code.get(code, 0), n), "first_failure": first.get(code, 0)} for code in FIRST}
    checks = {}
    for code, rule in CHECKS.items():
        auto = {tid for tid, t in by_id.items() if rule(t["signals"])}
        manual = {tid for tid, entry in codes["traces"].items() if code in entry["codes"]}
        tp, fp, fn = len(auto & manual), len(auto - manual), len(manual - auto)
        checks[code] = {"auto": len(auto), "manual": len(manual), "both": tp, "auto_only": sorted(auto - manual),
                        "manual_only": sorted(manual - auto), "precision": round(tp / (tp + fp), 3) if tp + fp else None,
                        "recall": round(tp / (tp + fn), 3) if tp + fn else None}
    by_set = {}
    for t in traces:
        key = f"{t['set']}/{t['style']}"
        row = by_set.setdefault(key, {"n": 0, "ok": 0})
        row["n"] += 1
        row["ok"] += codes["traces"][t["trace_id"]]["codes"] == ["ok"]
    example = [t["trace_id"] for t in traces if t["signals"]["cites_prompt_example"]]
    return {"task": "T37", "traces": n, "coder": codes["coder"], "taxonomy": taxonomy, "checks": checks, "ok_by_set": by_set,
            "manual_only_codes": [c for c in FIRST if c not in CHECKS and c != "ok"],
            "cause_prompt_example": {"traces": len(example), "of_bad_citation": len(set(example) & {tid for tid, e in codes["traces"].items() if "bad_citation" in e["codes"]}),
                                     "ids": example}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("collect", help="answer the questions with the local model and save traces")
    c.add_argument("--out", default=str(ROOT / "artifacts/M11/error-analysis"))
    c.add_argument("--limit", type=int)
    c.add_argument("--threads", type=int, default=THREADS)
    r = sub.add_parser("report", help="taxonomy and check agreement from traces and codes")
    r.add_argument("--folder", default=str(FROZEN))
    args = parser.parse_args()
    if args.command == "collect":
        traces = collect(args.out, args.threads, args.limit)
        print(f"{len(traces)} traces -> {args.out}")
    else:
        result = report(args.folder)
        out = ROOT / "artifacts/M11/error-analysis"
        out.mkdir(parents=True, exist_ok=True)
        (out / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({k: v["traces"] for k, v in result["taxonomy"].items()}, ensure_ascii=False))
