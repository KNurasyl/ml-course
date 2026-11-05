"""Structured output from a small local model: prompt only, JSON mode, JSON Schema, a GBNF grammar and validation with
retry (extension 1.2.0, T31, lesson M09-L08X).

The contract is the one of M09-L05: exactly {"feedback": non-empty string, "score": integer 0, 1 or 2}, checked by
demo.validate_feedback; an invalid answer becomes the terminal needs_review state. 24 interview answers with an author's
gold score (8 correct, 8 partial, 8 wrong) were written for this lab and fixed before the model ran. Modes:
- prompt_only: the system prompt asks for JSON, nothing constrains the sampler;
- json_object: response_format json_object — llama-server allows only syntactically valid JSON;
- json_schema: response_format json_schema — llama-server turns the schema into a grammar;
- gbnf: the hand-written grammar GBNF below, stricter than the schema (feedback starts with a non-space character);
- retry: prompt_only plus validation; an invalid answer goes back with the reason, at most two retries.
Qwen2.5-0.5B Q4_K_M runs every mode; Qwen3.5-4B Q4_K_M runs prompt_only and json_schema, thinking switched off.
Greedy decoding, at most 160 new tokens, 4 threads; modes alternate inside every case so that CPU slowdown from heat
spreads evenly. Valid form says nothing about a correct score: both are reported.
"""
from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import re
import statistics
import time
from pathlib import Path

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/M09/structured"
MAX_TOKENS, RETRIES, THREADS = 160, 2, 4
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["feedback", "score"],
          "properties": {"feedback": {"type": "string", "minLength": 1, "maxLength": 200},
                         "score": {"type": "integer", "minimum": 0, "maximum": 2}}}
GBNF = r'''root ::= "{" space "\"feedback\"" space ":" space feedback "," space "\"score\"" space ":" space score space "}"
feedback ::= "\"" first char{0,199} "\""
first ::= [^"\\\x7F\x00-\x20] | escape
char ::= [^"\\\x7F\x00-\x1F] | escape
escape ::= "\\" (["\\/bfnrt] | "u" [0-9a-fA-F]{4})
score ::= [0-2]
space ::= " "?
'''
SYSTEM = ("Ты проверяешь ответ кандидата на техническом собеседовании. Верни только JSON-объект с двумя полями: "
          "feedback — одна-две фразы обратной связи на русском, score — целое число: 0 — неверно, 1 — частично верно, "
          "2 — верно. Никакого текста до или после JSON.")
CASES = [  # (question, answer, gold score)
    ("Что такое индекс в базе данных?", "Индекс — отдельная структура, например B-дерево, которая позволяет находить строки по значению столбца без полного просмотра таблицы; за это платят местом и замедлением записи.", 2),
    ("Зачем нужна транзакция?", "Чтобы группа операций выполнилась целиком или не выполнилась совсем, и другие запросы не видели промежуточного состояния.", 2),
    ("Что значит идемпотентный HTTP-метод?", "Повтор того же запроса оставляет состояние сервера таким же, как после первого; например, PUT и DELETE.", 2),
    ("Чем p95 задержки отличается от среднего?", "p95 — значение, быстрее которого выполняются 95 % запросов; в отличие от среднего он показывает медленный хвост.", 2),
    ("Когда кэш вредит?", "Когда данные часто меняются, а кэш не инвалидируется: пользователи получают устаревшие ответы; ещё когда попаданий мало, а память тратится.", 2),
    ("Что делает EXPLAIN в PostgreSQL?", "Показывает план выполнения запроса: узлы, сканирование по индексу или последовательное, оценки стоимости; с ANALYZE — фактическое время.", 2),
    ("Зачем повторам экспоненциальная пауза (backoff)?", "Чтобы повторы после сбоя не обрушили сервис волной запросов: паузы растут, а случайный разброс разводит клиентов во времени.", 2),
    ("Что такое проблема N+1 запросов?", "Для списка из N объектов делают один запрос за список и ещё по запросу на каждый объект вместо одного запроса с JOIN или IN.", 2),
    ("Что такое индекс в базе данных?", "Индекс ускоряет поиск в таблице.", 1),
    ("Зачем нужна транзакция?", "Чтобы при ошибке можно было откатить изменения.", 1),
    ("Что значит идемпотентный HTTP-метод?", "Повторный вызов даёт тот же результат, например GET.", 1),
    ("Чем p95 задержки отличается от среднего?", "p95 — это почти максимальная задержка.", 1),
    ("Когда кэш вредит?", "Когда данные устаревают.", 1),
    ("Что делает EXPLAIN в PostgreSQL?", "Показывает, использует ли запрос индекс.", 1),
    ("Зачем повторам экспоненциальная пауза (backoff)?", "Чтобы не повторять запрос слишком часто.", 1),
    ("Что такое проблема N+1 запросов?", "Это когда запросов к базе слишком много.", 1),
    ("Что такое индекс в базе данных?", "Индекс — это первичный ключ таблицы, он бывает только один.", 0),
    ("Зачем нужна транзакция?", "Транзакция нужна, чтобы запрос выполнялся быстрее.", 0),
    ("Что значит идемпотентный HTTP-метод?", "Это метод, который всегда возвращает код 200.", 0),
    ("Чем p95 задержки отличается от среднего?", "p95 — это среднее время ответа за 95 секунд.", 0),
    ("Когда кэш вредит?", "Кэш никогда не вредит, он только ускоряет.", 0),
    ("Что делает EXPLAIN в PostgreSQL?", "EXPLAIN выполняет запрос дважды и выбирает более быстрый вариант.", 0),
    ("Зачем повторам экспоненциальная пауза (backoff)?", "Backoff нужен, чтобы шифровать повторные запросы.", 0),
    ("Что такое проблема N+1 запросов?", "N+1 — это версия базы данных PostgreSQL.", 0),
]
MODES = ["prompt_only", "json_object", "json_schema", "gbnf", "retry"]
MODELS = [("qwen2.5-0.5b-instruct-q4_k_m", MODES), ("qwen3.5-4b-q4_k_m", ["prompt_only", "json_schema"])]


def load(name):
    spec = importlib.util.spec_from_file_location(f"m09_{name}", Path(__file__).with_name(f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def constraint(mode):
    """Request fields that constrain sampling; prompt_only and retry add none."""
    if mode == "json_object":
        return {"response_format": {"type": "json_object"}}
    if mode == "json_schema":
        return {"response_format": {"type": "json_schema", "json_schema": {"name": "feedback", "strict": True, "schema": SCHEMA}}}
    if mode == "gbnf":
        return {"grammar": GBNF}
    return {}


def check(text, validate_feedback):
    """Three separate questions: does it parse, does it match the JSON Schema, does it pass the M09-L05 contract."""
    try:
        value = json.loads(text)
        parsed = True
    except (json.JSONDecodeError, TypeError):
        value, parsed = None, False
    schema_valid = parsed and Draft202012Validator(SCHEMA).is_valid(value)
    contract = validate_feedback(text)
    return {"parsed": parsed, "schema_valid": bool(schema_valid), "contract_valid": contract["valid"],
            "reason": contract.get("reason"), "value": contract.get("value")}


def complete(local, model, messages, mode, extra):
    payload = {"model": model, "messages": messages, "max_tokens": MAX_TOKENS, "temperature": 0, "seed": 17,
               "cache_prompt": False, **constraint(mode), **extra}
    started = time.perf_counter()
    with local.request("/v1/chat/completions", payload) as response:
        data = json.load(response)
    timings = data.get("timings") or {}
    return {"text": data["choices"][0]["message"].get("content") or "", "finish_reason": data["choices"][0].get("finish_reason"),
            "ms": (time.perf_counter() - started) * 1000, "generated_tokens": timings.get("predicted_n"),
            "decode_tps": timings.get("predicted_per_second")}


def run_case(local, model, mode, case, validate_feedback, feedback_or_abstain, extra):
    question, answer, gold = case
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"Вопрос: {question}\nОтвет кандидата: {answer}"}]
    attempts = []
    for _ in range(1 + (RETRIES if mode == "retry" else 0)):
        out = complete(local, model, messages, mode, extra)
        verdict = check(out["text"], validate_feedback)
        attempts.append({**out, **{k: verdict[k] for k in ("parsed", "schema_valid", "contract_valid", "reason")}})
        if verdict["contract_valid"] or mode != "retry":
            break
        messages = messages + [{"role": "assistant", "content": out["text"]},
                               {"role": "user", "content": f"Ответ не прошёл проверку: {verdict['reason']}. Верни только "
                                                           "JSON-объект с полями feedback и score, без другого текста."}]
    final = check(attempts[-1]["text"], validate_feedback)
    response = feedback_or_abstain(attempts[-1]["text"])
    return {"question": question, "answer": answer, "gold": gold, "attempts": attempts, **{k: final[k] for k in ("parsed", "schema_valid", "contract_valid", "reason")},
            "status": response["status"], "score": response["score"], "score_correct": response["score"] == gold,
            "ms_total": sum(a["ms"] for a in attempts)}


NEGATIVE = re.compile(r"\bне\s*вер|\bневер", re.IGNORECASE)


def contradicts(row):
    """Heuristic: the feedback calls the answer wrong ("неверно", "не верен") while the score says 2."""
    return row["score"] == 2 and bool(NEGATIVE.search(row["attempts"][-1]["text"]))


def summarize(rows):
    n = len(rows)
    valid = [r for r in rows if r["contract_valid"]]
    tps = [a["decode_tps"] for r in rows for a in r["attempts"] if a.get("decode_tps")]
    by_gold = {str(g): dict(collections.Counter(str(r["score"]) for r in rows if r["gold"] == g)) for g in (0, 1, 2)}
    return {"n": n, "by_gold": by_gold, "feedback_contradicts_score": sum(contradicts(r) for r in valid),
            "parsed": sum(r["parsed"] for r in rows), "schema_valid": sum(r["schema_valid"] for r in rows),
            "contract_valid": len(valid), "needs_review": sum(r["status"] == "needs_review" for r in rows),
            "score_correct": sum(r["score_correct"] for r in rows),
            "score_correct_among_valid": round(sum(r["score_correct"] for r in valid) / len(valid), 3) if valid else None,
            "attempts_mean": round(statistics.mean(len(r["attempts"]) for r in rows), 2),
            "cut_by_max_tokens": sum(a["finish_reason"] == "length" for r in rows for a in r["attempts"]),
            "ms_median": round(statistics.median(r["ms_total"] for r in rows)), "decode_tps_median": round(statistics.median(tps), 1) if tps else None,
            "reasons": dict(collections.Counter(r["reason"] for r in rows if not r["contract_valid"])),
            "scores": dict(collections.Counter(str(r["score"]) for r in rows))}


def run(models=MODELS, cases=CASES, threads=THREADS, out_dir=None):
    local, serving, demo = load("llama_local"), load("llama_serving"), load("demo")
    results, status = {}, {}
    for name, modes in models:
        entry = serving.gguf_entry(name)
        model_path = ROOT / entry["path"] / entry["file"]
        if not model_path.exists():
            status[name] = "not_downloaded"
            continue
        try:
            local.server_exe()
        except FileNotFoundError:
            status[name] = "llama_cpp_missing"
            continue
        extra = {"chat_template_kwargs": {"enable_thinking": False}} if name.startswith("qwen3.5") else {}
        rows = {mode: [] for mode in modes}
        with local.Server(model_path, threads=threads, ctx=4096, alias=name):
            # Modes alternate inside every case: a laptop slows down as it heats, and a block per mode would
            # charge that drift to whichever mode ran last.
            for case in cases:
                for mode in modes:
                    rows[mode].append(run_case(local, name, mode, case, demo.validate_feedback, demo.feedback_or_abstain, extra))
        for mode in modes:
            summary = summarize(rows[mode])
            summary["same_text_as_prompt_only"] = sum(a["attempts"][0]["text"] == b["attempts"][0]["text"]
                                                      for a, b in zip(rows[mode], rows["prompt_only"]))
            results[f"{name}/{mode}"] = {"summary": summary, "rows": rows[mode]}
        status[name] = "measured"
    report = {"task": "T31", "contract": "M09-L05: exactly feedback (non-empty string) and score (integer 0..2)", "schema": SCHEMA,
              "gbnf": GBNF, "system": SYSTEM, "cases": len(cases), "gold": dict(collections.Counter(c[2] for c in cases)),
              "max_tokens": MAX_TOKENS, "retries": RETRIES, "threads": threads, "decoding": "greedy, seed 17, cache_prompt off",
              "llama_cpp": local.RELEASE["tag"], "status": status,
              "fixed_before_running": "contract, schema, grammar, prompts, 24 cases with gold scores, modes, max_tokens 160, retries 2",
              "results": results,
              "limits": "24 synthetic Russian interview answers, one gold label each by the course author; a small model on CPU. "
                        "Format validity and score agreement on this set, not a quality benchmark."}
    out = Path(out_dir) if out_dir else OUT
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def table_text(report):
    lines = ["| Модель / режим | JSON разобран | Схема | Контракт | needs_review | Оценка верна | Попыток | мс, медиана |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for key, result in report["results"].items():
        s = result["summary"]
        lines.append(f"| {key} | {s['parsed']}/{s['n']} | {s['schema_valid']} | {s['contract_valid']} | {s['needs_review']} | "
                     f"{s['score_correct']} | {s['attempts_mean']} | {s['ms_median']} |")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--only-small", action="store_true", help="skip Qwen3.5-4B")
    parser.add_argument("--cases", type=int, help="first N cases only, for a quick check")
    parser.add_argument("--threads", type=int, default=THREADS)
    args = parser.parse_args()
    report = run(MODELS[:1] if args.only_small else MODELS, CASES[:args.cases] if args.cases else CASES, args.threads)
    print(table_text(report))
    print(json.dumps(report["status"]))
