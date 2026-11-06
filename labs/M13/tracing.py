"""Tracing an LLM pipeline with OpenTelemetry GenAI semantic conventions into JSONL (extension 1.2.0, T39, lesson M13-L07X).

The course path needs no collector, no Docker and no new package. A small tracer from the standard library keeps the
current span in contextvars and writes every finished span as one JSON line. Field names follow the OpenTelemetry trace
data model: trace_id, span_id, parent_span_id, kind, start and end in Unix nanoseconds, status, attributes. Attribute
names follow the GenAI semantic conventions (repository open-telemetry/semantic-conventions-genai, commit e6658f1 of
06.11.2026, status Development). Evaluation results are separate records linked to a span by trace_id and span_id, the way
OpenTelemetry events are log records.

collect: the RAG pipeline of M11-L07X answers its 66 questions through llama-server (Qwen2.5-0.5B Q4_K_M). Each question
gives one trace: invoke_workflow rag-answer -> retrieval -> chat {model} -> evaluate (the checks of M11-L07X). Then one
request runs with a 50 ms client deadline, to show a failed call, and the same question runs once more without it.
analyze: stage durations with p50 and p95 (nearest rank, as in M13-L04), the share of the chat span, prefill and decode
from llama.cpp timings, the gap between client and server time, and latency and tokens split by the evaluation verdict.
otlp: converts spans and events into OTLP/JSON lines (traces and logs). An OpenTelemetry Collector or any OTLP/HTTP backend,
Langfuse among them, accepts this format; send posts it. Both are optional and outside the course path.

    python labs/M13/tracing.py collect            # needs llama.cpp and the GGUF of M09-L07X
    python labs/M13/tracing.py analyze            # frozen traces in quality/tracing, no model
    python labs/M13/tracing.py otlp
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import contextvars
import importlib.util
import json
import math
import os
import secrets
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / "quality/tracing"
OUT = ROOT / "artifacts/M13/tracing"
SEMCONV = {"repository": "https://github.com/open-telemetry/semantic-conventions-genai",
           "commit": "e6658f14c9d9ea0760825f029cb82e7673f48fab", "date": "2026-11-06", "status": "Development"}
KINDS = {"INTERNAL": 1, "SERVER": 2, "CLIENT": 3}  # OTLP SpanKind numbers
STATUS = {"UNSET": 0, "OK": 1, "ERROR": 2}
DEADLINE_S = 0.05
CURRENT: contextvars.ContextVar = contextvars.ContextVar("current_span", default=None)


def load(path):
    spec = importlib.util.spec_from_file_location(f"m13_{Path(path).stem}", ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def error_type(exc):
    """Low-cardinality error.type: a timeout is a timeout whichever layer raised it."""
    reason = getattr(exc, "reason", None)
    if isinstance(exc, TimeoutError) or isinstance(reason, TimeoutError):  # socket.timeout is TimeoutError
        return "timeout"
    if isinstance(exc, urllib.error.HTTPError):
        return str(exc.code)
    return type(exc).__qualname__


class Span:
    def __init__(self, trace_id, span_id, parent_span_id, name, kind, attributes):
        self.trace_id, self.span_id, self.parent_span_id = trace_id, span_id, parent_span_id
        self.name, self.kind, self.attributes = name, kind, dict(attributes or {})
        self.status = {"code": "UNSET"}

    def record(self):
        return {"trace_id": self.trace_id, "span_id": self.span_id, "parent_span_id": self.parent_span_id, "name": self.name,
                "kind": self.kind, "start_time_unix_nano": self.start, "end_time_unix_nano": self.end,
                "duration_ms": round((self.end - self.start) / 1e6, 3), "status": self.status,
                "attributes": {k: v for k, v in self.attributes.items() if v is not None}}


class Tracer:
    """Spans to spans.jsonl, evaluation events to events.jsonl; the parent is whatever span is current in this context."""

    def __init__(self, folder, new_id=secrets.token_hex):
        self.folder, self.new_id = Path(folder), new_id
        self.folder.mkdir(parents=True, exist_ok=True)
        self.spans = (self.folder / "spans.jsonl").open("w", encoding="utf-8", newline="\n")
        self.events = (self.folder / "events.jsonl").open("w", encoding="utf-8", newline="\n")

    def close(self):
        self.spans.close()
        self.events.close()

    @contextlib.contextmanager
    def span(self, name, kind="INTERNAL", attributes=None):
        parent = CURRENT.get()
        span = Span(parent.trace_id if parent else self.new_id(16), self.new_id(8), parent.span_id if parent else None,
                    name, kind, attributes)
        token = CURRENT.set(span)
        span.start, started = time.time_ns(), time.perf_counter_ns()
        try:
            yield span
        except Exception as exc:
            span.status = {"code": "ERROR", "message": f"{type(exc).__name__}: {exc}"[:200]}
            span.attributes.setdefault("error.type", error_type(exc))
            raise
        finally:
            # Wall clock places the span in time, the monotonic clock measures it: the end never jumps with clock corrections.
            span.end = span.start + (time.perf_counter_ns() - started)
            CURRENT.reset(token)
            self.spans.write(json.dumps(span.record(), ensure_ascii=False) + "\n")

    def event(self, name, attributes, span):
        self.events.write(json.dumps({"trace_id": span.trace_id, "span_id": span.span_id, "time_unix_nano": time.time_ns(),
                                      "event_name": name, "attributes": attributes}, ensure_ascii=False) + "\n")


def traceparent(span):
    """W3C Trace Context header that carries this span to another process as the parent of its spans."""
    return f"00-{span.trace_id}-{span.span_id}-01"


def answer(tracer, pipeline, question, local, timeout=600, capture_content=False):
    """One question through the M11-L07X pipeline, one trace. pipeline: (error_analysis module, retriever, rows_by_id)."""
    ea, retriever, rows_by_id = pipeline
    root_attrs = {"gen_ai.operation.name": "invoke_workflow", "gen_ai.agent.name": "rag-answer",
                  "course.question.id": question["id"], "course.question.set": question["set"]}
    with tracer.span("invoke_workflow rag-answer", "INTERNAL", root_attrs) as root:
        with tracer.span("retrieval", "INTERNAL", {"gen_ai.operation.name": "retrieval", "gen_ai.retrieval.top_k": ea.TOP,
                                                   "gen_ai.data_source.id": "m10-ru-paragraphs"}) as span:
            retrieved = retriever.top(question["text"])
            # Opt-In attribute; ids and ranks only, never the paragraph text
            span.attributes["gen_ai.retrieval.documents"] = [{"id": cid, "rank": i} for i, cid in enumerate(retrieved, 1)]
            if capture_content:
                span.attributes["gen_ai.retrieval.query.text"] = question["text"]
        user = ea.prompt(question["text"], retrieved, rows_by_id)
        payload = {"model": ea.MODEL, "max_tokens": ea.MAX_TOKENS, "temperature": 0, "seed": 17, "cache_prompt": False,
                   "messages": [{"role": "system", "content": ea.SYSTEM}, {"role": "user", "content": user}]}
        chat_attrs = {"gen_ai.operation.name": "chat", "gen_ai.provider.name": "llama.cpp", "gen_ai.request.model": ea.MODEL,
                      "gen_ai.request.max_tokens": ea.MAX_TOKENS, "gen_ai.request.temperature": 0.0, "gen_ai.request.seed": 17,
                      "gen_ai.output.type": "text", "server.address": local.HOST, "server.port": local.PORT}
        with tracer.span(f"chat {ea.MODEL}", "CLIENT", chat_attrs) as chat:
            if capture_content:
                chat.attributes["gen_ai.system_instructions"] = [{"type": "text", "content": ea.SYSTEM}]
                chat.attributes["gen_ai.input.messages"] = [{"role": "user", "parts": [{"type": "text", "content": user}]}]
            with local.request("/v1/chat/completions", payload, timeout=timeout) as response:
                data = json.load(response)
            choice, usage, timings = data["choices"][0], data.get("usage") or {}, data.get("timings") or {}
            text = (choice["message"].get("content") or "").strip()
            chat.attributes.update({"gen_ai.response.id": data.get("id"), "gen_ai.response.model": data.get("model"),
                                    "gen_ai.response.finish_reasons": [choice.get("finish_reason")],
                                    "gen_ai.usage.input_tokens": usage.get("prompt_tokens"),
                                    "gen_ai.usage.output_tokens": usage.get("completion_tokens"),
                                    "llama_cpp.prompt_ms": timings.get("prompt_ms"), "llama_cpp.predicted_ms": timings.get("predicted_ms")})
            if capture_content:
                chat.attributes["gen_ai.output.messages"] = [{"role": "assistant", "parts": [{"type": "text", "content": text}],
                                                              "finish_reason": choice.get("finish_reason")}]
        with tracer.span("evaluate", "INTERNAL", {"course.checks": len(ea.CHECKS)}) as span:
            signals = ea.signals(text, retrieved, question["relevant"], rows_by_id, choice.get("finish_reason"), question["text"])
            failed = [name for name, rule in ea.CHECKS.items() if rule(signals)]
            for name in ea.CHECKS:
                # A quality verdict is an event about the chat span, not an error status: the call itself succeeded.
                tracer.event("gen_ai.evaluation.result", {"gen_ai.evaluation.name": name,
                                                          "gen_ai.evaluation.score.label": "fail" if name in failed else "pass",
                                                          "gen_ai.evaluation.score.value": 0.0 if name in failed else 1.0,
                                                          "gen_ai.response.id": data.get("id")}, chat)
            span.attributes["course.checks.failed"] = failed
        root.attributes["course.checks.failed_count"] = len(failed)
    return text, failed


def tracer_overhead(n=2000):
    """Microseconds per empty span written to a temporary file: what the tracing itself costs."""
    folder = OUT / "overhead"
    tracer = Tracer(folder)
    started = time.perf_counter()
    for _ in range(n):
        with tracer.span("noop"):
            pass
    tracer.close()
    return round((time.perf_counter() - started) / n * 1e6, 1)


def collect(out_dir=OUT, threads=4, limit=None, capture_content=False):
    ea = load("labs/M11/error_analysis.py")
    local, serving = load("labs/M09/llama_local.py"), load("labs/M09/llama_serving.py")
    rows, questions = ea.dataset()
    rows_by_id = {r["id"]: r for r in rows}
    pipeline = (ea, ea.Retriever(rows), rows_by_id)
    entry = serving.gguf_entry(ea.MODEL)
    model = ROOT / entry["path"] / entry["file"]
    if not model.exists():
        raise SystemExit("Run: .\\.venv\\Scripts\\python.exe tools\\download_models.py --gguf")
    out = Path(out_dir)
    tracer = Tracer(out)
    deadline = {"question": questions[0]["id"], "client_timeout_s": DEADLINE_S}
    try:
        with local.Server(model, threads=threads, ctx=4096, alias=ea.MODEL, log=OUT / "server.log"):  # the log stays in artifacts
            for q in questions[:limit]:
                answer(tracer, pipeline, q, local, capture_content=capture_content)
            try:
                answer(tracer, pipeline, questions[0], local, timeout=DEADLINE_S, capture_content=capture_content)
            except (urllib.error.URLError, TimeoutError) as exc:
                deadline["error_type"] = error_type(exc)
            answer(tracer, pipeline, questions[0], local, capture_content=capture_content)
    finally:
        tracer.close()
    meta = {"task": "T39", "semconv": SEMCONV, "resource": {"service.name": "course-rag", "service.version": "1.2.0"},
            "model": ea.MODEL, "revision": entry["revision"], "llama_cpp": local.RELEASE["tag"], "threads": threads,
            "questions": len(questions[:limit]), "deadline": deadline, "capture_content": capture_content,
            "tracer_overhead_us_per_span": tracer_overhead(), "collected": time.strftime("%Y-%m-%d")}
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return meta


def read(folder):
    folder = Path(folder)
    lines = lambda name: [json.loads(x) for x in (folder / name).read_text(encoding="utf-8").splitlines() if x.strip()]
    return lines("spans.jsonl"), lines("events.jsonl"), json.loads((folder / "meta.json").read_text(encoding="utf-8"))


def nearest_rank(values, q):
    values = sorted(values)
    return values[max(1, math.ceil(q * len(values))) - 1]


def stats(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return {"n": len(values), "p50": round(nearest_rank(values, 0.5), 1), "p95": round(nearest_rank(values, 0.95), 1),
            "max": round(max(values), 1)}


def stage(span):
    return span["attributes"].get("gen_ai.operation.name") or span["name"]


def analyze(folder=FROZEN):
    spans, events, meta = read(folder)
    traces = collections.defaultdict(list)
    for s in spans:
        traces[s["trace_id"]].append(s)
    ids = {s["span_id"] for s in spans}
    assert all(s["parent_span_id"] in ids for s in spans if s["parent_span_id"]), "every parent must be in the file"
    roots = {tid: next(s for s in group if s["parent_span_id"] is None) for tid, group in traces.items()}
    complete = {tid for tid, group in traces.items() if roots[tid]["status"]["code"] != "ERROR"}
    by_stage = collections.defaultdict(list)
    for tid in complete:
        for s in traces[tid]:
            by_stage[stage(s)].append(s["duration_ms"])
    chats = {tid: next(s for s in traces[tid] if stage(s) == "chat") for tid in complete}
    server_ms = {tid: c["attributes"]["llama_cpp.prompt_ms"] + c["attributes"]["llama_cpp.predicted_ms"] for tid, c in chats.items()}
    failed = collections.defaultdict(set)
    for e in events:
        if e["attributes"]["gen_ai.evaluation.score.label"] == "fail":
            failed[e["trace_id"]].add(e["attributes"]["gen_ai.evaluation.name"])
    split = {}
    for label, group in (("no_failed_check", [t for t in complete if not failed[t]]), ("any_failed_check", [t for t in complete if failed[t]]),
                         ("no_answer", [t for t in complete if "no_answer" in failed[t]])):
        split[label] = {"traces": len(group), "workflow_ms": stats([roots[t]["duration_ms"] for t in group]),
                        "output_tokens_p50": stats([chats[t]["attributes"]["gen_ai.usage.output_tokens"] for t in group])["p50"] if group else None}
    slowest = max(complete, key=lambda t: roots[t]["duration_ms"])
    errors = [{"question": roots[t]["attributes"].get("course.question.id"), "spans": len(traces[t]),
               "error_spans": [(s["name"], s["attributes"].get("error.type")) for s in traces[t] if s["status"]["code"] == "ERROR"],
               "workflow_ms": roots[t]["duration_ms"]} for t in traces if t not in complete]
    order = sorted(complete, key=lambda t: roots[t]["start_time_unix_nano"])
    after_deadline = order[-1] if errors else None
    return {"task": "T39", "semconv": meta["semconv"], "model": meta["model"], "traces": len(traces), "complete": len(complete),
            "spans": len(spans), "events": len(events), "stages_ms": {k: stats(v) for k, v in sorted(by_stage.items())},
            "chat_share_of_workflow": round(sum(c["duration_ms"] for c in chats.values()) / sum(roots[t]["duration_ms"] for t in complete), 3),
            "prefill_ms": stats([c["attributes"]["llama_cpp.prompt_ms"] for c in chats.values()]),
            "decode_ms": stats([c["attributes"]["llama_cpp.predicted_ms"] for c in chats.values()]),
            # tokens per second of decoding: the same model on a busy machine is slower, the trace shows it
            "decode_tokens_per_s": stats([c["attributes"]["gen_ai.usage.output_tokens"] / c["attributes"]["llama_cpp.predicted_ms"] * 1000
                                          for c in chats.values() if c["attributes"]["llama_cpp.predicted_ms"]]),
            "client_minus_server_ms": stats([chats[t]["duration_ms"] - server_ms[t] for t in complete]),
            "input_tokens": stats([c["attributes"]["gen_ai.usage.input_tokens"] for c in chats.values()]),
            "output_tokens": stats([c["attributes"]["gen_ai.usage.output_tokens"] for c in chats.values()]),
            "by_verdict": split, "errors": errors,
            "after_deadline": None if after_deadline is None else {
                "question": roots[after_deadline]["attributes"]["course.question.id"], "chat_ms": chats[after_deadline]["duration_ms"],
                "server_ms": round(server_ms[after_deadline], 1),
                "client_minus_server_ms": round(chats[after_deadline]["duration_ms"] - server_ms[after_deadline], 1)},
            "slowest": {"question": roots[slowest]["attributes"]["course.question.id"], "workflow_ms": roots[slowest]["duration_ms"],
                        "output_tokens": chats[slowest]["attributes"]["gen_ai.usage.output_tokens"],
                        "finish_reasons": chats[slowest]["attributes"]["gen_ai.response.finish_reasons"]},
            "tracer_overhead_us_per_span": meta.get("tracer_overhead_us_per_span")}


def any_value(value):
    """OTLP/JSON AnyValue: 64-bit integers travel as strings, lists and dicts as nested values."""
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        return {"intValue": str(value)}
    if isinstance(value, float):
        return {"doubleValue": value}
    if isinstance(value, (list, tuple)):
        return {"arrayValue": {"values": [any_value(v) for v in value]}}
    if isinstance(value, dict):
        return {"kvlistValue": {"values": attributes(value)}}
    return {"stringValue": str(value)}


def attributes(mapping):
    return [{"key": k, "value": any_value(v)} for k, v in mapping.items() if v is not None]


def otlp(folder=FROZEN, out_dir=OUT):
    """One OTLP/JSON ExportTraceServiceRequest per trace and one ExportLogsServiceRequest per trace, as JSON lines."""
    spans, events, meta = read(folder)
    resource = {"attributes": attributes(meta["resource"])}
    scope = {"name": "course.m13.tracing", "version": "1.2.0"}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    by_trace, logs = collections.defaultdict(list), collections.defaultdict(list)
    for s in spans:
        span = {"traceId": s["trace_id"], "spanId": s["span_id"], "name": s["name"], "kind": KINDS[s["kind"]],
                "startTimeUnixNano": str(s["start_time_unix_nano"]), "endTimeUnixNano": str(s["end_time_unix_nano"]),
                "attributes": attributes(s["attributes"]), "status": {"code": STATUS[s["status"]["code"]]}}
        if s["parent_span_id"]:
            span["parentSpanId"] = s["parent_span_id"]
        if s["status"].get("message"):
            span["status"]["message"] = s["status"]["message"]
        by_trace[s["trace_id"]].append(span)
    for e in events:
        logs[e["trace_id"]].append({"timeUnixNano": str(e["time_unix_nano"]), "traceId": e["trace_id"], "spanId": e["span_id"],
                                    "eventName": e["event_name"], "attributes": attributes(e["attributes"])})
    with (out / "otlp-traces.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for group in by_trace.values():
            handle.write(json.dumps({"resourceSpans": [{"resource": resource, "scopeSpans": [{"scope": scope, "spans": group}]}]},
                                    ensure_ascii=False) + "\n")
    with (out / "otlp-logs.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for group in logs.values():
            handle.write(json.dumps({"resourceLogs": [{"resource": resource, "scopeLogs": [{"scope": scope, "logRecords": group}]}]},
                                    ensure_ascii=False) + "\n")
    return {"traces": len(by_trace), "log_batches": len(logs), "folder": str(out)}


def send(endpoint, folder=OUT, headers_env="OTEL_EXPORTER_OTLP_HEADERS"):
    """POST every OTLP/JSON line to {endpoint}/v1/traces and /v1/logs. Extra headers, such as Langfuse's Basic auth, come
    from an environment variable in the OpenTelemetry format key=value,key=value and are never printed."""
    headers = {"Content-Type": "application/json"}
    for pair in filter(None, os.environ.get(headers_env, "").split(",")):
        key, _, value = pair.partition("=")
        headers[key.strip()] = value.strip()
    sent = collections.Counter()
    for name, path in (("traces", "otlp-traces.jsonl"), ("logs", "otlp-logs.jsonl")):
        for line in (Path(folder) / path).read_text(encoding="utf-8").splitlines():
            req = urllib.request.Request(f"{endpoint.rstrip('/')}/v1/{name}", data=line.encode("utf-8"), headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=30) as response:
                sent[f"{name}_{response.status}"] += 1
    return dict(sent)


def table_text(report):
    lines = ["| Этап | n | p50, мс | p95, мс | max, мс |", "| --- | --- | --- | --- | --- |"]
    for name, s in report["stages_ms"].items():
        lines.append(f"| {name} | {s['n']} | {s['p50']} | {s['p95']} | {s['max']} |")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["collect", "analyze", "otlp", "send"])
    parser.add_argument("--folder", type=Path, help="traces to analyze or export; default quality/tracing")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--limit", type=int, help="collect: first N questions only")
    parser.add_argument("--capture-content", action="store_true", help="collect: also record prompts and answers (Opt-In)")
    parser.add_argument("--endpoint", help="send: OTLP/HTTP base URL, e.g. http://localhost:3000/api/public/otel")
    args = parser.parse_args()
    if args.command == "collect":
        print(json.dumps(collect(args.out, limit=args.limit, capture_content=args.capture_content), ensure_ascii=False, indent=2))
    elif args.command == "analyze":
        result = analyze(args.folder or FROZEN)
        print(table_text(result))
        print(json.dumps({k: v for k, v in result.items() if k != "stages_ms"}, ensure_ascii=False, indent=2))
    elif args.command == "otlp":
        print(otlp(args.folder or FROZEN, args.out))
    else:
        print(send(args.endpoint, args.out))
