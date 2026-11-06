"""Tracing lab (T39): one span tree per request, clocks, errors, content capture off by default, OTLP/JSON shape, sending,
and the analysis of the frozen traces in quality/tracing."""
from __future__ import annotations

import contextlib
import http.server
import importlib.util
import io
import itertools
import json
import re
import threading
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, SOURCE / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


lab = load("labs/M13/tracing.py", "m13_tracing")
ea = load("labs/M11/error_analysis.py", "m11_error_analysis_for_tracing")
ROWS, QUESTIONS = ea.dataset()
BY_ID = {r["id"]: r for r in ROWS}


def counter_ids():
    numbers = itertools.count(1)
    return lambda size: f"{next(numbers):0{size * 2}x}"


class FakeLocal:
    """Stands in for labs/M09/llama_local.py: same HOST, PORT and request(), a canned llama-server answer."""
    HOST, PORT = "127.0.0.1", 18080

    def __init__(self, text="[index-v2-c1]", fail=None):
        self.text, self.fail = text, fail

    @contextlib.contextmanager
    def request(self, path, payload, timeout=600):
        if self.fail:
            raise self.fail
        body = {"id": "chatcmpl-test", "model": payload["model"],
                "choices": [{"message": {"content": self.text}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 400, "completion_tokens": 9}, "timings": {"prompt_ms": 3000.0, "predicted_ms": 500.0}}
        yield io.BytesIO(json.dumps(body).encode("utf-8"))


class FixedRetriever:
    def top(self, text, k=3):
        return ["index-v2-c1", "cache-v2-c1", "rag-v2-c1"]


PIPELINE = (ea, FixedRetriever(), BY_ID)


def spans_of(folder):
    return [json.loads(x) for x in (folder / "spans.jsonl").read_text(encoding="utf-8").splitlines()]


def test_spans_form_one_tree_and_the_context_is_restored(tmp_path):
    tracer = lab.Tracer(tmp_path, new_id=counter_ids())
    with tracer.span("root") as root:
        with tracer.span("child"), tracer.span("grandchild"):
            pass
        assert lab.CURRENT.get() is root
    tracer.close()
    assert lab.CURRENT.get() is None
    grandchild, child, parent = spans_of(tmp_path)  # a span is written when it ends: children first
    assert {s["trace_id"] for s in (grandchild, child, parent)} == {parent["trace_id"]} and len(parent["trace_id"]) == 32
    assert parent["parent_span_id"] is None and child["parent_span_id"] == parent["span_id"] and grandchild["parent_span_id"] == child["span_id"]


def test_the_end_comes_from_the_monotonic_clock(tmp_path, monkeypatch):
    wall = iter([5_000_000_000, 1_000_000_000])     # the wall clock jumps back by four seconds during the span
    ticks = iter([0, 7_000_000])                     # the monotonic clock counts 7 ms
    monkeypatch.setattr(lab.time, "time_ns", lambda: next(wall))
    monkeypatch.setattr(lab.time, "perf_counter_ns", lambda: next(ticks))
    tracer = lab.Tracer(tmp_path)
    with tracer.span("work"):
        pass
    tracer.close()
    (span,) = spans_of(tmp_path)
    assert span["end_time_unix_nano"] - span["start_time_unix_nano"] == 7_000_000 and span["duration_ms"] == 7.0


def test_a_failed_call_marks_its_span_and_the_parents(tmp_path):
    tracer = lab.Tracer(tmp_path)
    with pytest.raises(TimeoutError):
        lab.answer(tracer, PIPELINE, QUESTIONS[0], FakeLocal(fail=TimeoutError("timed out")))
    tracer.close()
    spans = {s["name"].split()[0]: s for s in spans_of(tmp_path)}
    assert set(spans) == {"retrieval", "chat", "invoke_workflow"}  # evaluate never ran
    for name in ("chat", "invoke_workflow"):
        assert spans[name]["status"]["code"] == "ERROR" and spans[name]["attributes"]["error.type"] == "timeout"
    assert spans["retrieval"]["status"]["code"] == "UNSET"
    assert (tmp_path / "events.jsonl").read_text(encoding="utf-8") == ""


def test_error_type_is_low_cardinality():
    import urllib.error
    assert lab.error_type(urllib.error.URLError(TimeoutError("timed out"))) == "timeout"
    assert lab.error_type(urllib.error.HTTPError("http://x", 503, "busy", {}, None)) == "503"
    assert lab.error_type(ConnectionRefusedError()) == "ConnectionRefusedError"


def test_answer_follows_the_genai_conventions_and_keeps_content_out_by_default(tmp_path):
    tracer = lab.Tracer(tmp_path / "default")
    text, failed = lab.answer(tracer, PIPELINE, QUESTIONS[0], FakeLocal())
    tracer.close()
    spans = {s["name"]: s for s in spans_of(tmp_path / "default")}
    chat = spans[f"chat {ea.MODEL}"]
    assert chat["kind"] == "CLIENT" and spans["retrieval"]["kind"] == "INTERNAL"
    a = chat["attributes"]
    assert a["gen_ai.operation.name"] == "chat" and a["gen_ai.provider.name"] == "llama.cpp" and a["gen_ai.request.model"] == ea.MODEL
    assert a["gen_ai.usage.input_tokens"] == 400 and a["gen_ai.response.finish_reasons"] == ["stop"]
    assert not any(k in a for k in ("gen_ai.input.messages", "gen_ai.output.messages", "gen_ai.system_instructions"))
    assert "gen_ai.retrieval.query.text" not in spans["retrieval"]["attributes"]
    root = spans["invoke_workflow rag-answer"]
    assert root["attributes"]["gen_ai.operation.name"] == "invoke_workflow" and root["attributes"]["course.checks.failed_count"] == len(failed)
    events = [json.loads(x) for x in (tmp_path / "default/events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(events) == len(ea.CHECKS) and {e["span_id"] for e in events} == {chat["span_id"]}
    assert sorted(e["attributes"]["gen_ai.evaluation.name"] for e in events if e["attributes"]["gen_ai.evaluation.score.label"] == "fail") == sorted(failed)
    assert "no_answer" in failed and text == "[index-v2-c1]"
    tracer = lab.Tracer(tmp_path / "opt-in")
    lab.answer(tracer, PIPELINE, QUESTIONS[0], FakeLocal(), capture_content=True)
    tracer.close()
    chat = next(s for s in spans_of(tmp_path / "opt-in") if s["kind"] == "CLIENT")
    assert chat["attributes"]["gen_ai.output.messages"][0]["parts"][0]["content"] == "[index-v2-c1]"


def test_traceparent_and_nearest_rank():
    span = lab.Span("4bf92f3577b34da6a3ce929d0e0e4736", "00f067aa0ba902b7", None, "x", "INTERNAL", {})
    assert lab.traceparent(span) == "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
    values = [10, 10, 12, 14, 100]  # the example of M13-L04
    assert lab.nearest_rank(values, 0.95) == 100 and lab.nearest_rank(values, 0.5) == 12


def test_otlp_json_has_the_shape_of_the_proto_examples(tmp_path):
    result = lab.otlp(lab.FROZEN, tmp_path)
    traces = [json.loads(x) for x in (tmp_path / "otlp-traces.jsonl").read_text(encoding="utf-8").splitlines()]
    logs = [json.loads(x) for x in (tmp_path / "otlp-logs.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(traces) == result["traces"] and len(logs) == result["log_batches"]
    resource = traces[0]["resourceSpans"][0]["resource"]["attributes"]
    assert {"key": "service.name", "value": {"stringValue": "course-rag"}} in resource
    spans = [s for t in traces for s in t["resourceSpans"][0]["scopeSpans"][0]["spans"]]
    for s in spans:
        assert re.fullmatch(r"[0-9a-f]{32}", s["traceId"]) and re.fullmatch(r"[0-9a-f]{16}", s["spanId"])
        assert s["kind"] in (1, 3) and isinstance(s["startTimeUnixNano"], str) and int(s["endTimeUnixNano"]) >= int(s["startTimeUnixNano"])
    chat = next(s for s in spans if s["kind"] == 3)
    values = {a["key"]: a["value"] for a in chat["attributes"]}
    assert values["gen_ai.usage.input_tokens"].keys() == {"intValue"} and isinstance(values["gen_ai.usage.input_tokens"]["intValue"], str)
    assert "arrayValue" in values["gen_ai.response.finish_reasons"] and "doubleValue" in values["gen_ai.request.temperature"]
    record = logs[0]["resourceLogs"][0]["scopeLogs"][0]["logRecords"][0]
    assert record["eventName"] == "gen_ai.evaluation.result" and len(record["spanId"]) == 16


def test_send_posts_every_line_with_headers_from_the_environment(tmp_path, monkeypatch):
    lab.otlp(lab.FROZEN, tmp_path)
    seen = []

    class Receiver(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            seen.append((self.path, self.headers.get("Authorization"), json.loads(body)))
            self.send_response(200)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Receiver)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_HEADERS", "Authorization=Basic dGVzdDp0ZXN0")  # base64 of the test pair test:test
    try:
        sent = lab.send(f"http://127.0.0.1:{server.server_port}/api/public/otel", tmp_path)
    finally:
        server.shutdown()
    lines = len((tmp_path / "otlp-traces.jsonl").read_text(encoding="utf-8").splitlines())
    assert sent["traces_200"] == lines and {p for p, _, _ in seen} == {"/api/public/otel/v1/traces", "/api/public/otel/v1/logs"}
    assert {a for _, a, _ in seen} == {"Basic dGVzdDp0ZXN0"}


CONTENT = ("gen_ai.input.messages", "gen_ai.output.messages", "gen_ai.system_instructions", "gen_ai.retrieval.query.text")


def test_frozen_traces_follow_the_conventions_and_hold_no_content():
    spans, events, meta = lab.read(lab.FROZEN)
    assert meta["capture_content"] is False and meta["semconv"]["commit"].startswith("e6658f1")
    for s in spans:
        a = s["attributes"]
        assert not any(k in a for k in CONTENT)
        if s["name"] != "evaluate":
            assert a["gen_ai.operation.name"] in ("invoke_workflow", "retrieval", "chat")
        if s["kind"] == "CLIENT":
            assert a["gen_ai.provider.name"] == "llama.cpp" and s["name"] == f"chat {a['gen_ai.request.model']}"
    assert {e["event_name"] for e in events} == {"gen_ai.evaluation.result"}


def test_frozen_traces_give_the_numbers_of_the_lesson():
    r = lab.analyze(lab.FROZEN)
    assert (r["traces"], r["complete"], r["spans"], r["events"]) == (68, 67, 271, 402)
    assert r["chat_share_of_workflow"] == 0.999 and r["stages_ms"]["chat"]["p50"] == 1759.5
    assert r["stages_ms"]["invoke_workflow"]["p95"] == 3264.6 and r["prefill_ms"]["p50"] == 1425.3 and r["decode_ms"]["p50"] == 282.6
    assert r["client_minus_server_ms"]["p50"] == 8.1 and r["after_deadline"]["client_minus_server_ms"] == 1639.9
    assert r["errors"][0]["error_spans"] == [(f"chat {ea.MODEL}", "timeout"), ("invoke_workflow rag-answer", "timeout")]
    assert r["slowest"]["question"] == "p-retry" and r["slowest"]["finish_reasons"] == ["length"]
    assert r["by_verdict"]["no_answer"]["traces"] == 43 and r["by_verdict"]["no_failed_check"]["traces"] == 7
    noisy = lab.analyze(SOURCE / "quality/tracing-noisy")  # the discarded run: same answers, a busy CPU
    assert noisy["output_tokens"] == r["output_tokens"] and noisy["by_verdict"]["no_answer"]["traces"] == 43
    assert noisy["decode_tokens_per_s"]["p50"] < 0.5 * r["decode_tokens_per_s"]["p50"]
