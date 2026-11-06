"""MCP lab (T42): the server wraps the course's tool contracts, keeps the session scope out of tool arguments, answers both
protocol revisions; the planner's action schema comes from the tools' input schemas; frozen wire logs and runs."""
from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

Client = pytest.importorskip("mcp", reason="mcp is optional: requirements-optional.in").Client
SOURCE = Path(__file__).resolve().parents[1]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, SOURCE / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


server_lab = load("labs/M14/mcp_server.py", "m14_mcp_server")
planner = load("labs/M14/mcp_planner.py", "m14_mcp_planner")


def call(session, mode, steps):
    async def go():
        async with Client(server_lab.build(session), mode=mode) as client:
            return client.protocol_version, [await step(client) for step in steps]
    return asyncio.run(go())


def test_tools_wrap_the_course_contracts_and_declare_read_only():
    version, (listed,) = call("s1", "auto", [lambda c: c.list_tools()])
    tools = {t.name: t for t in listed.tools}
    assert version == "2026-07-28" and list(tools) == ["search_questions", "get_question", "read_document"]
    assert all(t.annotations.read_only_hint and not t.annotations.destructive_hint for t in tools.values())
    assert all(t.output_schema for t in tools.values()) and tools["search_questions"].input_schema["required"] == ["query"]


def test_session_scope_comes_from_the_server_not_from_arguments():
    for session, readable, hidden in (("s1", "d1", "d2"), ("s2", "d2", "d1")):
        _, (ok, denied) = call(session, "auto", [lambda c, d=readable: c.call_tool("read_document", {"document_id": d}),
                                                 lambda c, d=hidden: c.call_tool("read_document", {"document_id": d})])
        assert "text" in ok.structured_content and denied.structured_content == {"document_id": hidden, "error": "not_available"}
    _, (extra,) = call("s1", "auto", [lambda c: c.call_tool("read_document", {"document_id": "d2", "session": "s2"})])
    assert "text" not in (extra.structured_content or {})  # an extra argument cannot widen the scope


def test_search_get_resource_and_a_readable_error():
    _, (found, question, missing, resource) = call("s1", "auto", [
        lambda c: c.call_tool("search_questions", {"query": "Что такое идемпотентность операции?"}),
        lambda c: c.call_tool("get_question", {"question_id": "HR01"}),
        lambda c: c.call_tool("get_question", {"question_id": "NOPE"}),
        lambda c: c.read_resource("interview://questions/TECH04")])
    assert found.structured_content["result"][0]["id"] == "TECH04"
    assert question.structured_content["followups"] == ["какой был результат измерений?"]
    assert missing.is_error and "нет вопроса NOPE" in missing.content[0].text
    assert json.loads(resource.contents[0].text)["id"] == "TECH04"


def test_both_protocol_revisions_give_the_same_tools_and_results():
    results = {}
    for mode in ("auto", "legacy"):
        version, (listed, doc) = call("s1", mode, [lambda c: c.list_tools(), lambda c: c.call_tool("read_document", {"document_id": "d1"})])
        results[version] = ([t.name for t in listed.tools], doc.structured_content)
    assert set(results) == {"2026-07-28", "2025-11-25"} and len({json.dumps(v, ensure_ascii=False) for v in results.values()}) == 1


def test_a_real_stdio_server_process():
    async def go():
        async with Client(planner.server_params("s1")) as client:
            return client.protocol_version, (await client.call_tool("read_document", {"document_id": "d1"})).structured_content
    version, doc = asyncio.run(go())
    assert version == "2026-07-28" and doc["text"].startswith("HTTP PUT")


def test_the_action_schema_comes_from_the_tools():
    _, (listed,) = call("s1", "auto", [lambda c: c.list_tools()])
    validator = Draft202012Validator(planner.action_schema(listed.tools))
    assert validator.is_valid({"tool": "read_document", "args": {"document_id": "d1"}})
    assert validator.is_valid({"tool": "finish", "args": {"answer": "TECH04"}})
    for bad in ({"tool": "delete_document", "args": {"document_id": "d1"}}, {"tool": "read_document", "args": {"document_id": "d1", "session": "s2"}},
                {"tool": "search_questions", "args": {"query": "x", "track": "devops"}}, {"tool": "finish", "args": {"answer": ""}}):
        assert not validator.is_valid(bad), bad


@pytest.mark.parametrize("task_id, calls, answer, passed", [
    ("find-idempotency", ["search_questions"], "Это вопрос TECH04.", True),
    ("find-idempotency", [], "TECH04", False),                      # the answer without the tool call is a guess
    ("read-d2", ["read_document"], "Документ недоступен.", True),
    ("read-d2", ["read_document"], "Private answer from another practice session.", False),
    ("criteria-tech01", ["get_question"], None, False),              # never finished
])
def test_task_checks(task_id, calls, answer, passed):
    task = next(t for t in planner.TASKS if t["id"] == task_id)
    assert planner.check(task, [{"tool": c} for c in calls], answer) is passed


def test_frozen_wire_logs_show_what_changed():
    kinds = {mode: [(m["dir"], m["kind"]) for m in planner.describe(SOURCE / f"quality/mcp/wire-{mode}.jsonl")] for mode in ("auto", "legacy")}
    assert [k for _, k in kinds["auto"]] == ["server/discover", "result", "tools/list", "result", "tools/call", "result"]
    assert [k for _, k in kinds["legacy"]] == ["initialize", "result", "notifications/initialized", "tools/list", "result", "tools/call", "result"]
    modern = planner.describe(SOURCE / "quality/mcp/wire-auto.jsonl")
    assert all("io.modelcontextprotocol/protocolVersion" in m["meta"] for m in modern if m["dir"] == "client->server")
    assert all("resultType" in m["fields"] for m in modern if m["kind"] == "result")


def test_frozen_planner_runs():
    report = planner.report(SOURCE / "quality/mcp")["models"]
    assert (report["tiny"]["passed"], report["tiny"]["finished"], report["tiny"]["tool_calls"]) == (0, 0, 28)
    assert (report["strong"]["passed"], report["strong"]["finished"], report["strong"]["tool_calls"]) == (6, 7, 9)
    assert report["strong"]["failed"] == ["find-help"] and report["strong"]["protocol"] == "2026-07-28"
    runs = [json.loads(x) for x in (SOURCE / "quality/mcp/planner.jsonl").read_text(encoding="utf-8").splitlines()]
    miss = next(r for r in runs if r["model"] == "strong" and r["task"] == "find-help")
    assert all(c["tool"] == "search_questions" and c["result"] == {"result": []} for c in miss["calls"])  # the tool found nothing
