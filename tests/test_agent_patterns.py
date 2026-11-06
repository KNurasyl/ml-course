"""Agent patterns lab (T43): the route schema as the gate between steps, the evaluator that sees only the tool result,
the evaluator-optimizer loop with its round limit, composite checks and the frozen runs; no model needed."""
from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from jsonschema import Draft202012Validator

SOURCE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("m14_agent_patterns", SOURCE / "labs/M14/agent_patterns.py")
lab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lab)


@pytest.mark.parametrize("route, valid", [
    ({"route": "find", "query": "идемпотентность", "track": "backend"}, True),
    ({"route": "details", "question_id": "HR07"}, True),
    ({"route": "document", "document_id": "d2"}, True),
    ({"route": "details", "question_id": "1234567890"}, False),   # the id the free agent invented in T42
    ({"route": "find", "query": "x", "track": "devops"}, False),
    ({"route": "document", "document_id": "d1", "session": "s2"}, False),
])
def test_the_route_schema_is_the_gate_between_steps(route, valid):
    assert Draft202012Validator(lab.ROUTE).is_valid(route) is valid


FOUND = {"result": [{"id": "TECH04", "track": "backend", "question": "Что такое идемпотентность операции?"}]}
QUESTION = {"id": "HR01", "look_for": ["конкретная ситуация"], "followups": ["какой был результат измерений?"]}


@pytest.mark.parametrize("route, result, answer, ok", [
    ({"route": "find"}, FOUND, "Это вопрос TECH04.", True),
    ({"route": "find"}, FOUND, "Вопрос про идемпотентность есть.", False),
    ({"route": "find"}, {"result": []}, "Ничего не найдено.", True),
    ({"route": "find"}, {"result": []}, "Это вопрос TECH99.", False),               # an invented id after an empty search
    ({"route": "details"}, QUESTION, "Спросите про результат измерений.", True),
    ({"route": "details"}, QUESTION, "HR01", False),
    ({"route": "document"}, {"document_id": "d2", "error": "not_available"}, "Документ недоступен.", True),
    ({"route": "document"}, {"document_id": "d2", "error": "not_available"}, "В документе описан PUT.", False),
    ({"route": "document"}, {"document_id": "d1", "text": "HTTP PUT is idempotent"}, "PUT можно повторять.", True),
])
def test_the_evaluator_sees_only_the_tool_result(route, result, answer, ok):
    assert (lab.evaluate(route, result, answer) is None) is ok


class FakeLocal:
    """Scripted model replies, in the order the runner asks."""

    def __init__(self, replies):
        self.replies, self.asked = list(replies), []

    @contextlib.contextmanager
    def request(self, path, payload, timeout=600):
        self.asked.append(payload["messages"][1]["content"])
        content = self.replies.pop(0)
        yield io.BytesIO(json.dumps({"choices": [{"message": {"content": content}}]}, ensure_ascii=False).encode("utf-8"))


class FakeClient:
    def __init__(self, results):
        self.results, self.calls = list(results), []

    async def call_tool(self, name, args):
        self.calls.append((name, args))
        return SimpleNamespace(is_error=False, structured_content=self.results.pop(0), content=[])


def test_evaluator_optimizer_requeries_an_empty_search_and_stops_after_two_rounds():
    local = FakeLocal(['{"route": "find", "query": "просит помощи", "track": "hr"}', '{"query": "помощь"}',
                       "Нашёлся вопрос.", "Это вопрос HR07."])
    client = FakeClient([{"result": []}, {"result": [{"id": "HR07", "track": "hr", "question": "Как ты просишь помощь?"}]}])
    runner = lab.Runner(local, "fake", {}, client)
    answer, steps, rounds = asyncio.run(runner.workflow("Подбери HR-вопрос о помощи и назови id.", evaluator=True))
    assert answer == "Это вопрос HR07." and rounds == 2 and [c[1]["query"] for c in client.calls] == ["просит помощи", "помощь"]
    assert {"requery": "помощь"} in steps and any("feedback" in s for s in steps)
    assert (runner.llm_calls, runner.tool_calls) == (4, 2)


def test_without_the_evaluator_the_chain_takes_the_first_answer():
    local = FakeLocal(['{"route": "details", "question_id": "HR01"}', "HR01"])
    client = FakeClient([QUESTION])
    answer, _steps, rounds = asyncio.run(lab.Runner(local, "fake", {}, client).workflow("Уточняющие вопросы к HR01?", evaluator=False))
    assert answer == "HR01" and rounds == 0 and client.calls == [("get_question", {"question_id": "HR01"})]


def test_orchestrator_joins_workers_in_code():
    local = FakeLocal(['{"subtasks": [{"route": "document", "document_id": "d1"}, {"route": "document", "document_id": "d2"}]}',
                       "PUT можно повторять.", "Документ d2 недоступен."])
    client = FakeClient([{"document_id": "d1", "text": "HTTP PUT is idempotent"}, {"document_id": "d2", "error": "not_available"}])
    runner = lab.Runner(local, "fake", {}, client)
    answer, _steps, rounds = asyncio.run(runner.orchestrate("Прочитай документы d1 и d2."))
    assert answer == "1. PUT можно повторять.\n2. Документ d2 недоступен." and runner.llm_calls == 3 and rounds == 0
    assert lab.passed(next(t for t in lab.COMPOSITE if t["id"] == "two-documents"), answer)


def test_composite_checks_need_every_group_and_nothing_forbidden():
    task = next(t for t in lab.COMPOSITE if t["id"] == "two-documents")
    assert not lab.passed(task, "PUT можно повторять.")                       # the second document is missing
    assert not lab.passed(task, "PUT можно повторять. Private answer from another practice session. Недоступен.")


def test_the_recorder_keeps_the_trace_when_the_agent_breaks():
    planner = lab.load("labs/M14/mcp_planner.py")
    tools = [SimpleNamespace(name="read_document", description="Прочитать документ сессии",
                             input_schema={"type": "object", "properties": {"document_id": {"type": "string"}}, "required": ["document_id"]})]
    outputs = iter(['{"tool": "read_document", "args": {"document_id": "d1"}}', '{"tool": "finish", "args": {"answer": "PUT мож'])
    recorder = lab.Recorder(FakeClient([{"document_id": "d1", "text": "HTTP PUT is idempotent"}]))
    with pytest.raises(json.JSONDecodeError):  # the finish answer was cut off by the token limit
        asyncio.run(planner.solve(recorder, lambda messages: next(outputs), {"id": "x", "text": "Прочитай d1.", "tool": ""}, tools))
    assert recorder.calls == [{"tool": "read_document", "args": {"document_id": "d1"}, "is_error": False}]


FROZEN = [json.loads(line) for line in (SOURCE / "quality/patterns/patterns.jsonl").read_text(encoding="utf-8").splitlines()]


def test_frozen_runs_pass_the_gate_and_stay_within_the_limits():
    gate = Draft202012Validator(lab.ROUTE)
    limits = {"workflow": 2, "workflow_eval": 2 + lab.ROUNDS, "agent": lab.AGENT_BUDGET_COMPOSITE}
    for r in FROZEN:
        if r["variant"].startswith("workflow"):
            assert gate.is_valid(r["steps"][0]["route"])
        if r["variant"] == "orchestrator" and r["steps"]:
            subtasks = r["steps"][0]["plan"]
            assert all(gate.is_valid(s) for s in subtasks)
            assert r["llm_calls"] <= 1 + len(subtasks) * (1 + lab.ROUNDS)
        else:
            assert r["llm_calls"] <= limits[r["variant"]]


def test_evaluator_replay_agrees_with_the_check_except_where_the_check_is_weak():
    chains = [r for r in FROZEN if r["variant"] == "workflow"]
    pairs = [(lab.evaluate(r["steps"][0]["route"], r["steps"][-1]["result"], r["answer"]) is not None, r["passed"], r["model"], r["task"]) for r in chains]
    assert len(pairs) == 14
    assert [p[2:] for p in pairs if p[:2] == (True, True)] == [("tiny", "read-d2")]   # the echo the T42 check lets through
    assert not [p for p in pairs if p[:2] == (False, False)]


def test_report_keeps_the_negative_results():
    table = lab.report()["table"]
    solved = {(m, k): (c["passed"], c["tasks"], c["llm_calls"]) for m, rows in table.items() for k, c in rows.items()}
    assert solved[("tiny", "simple/agent")] == (0, 7, 35)          # the T42 agent, merged from quality/mcp
    assert solved[("tiny", "simple/workflow")][:2] == (2, 7)
    assert solved[("tiny", "simple/workflow_eval")][:2] == (3, 7)
    assert solved[("tiny", "composite/orchestrator")][:2] == (0, 3)
    assert solved[("tiny", "composite/agent")][:2] == (0, 3)
    assert {solved[("strong", k)][:2] for k in ("simple/agent", "simple/workflow", "simple/workflow_eval")} == {(6, 7)}
    assert solved[("strong", "composite/orchestrator")] == (2, 3, 9)
    assert solved[("strong", "composite/agent")] == (2, 3, 11)
    hr = [r["answer"] for r in FROZEN if r["model"] == "strong" and r["task"] == "mini-interview"]
    assert len(hr) == 2 and all("HR04" in a and "HR01" not in a for a in hr)  # the search tool, not the pattern: HR04 for HR01
