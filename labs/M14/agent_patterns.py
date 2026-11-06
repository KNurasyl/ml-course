"""Agent patterns on the tasks where the free planner failed (extension 1.2.0, T43, lesson M14-L10X).

Baseline is the agent of T42 (labs/M14/mcp_planner.py): the model picks the next tool call at every step until it calls
finish. On the 7 MCP tasks the 0.5B solved 0 of 7, looping on one call, and the 4B solved 6 of 7 (quality/mcp). Here the
same models, MCP server, tools and checks run in workflows, where code owns the control flow (Anthropic, "Building
effective agents", 2024):
- routing: one constrained call sorts the task into find, details or document and extracts its single argument;
- prompt chaining: the route goes to code, code calls the one tool, and the model writes the answer from the tool
  result. The gate between steps is the route's JSON Schema: an id of the wrong shape cannot pass;
- evaluator-optimizer: a code evaluator that sees the tool result, never the expected answer, sends feedback. An empty
  search asks for new keywords; an answer without a found id, a criterion or the document's content is rewritten. At most
  two rounds;
- orchestrator-workers: for three composite tasks one call splits the task into routed subtasks; each runs as a worker
  chain with the evaluator, and code joins the answers. The agent runs the composite tasks too, with a budget of 8 steps.
All variants are judged by deterministic checks of the answer. Negative results stay in quality/patterns.

    python labs/M14/agent_patterns.py run          # needs mcp, llama.cpp and the GGUF of M09-L07X; --models tiny for 0.5B
    python labs/M14/agent_patterns.py report       # frozen runs in quality/patterns plus the T42 agent, no model
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import importlib.util
import json
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / "quality/patterns"
OUT = ROOT / "artifacts/M14/patterns"
MODELS = {"tiny": "qwen2.5-0.5b-instruct-q4_k_m", "strong": "qwen3.5-4b-q4_k_m"}
ROUNDS, AGENT_BUDGET_COMPOSITE = 2, 8
# The agent writes the whole composite answer in one finish step. With the T42 limit of 200 tokens the 4B answer to
# mini-interview was cut off mid-string and broke the JSON (first run, 07.11.2026), so composite steps get 400.
AGENT_MAX_TOKENS_COMPOSITE = 400
ROUTE = {"anyOf": [
    {"type": "object", "additionalProperties": False, "required": ["route", "query", "track"],
     "properties": {"route": {"const": "find"}, "query": {"type": "string", "minLength": 2, "maxLength": 60},
                    "track": {"enum": ["backend", "hr", "any"]}}},
    {"type": "object", "additionalProperties": False, "required": ["route", "question_id"],
     "properties": {"route": {"const": "details"}, "question_id": {"type": "string", "pattern": "^(TECH|HR)[0-9]{2}$"}}},
    {"type": "object", "additionalProperties": False, "required": ["route", "document_id"],
     "properties": {"route": {"const": "document"}, "document_id": {"type": "string", "pattern": "^d[0-9]{1,2}$"}}},
]}
PLAN = {"type": "object", "additionalProperties": False, "required": ["subtasks"],
        "properties": {"subtasks": {"type": "array", "minItems": 1, "maxItems": 4, "items": ROUTE}}}
QUERY = {"type": "object", "additionalProperties": False, "required": ["query"],
         "properties": {"query": {"type": "string", "minLength": 2, "maxLength": 60}}}
ROUTES_TEXT = ("route=find — найти вопрос собеседования по теме: query — одно-три ключевых слова темы на русском, track — backend, "
               "hr или any. route=details — критерии и уточняющие вопросы известного вопроса: question_id вида TECH01 или HR01. "
               "route=document — прочитать документ сессии: document_id вида d1.")
ROUTER = "Определи, что нужно сделать по запросу пользователя, и верни один JSON-объект. " + ROUTES_TEXT
ORCHESTRATOR = ("Разбей запрос пользователя на отдельные шаги, каждый шаг — один маршрут, и верни JSON {\"subtasks\": [...]}. "
                + ROUTES_TEXT)
REQUERY = ("Поиск вопросов по ключевым словам ничего не нашёл. Предложи другие ключевые слова: другую форму слова или синоним, "
           "одно-три слова на русском. Верни JSON {\"query\": ...}.")
WRITER = ("Ты помощник интервьюера. Ответь на запрос одной-тремя фразами на русском, опираясь только на результат инструмента. "
          "Если результат пуст или содержит ошибку, прямо скажи об этом. Id вопросов называй как есть.")
COMPOSITE = [
    {"id": "mini-interview", "text": "Подготовь мини-собеседование: найди один backend-вопрос про идемпотентность и один HR-вопрос о "
                                     "сложной задаче; для каждого назови id и что искать в ответе.",
     "groups": [["TECH04"], ["HR01"], ["повтор", "эффект"], ["ситуац", "действи"]]},
    {"id": "two-criteria", "text": "Для вопросов TECH01 и TECH05 перечисли, что искать в ответе кандидата.",
     "groups": [["параметр", "инференс", "обучен"], ["структур", "sql", "отделя"]]},
    {"id": "two-documents", "text": "Прочитай документы d1 и d2 и скажи, что в них.",
     "groups": [["put", "идемпот", "idempot", "повтор"], ["недоступ", "нет доступа", "не найден", "not_available"]],
     "forbid": ["private answer"]},
]


def load(path):
    spec = importlib.util.spec_from_file_location(f"m14_{Path(path).stem}", ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stems(text):
    return {w[:5] for w in re.findall(r"[a-zа-яё0-9]+", text.lower()) if len(w) >= 3}


def evaluate(route, result, answer):
    """Feedback from the tool result alone, or None when the answer uses it. The expected answer is never consulted."""
    text = answer.lower()
    if route["route"] == "find":
        ids = [hit["id"] for hit in (result or {}).get("result", [])]
        if not ids:
            return None if re.search(r"не найден|ничего|нет", text) else "Поиск ничего не нашёл: так и скажи, не придумывай id."
        return None if any(i.lower() in text for i in ids) else f"Назови id найденных вопросов: {', '.join(ids)}."
    if route["route"] == "details":
        if not isinstance(result, dict) or "look_for" not in result:
            return None if re.search(r"нет|не найден|ошибк", text) else "Скажи, что такого вопроса нет."
        items = " ".join(result["look_for"] + result["followups"])
        return None if stems(items) & stems(answer) else "Перечисли, что искать в ответе и уточняющие вопросы из результата."
    if (result or {}).get("error"):
        return None if re.search(r"недоступ|нет доступа|не найден|нельзя", text) else "Скажи, что документ недоступен, и не пересказывай его."
    return None if stems(result["text"]) & stems(answer) else "Перескажи документ по его тексту."


def task_text(route):
    return {"find": f"Найди вопрос собеседования по теме «{route.get('query')}» и назови его id.",
            "details": f"Что искать в ответе на вопрос {route.get('question_id')} и какие есть уточняющие вопросы?",
            "document": f"Прочитай документ {route.get('document_id')} и перескажи его."}[route["route"]]


class Recorder:
    """MCP client proxy that keeps every tool call, so a run that breaks midway still leaves its trace."""

    def __init__(self, client):
        self.client, self.calls = client, []

    async def call_tool(self, name, args):
        result = await self.client.call_tool(name, args)
        self.calls.append({"tool": name, "args": args, "is_error": bool(result.is_error)})
        return result


class Runner:
    """One model behind llama-server and one MCP client; counts model and tool calls per task."""

    def __init__(self, local, name, extra, client):
        self.local, self.name, self.extra, self.client = local, name, extra, client
        self.llm_calls = self.tool_calls = 0

    def ask(self, system, user, schema=None, max_tokens=160):
        payload = {"model": self.name, "max_tokens": max_tokens, "temperature": 0, "seed": 17, "cache_prompt": False,
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}], **self.extra}
        if schema:
            payload["response_format"] = {"type": "json_schema", "json_schema": {"name": "out", "schema": schema}}
        self.llm_calls += 1
        with self.local.request("/v1/chat/completions", payload) as response:
            text = json.load(response)["choices"][0]["message"].get("content") or ""
        return json.loads(text) if schema else text.strip()

    async def tool(self, route):
        name, args = {"find": ("search_questions", {"query": route.get("query", ""), "track": route.get("track", "any")}),
                      "details": ("get_question", {"question_id": route.get("question_id", "")}),
                      "document": ("read_document", {"document_id": route.get("document_id", "")})}[route["route"]]
        self.tool_calls += 1
        result = await self.client.call_tool(name, args)
        if result.is_error:
            return {"error": result.content[0].text if result.content else "error"}
        return result.structured_content

    async def chain(self, task, route, evaluator):
        """Prompt chaining: route -> tool -> answer; with the evaluator, empty searches and unsupported answers get feedback."""
        steps = [{"route": route}]
        result = await self.tool(route)
        rounds = 0
        while evaluator and route["route"] == "find" and not (result or {}).get("result") and rounds < ROUNDS:
            rounds += 1
            new = self.ask(REQUERY, f"Запрос пользователя: {task}\nПрежние ключевые слова: {route['query']}", QUERY)
            route = {**route, "query": new["query"]}
            steps.append({"requery": new["query"]})
            result = await self.tool(route)
        user = f"Запрос: {task}\nРезультат инструмента: {json.dumps(result, ensure_ascii=False)}"
        answer = self.ask(WRITER, user)
        while evaluator and rounds < ROUNDS and (feedback := evaluate(route, result, answer)):
            rounds += 1
            steps.append({"feedback": feedback})
            answer = self.ask(WRITER, user + f"\nПрежний ответ: {answer}\nЗамечание проверки: {feedback}")
        steps.append({"result": result})
        return answer, steps, rounds

    async def workflow(self, task, evaluator):
        route = self.ask(ROUTER, task, ROUTE)
        return await self.chain(task, route, evaluator)

    async def orchestrate(self, task):
        plan = self.ask(ORCHESTRATOR, task, PLAN, max_tokens=240)
        parts, steps, rounds = [], [{"plan": plan["subtasks"]}], 0
        for i, route in enumerate(plan["subtasks"], 1):
            answer, worker_steps, worker_rounds = await self.chain(task_text(route), route, evaluator=True)
            parts.append(f"{i}. {answer}")
            steps.append({"worker": i, "steps": worker_steps})
            rounds += worker_rounds
        return "\n".join(parts), steps, rounds  # code joins the workers' answers: no extra model call


def passed(task, answer, calls=None):
    """The T42 check for its tasks; for composite tasks every group must be present and nothing forbidden."""
    if "groups" not in task:
        return load("labs/M14/mcp_planner.py").check(task, calls or [], answer)
    text = (answer or "").lower()
    return all(any(s.lower() in text for s in group) for group in task["groups"]) and not any(s in text for s in task.get("forbid", []))


async def run_model(label, local, serving, out):
    from mcp import Client
    planner, server = load("labs/M14/mcp_planner.py"), load("labs/M14/mcp_server.py")
    name = MODELS[label]
    entry = serving.gguf_entry(name)
    path = ROOT / entry["path"] / entry["file"]
    if not path.exists():
        return []
    extra = {"chat_template_kwargs": {"enable_thinking": False}} if label == "strong" else {}
    records = []
    with local.Server(path, threads=4, ctx=4096, alias=name, log=out / f"{label}.server.log"):
        async with Client(server.build("s1")) as client:
            for variant, tasks in (("workflow", planner.TASKS), ("workflow_eval", planner.TASKS), ("orchestrator", COMPOSITE), ("agent", COMPOSITE)):
                for task in tasks:
                    runner = Runner(local, name, extra, client)
                    calls, steps = None, []
                    try:
                        if variant == "agent":
                            planner.BUDGET = AGENT_BUDGET_COMPOSITE
                            tools = (await client.list_tools()).tools
                            schema = planner.action_schema(tools)
                            recorder, raws = Recorder(client), []

                            def propose(messages, runner=runner, schema=schema, raws=raws):
                                runner.llm_calls += 1
                                payload = {"model": name, "messages": messages, "max_tokens": AGENT_MAX_TOKENS_COMPOSITE, "temperature": 0,
                                           "seed": 17, "cache_prompt": False, "response_format": {"type": "json_schema",
                                                                                                "json_schema": {"name": "action", "schema": schema}}, **extra}
                                with local.request("/v1/chat/completions", payload) as response:
                                    raws.append(json.load(response)["choices"][0]["message"].get("content") or "")
                                return raws[-1]

                            # solve() also runs the T42 check, which needs a tool name; the composite check below replaces it
                            steps, rounds = recorder.calls, 0
                            runner.tool_calls = 0
                            try:
                                solved = await planner.solve(recorder, propose, {**task, "tool": ""}, tools)
                            finally:
                                runner.tool_calls = len(recorder.calls)
                                if raws:
                                    steps.append({"last_output": raws[-1][-300:]})
                            answer, calls = solved["answer"], solved["calls"]
                        elif variant == "orchestrator":
                            answer, steps, rounds = await runner.orchestrate(task["text"])
                        else:
                            answer, steps, rounds = await runner.workflow(task["text"], evaluator=variant == "workflow_eval")
                            calls = [{"tool": {"find": "search_questions", "details": "get_question", "document": "read_document"}[steps[0]["route"]["route"]]}]
                        error = None
                    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:  # a model output that broke the flow
                        answer, rounds, error = None, 0, f"{type(exc).__name__}: {exc}"[:200]  # steps keep what ran before the break
                    records.append({"model": label, "variant": variant, "task": task["id"], "answer": answer, "error": error,
                                    "passed": bool(answer) and passed(task, answer, calls), "llm_calls": runner.llm_calls,
                                    "tool_calls": runner.tool_calls, "rounds": rounds, "steps": steps})
    return records


def run(models=tuple(MODELS), out_dir=OUT):
    local, serving = load("labs/M09/llama_local.py"), load("labs/M09/llama_serving.py")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    records, status = [], {}
    for label in models:
        started = time.perf_counter()
        got = asyncio.run(run_model(label, local, serving, out))
        status[label] = {"records": len(got), "minutes": round((time.perf_counter() - started) / 60, 1)}
        records += got
    meta = {"task": "T43", "models": {k: MODELS[k] for k in models}, "llama_cpp": local.RELEASE["tag"], "status": status,
            "rounds": ROUNDS, "agent_budget_composite": AGENT_BUDGET_COMPOSITE, "agent_max_tokens_composite": AGENT_MAX_TOKENS_COMPOSITE,
            "first_run": "with 200 tokens per agent step the 4B finish answer to mini-interview was cut off and broke the JSON",
            "decoding": "greedy, seed 17",
            "agent_on_simple_tasks": "quality/mcp/planner.jsonl (T42)", "collected": time.strftime("%Y-%m-%d")}
    (out / "patterns-meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    with (out / "patterns.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for r in records:
            handle.write(json.dumps(r, ensure_ascii=False) + "\n")
    return report(out)


def report(folder=FROZEN):
    records = [json.loads(x) for x in (Path(folder) / "patterns.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    agent = [json.loads(x) for x in (ROOT / "quality/mcp/planner.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    for r in agent:  # the T42 agent on the simple tasks, one model call per step
        records.append({"model": r["model"], "variant": "agent", "task": r["task"], "passed": r["passed"], "error": None,
                        "llm_calls": r["steps"] + (r["stop"] != "finished"), "tool_calls": len(r["calls"]), "rounds": 0, "simple": True})
    simple = {t["id"] for t in load("labs/M14/mcp_planner.py").TASKS}
    table = collections.defaultdict(lambda: collections.defaultdict(lambda: {"tasks": 0, "passed": 0, "llm_calls": 0, "tool_calls": 0, "errors": 0}))
    for r in records:
        cell = table[r["model"]][f"{'simple' if r['task'] in simple else 'composite'}/{r['variant']}"]
        cell["tasks"] += 1
        cell["passed"] += bool(r["passed"])
        cell["llm_calls"] += r["llm_calls"]
        cell["tool_calls"] += r["tool_calls"]
        cell["errors"] += bool(r.get("error"))
    return {"task": "T43", "table": {m: dict(sorted(v.items())) for m, v in table.items()}}


def table_text(result):
    lines = ["| Модель | Задачи / вариант | Решено | Вызовов модели | Вызовов инструментов | Сбоев |", "| --- | --- | --- | --- | --- | --- |"]
    for model, rows in result["table"].items():
        for key, c in rows.items():
            lines.append(f"| {model} | {key} | {c['passed']}/{c['tasks']} | {c['llm_calls']} | {c['tool_calls']} | {c['errors']} |")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["run", "report"])
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    args = parser.parse_args()
    print(table_text(run(tuple(args.models)) if args.command == "run" else report()))
