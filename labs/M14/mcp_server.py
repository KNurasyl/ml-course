"""MCP server "interview question base" (extension 1.2.0, T42, lesson M14-L09X).

The tools wrap contracts that already exist in the course instead of adding new behaviour:
- read_document calls the M14 executor (labs/M14/demo.execute). The practice session comes from the server's own
  configuration, never from a tool argument, as M14-L03 requires;
- search_questions and get_question read the synthetic interview sets in data/interview (12 backend, 8 HR questions).
The resource template interview://questions/{question_id} exposes the same question as a resource, which the client
reads on demand instead of the model calling a tool.

Protocol revision 2026-07-28 is stateless: there is no protocol-level session, every request carries its protocol version
in _meta, and list results are the same for every connection. So a stdio server process stands for one practice session,
and --session fixes its scope when the process starts. All tools only read, which their annotations declare.

    python labs/M14/mcp_server.py --session s1      # stdio server; an MCP client starts it as a subprocess
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
from pathlib import Path
from typing import Literal

from mcp.server.caching import CacheHint
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ResourceNotFoundError, ToolError
from mcp.types import ToolAnnotations
from typing_extensions import TypedDict  # pydantic needs this TypedDict on Python 3.11

ROOT = Path(__file__).resolve().parents[2]
READ_ONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False)


# Typed results give every tool an outputSchema, and the client gets structuredContent besides the text.
class Hit(TypedDict):
    id: str
    track: str
    question: str


class Question(Hit):
    look_for: list[str]
    followups: list[str]


class DocumentId(TypedDict):
    document_id: str


class Document(DocumentId, total=False):  # text when readable, otherwise the M14 stop reason
    text: str
    error: str


def load(path):
    spec = importlib.util.spec_from_file_location(f"m14_{Path(path).stem}", ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def questions():
    """Both synthetic sets as one list; the fields that help an interviewer are kept, synthetic answers are not."""
    out = []
    for track, name in (("backend", "backend-development.json"), ("hr", "hr-development.json")):
        data = json.loads((ROOT / "data/interview" / name).read_text(encoding="utf-8"))
        for case in data["cases"]:
            out.append({"id": case["id"], "track": track, "question": case["question"],
                        "look_for": case.get("expected_concepts") or case.get("evidence_to_seek") or [],
                        "followups": case.get("followups", [])})
    return out


def words(text):
    return {w[:6] for w in re.findall(r"[a-zа-яё0-9]+", text.lower()) if len(w) > 2}  # a crude Russian stem: six letters


def build(session="s1"):
    executor = load("labs/M14/demo.py").execute
    bank = {q["id"]: q for q in questions()}
    # The tool list never changes while the server runs, so a client may cache it for an hour (ttlMs, 2026-07-28).
    server = MCPServer("interview-questions", version="1.2.0",
                       instructions="База вопросов собеседования курса: поиск вопроса, его критерии и документы сессии.",
                       cache_hints={"tools/list": CacheHint(ttl_ms=3_600_000, scope="private")})

    @server.tool(annotations=READ_ONLY)
    def search_questions(query: str, track: Literal["backend", "hr", "any"] = "any", limit: int = 3) -> list[Hit]:
        """Найти вопросы собеседования по словам запроса. Возвращает id, трек и текст вопроса, лучшие первыми."""
        wanted = words(query)
        scored = [(len(wanted & words(q["question"] + " " + " ".join(q["look_for"]))), q) for q in bank.values()
                  if track in ("any", q["track"])]
        best = sorted((s for s in scored if s[0] > 0), key=lambda s: (-s[0], s[1]["id"]))[:max(1, min(limit, 10))]
        return [{"id": q["id"], "track": q["track"], "question": q["question"]} for _, q in best]

    @server.tool(annotations=READ_ONLY)
    def get_question(question_id: str) -> Question:
        """Вопрос по id: текст, что искать в ответе кандидата и уточняющие вопросы."""
        if question_id not in bank:
            raise ToolError(f"нет вопроса {question_id}")  # an error the model can read, not a server crash
        return bank[question_id]

    @server.tool(annotations=READ_ONLY)
    def read_document(document_id: str) -> Document:
        """Прочитать документ текущей практической сессии. Документы других сессий недоступны."""
        result = executor([{"tool": "read_document", "args": {"document_id": document_id}}], session=session)
        if result["observations"]:
            return {"document_id": document_id, "text": result["observations"][0]}
        return {"document_id": document_id, "error": result["stop"]}  # the M14 stop reason: not_available, denied

    @server.resource("interview://questions/{question_id}", mime_type="application/json")
    def question_resource(question_id: str) -> str:
        """The same question as a resource."""
        if question_id not in bank:
            raise ResourceNotFoundError(f"нет вопроса {question_id}")
        return json.dumps(bank[question_id], ensure_ascii=False)

    return server


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", default="s1", help="the practice session this server process serves")
    build(parser.parse_args().session).run("stdio")
