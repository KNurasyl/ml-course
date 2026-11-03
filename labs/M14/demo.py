"""Bounded tool executor. The planner is SCRIPTED, not an LLM."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCUMENTS = {"s1": {"d1": "HTTP PUT is idempotent; retries need application-level checks."},
             "s2": {"d2": "Private answer from another practice session."}}


def execute(actions: list[dict], session: str = "s1", budget: int = 4) -> dict:
    trace, seen, observations = [], set(), []
    for step, action in enumerate(actions):
        if step >= budget:
            return {"stop": "budget", "trace": trace, "observations": observations}
        if not isinstance(action, dict) or set(action) != {"tool", "args"}:
            return {"stop": "invalid_action", "trace": trace, "observations": observations}
        name, args = action["tool"], action["args"]
        if name == "finish" and isinstance(args, dict) and set(args) == {"answer"} and isinstance(args["answer"], str):
            return {"stop": "finished", "trace": trace, "observations": observations, "answer": args["answer"]}
        if name != "read_document" or not isinstance(args, dict) or set(args) != {"document_id"} or not isinstance(args["document_id"], str):
            return {"stop": "denied", "trace": trace, "observations": observations}
        key = (name, args["document_id"])
        if key in seen:
            return {"stop": "repeated_call", "trace": trace, "observations": observations}
        seen.add(key)
        # Scope comes from the server session, never from model arguments.
        document = DOCUMENTS.get(session, {}).get(args["document_id"])
        if document is None:
            return {"stop": "not_available", "trace": trace, "observations": observations}
        observations.append(document)
        trace.append({"step": step, "tool": name, "document_id": args["document_id"], "status": "ok"})
    return {"stop": "no_final_action", "trace": trace, "observations": observations}


def run() -> dict:
    read = {"tool": "read_document", "args": {"document_id": "d1"}}
    finish = {"tool": "finish", "args": {"answer": "PUT may be retried when the operation preserves idempotency."}}
    results = {
        "happy": execute([read, finish]),
        "unknown_tool": execute([{"tool": "delete_all", "args": {}}]),
        "cross_session": execute([{"tool": "read_document", "args": {"document_id": "d2"}}]),
        "repeated": execute([read, read, finish]),
        "budget": execute([read, finish], budget=1),
        "wrong_type": execute([{"tool": "read_document", "args": {"document_id": 1}}]),
        "extra_scope": execute([{"tool": "read_document", "args": {"document_id": "d1", "session": "s2"}}]),
        "unfinished": execute([read]),
    }
    expected = {"happy": "finished", "unknown_tool": "denied", "cross_session": "not_available", "repeated": "repeated_call", "budget": "budget", "wrong_type": "denied", "extra_scope": "denied", "unfinished": "no_final_action"}
    assert {k: v["stop"] for k, v in results.items()} == expected
    assert not results["cross_session"]["observations"]
    out = ROOT / "artifacts" / "M14"
    out.mkdir(parents=True, exist_ok=True)
    report = {"planner": "scripted deterministic test harness, NOT an LLM", "checks": len(results), "results": results}
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
