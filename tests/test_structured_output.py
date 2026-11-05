"""Structured output lab (T31): the contract, the schema and the grammar disagree in known places; request fields per mode."""
from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(f"t31_{name}", SOURCE / f"labs/M09/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


lab, demo = load("structured_output"), load("demo")


@pytest.mark.parametrize("text, parsed, schema_valid, contract_valid", [
    ('{"feedback": "Назови цену записи", "score": 1}', True, True, True),
    ('{"feedback": "ok", "score": 1.0}', True, True, False),     # JSON Schema counts 1.0 as an integer, the contract does not
    ('{"feedback": "   ", "score": 2}', True, True, False),      # minLength 1 accepts spaces, the contract strips them
    ('{"feedback": "ok", "score": true}', True, False, False),   # a boolean is not an integer for either
    ('{"feedback": "ok", "score": 3}', True, False, False),
    ('{"feedback": "ok", "score": 1, "extra": 0}', True, False, False),
    ('```json\n{"feedback": "ok", "score": 1}\n```', False, False, False),
    ('{"feedback": "ok", "score": ', False, False, False),
])
def test_parse_schema_and_contract_are_three_different_checks(text, parsed, schema_valid, contract_valid):
    verdict = lab.check(text, demo.validate_feedback)
    assert (verdict["parsed"], verdict["schema_valid"], verdict["contract_valid"]) == (parsed, schema_valid, contract_valid)


def test_constraint_fields_per_mode():
    assert lab.constraint("prompt_only") == {} and lab.constraint("retry") == {}
    assert lab.constraint("json_object") == {"response_format": {"type": "json_object"}}
    schema = lab.constraint("json_schema")["response_format"]
    assert schema["type"] == "json_schema" and schema["json_schema"]["schema"] is lab.SCHEMA  # the OpenAI form the server enforces
    assert lab.constraint("gbnf") == {"grammar": lab.GBNF}


def test_grammar_defines_every_rule_it_uses_and_allows_only_scores_0_to_2():
    rules = dict(re.findall(r"(?m)^([a-z-]+) ::= (.*)$", lab.GBNF))
    body = re.sub(r"\[(?:\\.|[^\]\\])*\]", " ", " ".join(rules.values()))  # character classes first: they may hold a quote
    body = re.sub(r'"(?:\\.|[^"\\])*"', " ", body)                          # then string literals
    used = set(re.findall(r"[a-z][a-z-]*", body))
    assert "root" in rules and used <= set(rules), used - set(rules)
    assert rules["score"] == "[0-2]"
    assert "first" in rules["feedback"]  # stricter than the schema: no leading space in feedback


def test_cases_are_balanced_and_fixed():
    gold = [case[2] for case in lab.CASES]
    assert len(lab.CASES) == 24 and gold.count(0) == gold.count(1) == gold.count(2) == 8
    assert len({(q, a) for q, a, _ in lab.CASES}) == 24 and all(q.endswith("?") for q, _, _ in lab.CASES)
    assert lab.MODELS[0][1] == lab.MODES and set(lab.MODELS[1][1]) == {"prompt_only", "json_schema"}


def test_summary_counts_form_and_score_separately():
    def row(status, valid, score, gold, text, attempts=1, reason=None):
        return {"parsed": valid, "schema_valid": valid, "contract_valid": valid, "reason": reason, "status": status,
                "score": score, "gold": gold, "score_correct": score == gold, "ms_total": 100.0,
                "attempts": [{"finish_reason": "stop", "decode_tps": 20.0, "text": text}] * attempts}
    rows = [row("ok", True, 2, 2, '{"feedback": "Верно", "score": 2}'),
            row("ok", True, 2, 0, '{"feedback": "Ответ кандидата не верен.", "score": 2}'),
            row("needs_review", False, None, 1, "feedback: x", 3, "invalid_json")]
    summary = lab.summarize(rows)
    assert summary["contract_valid"] == 2 and summary["needs_review"] == 1 and summary["score_correct"] == 1
    assert summary["score_correct_among_valid"] == 0.5 and summary["attempts_mean"] == round(5 / 3, 2)
    assert summary["reasons"] == {"invalid_json": 1} and summary["scores"] == {"2": 2, "None": 1}
    assert summary["by_gold"] == {"0": {"2": 1}, "1": {"None": 1}, "2": {"2": 1}}
    assert summary["feedback_contradicts_score"] == 1  # "не верен" with score 2


def test_run_without_models_reports_status(tmp_path, monkeypatch):
    serving = load("llama_serving")
    fake = tmp_path / "root"
    (fake / "metadata").mkdir(parents=True)
    (fake / "metadata/reference-models.json").write_text((SOURCE / "metadata/reference-models.json").read_text(encoding="utf-8"),
                                                         encoding="utf-8")
    monkeypatch.setattr(lab, "ROOT", fake)
    monkeypatch.setattr(lab, "load", lambda name: serving if name == "llama_serving" else load(name))
    monkeypatch.setattr(serving, "ROOT", fake)
    report = lab.run(out_dir=tmp_path / "out")
    assert report["status"] == {"qwen2.5-0.5b-instruct-q4_k_m": "not_downloaded", "qwen3.5-4b-q4_k_m": "not_downloaded"}
    assert report["results"] == {} and json.loads((tmp_path / "out/report.json").read_text(encoding="utf-8"))["task"] == "T31"
