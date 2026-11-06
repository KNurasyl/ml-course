"""promptfoo layer (T38): generated files follow the Python source, the JS contract equals the Python one, the config talks
only to local servers, and the comparison lists every disagreement."""
from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

SOURCE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("m11_promptfoo_eval", SOURCE / "labs/M11/promptfoo_eval.py")
lab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lab)
CONFIG = yaml.safe_load((lab.CONFIG_DIR / "promptfooconfig.yaml").read_text(encoding="utf-8"))

EDGE_CASES = [
    '{"feedback": "Назови цену записи", "score": 1}',
    '  {"score": 0, "feedback": "x"}\n',
    '{"feedback": "ok", "score": 1.0}',       # JSON.parse gives 1, Python gives the float 1.0
    '{"feedback": "ok", "score": 1e0}',
    '{"feedback": "ok", "score": -0}',
    '{"feedback": "ok", "score": true}',
    '{"feedback": "ok", "score": 3}',
    '{"feedback": "ok", "score": "1"}',
    '{"feedback": "   ", "score": 2}',
    '{"feedback": "\\u001f", "score": 2}',    # whitespace for str.strip(), not for String.prototype.trim()
    '{"feedback": "\\u0085", "score": 2}',
    '{"feedback": "\\ufeff", "score": 2}',    # the other way round
    '{"feedback": "ok", "score": 1, "extra": 0}',
    '{"feedback": "ok", "score": 1, "score": 2}',
    '{"inner": {"score": 1.0}, "feedback": "ok"}',
    '[{"feedback": "ok", "score": 1}]',
    'null',
    '```json\n{"feedback": "ok", "score": 1}\n```',
    '{"feedback": "ok", "score": 1',
    '',
]


def test_generated_files_follow_the_python_source():
    assert lab.sync(write=False) == []
    cases = [json.loads(line) for line in (lab.CONFIG_DIR / "cases.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [(c["vars"]["question"], c["vars"]["answer"], c["vars"]["gold"]) for c in cases] == lab.so.CASES
    prompt = json.loads((lab.CONFIG_DIR / "prompt.json").read_text(encoding="utf-8"))
    assert prompt[0]["content"] == lab.so.SYSTEM and lab.expected_messages("q", "a")[1]["content"] == "Вопрос: q\nОтвет кандидата: a"


def test_variables_that_would_break_the_json_prompt_are_refused(monkeypatch):
    monkeypatch.setattr(lab.so, "CASES", [("Что такое \"индекс\"?", "Ответ", 1)])
    with pytest.raises(ValueError, match="quotes"):
        lab.generated()


def test_config_talks_only_to_local_servers_with_the_t31_request():
    providers = {p["label"]: p for p in CONFIG["providers"]}
    assert set(providers) == set(lab.PROFILES)
    for label, (model, port) in lab.PROFILES.items():
        p, c = providers[label], providers[label]["config"]
        assert p["id"] == f"openai:chat:{model}" and c["apiBaseUrl"] == f"http://127.0.0.1:{port}/v1"
        assert c["apiKeyRequired"] is False and c["useDefaultApiKey"] is False and "apiKey" not in c
        assert c["temperature"] == 0 and c["max_tokens"] == lab.so.MAX_TOKENS
        assert c["passthrough"]["seed"] == 17 and c["passthrough"]["cache_prompt"] is False
    assert providers["strong"]["config"]["passthrough"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert [a["metric"] for a in CONFIG["defaultTest"]["assert"]] == list(lab.METRICS)
    assert CONFIG["tests"] == "file://cases.jsonl" and CONFIG["evaluateOptions"]["maxConcurrency"] == 1


NO_NODE = pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is optional; the Python contract is the primary one")


def node_map(function):
    """Apply a JS function to every edge case in a real Node process."""
    script = f"const xs = JSON.parse(require('fs').readFileSync(0, 'utf8')); const f = {function};"              "process.stdout.write(JSON.stringify(xs.map(f)));"
    done = subprocess.run(["node", "-e", script], input=json.dumps(EDGE_CASES), capture_output=True, text=True,
                          encoding="utf-8", check=True, cwd=lab.CONFIG_DIR)
    return json.loads(done.stdout)


@NO_NODE
def test_js_contract_equals_the_python_contract():
    for text, js in zip(EDGE_CASES, node_map("require('./checks.js').validateFeedback"), strict=True):
        py = lab.demo.validate_feedback(text)
        assert (js["valid"], js.get("reason")) == (py["valid"], py.get("reason")), text
    nan = '{"feedback": "ok", "score": NaN}'  # Python's json accepts NaN, JSON.parse does not: same verdict, other reason
    assert lab.demo.validate_feedback(nan)["reason"] == "score_out_of_range"


@NO_NODE
def test_the_obvious_port_disagrees_on_five_edge_cases():
    naive = ("x => { let v; try { v = JSON.parse(x) } catch { return false }"
             " return v !== null && typeof v === 'object' && !Array.isArray(v) && Object.keys(v).sort().join() === 'feedback,score'"
             " && typeof v.feedback === 'string' && v.feedback.trim() !== '' && Number.isInteger(v.score) && v.score >= 0 && v.score <= 2 }")
    differ = [i for i, (text, js) in enumerate(zip(EDGE_CASES, node_map(naive), strict=True))
              if js != lab.demo.validate_feedback(text)["valid"]]
    assert differ == [2, 3, 9, 10, 11]  # 1.0, 1e0 and three whitespace characters


def fake_row(label, case, output, pf):
    question, answer, gold = case
    return {"provider": {"id": f"openai:chat:{lab.PROFILES[label][0]}", "label": label},
            "vars": {"case_id": "c01", "question": question, "answer": answer, "gold": gold},
            "prompt": {"raw": json.dumps(lab.expected_messages(question, answer), ensure_ascii=False)},
            "response": {"output": output}, "success": all(pf.values()), "latencyMs": 100,
            "gradingResult": {"componentResults": [{"pass": v, "assertion": {"metric": k}} for k, v in pf.items()]}}


def test_comparison_lists_every_disagreement(tmp_path):
    case = lab.so.CASES[0]
    rows = [fake_row("tiny", case, '{"feedback": "ok", "score": 2}', {"schema": True, "contract": True, "score": True}),
            # a checker that forgot the 1.0 case: promptfoo passes, the Python contract does not
            fake_row("strong", case, '{"feedback": "ok", "score": 2.0}', {"schema": True, "contract": True, "score": True})]
    results = tmp_path / "results.json"
    results.write_text(json.dumps({"results": {"results": rows}}, ensure_ascii=False), encoding="utf-8")
    t31 = tmp_path / "report.json"
    t31.write_text(json.dumps({"results": {f"{lab.PROFILES['tiny'][0]}/prompt_only": {"rows": [
        {"question": case[0], "answer": case[1], "attempts": [{"text": '{"feedback": "ok", "score": 2}'}]}]}}}), encoding="utf-8")
    report = lab.compare(results, t31)
    tiny, strong = report["profiles"]["tiny"], report["profiles"]["strong"]
    assert tiny["disagreements"] == [] and tiny["same_text_as_t31_prompt_only"] == 1 and tiny["prompt_as_expected"] == 1
    assert [(d["metric"], d["promptfoo"], d["own"]) for d in strong["disagreements"]] == [("contract", True, False), ("score", True, False)]
    assert strong["own"] == {"schema": 1, "contract": 0, "score": 0} and strong["scores_by_gold"] == {"2": {"None": 1}}
    assert lab.compare(results, tmp_path / "missing.json")["profiles"]["tiny"]["same_text_as_t31_prompt_only"] is None
