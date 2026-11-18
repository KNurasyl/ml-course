"""R07: company data is deterministic and the MP1-MP3 checks run, pass good work and catch broken work."""
from __future__ import annotations

import importlib.util
import json
import statistics
import subprocess
import sys
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1]
MINI = SOURCE / "projects/mini"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


generator = load(SOURCE / "data/company/generate.py", "company_generate")
common = load(MINI / "common.py", "mini_common")
mp1 = load(MINI / "MP1-model-passport/check.py", "mp1_check")
mp2 = load(MINI / "MP2-eval-suite/check.py", "mp2_check")
mp3 = load(MINI / "MP3-support-operator/check.py", "mp3_check")


def test_company_data_is_deterministic_and_current():
    first, second = generator.build(), generator.build()
    assert first == second
    stale = [name for name, text in first.items() if (SOURCE / "data/company" / name).read_text(encoding="utf-8") != text]
    assert stale == []
    kb = common.kb_documents()
    assert 30 <= len(kb) <= 50 and all(doc["text"] for doc in kb.values())
    tickets = common.read_jsonl(SOURCE / "data/company/tickets.jsonl")
    assert len(tickets) == 600 and {t["label"]["category"] for t in tickets} == set(generator.TICKET_CATEGORIES)
    assert len(common.operator_requests()) == 20
    text = "".join(first.values())
    assert "@" not in text.replace(f"@{generator.DOMAIN}", "")  # no contact data besides the reserved .test domain


def test_cohen_kappa():
    assert common.cohen_kappa([1, 1, 0, 0], [1, 1, 0, 0]) == 1.0
    assert common.cohen_kappa([1, 0, 1, 0], [0, 1, 0, 1]) == -1.0
    assert round(common.cohen_kappa([2, 2, 1, 0, 2, 1], [2, 1, 1, 0, 2, 2]), 3) == 0.455


# ------------------------------------------------------------------ MP1

def passport() -> dict:
    requests = common.operator_requests()
    configs = []
    for n, (name, device) in enumerate((("qwen2.5-0.5b-cpu", "cpu"), ("qwen3.5-4b-gpu", "gpu"), ("qwen3.5-4b-cpu", "cpu"))):
        outputs = []
        for i, r in enumerate(requests):
            good = i % (n + 2) != 0
            raw = json.dumps({"category": r["expected"]["category"], "order_id": r["expected"]["order_id"],
                              "urgency": r["expected"]["urgency"]}) if good else "Категория: доставка"
            outputs.append({"request_id": r["request_id"], "raw": raw, "ttft_ms": 100.0 + i, "tps": 40.0, "quality": 2 if good else 0})
        valid = sum(json.loads(o["raw"]) is not None for o in outputs if o["raw"].startswith("{")) / len(outputs)
        configs.append({"name": name, "device": device, "ram_mib": 500, "vram_mib": 2900 if device == "gpu" else None,
                        "ttft_ms_median": statistics.median(o["ttft_ms"] for o in outputs), "tps_median": 40.0,
                        "json_valid_rate": valid, "category_accuracy": valid,
                        "quality": statistics.mean(o["quality"] for o in outputs), "outputs": outputs})
    conclusion = ("Беру qwen3.5-4b-gpu: при 2900 МиБ видеопамяти она отвечает быстрее всех и чаще выдаёт валидный JSON. "
                  "Цена выбора — занятая видеокарта: пока работает оператор, другие GPU-задачи на ноутбуке не запустить, "
                  "а запасной вариант на CPU в восемь раз медленнее и годится только для отладки и редких запросов.")
    return {"configs": configs, "choice": "qwen3.5-4b-gpu", "conclusion": conclusion,
            "rubric": "2 — категория, номер заказа и срочность верны и ответ без лишнего текста; 1 — одно поле неверно; "
                      "0 — нет JSON или неверна категория."}


def test_mp1_accepts_a_consistent_passport():
    report = mp1.check(passport())
    assert all(ok for ok, *_ in report.rows), [r for r in report.rows if not r[0]]


def test_mp1_catches_wrong_numbers_and_missing_parts():
    data = passport()
    data["configs"][0]["json_valid_rate"] = 0.99
    data["configs"][1]["vram_mib"] = None
    data["configs"][2]["outputs"] = data["configs"][2]["outputs"][:10]
    data["conclusion"] = "TODO"
    failed = [text for ok, text, _ in mp1.check(data).rows if not ok]
    assert any("доля валидного JSON" in t for t in failed)
    assert any("видеопамять" in t for t in failed)
    assert any("20 запросов" in t for t in failed)
    assert any("вывод" in t for t in failed)


# ------------------------------------------------------------------ MP2 and MP3

def mp2_folder(root: Path) -> Path:
    folder = root / "MP2"
    folder.mkdir(parents=True)
    kb = sorted(common.kb_documents())
    cases = [{"id": f"C{i:03}", "question": f"Вопрос {i}", "answerable": i >= 10,
              "expected_sources": [kb[i % len(kb)]] if i >= 10 else [], "expected_facts": ["факт"] if i >= 10 else []}
             for i in range(42)]
    (folder / "cases.jsonl").write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in cases), encoding="utf-8")
    (folder / "rubric.md").write_text("# Рубрика\n\n## Опора на источник\n\n2 — да.\n\n## Отказ, когда ответа нет\n\n2 — да.\n\n"
                                      "## Полнота\n\n2 — да.\n", encoding="utf-8")
    manual = [2, 2, 1, 0, 2, 1, 2, 0, 1, 2, 2, 1, 0, 2, 1, 2, 0, 1, 2, 2]
    auto = [2, 1, 1, 0, 2, 2, 2, 0, 1, 2, 2, 1, 1, 2, 1, 2, 0, 1, 2, 1]
    (folder / "annotations.jsonl").write_text("".join(json.dumps({"case_id": f"C{i:03}", "manual": m, "auto": a}) + "\n"
                                                      for i, (m, a) in enumerate(zip(manual, auto))), encoding="utf-8")
    kappa = common.cohen_kappa(manual, auto)
    (folder / "agreement.json").write_text(json.dumps({"kappa": kappa, "n": 20, "note": "Судья согласен с ручной разметкой "
                                                       "умеренно, на отказах ошибается чаще, поэтому отказы проверяю вручную."},
                                                      ensure_ascii=False), encoding="utf-8")
    (folder / "test_regression.py").write_text("def test_one():\n    assert True\n\n\ndef test_two():\n    assert 1 + 1 == 2\n",
                                               encoding="utf-8")
    return folder


def test_mp2_accepts_a_complete_set_and_catches_gaps(tmp_path):
    folder = mp2_folder(tmp_path)
    report = mp2.check(folder)
    assert all(ok for ok, *_ in report.rows), [r for r in report.rows if not r[0]]
    lines = (folder / "cases.jsonl").read_text(encoding="utf-8").splitlines()
    broken = [json.loads(line) for line in lines[:30]]
    broken[12]["expected_sources"] = ["KB-999"]
    (folder / "cases.jsonl").write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in broken), encoding="utf-8")
    (folder / "agreement.json").write_text(json.dumps({"kappa": 0.9, "n": 20, "note": "x"}), encoding="utf-8")
    failed = [text for ok, text, _ in mp2.check(folder, run_tests=False).rows if not ok]
    assert any("40 случаев" in t for t in failed) and any("expected_sources" in t for t in failed)
    assert any("каппа" in t for t in failed) and any("вывод о согласии" in t for t in failed)


def mp3_run(cases, refuse_all=False, wrong=False):
    rows = []
    for c in cases:
        cited = [] if not c["answerable"] else (["KB-999"] if wrong else c["expected_sources"])
        rows.append({"case_id": c["id"], "answer": "...", "cited": cited, "retrieved": c["expected_sources"],
                     "refused": refuse_all or not c["answerable"]})
    return rows


def test_mp3_recomputes_metrics_and_requires_honest_changes(tmp_path):
    cases_path = mp2_folder(tmp_path) / "cases.jsonl"
    cases = common.read_jsonl(cases_path)
    kb = common.kb_documents()
    folder = tmp_path / "MP3"
    (folder / "runs").mkdir(parents=True)
    runs = {"baseline": mp3_run(cases, wrong=True), "improved": mp3_run(cases)}
    for name, rows in runs.items():
        (folder / "runs" / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    stated = {name: mp3.metrics(cases, rows, kb) for name, rows in runs.items()}
    stated["changes"] = ("Заменил поиск по словам на BM25, добавил в промпт требование ссылаться только на найденные документы "
                         "и проверку ссылок кодом после ответа; отказ теперь проверяется по точной фразе.")
    (folder / "metrics.json").write_text(json.dumps(stated, ensure_ascii=False), encoding="utf-8")
    report = mp3.check(folder, cases_path)
    assert all(ok for ok, *_ in report.rows), [r for r in report.rows if not r[0]]
    assert stated["baseline"]["citation_valid"] < 1.0 and stated["improved"]["source_hit"] == 1.0
    # A hidden drop: the improved run refuses everything, false_refusal rises and is not named in changes.
    runs["improved"] = mp3_run(cases, refuse_all=True)
    (folder / "runs/improved.jsonl").write_text("".join(json.dumps(r) + "\n" for r in runs["improved"]), encoding="utf-8")
    stated["improved"] = mp3.metrics(cases, runs["improved"], kb)
    (folder / "metrics.json").write_text(json.dumps(stated, ensure_ascii=False), encoding="utf-8")
    failed = [text for ok, text, _ in mp3.check(folder, cases_path).rows if not ok]
    assert failed == ["ухудшение метрик не скрыто: каждая упавшая метрика названа в changes"]


@pytest.mark.parametrize("project", ["MP1-model-passport", "MP2-eval-suite", "MP3-support-operator"])
def test_checks_run_as_scripts_and_report_missing_work(project, tmp_path):
    args = {"MP1-model-passport": ["--file", str(tmp_path / "none.json")], "MP2-eval-suite": ["--dir", str(tmp_path)],
            "MP3-support-operator": ["--dir", str(tmp_path), "--cases", str(tmp_path / "none.jsonl")]}[project]
    done = subprocess.run([sys.executable, str(MINI / project / "check.py"), *args], capture_output=True, text=True,
                          encoding="utf-8", timeout=120)
    assert done.returncode == 1, done.stderr
    assert "Traceback" not in done.stderr
