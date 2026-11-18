"""MP1 check: structure of submissions/MP1/passport.json and every number recomputed from the raw outputs.

    python projects/mini/MP1-model-passport/check.py [--file path/to/passport.json]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import SUBMISSIONS, Report, operator_requests  # noqa: E402

FIELDS = ("category", "order_id", "urgency")
CATEGORIES = {"delivery_delay", "return_request", "warranty_repair", "payment_issue", "installation", "product_question",
              "order_change", "complaint_other"}


def valid_json(raw: str) -> dict | None:
    """The check's own reading: the answer, without code fences, is one JSON object with the three fields."""
    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("\n") + 1:] if "\n" in text else text
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) and all(k in value for k in FIELDS) else None


def close(a, b, tolerance=0.01) -> bool:
    return isinstance(a, (int, float)) and isinstance(b, (int, float)) and abs(a - b) <= tolerance


def check(passport: dict) -> Report:
    report = Report("MP1 «Паспорт моделей»: проверка паспорта")
    expected = {r["request_id"]: r["expected"] for r in operator_requests()}
    configs = passport.get("configs") or []
    names = [c.get("name") for c in configs]
    report.check(len(set(names)) >= 3 and len(set(names)) == len(names), "три разные конфигурации и больше", f"есть: {names}")
    for c in configs:
        name = c.get("name")
        outputs = c.get("outputs") or []
        ids = [o.get("request_id") for o in outputs]
        report.check(sorted(ids) == sorted(expected), f"{name}: по ответу на каждый из 20 запросов", f"ответов {len(ids)}")
        report.check(isinstance(c.get("ram_mib"), (int, float)) and c["ram_mib"] > 0
                     and isinstance(c.get("ttft_ms_median"), (int, float)) and isinstance(c.get("tps_median"), (int, float)),
                     f"{name}: память RAM, время до первого токена и токены в секунду указаны")
        if c.get("device") == "gpu":
            report.check(isinstance(c.get("vram_mib"), (int, float)) and c["vram_mib"] > 0, f"{name}: видеопамять указана")
        ttft = [o["ttft_ms"] for o in outputs if isinstance(o.get("ttft_ms"), (int, float))]
        if ttft:
            report.check(close(statistics.median(ttft), c.get("ttft_ms_median"), 1.0), f"{name}: медиана времени до первого токена совпадает",
                         f"пересчёт {statistics.median(ttft):.1f}, указано {c.get('ttft_ms_median')}")
        parsed = [valid_json(o.get("raw")) for o in outputs]
        rate = sum(p is not None for p in parsed) / len(outputs) if outputs else 0
        report.check(close(rate, c.get("json_valid_rate")), f"{name}: доля валидного JSON совпадает с пересчётом",
                     f"пересчёт {rate:.2f}, указано {c.get('json_valid_rate')}")
        right = sum(1 for o, p in zip(outputs, parsed)
                    if p and p.get("category") == expected.get(o.get("request_id"), {}).get("category"))
        accuracy = right / len(outputs) if outputs else 0
        report.check(close(accuracy, c.get("category_accuracy")), f"{name}: доля верных категорий совпадает с пересчётом",
                     f"пересчёт {accuracy:.2f}, указано {c.get('category_accuracy')}")
        scores = [o.get("quality") for o in outputs]
        rated = bool(scores) and all(s in (0, 1, 2) for s in scores)
        report.check(rated and close(statistics.mean(scores), c.get("quality")),
                     f"{name}: каждый ответ оценён по рубрике 0–2, среднее совпадает",
                     "есть неоценённые ответы" if not rated else f"пересчёт {statistics.mean(scores):.2f}")
    rubric = str(passport.get("rubric") or "")
    report.check(len(rubric.split()) >= 15 and "TODO" not in rubric, "рубрика качества записана (от 15 слов)")
    conclusion = str(passport.get("conclusion") or "")
    choice = passport.get("choice")
    report.check(choice in names, "выбранная конфигурация есть среди замеренных", f"choice = {choice!r}")
    report.check(len(conclusion.split()) >= 40 and "TODO" not in conclusion and str(choice) in conclusion,
                 "вывод «какую беру и почему» от 40 слов и называет выбор", f"слов {len(conclusion.split())}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, default=SUBMISSIONS / "MP1/passport.json")
    path = parser.parse_args().file
    if not path.is_file():
        sys.stdout.reconfigure(encoding="utf-8")
        print(f"Нет файла {path}. Начни со starter.py (README.md, раздел «Как начать»).")
        return 1
    return check(json.loads(path.read_text(encoding="utf-8"))).finish()


if __name__ == "__main__":
    sys.exit(main())
