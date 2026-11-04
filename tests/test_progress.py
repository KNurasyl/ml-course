"""The electronic journal imports real attempts only and schedules reviews 1-3-7-21."""
from __future__ import annotations
from datetime import date
import importlib.util
import json
from pathlib import Path

import pytest


SOURCE = Path(__file__).resolve().parents[1]


def load_tool(name):
    spec = importlib.util.spec_from_file_location(name, SOURCE / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


progress = load_tool("progress")
JOURNAL = """# Прогресс

| Задание | Дата и время | Работа | Самооценка | Помощь | Отложенная проверка |
| --- | --- | --- | --- | --- | --- |
| D00 | 2026-11-05, 40 мин | submissions/D00.md | Не проверено | — | — |
| P01-HW01 | 2026-11-06, 150 мин | submissions/P01.ipynb | С поддержкой | подсказка 2 | 2026-11-12 |
| P02-HW01 | — | Не выполнена | Не проверено | — | — |

## Запись одной попытки

- Дата, длительность и версия задания:
- Путь к работе:
- Самооценка и её основание:

- Дата, длительность и версия задания: 2026-11-07, 90 мин, 1.2.0
- Путь к работе: submissions/M00-L03-A.md
- Проверяемое умение: MSE
- Ошибка и её причина: забыл возвести в квадрат
- Самооценка и её основание: самостоятельно — самооценка, новый вариант решён
- Внешняя проверка, если действительно была:
"""


@pytest.fixture
def course(tmp_path, monkeypatch):
    (tmp_path / "progress").mkdir()
    (tmp_path / "progress/progress.md").write_text(JOURNAL, encoding="utf-8")
    monkeypatch.setattr(progress, "ROOT", tmp_path)
    return tmp_path


def test_import_keeps_every_filled_attempt_and_is_idempotent(course):
    records = progress.import_journal(course)
    assert [(r["task"], r["state"], r["minutes"]) for r in records] == [
        ("D00", "unchecked", 40), ("P01-HW01", "supported", 150), ("M00-L03-A", "independent", 90)]
    assert records[2]["error"] == "забыл возвести в квадрат" and records[2]["external_check"] is None
    progress.append(course, records)
    assert progress.import_journal(course) == []


def test_committed_dashboard_is_built_from_the_journal():
    assert (SOURCE / "progress/dashboard.md").read_text(encoding="utf-8") == \
        progress.render_dashboard(progress.read_log(SOURCE))


def test_schedule_follows_1_3_7_21_and_page_example():
    days = ["2026-11-10", "2026-11-11", "2026-11-14", "2026-11-21", "2026-12-12"]
    records, dues = [], []
    for day in days:
        records.append(progress.make_record("M02-L01-B", date.fromisoformat(day), 30, "independent"))
        dues.append(progress.schedule(records)["M02-L01"]["due"])
    assert dues == ["2026-11-11", "2026-11-14", "2026-11-21", "2026-12-12", None]
    records.append(progress.make_record("M03-L02-A", date(2026, 11, 10), 30, "3"))
    records.append(progress.make_record("M03-L02-B", date(2026, 11, 11), 30, "1"))
    assert progress.schedule(records)["M03-L02"] == {"stage": 0, "due": "2026-11-12", "module": "M03",
                                                     "last": "2026-11-11", "task": "M03-L02-B"}


def test_mixed_sessions_interleave_modules():
    items = [{"lesson": lesson, "module": lesson[:3], "due": "2026-11-12"}
             for lesson in ("M02-L01", "M02-L02", "M02-L03", "M03-L01", "M05-L02")]
    sessions = progress.mixed_sessions(items)
    assert [i["module"] for i in sessions[0]] == ["M02", "M03", "M05"]
    assert sum(len(s) for s in sessions) == 5


def test_cli_add_accepts_plan_syntax_and_rejects_bad_values(course, capsys):
    assert progress.main(["add", "M10-L03-A", "--date", "2026-11-08", "--min", "140", "--self", "3",
                          "--help", "hint2", "--route", "full", "--error", "перепутал знаменатель"]) == 0
    record = progress.read_log(course)[-1]
    assert (record["lesson"], record["state_label"], record["help"]) == ("M10-L03", "Самостоятельно — самооценка", "hint2")
    with pytest.raises(ValueError):
        progress.make_record("M10-L03-A", date(2026, 11, 8), 10, "7")
    with pytest.raises(ValueError):
        progress.parse_task("lesson three")
    assert progress.main(["dashboard"]) == 0
    assert "| M10 | 1 | 140 |" in (course / "progress/dashboard.md").read_text(encoding="utf-8")
