"""R04: /study picks the next mission deterministically from route 2.0 and the journal."""
from __future__ import annotations

from datetime import date
import importlib.util
import shutil
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1]


def load_tool(name):
    spec = importlib.util.spec_from_file_location(name, SOURCE / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


progress = load_tool("progress")
DAY = date(2026, 11, 20)


@pytest.fixture
def course(tmp_path):
    (tmp_path / "course").mkdir()
    shutil.copy(SOURCE / "course/route-v2.yaml", tmp_path / "course/route-v2.yaml")
    return tmp_path


def add(root, task, day, state, help_level="none"):
    progress.append(root, [progress.make_record(task, day, 40, state, help_level)])


def test_empty_journal_offers_the_level_zero_mission(course):
    step = progress.next_step(course, DAY)
    assert step["type"] == "mission" and step["level"] == 0
    assert step["id"] == "M09-L07X-M0" and step["kind"] == "launch"
    assert step["progress"]["passed"] == 0 and step["progress"]["total"] > 100


def test_saved_attempt_moves_to_the_next_mission(course):
    add(course, "M09-L07X-M0", DAY, "3")
    assert progress.next_step(course, DAY)["id"] == "MP0"
    add(course, "MP0", DAY, "2", "hint1")
    step = progress.next_step(course, DAY)
    assert (step["id"], step["level"], step["kind"]) == ("M09-L01-M1", 1, "entry")
    assert step["progress"]["passed"] == 2


def test_failed_attempt_keeps_the_mission(course):
    add(course, "M09-L07X-M0", DAY, "1")
    step = progress.next_step(course, DAY)
    assert step["id"] == "M09-L07X-M0" and step["previous_attempts"] == 1


def test_overdue_delayed_check_comes_first(course):
    for task in ("M09-L07X-M0", "MP0", "M09-L01-M1"):
        add(course, task, date(2026, 11, 10), "3")
    add(course, "M09-L01-M2", date(2026, 11, 10), "3")  # own work: C is due in 1 day
    step = progress.next_step(course, DAY)
    assert step["type"] == "review" and step["task"] == "M09-L01-C"
    assert step["due"] == "2026-11-11" and step["overdue_days"] == 9
    assert step["section"] == "Новая и отложенная проверка"
    # After C is solved on its own, the next check moves to 3 days later and the route continues.
    add(course, "M09-L01-C", DAY, "4")
    step = progress.next_step(course, DAY)
    assert step["type"] == "mission" and step["id"] == "M09-L01-M3"
    assert progress.next_step(course, date(2026, 11, 23))["task"] == "M09-L01-C"


def test_entry_and_lab_missions_do_not_start_reviews(course):
    add(course, "M09-L07X-M0", date(2026, 11, 1), "3")
    add(course, "MP0", date(2026, 11, 1), "3")
    add(course, "M09-L01-M1", date(2026, 11, 1), "3")
    assert progress.next_step(course, DAY)["type"] == "mission"


def test_project_ids_are_journal_tasks():
    assert progress.parse_task("MP1") == {"task": "MP1", "module": "MP1", "lesson": "MP1"}
    assert progress.parse_task("M10-FT-M1")["lesson"] == "M10"


def test_cli_prints_utf8_text(course, capsys, monkeypatch):
    monkeypatch.setattr(progress, "ROOT", course)
    assert progress.main(["next", "--date", "2026-11-20"]) == 0
    out = capsys.readouterr().out
    assert "Миссия M09-L07X-M0 (launch)" in out and "Пройдено миссий: 0 из" in out


def test_study_skill_runs_the_tutor_protocol():
    skill = (SOURCE / ".claude/skills/study/SKILL.md").read_text(encoding="utf-8")
    assert skill.startswith("---\nname: study\n")
    assert "!`.venv/Scripts/python.exe tools/progress.py next`" in skill
    assert "course/tutor-protocol.md" in skill
    protocol = (SOURCE / "course/tutor-protocol.md").read_text(encoding="utf-8")
    for rule in ("1–3–7–21", "не больше 150 слов", "tools/progress.py add", "до сохранённой попытки"):
        assert rule in protocol
