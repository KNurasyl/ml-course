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
    for task in ("M09-L07X-M0", "MP0"):
        add(course, task, date(2026, 11, 10), "3")
    add(course, "M09-L01-M1", date(2026, 11, 10), "3", "hint1")  # with a hint: the skip rule does not apply
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
    add(course, "M09-L01-M1", date(2026, 11, 1), "3", "hint1")  # an entry credited by the skip rule would start C
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


def start_level_one(root, day):
    for task in ("M09-L07X-M0", "MP0"):
        add(root, task, day, "3")


def test_entry_solved_alone_credits_m2_and_m3(course):
    start_level_one(course, DAY)
    add(course, "M09-L01-M1", DAY, "3", "none")
    step = progress.next_step(course, DAY)
    assert step["id"] == "M09-L02-M1"  # M2 and M3 of M09-L01 are skipped
    assert step["progress"]["passed"] == 5 and step["progress"]["credited"] == 2
    status = progress.mission_status(progress.read_log(course), progress.route_missions(course))
    assert status["M09-L01-M1"] == "passed"
    assert status["M09-L01-M2"] == status["M09-L01-M3"] == "credited"
    assert "из них зачтено по входу: 2" in progress.describe(step)


@pytest.mark.parametrize("state, help_level", [("3", "hint1"), ("2", "none"), ("4", "none")])
def test_skip_rule_needs_self_3_and_no_help(course, state, help_level):
    start_level_one(course, DAY)
    add(course, "M09-L01-M1", DAY, state, help_level)
    step = progress.next_step(course, DAY)
    assert step["id"] == "M09-L01-M2" and step["progress"]["credited"] == 0


def test_credited_lesson_keeps_the_delayed_check(course):
    start_level_one(course, date(2026, 11, 10))
    add(course, "M09-L01-M1", date(2026, 11, 10), "3", "none")
    step = progress.next_step(course, DAY)
    assert step["type"] == "review" and step["task"] == "M09-L01-C" and step["due"] == "2026-11-11"
    # Before the due date the route goes on to the next lesson.
    other = progress.next_step(course, date(2026, 11, 10))
    assert other["type"] == "mission" and other["id"] == "M09-L02-M1"


def test_credited_missions_count_on_the_progress_map(course, monkeypatch):
    spec = importlib.util.spec_from_file_location("map_for_skip", SOURCE / "tools/progress_map.py")
    progress_map = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(progress_map)
    start_level_one(course, DAY)
    add(course, "M09-L01-M1", DAY, "3", "none")
    assert progress_map.stats(course)["passed"] == 5


def test_tutor_protocol_states_the_skip_rule():
    protocol = (SOURCE / "course/tutor-protocol.md").read_text(encoding="utf-8")
    assert "зачтено по входу" in protocol and "--self 3 --help none" in protocol
    assert "Отложенная проверка C остаётся" in protocol


def test_mini_projects_are_split_into_sessions_with_one_result(course):
    missions = progress.route_missions(course)
    for pid, count in (("MP1", 6), ("MP2", 6), ("MP3", 5)):
        ids = [m["id"] for m in missions if m["id"].startswith(pid)]
        assert ids == [f"{pid}-S{n}" for n in range(1, count + 1)] + [pid]
        sessions = [m for m in missions if m["id"].startswith(f"{pid}-S")]
        assert all(m["kind"] == "project" and m["result"] for m in sessions)
        final = next(m for m in missions if m["id"] == pid)
        assert final["check"].startswith("projects/mini/") and final["check"].endswith("/check.py")
    assert "check" not in next(m for m in missions if m["id"] == "MP0")


def test_session_passes_by_a_record_project_only_by_a_green_check(course):
    missions = progress.route_missions(course)
    add(course, "MP1-S1", DAY, "3")
    add(course, "MP1", DAY, "4")  # a plain record, as the old rule allowed
    status = progress.mission_status(progress.read_log(course), missions)
    assert status.get("MP1-S1") == "passed" and "MP1" not in status
    red = dict(progress.make_record("MP1", DAY, 30, "3"), check="red")
    green = dict(progress.make_record("MP1", DAY, 30, "3"), check="green")
    assert "MP1" not in progress.mission_status([red], missions)
    assert progress.mission_status([green], missions)["MP1"] == "passed"
    weak = dict(progress.make_record("MP1", DAY, 30, "1"), check="green")
    assert "MP1" not in progress.mission_status([weak], missions)


def test_describe_shows_the_session_result_and_the_check(course):
    missions = {m["id"]: m for m in progress.route_missions(course)}
    step = {"type": "mission", **missions["MP1-S2"], "previous_attempts": 0, "progress": {"passed": 0, "total": 1}}
    assert "Результат занятия: прогон Qwen3.5-4B на GPU" in progress.describe(step)
    step = {"type": "mission", **missions["MP1"], "previous_attempts": 0, "progress": {"passed": 0, "total": 1}}
    assert "tools/progress.py check MP1" in progress.describe(step)


@pytest.fixture
def checked(course, monkeypatch):
    script = course / "projects/mini/MP1-model-passport/check.py"
    script.parent.mkdir(parents=True)
    monkeypatch.setattr(progress, "ROOT", course)
    return script


def test_add_refuses_a_project_with_check(checked):
    with pytest.raises(SystemExit, match="progress.py check MP1"):
        progress.main(["add", "MP1", "--min", "30", "--self", "3"])
    assert progress.main(["add", "MP1-S1", "--min", "30", "--self", "3"]) == 0


@pytest.mark.parametrize("code, verdict", [(0, "green"), (1, "red")])
def test_check_command_runs_check_py_and_records_its_result(checked, capsys, code, verdict):
    checked.write_text(f"import sys\nprint('Критериев выполнено: {5 - code} из 5.')\nsys.exit({code})\n", encoding="utf-8")
    assert progress.main(["check", "MP1", "--min", "35", "--self", "3", "--date", "2026-11-20"]) == code
    (record,) = progress.read_log(checked.parents[3])
    assert (record["task"], record["check"], record["source"]) == ("MP1", verdict, "check.py")
    assert record["note"] == f"Критериев выполнено: {5 - code} из 5."
    assert "Критериев выполнено" in capsys.readouterr().out
    passed = "MP1" in progress.mission_status([record], progress.route_missions(checked.parents[3]))
    assert passed is (verdict == "green")


def test_tutor_protocol_explains_project_sessions():
    protocol = (SOURCE / "course/tutor-protocol.md").read_text(encoding="utf-8")
    assert "MP1-S1" in protocol and r"tools\progress.py check MP1" in protocol
    assert "одной записью не проходится" in protocol
