"""The adaptive route for familiar topics must stay complete and point to real keys."""
from __future__ import annotations
import importlib.util
import json
import re
from pathlib import Path

import pytest


SOURCE = Path(__file__).resolve().parents[1]


def load_tool(name):
    spec = importlib.util.spec_from_file_location(name, SOURCE / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


validator = load_tool("validate_course")


def route_errors(root):
    errors = []
    curriculum = json.loads((root / "curriculum.json").read_text(encoding="utf-8"))
    validator._validate_route(root, curriculum, errors)
    return errors


def test_course_route_covers_every_familiar_lesson():
    assert route_errors(SOURCE) == []


@pytest.fixture
def small_route(tmp_path):
    """P01-P10 and one lesson per module M00-M06, each with a check block and key."""
    modules = [{"id": f"M{i:02}", "lessons": [{"id": f"M{i:02}-L01"}]} for i in range(7)]
    (tmp_path / "curriculum.json").write_text(json.dumps({"modules": modules}), encoding="utf-8")
    lines = ["version: 1", "checks:"]
    lessons = [f"P{i:02}" for i in range(1, 11)] + [m["lessons"][0]["id"] for m in modules]
    for lesson in lessons:
        pilot = lesson.startswith("P")
        lesson_file = f"course/pilot/{lesson}.md" if pilot else f"course/lessons/{lesson}.md"
        key_file = "solutions/pilot/answer-key.md" if pilot else f"solutions/modules/{lesson[:3]}.md"
        anchor = f"{lesson.lower()}-d"
        target = tmp_path / lesson_file
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"# {lesson}\n\n## Проверь себя сначала\n\nЗадача. [Ключ](../../{key_file}#{anchor}).\n\n"
                          "## Самостоятельная задача A\n\nТекст.\n", encoding="utf-8")
        key = tmp_path / key_file
        key.parent.mkdir(parents=True, exist_ok=True)
        with key.open("a", encoding="utf-8") as stream:
            stream.write(f"## {lesson}-D\n\nОтвет.\n\n")
        lines += [f"  - lesson: {lesson}", f"    check: {lesson}-D", f"    file: {lesson_file}",
                  f"    answer: {key_file}#{anchor}"]
    (tmp_path / "course/route.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return tmp_path


def test_synthetic_route_is_valid(small_route):
    assert route_errors(small_route) == []


def test_missing_block_key_or_lesson_is_reported(small_route):
    lesson = small_route / "course/lessons/M03-L01.md"
    lesson.write_text(lesson.read_text(encoding="utf-8").replace("## Проверь себя сначала", "## Повтор"), encoding="utf-8")
    key = small_route / "solutions/modules/M05.md"
    key.write_text(key.read_text(encoding="utf-8").replace("## M05-L01-D", "## M05-L01-A"), encoding="utf-8")
    route = small_route / "course/route.yaml"
    route.write_text(route.read_text(encoding="utf-8").split("  - lesson: M06-L01")[0], encoding="utf-8")
    errors = route_errors(small_route)
    assert "Missing first-step check block in course/lessons/M03-L01.md" in errors
    assert "Missing key section ## M05-L01-D in solutions/modules/M05.md" in errors
    assert any("must list P01-P10 and every M00-M06 lesson" in e for e in errors)


def test_check_after_independent_task_or_wrong_link_is_reported(small_route):
    lesson = small_route / "course/lessons/M01-L01.md"
    lesson.write_text("# M01-L01\n\n## Самостоятельная задача A\n\nТекст.\n\n## Проверь себя сначала\n\n"
                      "[Ключ](../../solutions/modules/M01.md#m01-l01-d).\n", encoding="utf-8")
    pilot = small_route / "course/pilot/P02.md"
    pilot.write_text(pilot.read_text(encoding="utf-8").replace("#p02-d", "#p03-d"), encoding="utf-8")
    errors = route_errors(small_route)
    assert "First-step check must precede the independent task in course/lessons/M01-L01.md" in errors
    assert any(e.startswith("First-step check in course/pilot/P02.md must link to its key") for e in errors)


def test_checkpoint_table_covers_every_pilot_to_module_link():
    curriculum = json.loads((SOURCE / "curriculum.json").read_text(encoding="utf-8"))
    text = (SOURCE / "course/checkpoints.md").read_text(encoding="utf-8")
    section = text.split("### Что сокращать после CP0")[1].split("\n## ")[0]
    rows = {}
    for line in section.splitlines():
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if line.startswith("| P") and len(cells) == 5:
            rows[cells[0]] = (re.findall(r"\[(M\d{2}-L\d{2})\]\(lessons/\1\.md\)", cells[1]), cells[3])
    assert set(rows) == set(curriculum["pilot_lesson_map"])
    for session, lessons in curriculum["pilot_lesson_map"].items():
        linked, start = rows[session]
        assert linked == lessons
        assert all(f"{lesson}-D" in start for lesson in lessons)
