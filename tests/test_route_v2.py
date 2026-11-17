"""Route 2.0 (R02): every lesson has one status and every mission points to a real section."""
from __future__ import annotations

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


validator = load_tool("validate_course")


def route_errors(root):
    errors = []
    curriculum = json.loads((root / "curriculum.json").read_text(encoding="utf-8"))
    validator._validate_route_v2(root, curriculum, errors)
    return errors


def test_course_route_v2_is_valid():
    assert route_errors(SOURCE) == []


def test_every_lesson_has_exactly_one_status():
    status = validator.route_v2_status(SOURCE)
    curriculum = json.loads((SOURCE / "curriculum.json").read_text(encoding="utf-8"))
    lessons = validator.PILOT_IDS + [l["id"] for m in curriculum["modules"] for l in m["lessons"]]
    assert sorted(status) == sorted(lessons)
    # Plan 2.0, section 4: what goes to the library and what stays on the route.
    assert all(status[p] == "library" for p in validator.PILOT_IDS)
    assert status["M00-L01"] == status["M08-L08X"] == status["M15-L01"] == status["M16-L01"] == "library"
    assert status["M12-L03"] == status["M12-L05"] == status["M12-L09X"] == "library"
    assert status["M09-L07X"] == status["M11-L07X"] == status["M14-L10X"] == status["M13-L07X"] == "core"


def test_level_zero_is_a_single_launch_mission():
    route = validator.load_route_v2(SOURCE)
    zero = route["levels"][0]
    missions = [m for entry in zero["lessons"] for m in entry["missions"]]
    assert [m["kind"] for m in missions] == ["launch"]
    assert [lv["level"] for lv in route["levels"]] == list(range(8))


LESSON = "# {lid}\n\n## Разобранный пример\n\nТекст.\n\n## Самостоятельная задача A — {lid}-A\n\nТекст.\n"


@pytest.fixture
def small(tmp_path):
    """Two modules: M09-L01 on the route, M00-L01 and P01-P10 in the library."""
    curriculum = {"modules": [{"id": "M00", "lessons": [{"id": "M00-L01"}]},
                              {"id": "M09", "lessons": [{"id": "M09-L01"}]}]}
    (tmp_path / "curriculum.json").write_text(json.dumps(curriculum), encoding="utf-8")
    lesson = tmp_path / "course/lessons/M09-L01.md"
    lesson.parent.mkdir(parents=True)
    lesson.write_text(LESSON.format(lid="M09-L01"), encoding="utf-8")
    (tmp_path / "labs/M09").mkdir(parents=True)
    (tmp_path / "labs/M09/demo.py").write_text("print(1)\n", encoding="utf-8")
    route = """version: 2
session_model: {pending: R06}
levels:
- level: 0
  title: Старт
  project: {id: MP0, title: Запуск, missions: [MP0], path: labs/M09/demo.py}
  lessons:
    - lesson: M09-L01
      missions:
        - {id: M09-L01-M1, kind: entry, file: course/lessons/M09-L01.md, section: Разобранный пример}
        - {id: M09-L01-M3, kind: lab, file: course/lessons/M09-L01.md, section: Самостоятельная задача A, lab: labs/M09/demo.py}
    - {lesson: M09-L09X, planned: R08, missions: []}
library: [P01, P02, P03, P04, P05, P06, P07, P08, P09, P10, M00-L01]
"""
    (tmp_path / "course/route-v2.yaml").write_text(route, encoding="utf-8")
    return tmp_path


def edit(root, old, new):
    path = root / "course/route-v2.yaml"
    text = path.read_text(encoding="utf-8")
    assert old in text
    path.write_text(text.replace(old, new), encoding="utf-8")


def test_synthetic_route_is_valid(small):
    assert route_errors(small) == []


def test_lesson_without_status_or_with_two_is_reported(small):
    edit(small, ", M00-L01]", ", M09-L01]")
    errors = route_errors(small)
    assert "course/route-v2.yaml: M00-L01 has no status (core or library)" in errors
    assert "course/route-v2.yaml: M09-L01 is both core and library" in errors


def test_broken_links_are_reported(small):
    edit(small, "section: Разобранный пример", "section: Нет такого раздела")
    edit(small, "lab: labs/M09/demo.py}", "lab: labs/M09/none.py}")
    edit(small, "path: labs/M09/demo.py}", "path: projects/mini/none.md}")
    errors = route_errors(small)
    assert any("M09-L01-M1 section not found" in e for e in errors)
    assert any("M09-L01-M3 lab file missing" in e for e in errors)
    assert any("level 0 project file missing" in e for e in errors)


def test_planned_flag_must_be_removed_when_lesson_appears(small):
    curriculum = json.loads((small / "curriculum.json").read_text(encoding="utf-8"))
    curriculum["modules"][1]["lessons"].append({"id": "M09-L09X"})
    (small / "curriculum.json").write_text(json.dumps(curriculum), encoding="utf-8")
    errors = route_errors(small)
    assert any("M09-L09X is registered in curriculum.json, remove 'planned: R08'" in e for e in errors)
    assert any("M09-L09X has no status" in e for e in errors)


def test_mission_ids_are_unique_and_loggable(small):
    edit(small, "id: M09-L01-M3", "id: M09-L01-M1")
    edit(small, "missions: [MP0]", "missions: [mp-zero]")
    errors = route_errors(small)
    assert "course/route-v2.yaml: duplicate mission id M09-L01-M1" in errors
    assert any("project mission id 'mp-zero'" in e for e in errors)
