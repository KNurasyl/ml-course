"""Exercise failure modes that would break an independent offline learner."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import shutil

import pytest


SOURCE = Path(__file__).resolve().parents[1]


def load_tool(name):
    spec = importlib.util.spec_from_file_location(name, SOURCE / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = load_tool("build_manifest")
validator = load_tool("validate_course")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


@pytest.fixture
def local_course(tmp_path):
    shutil.copyfile(SOURCE / "curriculum.json", tmp_path / "curriculum.json")
    (tmp_path / "tools").mkdir()
    shutil.copyfile(SOURCE / "tools/build_reader.py", tmp_path / "tools/build_reader.py")
    mapping = {
        "HOME": "course/README.md", "DIAGNOSTIC": "course/diagnostic.md",
        "GLOSSARY": "course/glossary.md", "PROJECT": "course/project.md",
        "MENTOR": "course/self-study.md", "MENTOR-KEYS": "solutions/pilot/answer-key.md",
        "PROGRESS": "progress/progress.md",
        **{f"M{i:02}": f"course/modules/M{i:02}.md" for i in range(17)},
        **{f"P{i:02}": f"course/pilot/P{i:02}.md" for i in range(1, 11)},
    }
    registry = {}
    for key, path in mapping.items():
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"# {key}\n\nЗадание {key}-HW01. Объясни полученный результат.\n", encoding="utf-8")
        registry[key] = {"id": key, "title": key, "path": path, "version": "1.0.0", "status": "ready"}
    write_json(tmp_path / "metadata/material-registry.json", registry)
    inventory = []
    for i in range(1, 11):
        key = f"P{i:02}"
        for folder in ("notebooks/pilot", "solutions/pilot"):
            write_json(tmp_path / folder / f"{key}.ipynb", {"cells": [
                {"cell_type": "markdown", "source": [f"Работа {key}.\n"]},
                {"cell_type": "code", "source": ["assert 1 + 1 == 2"], "outputs": []},
            ]})
        inventory.append({"id": key, "notebook": f"notebooks/pilot/{key}.ipynb",
                          "solution": f"solutions/pilot/{key}.ipynb", "source": "https://example.org/optional"})
    write_json(tmp_path / "metadata/pilot-lab-inventory.json", inventory)
    (tmp_path / "README.md").write_text("[Начать](START_HERE.html#home)", encoding="utf-8")
    (tmp_path / "AGENTS.md").write_text("Читать локальный журнал.", encoding="utf-8")
    fingerprint = validator.reader_fingerprint(tmp_path)
    (tmp_path / "START_HERE.html").write_text(
        '<!doctype html><html lang="ru"><head>'
        f'<meta name="course-source-sha256" content="{fingerprint}">'
        '</head><body><main id="home">'
        '<a href="course/diagnostic.md">Диагностика</a></main></body></html>', encoding="utf-8")
    write_json(tmp_path / "course-manifest.json", builder.build_manifest(tmp_path))
    return tmp_path


def test_local_route_builds_without_remote_registry_or_authoring_cache(local_course):
    assert not (local_course / ".authoring").exists()
    assert not (local_course / "metadata/page-registry.json").exists()
    manifest = builder.build_manifest(local_course)
    assert manifest["delivery"]["status"] == "local_ready"
    assert manifest["full_course_ready"] is False
    assert validator.validate(local_course, require_local=True)["errors"] == []


def test_missing_lesson_is_not_a_ready_delivery(local_course):
    (local_course / "course/pilot/P03.md").unlink()
    manifest = builder.build_manifest(local_course)
    assert manifest["delivery"]["status"] == "incomplete"
    assert "P03" in manifest["delivery"]["missing_materials"]
    assert any("P03.md" in error for error in validator.validate(local_course, require_local=True)["errors"])


def test_unexplained_prerequisite_is_detected(local_course):
    path = local_course / "curriculum.json"
    curriculum = json.loads(path.read_text(encoding="utf-8"))
    curriculum["modules"][0]["lessons"][0]["prerequisites"] = ["M99-L99"]
    write_json(path, curriculum)
    assert "Unknown lesson dependency M99-L99" in validator.validate(local_course)["errors"]


def test_reintroducing_remote_lesson_link_breaks_offline_route(local_course):
    target = local_course / "course/pilot/P04.md"
    with target.open("a", encoding="utf-8") as stream:
        stream.write("\n[Нужная теория](https://chatgpt.com/space/page_old)\n")
    errors = validator.validate(local_course, require_local=True)["errors"]
    assert any("Remote Page dependency in course\\pilot\\P04.md" in error or
               "Remote Page dependency in course/pilot/P04.md" in error for error in errors)


def test_broken_assignment_link_is_detected(local_course):
    target = local_course / "course/diagnostic.md"
    with target.open("a", encoding="utf-8") as stream:
        stream.write("\n[Задание](missing-exercise.md)\n")
    assert any("Broken local link" in error and "missing-exercise.md" in error
               for error in validator.validate(local_course, require_local=True)["errors"])


def test_missing_reader_anchor_is_detected(local_course):
    (local_course / "README.md").write_text("[Начать](START_HERE.html#missing)", encoding="utf-8")
    assert any("Missing HTML anchor" in error
               for error in validator.validate(local_course, require_local=True)["errors"])


def test_required_online_javascript_breaks_offline_reader(local_course):
    target = local_course / "START_HERE.html"
    with target.open("a", encoding="utf-8") as stream:
        stream.write('<script src="https://example.invalid/required.js"></script>')
    assert any("Online asset in offline reader" in error
               for error in validator.validate(local_course, require_local=True)["errors"])


def test_reader_cannot_silently_depend_on_file_outside_repository(local_course):
    target = local_course / "course/diagnostic.md"
    with target.open("a", encoding="utf-8") as stream:
        stream.write("\n[Теория](../../outside.md)\n")
    assert any("Link leaves repository" in error
               for error in validator.validate(local_course, require_local=True)["errors"])


def test_changed_progress_requires_reader_refresh(local_course):
    with (local_course / "progress/progress.md").open("a", encoding="utf-8") as stream:
        stream.write("\nЗаписана самостоятельная попытка P01.\n")
    assert "Offline reader is stale: rebuild with python tools/build_reader.py" in \
        validator.validate(local_course, require_local=True)["errors"]
    assert builder.build_manifest(local_course)["delivery"]["status"] == "incomplete"


def test_registering_full_route_requires_remaining_lesson_files(local_course):
    path = local_course / "metadata/material-registry.json"
    registry = json.loads(path.read_text(encoding="utf-8"))
    source = local_course / "course/lessons/M00-L01.md"
    source.parent.mkdir(parents=True)
    source.write_text("# Первый урок", encoding="utf-8")
    registry["M00-L01"] = {"id":"M00-L01","title":"Первый урок","path":"course/lessons/M00-L01.md","version":"1.0.0","status":"ready"}
    write_json(path, registry)
    manifest = builder.build_manifest(local_course)
    assert manifest["delivery"]["status"] == "incomplete"
    assert "M00-L02" in manifest["delivery"]["missing_materials"]
    assert manifest["full_materials_ready"] is False
