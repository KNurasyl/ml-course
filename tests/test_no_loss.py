"""The no-loss guard must notice removed content and tolerate allowed renames."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path

import pytest


SOURCE = Path(__file__).resolve().parents[1]
BASELINE = SOURCE / "quality/snapshots/baseline-1.1.0.json"


def load_tool(name):
    spec = importlib.util.spec_from_file_location(name, SOURCE / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


guard = load_tool("check_no_loss")

LESSON = """# M00-L01. Модель как функция

## Объяснение

Модель сопоставляет входу x прогноз y. Для w=2 и b=1 прогноз при x=3 равен 7.

## Самостоятельная задача M00-L01-A01

Вычисли прогноз для x=4 и объясни каждый шаг.

## Повтор B и отложенная проверка C

**Вариант B:** Для w=3, b=0 найди прогноз при x=2.

**Вариант C через 3–7 дней:** Объясни, почему b сдвигает все прогнозы одинаково.
"""


@pytest.fixture
def small_course(tmp_path):
    curriculum = {"modules": [{"id": "M00", "homework": {"id": "M00-HW01"},
                               "lessons": [{"id": "M00-L01"}, {"id": "M00-L02"}]}]}
    files = {
        "curriculum.json": json.dumps(curriculum),
        "course/lessons/M00-L01.md": LESSON,
        "course/lessons/M00-L02.md": "# M00-L02\n\n## Самостоятельная задача A\n\nТекст.\n",
        "course/pilot/P01.md": "# P01\n\n## Самостоятельная работа P01-HW01\n\nЗадание.\n",
        "solutions/modules/M00.md": "# Решения\n\n## M00-L01-A\n\nОтвет 9.\n",
        "labs/M00/demo.py": "print(7)\n",
        "labs/M00/output/result.txt": "generated and ignored\n",
        "notebooks/pilot/P01.ipynb": json.dumps({"cells": [
            {"cell_type": "markdown", "source": ["Работа P01."]},
            {"cell_type": "code", "source": ["assert 2 * 3 + 1 == 7"], "outputs": []}]}),
    }
    for relative, text in files.items():
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    return tmp_path


def test_snapshot_counts_course_structure(small_course):
    state = guard.snapshot(small_course, include_tests=False)
    assert state["ids"]["lessons"] == ["M00-L01", "M00-L02"]
    assert state["ids"]["pilot_sessions"] == ["P01"]
    assert state["counts"]["notebooks"] == 1
    assert "labs/M00/output/result.txt" not in state["trees"]["labs"]
    lesson = state["files"]["course/lessons/M00-L01.md"]
    assert (lesson["tasks_a"], lesson["tasks_b"], lesson["tasks_c"]) == (1, 1, 1)
    assert guard.compare(state, guard.snapshot(small_course, include_tests=False))["status"] == "no_loss"


def test_removed_words_files_and_lessons_are_losses(small_course):
    before = guard.snapshot(small_course, include_tests=False)
    lesson = small_course / "course/lessons/M00-L01.md"
    lesson.write_text(LESSON.replace(" и объясни каждый шаг", ""), encoding="utf-8")
    (small_course / "labs/M00/demo.py").unlink()
    curriculum = json.loads((small_course / "curriculum.json").read_text(encoding="utf-8"))
    curriculum["modules"][0]["lessons"].pop()
    (small_course / "curriculum.json").write_text(json.dumps(curriculum), encoding="utf-8")
    result = guard.compare(before, guard.snapshot(small_course, include_tests=False))
    assert result["status"] == "lost_content"
    assert any("course/lessons/M00-L01.md: body_words" in loss for loss in result["losses"])
    assert any("labs/M00/demo.py" in loss for loss in result["losses"])
    assert any("ids lessons: removed M00-L02" in loss for loss in result["losses"])


def test_removed_retry_variant_is_a_loss(small_course):
    before = guard.snapshot(small_course, include_tests=False)
    lesson = small_course / "course/lessons/M00-L01.md"
    lesson.write_text(LESSON.replace("**Вариант C через 3–7 дней:**", "Ещё:"), encoding="utf-8")
    losses = guard.compare(before, guard.snapshot(small_course, include_tests=False))["losses"]
    assert any("tasks_c 1 -> 0" in loss for loss in losses)


def test_canonical_heading_rename_is_only_a_warning(small_course):
    before = guard.snapshot(small_course, include_tests=False)
    lesson = small_course / "course/lessons/M00-L01.md"
    lesson.write_text(LESSON.replace("## Повтор B и отложенная проверка C", "## Новая и отложенная проверка"),
                      encoding="utf-8")
    result = guard.compare(before, guard.snapshot(small_course, include_tests=False))
    assert result["losses"] == []
    assert len(result["warnings"]) == 1 and result["warnings"][0].startswith("course/lessons/M00-L01.md: words ")


def test_cli_compare_exits_nonzero_after_loss(small_course, tmp_path, monkeypatch):
    monkeypatch.setattr(guard, "ROOT", small_course)
    before = tmp_path / "before.json"
    guard.write_json(before, guard.snapshot(small_course, include_tests=False))
    assert guard.main(["compare", "--before", str(before), "--no-tests"]) == 0
    (small_course / "solutions/modules/M00.md").unlink()
    assert guard.main(["compare", "--before", str(before), "--no-tests"]) == 1


def test_current_course_keeps_everything_from_release_baseline():
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    assert baseline["course_version"] == "1.1.0" and baseline["counts"]["lessons"] == 103
    result = guard.compare(baseline, guard.snapshot(SOURCE, include_tests=False))
    assert result["losses"] == [], result["losses"]
    assert result["skipped"] == ["tests: not collected in the later snapshot"]


def test_only_task_headings_count_as_task_a():
    metrics = guard.markdown_metrics("## Самостоятельная задача A — M00-L01-A01\n\nТекст.\n\n"
                                     "## Критерии самостоятельной проверки\n\nТекст.\n\n"
                                     "## Основная самостоятельная работа M16-HW01\n\nТекст.\n")
    assert metrics["tasks_a"] == 2
