"""Every lesson keeps the canonical 1.2.0 template; gaps must be documented."""
from __future__ import annotations
import importlib.util
from pathlib import Path

import pytest


SOURCE = Path(__file__).resolve().parents[1]


def load_tool(name):
    spec = importlib.util.spec_from_file_location(name, SOURCE / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


linter = load_tool("lint_lessons")
SECTIONS = ["Результат и входные знания", "Объяснение", "Разобранный пример", "Практика с поддержкой",
            "Самостоятельная задача A — M00-L01-A01", "Подсказки по очереди", "Критерии и типичные ошибки",
            "Новая и отложенная проверка", "Источники и углубление"]


def lesson(sections, prefix=""):
    return "# M00-L01\n\n" + prefix + "".join(f"## {name}\n\nТекст.\n\n" for name in sections)


@pytest.fixture
def course(tmp_path):
    (tmp_path / "course/lessons").mkdir(parents=True)
    (tmp_path / "metadata").mkdir()
    return tmp_path


def write(course, text, lesson_id="M00-L01"):
    (course / f"course/lessons/{lesson_id}.md").write_text(text, encoding="utf-8")


def test_course_lessons_follow_template():
    result = linter.lint(SOURCE)
    assert result["lessons"] == 103 and result["errors"] == []


def test_canonical_lesson_passes_and_result_paragraph_counts(course):
    write(course, lesson(SECTIONS))
    write(course, lesson(SECTIONS[1:], prefix="**Результат:** вычислить ответ.\n\n"), "M00-L02")
    assert linter.lint(course)["errors"] == []


def test_missing_old_named_and_misordered_sections_fail(course):
    write(course, lesson([s for s in SECTIONS if s != "Объяснение"]))
    write(course, lesson([s.replace("Подсказки по очереди", "Подсказки") for s in SECTIONS]), "M00-L02")
    write(course, lesson([SECTIONS[0], SECTIONS[2], SECTIONS[1], *SECTIONS[3:]]), "M00-L03")
    write(course, lesson([s.replace("Разобранный пример", "Разобранный пример без модели") for s in SECTIONS]), "M00-L04")
    errors = linter.lint(course)["errors"]
    assert "M00-L01: missing section 'Объяснение'" in errors
    assert "M00-L02: missing section 'Подсказки по очереди'" in errors
    assert any(e.startswith("M00-L03: canonical sections are out of order") for e in errors)
    assert any(e.startswith("M00-L04: heading 'Разобранный пример без модели' looks canonical") for e in errors)


def test_combined_heading_and_criteria_before_hints_are_accepted(course):
    sections = SECTIONS[:5] + ["Критерии и типичные ошибки", "Подсказки по очереди; новая и отложенная проверка",
                               "Источники и углубление"]
    write(course, lesson(sections))
    assert linter.lint(course)["errors"] == []


def test_exceptions_need_reason_and_become_stale(course):
    write(course, lesson([s for s in SECTIONS if s != "Подсказки по очереди"]))
    write(course, lesson(SECTIONS), "M00-L02")
    (course / linter.EXCEPTIONS).write_text(
        "version: 1\nlessons:\n  M00-L01:\n    missing: [Подсказки по очереди]\n    reason: нет подсказок\n"
        "  M00-L02:\n    missing: [Подсказки по очереди]\n  M99-L01:\n    reason: урока нет\n", encoding="utf-8")
    errors = linter.lint(course)["errors"]
    assert not any(e.startswith("M00-L01") for e in errors)
    assert "M00-L02: exception needs a reason" in errors
    assert "M00-L02: exception for 'Подсказки по очереди' is stale, the section exists" in errors
    assert "M99-L01: exception for a lesson that does not exist" in errors


def test_lesson_with_code_needs_predict_line_in_practice(course):
    code = "```python\nprint(1)\n```\n\n"
    write(course, lesson(SECTIONS).replace("## Практика с поддержкой\n\nТекст.", "## Практика с поддержкой\n\n" + code))
    errors = linter.lint(course)["errors"]
    assert "M00-L01: practice of a lesson with code lacks the '**Предскажи до запуска.**' line" in errors
    write(course, lesson(SECTIONS).replace("## Практика с поддержкой\n\nТекст.",
                                           "## Практика с поддержкой\n\n**Предскажи до запуска.** Запиши.\n\n" + code))
    assert linter.lint(course)["errors"] == []
