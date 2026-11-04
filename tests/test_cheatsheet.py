"""Lesson cards build a cheatsheet and Anki rows; the committed cheatsheet stays current."""
from __future__ import annotations
import csv
import importlib.util
import io
import json
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[1]


def load_tool(name):
    spec = importlib.util.spec_from_file_location(name, SOURCE / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cheatsheet = load_tool("build_cheatsheet")
linter = load_tool("lint_lessons")
CARD = ("## Карточка урока\n\n| Поле | Содержание |\n| --- | --- |\n| Правило | MSE = среднее e² |\n"
        "| Смысл величин | e — остаток |\n| Когда применять | Сравнение кандидатов |\n| Главная ошибка | Нет квадрата |\n\n")


def test_committed_cheatsheet_is_current_and_counts_cards():
    markdown, _ = cheatsheet.build(SOURCE)
    assert (SOURCE / "course/cheatsheet.md").read_text(encoding="utf-8") == markdown
    lessons = sorted((SOURCE / "course/lessons").glob("*.md"))
    with_card = sum("## Карточка урока" in path.read_text(encoding="utf-8") for path in lessons)
    assert markdown.count("\n### ") == with_card >= 5


def test_cards_become_cheatsheet_sections_and_two_anki_rows(tmp_path):
    curriculum = {"modules": [{"id": "M00", "title": "Модель", "lessons": [{"id": "M00-L03", "title": "Ошибка"},
                                                                             {"id": "M00-L04", "title": "Без карточки"}]}]}
    (tmp_path / "curriculum.json").write_text(json.dumps(curriculum), encoding="utf-8")
    (tmp_path / "course/lessons").mkdir(parents=True)
    (tmp_path / "course/lessons/M00-L03.md").write_text("# M00-L03\n\n" + CARD + "## Источники и углубление\n",
                                                        encoding="utf-8")
    (tmp_path / "course/lessons/M00-L04.md").write_text("# M00-L04\n\n## Источники и углубление\n", encoding="utf-8")
    markdown, rows = cheatsheet.build(tmp_path)
    assert "Карточек: 1 из 2 уроков" in markdown
    assert "### [M00-L03. Ошибка](lessons/M00-L03.md)" in markdown and "- **Правило.** MSE = среднее e²." in markdown
    records = [row for row in csv.reader(io.StringIO(rows)) if row and not row[0].startswith("#")]
    assert len(records) == 2 and records[1][1].endswith("Главная ошибка: нет квадрата")


def test_lesson_card_rows_and_position_are_linted(tmp_path):
    (tmp_path / "course/lessons").mkdir(parents=True)
    sections = ["Результат и входные знания", "Объяснение", "Разобранный пример", "Практика с поддержкой",
                "Самостоятельная задача A", "Подсказки по очереди", "Критерии и типичные ошибки",
                "Новая и отложенная проверка", "Источники и углубление"]
    body = "# M00-L01\n\n" + "".join(f"## {name}\n\nТекст.\n\n" for name in sections)
    path = tmp_path / "course/lessons/M00-L01.md"
    path.write_text(body.replace("## Источники и углубление", CARD + "## Источники и углубление"), encoding="utf-8")
    assert linter.lint(tmp_path)["errors"] == []
    path.write_text(body.replace("## Практика с поддержкой", CARD.replace("| Главная ошибка | Нет квадрата |\n", "")
                                 + "## Практика с поддержкой"), encoding="utf-8")
    errors = linter.lint(tmp_path)["errors"]
    assert any("lesson card needs the rows" in error for error in errors)
    assert any("lesson card must stand after the delayed check" in error for error in errors)
