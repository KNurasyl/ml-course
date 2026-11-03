"""The reading copy must keep support hidden until the learner opens it."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import re

import mistune


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("course_reader", ROOT / "tools/build_reader.py")
reader = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reader)


def render(raw: str, source: Path, title: str, path_ids: dict[Path, str]) -> str:
    renderer = reader.CourseRenderer(ROOT, source, "P01", path_ids)
    markdown = mistune.create_markdown(renderer=renderer)
    return reader.fold_hints(markdown(reader.without_repeated_title(raw, title)))


def test_canonical_heading_levels_keep_hints_closed_without_hiding_the_next_task():
    raw = """# Учебное занятие

## Самостоятельная работа

Сначала сохрани свой ответ.

## Ошибки и подсказки

Первая подсказка доступна после попытки.

### Дополнительная подсказка

Рассмотри другой вход.

## Проверка результата

Сравни объяснение с критериями.
"""
    html = render(raw, ROOT / "course/pilot/P01.md", "Учебное занятие", {})
    folded = re.search(r'(<details\b[^>]*class="hints"[^>]*>)(.*?)</details>', html, re.DOTALL)
    assert folded is not None, "The standard ## hints section must become a collapsed disclosure"
    assert " open" not in folded.group(1)
    assert "Первая подсказка" in folded.group(2)
    assert re.search(r"<h3\b[^>]*>Дополнительная подсказка</h3>", folded.group(2))
    assert "Сначала сохрани свой ответ" not in folded.group(2)
    assert "Сравни объяснение с критериями" not in folded.group(2)
    assert re.search(r"<h2\b[^>]*>Проверка результата</h2>", html)
    assert "Учебное занятие" not in html, "The document title is supplied by the reader header"


def test_real_p01_lesson_has_collapsed_hints_and_visible_assessment():
    registry = json.loads((ROOT / "metadata/material-registry.json").read_text(encoding="utf-8"))
    item = registry["P01"]
    source = ROOT / item["path"]
    paths = {(ROOT / entry["path"]).resolve(): key for key, entry in registry.items()}
    html = render(source.read_text(encoding="utf-8"), source, item["title"], paths)
    folded = re.search(r'(<details\b[^>]*class="hints"[^>]*>)(.*?)</details>', html, re.DOTALL)
    assert folded is not None
    assert " open" not in folded.group(1)
    assert "Подсказка 1" in folded.group(2)
    assert "Проверка и следующий шаг" not in folded.group(2)
    assert "Проверка и следующий шаг" in html[folded.end():]
