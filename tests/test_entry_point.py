"""R03: one entry point. The learner reads at most 300 words before the first mission."""
from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1]


def load_tool(name):
    spec = importlib.util.spec_from_file_location(name, SOURCE / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def words(markdown: str) -> int:
    """Visible words: link targets, table rules, markup and progress-map comments do not count."""
    text = re.sub(r"<!--.*?-->", " ", markdown, flags=re.S)
    text = re.sub(r"\]\([^)]*\)", "]", text)
    text = re.sub(r"[|#*`\[\]>_-]+", " ", text)
    return len([w for w in text.split() if re.search(r"\w", w)])


def test_start_page_fits_the_word_budget_and_leads_to_level_zero():
    start = (SOURCE / "course/START.md").read_text(encoding="utf-8")
    # The generated progress map (R05) is not reading: it is a status table.
    reading = re.sub(r"<!-- progress-map:start -->.*?<!-- progress-map:end -->", "", start, flags=re.S)
    assert words(reading) <= 300
    assert "/study" in start
    assert "lessons/M09-L07X.md" in start
    for level in range(8):
        assert re.search(rf"(?m)^\| {level} \|", start), f"level {level} missing from the map"


def test_readme_is_one_screen_and_old_text_is_kept():
    readme = (SOURCE / "README.md").read_text(encoding="utf-8")
    assert words(readme) <= 150
    assert "course/START.md" in readme and "MAINTAINING.md" in readme
    maintaining = (SOURCE / "MAINTAINING.md").read_text(encoding="utf-8")
    for heading in ("## Начать самостоятельно", "## Оригиналы и копии", "## Проверка материалов",
                    "## После первых десяти занятий", "## Что изменилось после аудита"):
        assert heading in maintaining
    assert "START.md" in (SOURCE / "course/README.md").read_text(encoding="utf-8")


def test_reader_opens_on_start_and_puts_the_rest_in_the_library():
    reader = load_tool("build_reader")
    registry = json.loads((SOURCE / "metadata/material-registry.json").read_text(encoding="utf-8"))
    order = reader.ordered_materials(registry, SOURCE)
    assert order[0] == "START"
    html, _ = reader.build(SOURCE)
    assert html.index('id="START"') < html.index('id="HOME"')
    navigation = html[html.index('id="book-navigation"'):html.index("</nav>", html.index('id="book-navigation"'))]
    assert navigation.index('data-key="START"') < navigation.index("Библиотека") < navigation.index('data-key="HOME"')
    assert "const START = byId.has('START') ? 'START' : 'HOME';" in reader.SCRIPT
