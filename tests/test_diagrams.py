"""Mermaid schemas render offline from a verified local library and follow Mayer's limits."""
from __future__ import annotations
from hashlib import sha256
import importlib.util
from pathlib import Path
import re

import mistune


SOURCE = Path(__file__).resolve().parents[1]
VENDOR = SOURCE / "tools/vendor/mermaid"


def load_tool(name):
    spec = importlib.util.spec_from_file_location(name, SOURCE / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reader = load_tool("build_reader")
linter = load_tool("lint_lessons")
GOOD = """```mermaid
flowchart LR
    A["Вход"] -->|"fit"| B["Модель"]
    B --> C["Ответ"]
    classDef main stroke-width:2px
    class A,B main
```

Подпись: главный путь выделен.
"""


def test_vendored_mermaid_matches_recorded_hash_and_license():
    recorded = re.search(r"SHA-256 \| `([0-9a-f]{64})`", (VENDOR / "README.md").read_text(encoding="utf-8")).group(1)
    assert sha256((VENDOR / "mermaid.min.js").read_bytes()).hexdigest() == recorded
    assert (VENDOR / "LICENSE").read_text(encoding="utf-8").startswith("The MIT License")


def test_reader_loads_the_local_library_lazily():
    script = (SOURCE / "tools/reader.js").read_text(encoding="utf-8")
    assert "script.src='tools/vendor/mermaid/mermaid.min.js'" in script and "securityLevel:'strict'" in script
    assert "IntersectionObserver" in script and "cdn" not in script.lower()


def test_mermaid_block_becomes_escaped_diagram_source(tmp_path):
    source = tmp_path / "lesson.md"
    renderer = reader.CourseRenderer(tmp_path, source, "M00-L01", {})
    html = mistune.create_markdown(renderer=renderer)('```mermaid\nflowchart LR\n  A["a < b"] --> B\n```\n\n```python\nx=1\n```\n')
    assert ('<figure class="diagram"><pre class="mermaid" data-theme="light">'
            'flowchart LR\n  A[&quot;a &lt; b&quot;] --&gt; B\n</pre></figure>') in html
    assert '<code class="language-python">' in html


def lesson_with_schema(schema):
    sections = ["Результат и входные знания", "Объяснение", "Разобранный пример", "Схема", "Практика с поддержкой",
                "Самостоятельная задача A", "Подсказки по очереди", "Критерии и типичные ошибки",
                "Новая и отложенная проверка", "Источники и углубление"]
    return "# M05-L04\n\n" + "".join(f"## {name}\n\n{schema if name == 'Схема' else 'Текст.'}\n\n" for name in sections)


def test_schema_rules_blocks_main_path_caption_and_position(tmp_path):
    folder = tmp_path / "course/lessons"
    folder.mkdir(parents=True)
    write = lambda lesson_id, text: (folder / f"{lesson_id}.md").write_text(text, encoding="utf-8")
    write("M05-L01", lesson_with_schema(GOOD))
    assert linter.lint(tmp_path)["errors"] == []
    many = GOOD.replace('B --> C["Ответ"]', " --> ".join(f"N{i}" for i in range(9)))
    write("M05-L02", lesson_with_schema(many))
    write("M05-L03", lesson_with_schema(GOOD.replace("    class A,B main\n", "")))
    write("M05-L04", lesson_with_schema(GOOD.replace("Подпись: главный путь выделен.\n", "")))
    section = f"## Схема\n\n{GOOD}\n\n"
    moved = lesson_with_schema(GOOD).replace(section, "").replace("## Самостоятельная задача A", section + "## Самостоятельная задача A")
    write("M05-L05", moved)
    errors = linter.lint(tmp_path)["errors"]
    assert "M05-L02: schema has more than 9 blocks" in errors
    assert "M05-L03: schema must highlight the main path with classDef main" in errors
    assert "M05-L04: schema needs a caption next to the diagram" in errors
    assert "M05-L05: schema must stand after the worked example and before practice" in errors


def test_every_lesson_keeps_at_least_one_figure_and_course_schemas_pass():
    for path in sorted((SOURCE / "course/lessons").glob("*.md")):
        assert "![" in path.read_text(encoding="utf-8"), path.stem
    assert linter.lint(SOURCE)["errors"] == []
