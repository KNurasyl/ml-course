"""D6: Python examples are highlighted at build time; the visible code, every assert included, is unchanged."""
from __future__ import annotations

import importlib.util
import json
import re
from html import unescape
from html.parser import HTMLParser
from pathlib import Path

import mistune

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("reader_d6", ROOT / "tools/build_reader.py")
reader = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reader)
FENCE = re.compile(r"^```([^\n`]*)\n(.*?)^```", re.DOTALL | re.MULTILINE)


def course_blocks() -> list[tuple[str, str, str]]:
    registry = json.loads((ROOT / "metadata/material-registry.json").read_text(encoding="utf-8"))
    blocks = []
    for relative in sorted({item["path"] for item in registry.values()}):
        for info, code in FENCE.findall((ROOT / relative).read_text(encoding="utf-8")):
            blocks.append((relative, info.strip().split()[0] if info.strip() else "", code))
    return blocks


def visible_text(html: str) -> str:
    return unescape(re.sub(r"<[^>]+>", "", html))


class Tags(HTMLParser):
    def __init__(self):
        super().__init__()
        self.stack, self.problems, self.spans = [], [], 0

    def handle_starttag(self, tag, attrs):
        if tag == "span":
            self.spans += 1
            if "span" in self.stack or dict(attrs).get("class") not in {"k", "s", "n", "c"}:
                self.problems.append((tag, attrs))
        self.stack.append(tag)

    def handle_endtag(self, tag):
        if not self.stack or self.stack.pop() != tag:
            self.problems.append(("unbalanced", tag))


def test_every_python_example_is_highlighted_and_keeps_its_text():
    renderer = reader.CourseRenderer(ROOT, ROOT / "course/lessons/M00-L01.md", "M00-L01", {})
    python = [(relative, code) for relative, language, code in course_blocks() if language == "python"]
    assert len(python) == 143  # +1: M09-L09X (R08)
    asserts = 0
    for relative, code in python:
        html = renderer.block_code(code, "python")
        assert html.startswith('<pre><code class="language-python">') and html.endswith("</code></pre>\n"), relative
        inner = html[len('<pre><code class="language-python">'):-len("</code></pre>\n")]
        assert visible_text(inner) == code, relative
        parser = Tags()
        parser.feed(html)
        assert not parser.problems and not parser.stack and parser.spans, relative
        asserts += visible_text(inner).count("assert")
    assert asserts == sum(code.count("assert") for _, code in python) == 273  # +7: M09-L09X (R08)


def test_token_classes():
    code = ('import numpy as np  # note "q"\nx = 1e-5 + 0x1F + 1_000 + .5 + 3j + 2.\n'
            's = f"a{x}#b" + r\'\\d+\' + """multi\nline"""\nassert x2 is not None, \'fail: #1\'\n')
    html = reader.highlight_python(code)
    assert visible_text(html) == code
    spans = re.findall(r'<span class="(\w)">(.*?)</span>', html, re.DOTALL)
    assert [text for kind, text in spans if kind == "k"] == ["import", "as", "assert", "is", "not", "None"]
    assert [text for kind, text in spans if kind == "n"] == ["1e-5", "0x1F", "1_000", ".5", "3j", "2."]
    assert [unescape(text) for kind, text in spans if kind == "s"] == ['f"a{x}#b"', "r'\\d+'", '"""multi\nline"""', "'fail: #1'"]
    assert [unescape(text) for kind, text in spans if kind == "c"] == ['# note "q"']
    # Names with digits and names that start like a string prefix stay plain.
    assert "x2" not in "".join(text for _, text in spans) and "numpy" not in "".join(text for _, text in spans)


def test_a_broken_fragment_still_round_trips_and_other_languages_are_untouched():
    broken = "x = 'unterminated\nprint(\"ok\")  # ' not a string\n"
    assert visible_text(reader.highlight_python(broken)) == broken
    renderer = reader.CourseRenderer(ROOT, ROOT / "course/lessons/M00-L01.md", "M00-L01", {})
    plain = mistune.create_markdown()("```powershell\n$x = 1  # c\n```\n")
    assert renderer.block_code("$x = 1  # c\n", "powershell") == plain and "<span" not in plain


def test_token_colours_come_from_theme_tokens():
    css = (ROOT / "tools/reader.css").read_text(encoding="utf-8")
    for kind, token in (("k", "act"), ("s", "hint"), ("n", "accent"), ("c", "muted")):
        assert f".body-content pre code .{kind}{{color:var(--{token})}}" in css
