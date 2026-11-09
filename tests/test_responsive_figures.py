"""D3: drawn figures take theme tokens and the reader inlines them; matplotlib figures stay images until D4."""
from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import mistune
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("reader_d3", ROOT / "tools/build_reader.py")
reader = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reader)
MAPPING = json.loads((ROOT / "metadata/responsive-figures.json").read_text(encoding="utf-8"))
PAINT = re.compile(r'\s(fill|stroke|style)="([^"]*)"')
TOKEN_USE = re.compile(r"(fill|stroke):var\(--([\w-]+),(#[0-9a-f]{6})\)")


def light_tokens() -> dict[str, str]:
    css = (ROOT / "tools/reader.css").read_text(encoding="utf-8")
    block = re.search(r':root\[data-theme="light"\]\{color-scheme:light;(.*?)\}', css, re.DOTALL).group(1)
    return dict(re.findall(r"--([\w-]+):(#[0-9a-fA-F]{6})", block))


def inline_files() -> list[str]:
    return [entry[layout] for entry in MAPPING.values() if entry.get("inline") for layout in ("desktop", "mobile")]


def test_drawn_figures_take_colours_only_from_theme_tokens():
    light = light_tokens()
    files = inline_files()
    assert len(files) == 136
    for relative in files:
        svg = (ROOT / relative).read_text(encoding="utf-8")
        stem = Path(relative).stem
        assert 'fill="white"' not in svg and "<rect width=" not in svg, relative
        for attribute, value in PAINT.findall(svg):
            if attribute in ("fill", "stroke"):
                assert value == "none", (relative, attribute, value)
                continue
            # Only token uses, each with the light value of tools/reader.css as the standalone fallback.
            assert TOKEN_USE.sub("", value).strip(";") == "", (relative, value)
            for _, token, fallback in TOKEN_USE.findall(value):
                assert light[token] == fallback, (relative, token, fallback)
        ids = re.findall(r'\sid="([^"]+)"', svg)
        assert ids and all(identifier.startswith(stem + "-") for identifier in ids), relative
        references = re.findall(r"url\(#([^)]+)\)", svg) + re.search(r'aria-labelledby="([^"]+)"', svg).group(1).split()
        assert set(references) <= set(ids), relative


def test_only_drawn_figures_are_inline_and_matplotlib_figures_stay_images():
    drawn = set(inline_files())
    # Dark copies of the matplotlib figures (D4, <name>-dark.svg) are checked in tests/test_dark_figures.py.
    responsive = {path.relative_to(ROOT).as_posix() for path in (ROOT / "course/assets/responsive").glob("*.svg")
                  if not path.stem.endswith("-dark")}
    made_by_matplotlib = {relative for relative in responsive if "matplotlib" in (ROOT / relative).read_text(encoding="utf-8")[:3000]}
    assert len(responsive) == 208 and len(made_by_matplotlib) == 72
    assert drawn == responsive - made_by_matplotlib
    mapped = [entry[layout] for entry in MAPPING.values() for layout in ("desktop", "mobile")]
    assert sorted(mapped) == sorted(responsive)


def figure_course(tmp_path: Path, mobile_svg: str) -> reader.CourseRenderer:
    (tmp_path / "course/assets/responsive").mkdir(parents=True)
    (tmp_path / "course/lessons").mkdir(parents=True)
    (tmp_path / "metadata").mkdir()
    (tmp_path / "course/assets/x.svg").write_text("<svg/>", encoding="utf-8")
    desktop = ('<svg xmlns="http://www.w3.org/2000/svg" width="760" height="40" aria-labelledby="x-desktop-title">'
               '<title id="x-desktop-title">Схема</title><text style="fill:var(--ink,#1d2430)">Вход</text>'
               '<path d="M0,0 H9" fill="none" style="stroke:var(--act,#0b7478)"/></svg>\n')
    (tmp_path / "course/assets/responsive/x-desktop.svg").write_text(desktop, encoding="utf-8")
    (tmp_path / "course/assets/responsive/x-mobile.svg").write_text(mobile_svg, encoding="utf-8")
    mapping = {"course/assets/x.svg": {"desktop": "course/assets/responsive/x-desktop.svg",
                                       "mobile": "course/assets/responsive/x-mobile.svg", "caption": "Подпись", "inline": True}}
    (tmp_path / "metadata/responsive-figures.json").write_text(json.dumps(mapping, ensure_ascii=False), encoding="utf-8")
    return reader.CourseRenderer(tmp_path, tmp_path / "course/lessons/M00-L01.md", "M00-L01", {})


def test_reader_inlines_both_layouts_with_tokens_only(tmp_path):
    mobile = ('<svg xmlns="http://www.w3.org/2000/svg" width="360" height="40" aria-labelledby="x-mobile-title">'
              '<title id="x-mobile-title">Схема</title><rect style="fill:var(--accent-soft,#e4ecfb);stroke:var(--line,#d9d6cd)"/></svg>\n')
    html = mistune.create_markdown(renderer=figure_course(tmp_path, mobile))("![Схема входа](../assets/x.svg)\n")
    assert html.count('<span class="figure-variant figure-desktop"><svg ') == 1
    assert html.count('<span class="figure-variant figure-mobile"><svg ') == 1
    assert 'class="figure-expand figure-inline"' in html and 'data-title="Схема входа"' in html
    assert "style=\"fill:var(--ink)\"" in html and "fill:var(--accent-soft);stroke:var(--line)" in html
    assert not re.search(r"var\(--[\w-]+,", html) and "<picture>" not in html and "<img" not in html
    assert '<span class="figure-hint">Подпись · Открыть крупнее ↗</span>' in html


def test_reader_rejects_an_inline_figure_with_its_own_colour(tmp_path):
    mobile = '<svg xmlns="http://www.w3.org/2000/svg" width="360" height="40"><rect fill="#ffffff"/></svg>\n'
    with pytest.raises(ValueError, match="only from theme tokens"):
        mistune.create_markdown(renderer=figure_course(tmp_path, mobile))("![Схема](../assets/x.svg)\n")


def test_reader_picks_the_layout_by_column_and_keeps_figure_text_out_of_search():
    css = (ROOT / "tools/reader.css").read_text(encoding="utf-8")
    assert ".figure-inline{container:figure/inline-size}" in css
    assert "@container figure (max-width:599.98px){.figure-inline .figure-desktop{display:none}.figure-inline .figure-mobile{display:block}}" in css
    script = (ROOT / "tools/reader.js").read_text(encoding="utf-8")
    assert "closest('script, style, button, svg')" in script
    assert "source.querySelectorAll('svg').forEach(svg => svg.remove())" in script
    assert "figure-large-svg" in script and "id+'-zoom'" in script
