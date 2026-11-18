"""D4: every image figure of the reader has a dark copy in the 6.0 palette; the light files stay unchanged."""
from __future__ import annotations

import importlib.util
import json
import re
from hashlib import sha256
from pathlib import Path

import mistune
import pytest

ROOT = Path(__file__).resolve().parents[1]
FIGURES = json.loads((ROOT / "metadata/dark-figures.json").read_text(encoding="utf-8"))["figures"]
MAPPING = json.loads((ROOT / "metadata/responsive-figures.json").read_text(encoding="utf-8"))


def load_tool(name: str):
    spec = importlib.util.spec_from_file_location(f"d4_{name}", ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reader = load_tool("build_reader")
dark_figures = load_tool("build_dark_figures")


def dark_values() -> set[str]:
    css = (ROOT / "tools/reader.css").read_text(encoding="utf-8")
    block = re.search(r":root\{color-scheme:dark;(.*?)\}", css, re.DOTALL).group(1)
    return {value.lower() for value in re.findall(r"--[\w-]+:(#[0-9a-fA-F]{6})", block)}


def test_every_image_figure_has_a_dark_copy_and_the_light_file_is_unchanged():
    shown = {entry[layout] for entry in MAPPING.values() if not entry.get("inline") for layout in ("desktop", "mobile")}
    shown |= {path.as_posix() for path in (Path("course/assets") / p.name for p in dark_figures.hand_drawn())}
    assert len(shown) == 87 and set(FIGURES) == shown  # 86 + M09-L09X (R08)
    for light, entry in FIGURES.items():
        assert entry["dark"] == light.replace(".svg", "-dark.svg") and (ROOT / entry["dark"]).is_file(), light
        assert sha256((ROOT / light).read_bytes()).hexdigest() == entry["light_sha256"], f"light figure changed: {light}"


def test_dark_copies_use_only_the_dark_palette():
    allowed = dark_values()
    for entry in FIGURES.values():
        svg = re.sub(r"<image\b[^>]*>", "", (ROOT / entry["dark"]).read_text(encoding="utf-8"))
        colours = {colour.lower() for colour in re.findall(r"#[0-9a-fA-F]{6}\b", svg)}
        assert colours and colours <= allowed, (entry["dark"], sorted(colours - allowed))
        assert not re.search(r'(?:fill|stroke)\s*[:=]\s*"?\s*(?:white|black)\b', svg), entry["dark"]


def test_generators_skip_dark_copies_of_figures_the_reader_inlines():
    inline = {key for key, entry in MAPPING.items() if entry.get("inline")}
    for name, metadata in (("build_foundation_figures", "foundation-figures.json"), ("build_quantitative_figures", "quantitative-figures.json")):
        source = (ROOT / "tools" / f"{name}.py").read_text(encoding="utf-8")
        declared = set(re.findall(r"'([\w-]+)'", re.search(r"DRAWN_INLINE=\{(.*?)\}", source).group(1)))
        keys = json.loads((ROOT / "metadata" / metadata).read_text(encoding="utf-8"))
        assert declared == {Path(key).stem for key in keys if key in inline}, name


def test_hand_drawn_dark_copy_maps_roles_and_stops_on_an_unknown_colour():
    tokens = dark_figures.dark_tokens()
    light = ('<svg xmlns="http://www.w3.org/2000/svg"><rect fill="white"/><rect fill="#245bba"/>'
             '<text fill="white">1</text><rect fill="#9fb8e3"/><line stroke="#c9d1d9"/><rect fill="#c9d1d9"/></svg>')
    dark = dark_figures.darken(light, tokens, "x.svg")
    assert f'<rect fill="{tokens["paper"]}"/>' in dark and f'<rect fill="{tokens["accent"]}"/>' in dark
    assert f'<text fill="{tokens["bg"]}">' in dark
    assert f'<rect fill="{tokens["accent"]}" fill-opacity=".55"/>' in dark
    assert f'<line stroke="{tokens["line-strong"]}"/>' in dark and f'<rect fill="{tokens["muted"]}"/>' in dark
    assert "Dark copy of x.svg" in dark
    with pytest.raises(SystemExit, match="#123456"):
        dark_figures.darken('<svg xmlns="http://www.w3.org/2000/svg"><rect fill="#123456"/></svg>', tokens, "y.svg")


def test_reader_offers_the_dark_copy_by_theme_and_light_in_print(tmp_path):
    for relative in ("course/lessons", "course/assets/responsive", "metadata"):
        (tmp_path / relative).mkdir(parents=True, exist_ok=True)
    for relative in ("course/assets/x.png", "course/assets/responsive/x-desktop.svg", "course/assets/responsive/x-desktop-dark.svg",
                     "course/assets/responsive/x-mobile.svg", "course/assets/responsive/x-mobile-dark.svg"):
        (tmp_path / relative).write_text("<svg/>", encoding="utf-8")
    mapping = {"course/assets/x.png": {"desktop": "course/assets/responsive/x-desktop.svg",
                                       "mobile": "course/assets/responsive/x-mobile.svg", "caption": "Подпись"}}
    (tmp_path / "metadata/responsive-figures.json").write_text(json.dumps(mapping), encoding="utf-8")
    renderer = reader.CourseRenderer(tmp_path, tmp_path / "course/lessons/M00-L01.md", "M00-L01", {})
    html = mistune.create_markdown(renderer=renderer)("![График](../assets/x.png)\n")
    # D7: two pictures, the light one loaded eagerly so that a print, which waits for no source switch, has it.
    pictures = re.findall(r'<picture class="figure-(\w+)">(?:<source media="\(max-width: 600px\)" srcset="([^"]+)" data-layout="mobile">)?'
                          r'<img src="([^"]+)" alt="График" loading="(\w+)"></picture>', html)
    assert pictures == [("light", "course/assets/responsive/x-mobile.svg", "course/assets/responsive/x-desktop.svg", "eager"),
                        ("dark", "course/assets/responsive/x-mobile-dark.svg", "course/assets/responsive/x-desktop-dark.svg", "lazy")]
    assert 'class="figure-expand figure-themed"' in html and "data-variant" not in html


def test_css_shows_the_picture_of_the_theme_and_prints_the_light_one():
    css = (ROOT / "tools/reader.css").read_text(encoding="utf-8")
    assert (".figure-themed .figure-dark{display:none}:root[data-theme=\"dark\"] .figure-themed .figure-dark{display:inline}"
            ":root[data-theme=\"dark\"] .figure-themed .figure-light{display:none}") in css
    printing = css[css.index("@media print{body{"):]
    assert ".figure-themed .figure-light{display:inline!important}.figure-themed .figure-dark{display:none!important}" in printing
    script = (ROOT / "tools/reader.js").read_text(encoding="utf-8")
    assert "figure.querySelectorAll('source[data-layout=\"mobile\"]')" in script and "data-variant" not in script
