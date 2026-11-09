"""D7: print is light, without navigation or the theme switch, with light figures, drawn diagrams and link addresses."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSS = (ROOT / "tools/reader.css").read_text(encoding="utf-8")
SCRIPT = (ROOT / "tools/reader.js").read_text(encoding="utf-8")


def print_rules() -> str:
    start = CSS.index("@media print{body{")
    depth = 0
    for index in range(start, len(CSS)):
        depth += {"{": 1, "}": -1}.get(CSS[index], 0)
        if depth == 0 and CSS[index] == "}":
            return CSS[start:index + 1]
    raise AssertionError("print block is not closed")


def test_print_is_light_and_hides_navigation_and_the_theme_switch():
    assert re.search(r"@media print\{:root\{color-scheme:light;", CSS)
    hidden = re.search(r"([^{}]*)\{display:none!important\}", print_rules()).group(1).split(",")
    assert {".sidebar", ".topbar", ".lesson-outline", ".copy-code", "#figure-dialog", ".experiment"} <= set(hidden)
    # The theme switch lives in the top bar.
    assert re.search(r'<header class="topbar"[^>]*>.*?id="theme-toggle"', (ROOT / "tools/build_reader.py").read_text(encoding="utf-8"),
                     re.DOTALL)


def test_print_shows_external_link_addresses_and_light_figures():
    rules = print_rules()
    assert '.external-link::after{content:" (" attr(href) ")";' in rules
    assert ".external-link::after" not in CSS.replace(rules, "")
    assert ".figure-themed .figure-light{display:inline!important}.figure-themed .figure-dark{display:none!important}" in rules


def test_a_lesson_draws_its_diagrams_on_opening_and_the_print_button_waits_for_them():
    assert "rootMargin:'100000px 0px'" in SCRIPT and "diagramsSettled=()=>queue;" in SCRIPT
    assert ("Promise.race([diagramsSettled(),new Promise(resolve=>setTimeout(resolve,5000))]).then(()=>window.print())"
            in SCRIPT)
