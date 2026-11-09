"""F1/F2: the fonts are the recorded files within budget, ship with their OFL texts and are embedded offline."""
from __future__ import annotations

import base64
import importlib.util
import json
import re
import shutil
from hashlib import sha256
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FONTS = ROOT / "tools/fonts"
MANIFEST = json.loads((FONTS / "fonts.json").read_text(encoding="utf-8"))
BUDGET_BYTES = 350 * 1024  # IMPROVEMENT_PLAN.md, phase 6: F1 file budget and F2 reader growth
SPEC = importlib.util.spec_from_file_location("reader_fonts", ROOT / "tools/build_reader.py")
reader = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reader)


def test_font_files_match_recorded_size_and_hash():
    for item in MANIFEST["files"] + MANIFEST["licenses"]:
        data = (FONTS / item["file"]).read_bytes()
        assert len(data) == item["bytes"], item["file"]
        assert sha256(data).hexdigest() == item["sha256"], item["file"]
    for item in MANIFEST["files"]:
        assert (FONTS / item["file"]).read_bytes()[:4] == b"wOF2", item["file"]


def test_planned_families_weights_and_subsets_within_budget():
    found = {(item["family"], item["weight"], item["subset"]) for item in MANIFEST["files"]}
    planned = {(family, weight, subset)
               for family, weights in (("Golos Text", (400, 500, 700)), ("JetBrains Mono", (400, 600)))
               for weight in weights for subset in ("latin", "cyrillic")}
    assert found == planned
    total = sum(item["bytes"] for item in MANIFEST["files"])
    assert total == MANIFEST["total_font_bytes"] <= BUDGET_BYTES
    # Every file carries the range it covers, so F2 can declare one @font-face per file.
    assert all(re.fullmatch(r"U\+[0-9A-F]{4}(-[0-9A-F]{4})?(,U\+[0-9A-F]{4}(-[0-9A-F]{4})?)*", item["unicode_range"])
               for item in MANIFEST["files"])
    assert {item["unicode_range"] for item in MANIFEST["files"] if item["subset"] == "cyrillic"} == {
        "U+0301,U+0400-045F,U+0490-0491,U+04B0-04B1,U+2116"}


def test_each_family_ships_its_ofl_text_and_readme_lists_every_file():
    families = {item["family"] for item in MANIFEST["files"]}
    licensed = {item["family"] for item in MANIFEST["licenses"]}
    assert families == licensed
    for item in MANIFEST["licenses"]:
        text = (FONTS / item["file"]).read_text(encoding="utf-8")
        assert item["license"] == "OFL-1.1" and "SIL OPEN FONT LICENSE Version 1.1" in text
        assert text.startswith("Copyright ")
    readme = (FONTS / "README.md").read_text(encoding="utf-8")
    for item in MANIFEST["files"] + MANIFEST["licenses"]:
        assert item["file"] in readme and item["sha256"] in readme, item["file"]


def test_reader_embeds_each_font_once_as_a_data_url_with_swap():
    faces = reader.font_faces()
    assert faces.count("@font-face") == faces.count("font-display:swap") == len(MANIFEST["files"])
    for item in MANIFEST["files"]:
        encoded = base64.b64encode((FONTS / item["file"]).read_bytes()).decode("ascii")
        assert (f"src:url(data:font/woff2;base64,{encoded}) format('woff2');"
                f"unicode-range:{item['unicode_range']}") in faces, item["file"]
    assert faces.count("url(") == faces.count("url(data:font/woff2;base64,")
    assert len(faces.encode("utf-8")) <= BUDGET_BYTES
    assert reader.FONT_FACES in (ROOT / "START_HERE.html").read_text(encoding="utf-8")
    # Text and code use the token stacks, which keep Segoe UI and Consolas as fallbacks.
    assert "font:17px/1.75 var(--sans)" in reader.CSS and ".body-content code{font-family:var(--mono)" in reader.CSS
    assert "'Segoe UI'" in reader.CSS.split("--sans:")[1].split(";")[0]
    assert "Consolas" in reader.CSS.split("--mono:")[1].split(";")[0]


def test_changed_font_file_stops_the_build(tmp_path):
    copy = tmp_path / "fonts"
    shutil.copytree(FONTS, copy)
    target = copy / MANIFEST["files"][0]["file"]
    data = bytearray(target.read_bytes())
    data[-1] ^= 1
    target.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="differs from tools/fonts/fonts.json"):
        reader.font_faces(copy)
