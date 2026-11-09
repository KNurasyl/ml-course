"""F1: the embedded fonts are the recorded files, within budget, and ship with their OFL texts."""
from __future__ import annotations

import json
import re
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FONTS = ROOT / "tools/fonts"
MANIFEST = json.loads((FONTS / "fonts.json").read_text(encoding="utf-8"))
BUDGET_BYTES = 350 * 1024  # IMPROVEMENT_PLAN.md, phase 6, F1


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
