"""Tool-dependent lessons must state versions and an honest verification date."""
from __future__ import annotations
from datetime import date
import importlib.util
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[1]


def load_tool(name):
    spec = importlib.util.spec_from_file_location(name, SOURCE / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


freshness = load_tool("check_freshness")
BLOCK = ("> **Версии и дата проверки:** Python 3.11.9, PyTorch 2.15.0+cpu · проверено 04.11.2026 · "
         "источник: [PyTorch](https://docs.pytorch.org/docs/2.15/nn.html)\n\n")


def write_lessons(root, lessons):
    folder = root / "course/lessons"
    folder.mkdir(parents=True)
    for lesson_id, block in lessons.items():
        (folder / f"{lesson_id}.md").write_text(f"# {lesson_id}\n\n{block}## Объяснение\n\nТекст.\n", encoding="utf-8")


def test_every_m09_m16_lesson_has_a_valid_block():
    result = freshness.check(SOURCE)
    assert result["errors"] == []
    required = [p.stem for p in (SOURCE / "course/lessons").glob("*.md") if freshness.requires_block(p.stem)]
    assert len(required) >= 51 and result["lessons_with_block"] >= len(required)


def test_missing_and_malformed_blocks_are_errors(tmp_path):
    write_lessons(tmp_path, {
        "M05-L01": "",  # Before M09 the block is optional.
        "M10-L01": "",
        "M10-L07X": "",
        "M11-L01": "> **Версии и дата проверки:** Python 3.11.9, проверено вчера\n\n",
        "M12-L01": BLOCK.replace("[PyTorch](https://docs.pytorch.org/docs/2.15/nn.html)", "документация PyTorch"),
    })
    errors = freshness.check(tmp_path, date(2026, 11, 4))["errors"]
    assert "M10-L01: missing versions block" in errors
    assert "M10-L07X: missing versions block" in errors
    assert any(e.startswith("M11-L01: malformed versions block") for e in errors)
    assert "M12-L01: versions block source must be a link" in errors
    assert not any(e.startswith("M05-L01") for e in errors)


def test_old_date_warns_and_future_date_fails(tmp_path):
    write_lessons(tmp_path, {"M09-L01": BLOCK, "M09-L02": BLOCK.replace("04.11.2026", "05.11.2026")})
    fresh = freshness.check(tmp_path, date(2026, 11, 4))
    assert fresh["warnings"] == [] and fresh["errors"] == ["M09-L02: verification date 05.11.2026 is in the future"]
    stale = freshness.check(tmp_path, date(2027, 6, 1))
    assert stale["errors"] == []
    assert stale["warnings"][0].startswith("M09-L01: versions checked 04.11.2026, 209 days ago")


def test_cli_fails_on_stale_dates_only_in_strict_mode(tmp_path, monkeypatch):
    write_lessons(tmp_path, {"M09-L01": BLOCK})
    monkeypatch.setattr(freshness, "ROOT", tmp_path)
    assert freshness.main(["--today", "2027-06-01"]) == 0
    assert freshness.main(["--today", "2027-06-01", "--strict"]) == 1
