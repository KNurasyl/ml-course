"""R05: the progress map is built deterministically from the journal, without points or rewards."""
from __future__ import annotations

from datetime import date
import importlib.util
import shutil
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1]


def load_tool(name):
    spec = importlib.util.spec_from_file_location(name, SOURCE / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


progress = load_tool("progress")
progress_map = load_tool("progress_map")


@pytest.fixture
def course(tmp_path):
    (tmp_path / "course").mkdir()
    shutil.copy(SOURCE / "course/route-v2.yaml", tmp_path / "course/route-v2.yaml")
    shutil.copy(SOURCE / "course/START.md", tmp_path / "course/START.md")
    journal = [("M09-L07X-M0", "2026-11-18", "3"), ("MP0", "2026-11-19", "2"), ("M09-L01-M1", "2026-11-20", "1"),
               ("M09-L01-M1", "2026-11-20", "3"), ("M09-L01-M2", "2026-11-22", "3")]
    progress.append(tmp_path, [progress.make_record(t, date.fromisoformat(d), 40, s) for t, d, s in journal])
    return tmp_path


def test_synthetic_journal_gives_levels_percent_and_streak(course):
    data = progress_map.stats(course)
    zero, one = data["levels"][0], data["levels"][1]
    assert (zero["passed"], zero["total"], zero["percent"], zero["state"]) == (2, 2, 100, "done")
    assert one["passed"] == 2 and one["state"] == "active" and data["current"] == 1
    assert one["percent"] == 2 * 100 // one["total"]
    assert data["passed"] == 4 and data["attempts"] == 5
    # 18, 19, 20 in a row, then a gap: the streak ending on the last attempt (22) is one day.
    assert (data["streak"], data["last_day"]) == (1, "2026-11-22")
    assert progress_map.streak(["2026-11-18", "2026-11-19", "2026-11-20"]) == (3, "2026-11-20")
    assert progress_map.streak([]) == (0, None)


def test_map_is_deterministic_and_has_no_points(course):
    first = progress_map.render_start(course)
    assert first == progress_map.render_start(course)
    block = progress_map.BLOCK.search(first).group(0)
    assert "**Пройдено миссий:** 4 из" in block and "серия 1 дн. подряд по 2026-11-22" in block
    assert "| 1 | Выбирать и запускать модели | MP1 «Паспорт моделей» | 20 |" in block and "← сейчас" in block
    html = progress_map.html(progress_map.stats(course))
    assert 'class="map-node done"' in html and 'class="map-node active"' in html
    for word in ("очк", "наград", "трофе", "XP"):
        assert word not in block + html


def test_stale_map_is_detected_and_dashboard_rewrites_it(course, monkeypatch):
    assert progress_map.write_start(course, check=True) is False  # START was built from the empty journal
    monkeypatch.setattr(progress, "ROOT", course)
    (course / "progress").mkdir(exist_ok=True)
    assert progress.main(["dashboard"]) == 0
    assert progress_map.write_start(course, check=True) is True
    assert "4 из" in (course / "course/START.md").read_text(encoding="utf-8")


def test_repository_start_matches_its_journal():
    assert progress_map.write_start(SOURCE, check=True)


def test_reader_draws_the_map_on_start():
    reader = load_tool("build_reader")
    html, _ = reader.build(SOURCE)
    start = html[html.index('id="START"'):html.index('id="HOME"')]
    assert 'class="progress-map"' in start and "PROGRESS-MAP" not in start
    assert "progress-map:start" not in start
