"""Progress map of route 2.0 (R05): levels, share of passed missions and the day streak.

Built only from progress/progress.jsonl and course/route-v2.yaml, so the same journal always gives
the same map. No points, rewards or story: the map shows what the journal already says.

  python tools/progress_map.py           rewrite the map block in course/START.md
  python tools/progress_map.py --check   fail if the block is stale

The block sits between the markers below; `progress.py dashboard` rewrites it too. The reader
(build_reader.py) draws the same numbers as the level map from quality/design-v3/quest-mode.html.
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta
from html import escape
import importlib.util
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
START = "course/START.md"
BEGIN, END = "<!-- progress-map:start -->", "<!-- progress-map:end -->"
BLOCK = re.compile(re.escape(BEGIN) + r".*?" + re.escape(END), re.S)


def _progress():
    spec = importlib.util.spec_from_file_location("course_progress_tool", Path(__file__).with_name("progress.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def streak(days: list[str]) -> tuple[int, str | None]:
    """Consecutive calendar days with an attempt, ending on the latest attempt day."""
    if not days:
        return 0, None
    seen = {date.fromisoformat(d) for d in days}
    last = max(seen)
    count, day = 0, last
    while day in seen:
        count, day = count + 1, day - timedelta(days=1)
    return count, last.isoformat()


def stats(root: Path = ROOT) -> dict:
    import yaml

    progress = _progress()
    records = progress.read_log(root)
    route = yaml.safe_load((root / progress.ROUTE).read_text(encoding="utf-8"))
    missions = progress.route_missions(root)
    passed = {r["task"] for r in records if r["state"] in progress.MISSION_PASSED}
    levels = []
    for level in route["levels"]:
        ids = [m["id"] for m in missions if m["level"] == level["level"]]
        done = sum(1 for i in ids if i in passed)
        levels.append({"level": level["level"], "title": level["title"], "outcome": level.get("outcome", ""),
                       "card": level.get("card") or level["project"]["title"], "hours": level.get("hours"),
                       "passed": done, "total": len(ids), "percent": done * 100 // len(ids) if ids else 0})
    current = next((lv["level"] for lv in levels if lv["passed"] < lv["total"]), None)
    for lv in levels:
        lv["state"] = ("done" if lv["passed"] == lv["total"] else "active" if lv["level"] == current else
                       "started" if lv["passed"] else "ahead")
    days, last = streak([r["date"] for r in records])
    total = sum(lv["total"] for lv in levels)
    done = sum(lv["passed"] for lv in levels)
    return {"levels": levels, "passed": done, "total": total, "percent": done * 100 // total if total else 0,
            "streak": days, "last_day": last, "attempts": len(records), "current": current}


def markdown(data: dict) -> str:
    if data["attempts"]:
        streak_text = f"серия {data['streak']} дн. подряд по {data['last_day']}"
    else:
        streak_text = "журнал пока пуст: первую запись сделает тьютор"
    lines = [BEGIN,
             "<!-- Собирается tools/progress_map.py из progress/progress.jsonl; вручную не править. -->",
             "",
             f"**Пройдено миссий:** {data['passed']} из {data['total']} ({data['percent']} %) · {streak_text}.",
             "",
             "| Ур. | Что умеешь после | Мини-проект | Часы | Миссии |",
             "| --- | --- | --- | --- | --- |"]
    for lv in data["levels"]:
        mark = " ← сейчас" if lv["level"] == data["current"] else ""
        lines.append(f"| {lv['level']} | {lv['outcome']} | {lv['card']} | {lv['hours']} | "
                     f"{lv['passed']} из {lv['total']} ({lv['percent']} %){mark} |")
    lines.append(END)
    return "\n".join(lines)


def html(data: dict) -> str:
    """The level map of the quest-mode prototype, static: no points, no rewards."""
    streak_value = str(data["streak"]) if data["attempts"] else "—"
    stats_bar = (
        '<div class="map-stats" aria-label="Прогресс по журналу">'
        f'<div class="map-stat"><b>{data["passed"]} / {data["total"]}</b><span>миссий пройдено</span></div>'
        f'<div class="map-stat"><b>{data["percent"]} %</b><span>маршрута</span></div>'
        f'<div class="map-stat"><b>{streak_value}</b><span>дней подряд с попыткой</span></div></div>')
    nodes = []
    for lv in data["levels"]:
        label = {"done": "пройден", "active": "сейчас", "started": "начат", "ahead": "впереди"}[lv["state"]]
        nodes.append(
            f'<div class="map-node {lv["state"]}"><span class="map-id">Уровень {lv["level"]} · {label}</span>'
            f'<span class="map-title">{escape(lv["title"])}</span>'
            f'<span class="map-bar" role="img" aria-label="{lv["percent"]} %"><i style="width:{lv["percent"]}%"></i></span>'
            f'<span class="map-count">{lv["passed"]} из {lv["total"]} · {escape(lv["card"])}</span></div>')
    note = ('<p class="map-note">Полоса — доля миссий уровня, пройденных с записью в журнале. '
            'Чтение и запуск эталона полосу не двигают.</p>')
    return f'<section class="progress-map" aria-label="Карта прогресса">{stats_bar}<div class="map-levels">{"".join(nodes)}</div>{note}</section>'


def render_start(root: Path = ROOT) -> str:
    text = (root / START).read_text(encoding="utf-8")
    if not BLOCK.search(text):
        raise ValueError(f"{START} has no progress map markers")
    return BLOCK.sub(lambda _: markdown(stats(root)), text)


def write_start(root: Path = ROOT, check: bool = False) -> bool:
    """Rewrite the block; with check=True only report whether it is current."""
    target = root / START
    updated = render_start(root)
    current = target.read_text(encoding="utf-8") == updated
    if not check and not current:
        target.write_text(updated, encoding="utf-8", newline="\n")
    return current


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    current = write_start(ROOT, args.check)
    if args.check and not current:
        raise SystemExit("Progress map in course/START.md is stale. Run: python tools/progress_map.py")
    data = stats(ROOT)
    print(json.dumps({"start": START, "passed": data["passed"], "total": data["total"], "streak": data["streak"]},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
