"""Check the "versions and verification date" block of tool-dependent lessons.

Every lesson of M09-M16 and every extension lesson (ID with suffix X) carries

> **Версии и дата проверки:** <tools and models> · проверено DD.MM.YYYY · источник: <link>

A missing or malformed block is an error. A date older than six months is only
a warning: tool versions drift faster than the course text, and re-verifying
them is the author's decision, not an automatic failure.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
MAX_AGE_DAYS = 183
MARKER = "**Версии и дата проверки:**"
BLOCK = re.compile(r"(?m)^> \*\*Версии и дата проверки:\*\* (?P<tools>\S.*?) · проверено (?P<date>\d{2}\.\d{2}\.\d{4})"
                   r" · источник: (?P<source>\S.*)$")
LESSON = re.compile(r"M(?P<module>\d{2})-L\d{2}(?P<extension>X?)")
LINK = re.compile(r"\]\(https?://|https?://")


def requires_block(lesson_id: str) -> bool:
    match = LESSON.fullmatch(lesson_id)
    return bool(match) and (match["extension"] == "X" or 9 <= int(match["module"]) <= 16)


def check(root: Path = ROOT, today: date | None = None) -> dict:
    today = today or date.today()
    errors, warnings, checked = [], [], {}
    for path in sorted((root / "course/lessons").glob("*.md")):
        lesson_id = path.stem
        text = path.read_text(encoding="utf-8")
        blocks = list(BLOCK.finditer(text))
        if text.count(MARKER) != len(blocks):
            errors.append(f"{lesson_id}: malformed versions block; expected '> {MARKER} … · проверено ДД.ММ.ГГГГ · источник: <ссылка>'")
        if not blocks:
            if requires_block(lesson_id):
                errors.append(f"{lesson_id}: missing versions block")
            continue
        for block in blocks:
            try:
                verified = datetime.strptime(block["date"], "%d.%m.%Y").date()
            except ValueError:
                errors.append(f"{lesson_id}: invalid verification date {block['date']}")
                continue
            if not LINK.search(block["source"]):
                errors.append(f"{lesson_id}: versions block source must be a link")
            age = (today - verified).days
            if age < 0:
                errors.append(f"{lesson_id}: verification date {block['date']} is in the future")
            elif age > MAX_AGE_DAYS:
                warnings.append(f"{lesson_id}: versions checked {block['date']}, {age} days ago; re-verify tools and models")
            checked[lesson_id] = {"verified": verified.isoformat(), "age_days": age}
    return {"today": today.isoformat(), "max_age_days": MAX_AGE_DAYS, "lessons_with_block": len(checked),
            "errors": errors, "warnings": warnings}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--today", type=date.fromisoformat, help="Reference date YYYY-MM-DD (default: today)")
    parser.add_argument("--strict", action="store_true", help="Also fail on stale verification dates")
    args = parser.parse_args(argv)
    result = check(ROOT, args.today)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if result["errors"] or (args.strict and result["warnings"]) else 0


if __name__ == "__main__":
    sys.exit(main())
