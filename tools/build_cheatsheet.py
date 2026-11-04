"""Collect "Карточка урока" sections into course/cheatsheet.md and an Anki CSV.

Each card is a table with four rows: Правило, Смысл величин, Когда применять,
Главная ошибка. The cheatsheet is generated: edit the cards in the lessons and
rebuild. artifacts/anki-cards.csv is a build output for import into Anki.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = "course/cheatsheet.md"
ANKI = "artifacts/anki-cards.csv"
FIELDS = ["Правило", "Смысл величин", "Когда применять", "Главная ошибка"]
CARD = re.compile(r"(?ms)^## Карточка урока\n(.*?)(?=^## |\Z)")


def sentence(text: str) -> str:
    return text if text.endswith((".", "!", "?")) else text + "."


def parse_card(section: str) -> dict:
    """Field → text for every "| Поле | Содержание |" row of a card table."""
    rows = {}
    for line in section.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) == 2 and cells[0] in FIELDS:
            rows[cells[0]] = cells[1]
    return rows


def collect(root: Path = ROOT) -> list[dict]:
    curriculum = json.loads((root / "curriculum.json").read_text(encoding="utf-8"))
    cards = []
    for module in curriculum["modules"]:
        for lesson in module["lessons"]:
            path = root / f"course/lessons/{lesson['id']}.md"
            found = CARD.search(path.read_text(encoding="utf-8")) if path.is_file() else None
            if found:
                cards.append({"module": module, "lesson": lesson, "fields": parse_card(found.group(1))})
    return cards


def render(cards: list[dict], total: int) -> str:
    lines = ["# Шпаргалка: карточки уроков", "",
             "Страница собрана `tools/build_cheatsheet.py` из разделов «Карточка урока»; вручную её не правят. "
             f"Карточек: {len(cards)} из {total} уроков. Карточка помогает вспомнить правило перед "
             "отложенной проверкой, но не заменяет решение задачи: сначала попробуй восстановить правило сам, затем сверься.",
             "",
             "Те же карточки в виде пар «вопрос — ответ» сохраняются в `artifacts/anki-cards.csv` для импорта в Anki "
             "(Файл → Импорт; разделитель и столбцы указаны в заголовке файла)."]
    module_id = None
    for card in cards:
        module, lesson = card["module"], card["lesson"]
        if module["id"] != module_id:
            module_id = module["id"]
            lines += ["", f"## {module['id']}. {module['title']}"]
        lines += ["", f"### [{lesson['id']}. {lesson['title']}](lessons/{lesson['id']}.md)", ""]
        lines += [f"- **{field}.** {sentence(card['fields'][field])}" for field in FIELDS if field in card["fields"]]
    return "\n".join(lines) + "\n"


def anki_rows(cards: list[dict]) -> str:
    buffer = io.StringIO()
    buffer.write("#separator:Comma\n#html:true\n#columns:Вопрос,Ответ,Теги\n#tags column:3\n")
    writer = csv.writer(buffer, lineterminator="\n")
    for card in cards:
        lesson, fields = card["lesson"], card["fields"]
        topic = f"{lesson['id']} «{lesson['title']}»"
        tag = f"ml-llm-course {card['module']['id']} {lesson['id']}"
        writer.writerow([f"{topic}: правило и смысл величин?",
                         f"{fields.get('Правило', '')}<br><br>{fields.get('Смысл величин', '')}", tag])
        error = fields.get("Главная ошибка", "")
        writer.writerow([f"{topic}: когда применять и какая главная ошибка?",
                         f"{fields.get('Когда применять', '')}<br><br>Главная ошибка: {error[:1].lower() + error[1:]}", tag])
    return buffer.getvalue()


def build(root: Path = ROOT) -> tuple[str, str]:
    cards = collect(root)
    total = sum(len(m["lessons"]) for m in json.loads((root / "curriculum.json").read_text(encoding="utf-8"))["modules"])
    return render(cards, total), anki_rows(cards)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="Fail if course/cheatsheet.md is stale")
    args = parser.parse_args(argv)
    markdown, csv_text = build(ROOT)
    target = ROOT / OUTPUT
    if args.check:
        if not target.is_file() or target.read_text(encoding="utf-8") != markdown:
            raise SystemExit("Cheatsheet is stale. Run: python tools/build_cheatsheet.py")
    else:
        target.write_text(markdown, encoding="utf-8", newline="\n")
        anki = ROOT / ANKI
        anki.parent.mkdir(parents=True, exist_ok=True)
        anki.write_text(csv_text, encoding="utf-8", newline="\n")
    print(json.dumps({"cheatsheet": OUTPUT, "anki": ANKI, "cards": markdown.count("\n### "),
                      "mode": "check" if args.check else "build"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
