"""Electronic learning journal: attempts in progress/progress.jsonl, review dates and a dashboard.

progress/progress.md stays the canonical text journal; this log stores one JSON
line per real attempt. Nothing is recorded automatically: running a reference
lab or opening a lesson is not an attempt.

  add TASK --min 140 --self 3 --help hint2 --route full --error "..."   append an attempt
  import                                                                 copy filled rows from progress.md
  due [--date YYYY-MM-DD]                                                reviews due, grouped into mixed sessions
  dashboard [--check]                                                    write progress/dashboard.md

Review rule (course/review-schedule.md): after an independent attempt the next
check comes in 1, 3, 7 and then 21 days; a supported attempt or one that needs
repetition is checked again the next day without advancing the stage.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import date, timedelta
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
LOG = "progress/progress.jsonl"
JOURNAL = "progress/progress.md"
DASHBOARD = "progress/dashboard.md"
INTERVALS = (1, 3, 7, 21)
STATES = {
    "unchecked": "Не проверено",
    "repeat": "Требуется повторение",
    "supported": "С поддержкой",
    "independent": "Самостоятельно — самооценка",
    "confirmed": "Подтверждено позже — самооценка",
}
SELF_SCALE = {"0": "unchecked", "1": "repeat", "2": "supported", "3": "independent", "4": "confirmed"}
# Free text in progress.md is matched by keyword, most specific first.
STATE_WORDS = (("подтвержд", "confirmed"), ("самостоят", "independent"), ("поддерж", "supported"),
               ("повтор", "repeat"), ("не проверено", "unchecked"))
HELP = {"none": "без помощи", "hint1": "подсказка 1", "hint2": "подсказка 2", "hint3": "подсказка 3",
        "key": "полный ключ", "external": "внешняя помощь"}
TASK = re.compile(r"(?P<lesson>(?P<module>[MP]\d{2})(?:-L\d{2}X?)?)(?:-[A-Z]+\d*)*|D00")


def parse_task(task: str) -> dict:
    match = TASK.fullmatch(task)
    if not match:
        raise ValueError(f"Unknown task id {task!r}: expected e.g. M10-L03-A, M10-HW01, P01-HW01 or D00")
    module = match.group("module") or "D00"
    return {"task": task, "module": module, "lesson": match.group("lesson") or task}


def read_log(root: Path = ROOT) -> list[dict]:
    path = root / LOG
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append(root: Path, records: list[dict]) -> None:
    path = root / LOG
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def make_record(task: str, day: date, minutes: int | None, state: str, help_level: str = "none",
                route: str | None = None, error: str = "", note: str = "", external: str = "",
                source: str = "cli") -> dict:
    state = SELF_SCALE.get(state, state)
    if state not in STATES:
        raise ValueError(f"Unknown self-assessment {state!r}: use 1-4 or {', '.join(STATES)}")
    if help_level not in HELP:
        raise ValueError(f"Unknown help level {help_level!r}: use {', '.join(HELP)}")
    if minutes is not None and minutes < 0:
        raise ValueError("Minutes cannot be negative")
    return {"schema": 1, **parse_task(task), "date": day.isoformat(), "minutes": minutes,
            "state": state, "state_label": STATES[state], "help": help_level, "route": route,
            "error": error, "note": note, "external_check": external or None, "source": source}


def state_from_text(text: str) -> str:
    return next((state for word, state in STATE_WORDS if word in text.lower()), "unchecked")


def import_journal(root: Path = ROOT) -> list[dict]:
    """Filled table rows and attempt blocks of progress.md that are not yet in the log."""
    text = (root / JOURNAL).read_text(encoding="utf-8")
    known = {(r["task"], r["date"], r["source"]) for r in read_log(root)}
    found = []
    for line in text.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 6 or not TASK.fullmatch(cells[0]):
            continue
        day = re.search(r"\d{4}-\d{2}-\d{2}", cells[1])
        if not day:
            continue  # "—": the row is a plan, not an attempt.
        state = state_from_text(cells[3])
        minutes = re.search(r"(\d+)\s*мин", cells[1])
        found.append(make_record(cells[0], date.fromisoformat(day.group(0)), int(minutes.group(1)) if minutes else None,
                                 state, note=f"Работа: {cells[2]}; помощь: {cells[4]}; отложенная: {cells[5]}",
                                 source="progress.md"))
    for block in re.findall(r"(?m)^- Дата, длительность и версия задания:(.*(?:\n- .*)*)", text):
        fields = dict(re.findall(r"(?m)^- ?([^:\n]+):[ \t]*(.*)$", "- Дата, длительность и версия задания:" + block))
        day = re.search(r"\d{4}-\d{2}-\d{2}", fields.get("Дата, длительность и версия задания", ""))
        task = re.search(r"\b(?:[MP]\d{2}(?:-L\d{2}X?)?(?:-[A-Z]+\d*)*|D00)\b", fields.get("Путь к работе", "") +
                         " " + fields.get("Проверяемое умение", ""))
        if not day or not task:
            continue  # The empty template block.
        minutes = re.search(r"(\d+)\s*мин", fields.get("Дата, длительность и версия задания", ""))
        state = state_from_text(fields.get("Самооценка и её основание", ""))
        found.append(make_record(task.group(0), date.fromisoformat(day.group(0)),
                                 int(minutes.group(1)) if minutes else None, state,
                                 error=fields.get("Ошибка и её причина", ""),
                                 external=fields.get("Внешняя проверка, если действительно была", ""),
                                 source="progress.md"))
    return [r for r in found if (r["task"], r["date"], r["source"]) not in known]


def schedule(records: list[dict]) -> dict[str, dict]:
    """Next review per lesson: stage advances only on independent or confirmed attempts."""
    state = {}
    for record in sorted(records, key=lambda r: r["date"]):
        current = state.get(record["lesson"], {"stage": 0})
        day = date.fromisoformat(record["date"])
        if record["state"] in ("independent", "confirmed"):
            stage = current["stage"] + 1
            due = day + timedelta(days=INTERVALS[stage - 1]) if stage <= len(INTERVALS) else None
        else:
            stage, due = (0 if record["state"] == "repeat" else current["stage"]), day + timedelta(days=1)
        state[record["lesson"]] = {"stage": stage, "due": due.isoformat() if due else None,
                                   "module": record["module"], "last": record["date"], "task": record["task"]}
    return state


def mixed_sessions(items: list[dict], size: int = 3) -> list[list[dict]]:
    """Interleave reviews: each session takes items from different modules in turn."""
    by_module = defaultdict(list)
    for item in sorted(items, key=lambda i: (i["due"], i["lesson"])):
        by_module[item["module"]].append(item)
    queue, sessions = [], []
    while any(by_module.values()):
        for module in sorted(by_module):
            if by_module[module]:
                queue.append(by_module[module].pop(0))
    for start in range(0, len(queue), size):
        sessions.append(queue[start:start + size])
    return sessions


def render_dashboard(records: list[dict]) -> str:
    lines = ["# Дашборд прогресса", "",
             "Страница собрана `tools/progress.py dashboard` из `progress/progress.jsonl`; вручную её не правят. "
             "Здесь только реальные попытки, записанные учеником. Самооценка и внешняя проверка учитываются отдельно; "
             "запуск эталонного кода попыткой не считается. Правило повторения — [расписание повторения](../course/review-schedule.md).",
             ""]
    if not records:
        lines += ["Записей пока нет. После первой попытки добавь её командой:", "", "```powershell",
                  '.\\.venv\\Scripts\\python.exe tools\\progress.py add P01-HW01 --min 120 --self 2 --help hint1 --error "..."',
                  ".\\.venv\\Scripts\\python.exe tools\\progress.py dashboard", "```", ""]
        return "\n".join(lines)
    minutes = sum(r["minutes"] or 0 for r in records)
    routes = Counter(r["route"] or "не указан" for r in records)
    lines += [f"Попыток: {len(records)}, учтённого времени: {minutes} мин ({minutes / 60:.1f} ч). "
              f"Маршрут: {', '.join(f'{k} — {v}' for k, v in sorted(routes.items()))}. "
              f"Внешних проверок: {sum(1 for r in records if r['external_check'])}.", "",
              "| Модуль | Попыток | Минут | Последнее состояние | Помощь чаще всего |", "| --- | --- | --- | --- | --- |"]
    for module in sorted({r["module"] for r in records}):
        rows = [r for r in records if r["module"] == module]
        last = max(rows, key=lambda r: r["date"])
        top_help = Counter(r["help"] for r in rows).most_common(1)[0][0]
        lines.append(f"| {module} | {len(rows)} | {sum(r['minutes'] or 0 for r in rows)} | "
                     f"{last['state_label']} ({last['task']}, {last['date']}) | {HELP[top_help]} |")
    plan = schedule(records)
    upcoming = [dict(item, lesson=lesson) for lesson, item in plan.items() if item["due"]]
    done = sorted(lesson for lesson, item in plan.items() if not item["due"])
    lines += ["", "## Повторения", ""]
    if upcoming:
        lines += ["| Дата | Урок или задача | Модуль | Ступень (1–3–7–21) |", "| --- | --- | --- | --- |"]
        lines += [f"| {i['due']} | {i['lesson']} ({i['task']}) | {i['module']} | {i['stage']} из {len(INTERVALS)} |"
                  for i in sorted(upcoming, key=lambda i: (i["due"], i["lesson"]))]
        lines += ["", "Смешанные сеансы для накопившихся повторений: в каждом задачи из разных модулей по очереди.", ""]
        lines += [f"{n}. " + "; ".join(f"{i['lesson']} ({i['module']})" for i in session)
                  for n, session in enumerate(mixed_sessions(upcoming), 1)]
    else:
        lines.append("Назначенных повторений нет.")
    if done:
        lines += ["", f"Прошли все четыре ступени: {', '.join(done)}."]
    lines += ["", "## Ошибки из журнала", ""]
    errors = [r for r in records if r["error"]]
    lines += [f"- {r['date']} · {r['task']}: {r['error']}" for r in errors] or ["Ошибок не записано."]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    # No built-in -h here: "--help hint2" names the help level, as in the improvement plan.
    add = commands.add_parser("add", help="Append one real attempt", add_help=False)
    add.add_argument("task")
    add.add_argument("--date", type=date.fromisoformat, default=date.today())
    add.add_argument("--min", dest="minutes", type=int, required=True, help="Actual minutes spent")
    add.add_argument("--self", dest="state", required=True,
                     help="0 не проверено, 1 требуется повторение, 2 с поддержкой, 3 самостоятельно, 4 подтверждено позже")
    add.add_argument("--help-level", "--help", dest="help_level", default="none", choices=sorted(HELP))
    add.add_argument("--route", choices=["short", "full"], help="Short path after a first-step check or full lesson")
    add.add_argument("--error", default="")
    add.add_argument("--note", default="")
    add.add_argument("--external", default="", help="Only a real independent check: who and when")
    commands.add_parser("import", help="Copy filled rows and attempt blocks from progress/progress.md")
    due = commands.add_parser("due", help="Reviews due on a date, as mixed sessions")
    due.add_argument("--date", type=date.fromisoformat, default=date.today())
    board = commands.add_parser("dashboard", help="Write progress/dashboard.md")
    board.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "add":
        record = make_record(args.task, args.date, args.minutes, args.state, args.help_level, args.route,
                             args.error, args.note, args.external)
        append(ROOT, [record])
        print(json.dumps(record, ensure_ascii=False))
    elif args.command == "import":
        records = import_journal(ROOT)
        append(ROOT, records)
        print(json.dumps({"imported": len(records)}, ensure_ascii=False))
    elif args.command == "due":
        items = [dict(item, lesson=lesson) for lesson, item in schedule(read_log(ROOT)).items()
                 if item["due"] and item["due"] <= args.date.isoformat()]
        print(json.dumps({"date": args.date.isoformat(), "sessions": mixed_sessions(items)}, ensure_ascii=False, indent=2))
    else:
        markdown = render_dashboard(read_log(ROOT))
        target = ROOT / DASHBOARD
        if args.check:
            if not target.is_file() or target.read_text(encoding="utf-8") != markdown:
                raise SystemExit("Dashboard is stale. Run: python tools/progress.py dashboard")
        else:
            target.write_text(markdown, encoding="utf-8", newline="\n")
        print(json.dumps({"dashboard": DASHBOARD, "attempts": len(read_log(ROOT))}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
