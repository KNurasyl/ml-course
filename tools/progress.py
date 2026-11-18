"""Electronic learning journal: attempts in progress/progress.jsonl, review dates and a dashboard.

progress/progress.md stays the canonical text journal; this log stores one JSON
line per real attempt. Nothing is recorded automatically: running a reference
lab or opening a lesson is not an attempt.

  add TASK --min 140 --self 3 --help hint2 --route full --error "..."   append an attempt
  import                                                                 copy filled rows from progress.md
  due [--date YYYY-MM-DD]                                                reviews due, grouped into mixed sessions
  dashboard [--check]                                                    write progress/dashboard.md and the START map
  next [--date YYYY-MM-DD] [--json]                                      tutor: overdue review or next route mission
  check MP1 --min 30 --self 3                                            run a mini-project's check.py and record the result

Review rule (course/review-schedule.md): after an independent attempt the next
check comes in 1, 3, 7 and then 21 days; a supported attempt or one that needs
repetition is checked again the next day without advancing the stage.
"""
from __future__ import annotations

import argparse
import importlib.util
import subprocess
from collections import Counter, defaultdict
from datetime import date, timedelta
import json
import os
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
# MP0..MP7 are the mini-projects of route 2.0 (course/route-v2.yaml).
TASK = re.compile(r"(?P<lesson>(?P<module>[MP]\d{2}|MP\d+)(?:-L\d{2}X?)?)(?:-[A-Z]+\d*)*|D00")


def parse_task(task: str) -> dict:
    match = TASK.fullmatch(task)
    if not match:
        raise ValueError(f"Unknown task id {task!r}: expected e.g. M10-L03-A, M10-HW01, P01-HW01, MP1 or D00")
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


ROUTE = "course/route-v2.yaml"
# A mission counts as passed once an attempt with real work is saved; "repeat" means do it again.
MISSION_PASSED = {"supported", "independent", "confirmed"}
# Only one's own work schedules a delayed check C (rule 1-3-7-21): entry, lab, launch and project
# missions practise or explore and do not start the review clock.
NO_REVIEW_KINDS = {"entry", "lab", "launch", "project"}
# Skip rule: an entry mission solved on its own (--self 3 --help none) credits missions M2 and M3 of the
# same lesson ("зачтено по входу"). The entry then starts the delayed check C, which stays mandatory.
CREDIT_STATE, CREDIT_HELP = "independent", "none"
CREDITED_KINDS = {"own", "lab"}


def route_missions(root: Path = ROOT) -> list[dict]:
    """Missions of route 2.0 in course order: lesson missions of a level, then its project."""
    import yaml

    route = yaml.safe_load((root / ROUTE).read_text(encoding="utf-8"))
    missions = []
    for level in route["levels"]:
        for entry in level.get("lessons") or []:
            if entry.get("planned"):
                continue
            owner = entry.get("lesson") or entry.get("fast_track")
            for mission in entry.get("missions") or []:
                missions.append({"level": level["level"], "level_title": level["title"], "lesson": owner, **mission})
        project = level.get("project") or {}
        common = {"level": level["level"], "level_title": level["title"], "kind": "project", "file": project.get("path"),
                  "title": project.get("title"), "planned": project.get("planned")}
        # Sessions of a mini-project: one result each, passed by an ordinary saved attempt.
        for session in project.get("sessions") or []:
            missions.append({**common, "lesson": session["id"], "id": session["id"], "result": session["result"]})
        for pid in project.get("missions") or []:
            missions.append({**common, "lesson": pid, "id": pid, **({"check": project["check"]} if project.get("check") else {})})
    return missions


def check_passed(record: dict) -> bool:
    """A project with check.py is passed only by a record of `progress.py check` with a green result."""
    return record.get("check") == "green" and record["state"] in MISSION_PASSED


def credits_lesson(record: dict, kinds: dict[str, str]) -> bool:
    """An entry attempt solved on its own without help: the skip rule applies to its lesson."""
    return kinds.get(record["task"]) == "entry" and record["state"] == CREDIT_STATE and record.get("help") == CREDIT_HELP


def mission_status(records: list[dict], missions: list[dict]) -> dict[str, str]:
    """Closed missions: "passed" by an own saved attempt, "credited" by the entry of the same lesson."""
    kinds = {m["id"]: m["kind"] for m in missions}
    checked = {m["id"] for m in missions if m.get("check")}
    status = {r["task"]: "passed" for r in records
              if r["state"] in MISSION_PASSED and (r["task"] not in checked or check_passed(r))}
    credited = {r["lesson"] for r in records if credits_lesson(r, kinds)}
    for mission in missions:
        if mission["lesson"] in credited and mission["kind"] in CREDITED_KINDS and mission["id"] not in status:
            status[mission["id"]] = "credited"
    return status


def review_due(records: list[dict], day: date, missions: list[dict]) -> list[dict]:
    """Delayed checks due on or before the day, oldest first."""
    kinds = {m["id"]: m["kind"] for m in missions}
    own = [r for r in records if kinds.get(r["task"]) not in NO_REVIEW_KINDS or credits_lesson(r, kinds)]
    last_task = {r["lesson"]: r["task"] for r in sorted(own, key=lambda r: r["date"])}
    items = []
    for lesson, item in schedule(own).items():
        if not item["due"] or item["due"] > day.isoformat():
            continue
        lesson_level = re.fullmatch(r"[MP]\d{2}-L\d{2}X?|P\d{2}", lesson)
        if lesson.startswith("P"):
            file = f"course/pilot/{lesson}.md"
        elif lesson_level:
            file = f"course/lessons/{lesson}.md"
        else:
            file = f"course/modules/{lesson}.md"
        base = lesson if "-L" in lesson or lesson.startswith("P") else re.sub(r"-[A-C]$", "", last_task[lesson])
        items.append({"type": "review", "lesson": lesson, "task": f"{base}-C", "due": item["due"],
                      "overdue_days": (day - date.fromisoformat(item["due"])).days, "stage": item["stage"],
                      "file": file, "section": "Новая и отложенная проверка"})
    return sorted(items, key=lambda i: (i["due"], i["lesson"]))


def next_step(root: Path = ROOT, day: date | None = None) -> dict:
    """What the tutor does now: an overdue delayed check first, otherwise the first unpassed mission."""
    day = day or date.today()
    records = read_log(root)
    missions = route_missions(root)
    passed = mission_status(records, missions)
    progress = {"passed": sum(1 for m in missions if m["id"] in passed),
                "credited": sum(1 for m in missions if passed.get(m["id"]) == "credited"), "total": len(missions)}
    due = review_due(records, day, missions)
    if due:
        return {**due[0], "queue": len(due), "progress": progress}
    for mission in missions:
        if mission["id"] not in passed:
            tries = sum(1 for r in records if r["task"] == mission["id"])
            return {"type": "mission", **mission, "previous_attempts": tries, "progress": progress}
    return {"type": "finished", "progress": progress}


def describe(step: dict) -> str:
    done = f"Пройдено миссий: {step['progress']['passed']} из {step['progress']['total']}."
    if step["progress"].get("credited"):
        done = done[:-1] + f", из них зачтено по входу: {step['progress']['credited']}."
    if step["type"] == "review":
        late = f", просрочена на {step['overdue_days']} дн." if step["overdue_days"] else ", срок сегодня"
        return (f"Отложенная проверка {step['task']} (урок {step['lesson']}{late}; в очереди {step['queue']}).\n"
                f"Источник задачи: {step['file']} § {step['section']}, вариант C.\n{done}")
    if step["type"] == "finished":
        return f"Все миссии маршрута пройдены. {done}"
    where = f"{step['file']} § {step['section']}" if step.get("section") else step["file"]
    lines = [f"Миссия {step['id']} ({step['kind']}) — уровень {step['level']} «{step['level_title']}».",
             f"Источник: {where}."]
    if step.get("lab"):
        lines.append(f"Лабораторная: {step['lab']}.")
    if step.get("command"):
        lines.append(f"Команда: {step['command']}")
    if step.get("title"):
        lines.append(f"Проект: {step['title']}.")
    if step.get("result"):
        lines.append(f"Результат занятия: {step['result']}.")
    if step.get("check"):
        lines.append(f"Проект пройден, когда {step['check']} зелёный: tools/progress.py check {step['id']} --min … --self …")
    if step.get("planned"):
        lines.append(f"Материал ещё не написан (задача {step['planned']}): сообщи ученику и не придумывай проект сам.")
    if step["previous_attempts"]:
        lines.append(f"Прежних попыток этой миссии: {step['previous_attempts']}.")
    lines.append(done)
    return "\n".join(lines)


def checked_projects(root: Path = ROOT) -> dict[str, str]:
    """Mini-project id -> its check.py, for projects that route 2.0 splits into sessions."""
    return {m["id"]: m["check"] for m in route_missions(root) if m.get("check")} if (root / ROUTE).is_file() else {}


def run_check(root: Path, args) -> int:
    """Run the project's check.py, print its report and append the attempt; only green passes the project."""
    sys.stdout.reconfigure(encoding="utf-8")
    script = checked_projects(root).get(args.task)
    if not script:
        raise SystemExit(f"{args.task} has no check.py in {ROUTE}")
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}  # the report is read as UTF-8 whatever the console code page
    done = subprocess.run([sys.executable, str(root / script)], cwd=root, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", env=env)
    print(done.stdout.rstrip())
    green = done.returncode == 0
    lines = done.stdout.strip().splitlines()
    record = make_record(args.task, args.date, args.minutes, args.state, args.help_level,
                         note=args.note or (lines[-1] if lines else ""), source="check.py")
    record["check"] = "green" if green else "red"
    append(root, [record])
    print(json.dumps(record, ensure_ascii=False))
    return 0 if green else 1


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
    step = commands.add_parser("next", help="Tutor: overdue delayed check or the next mission of route 2.0")
    step.add_argument("--date", type=date.fromisoformat, default=date.today())
    step.add_argument("--json", action="store_true")
    run = commands.add_parser("check", help="Run a mini-project's check.py and record the attempt with its result",
                              add_help=False)
    run.add_argument("task")
    run.add_argument("--date", type=date.fromisoformat, default=date.today())
    run.add_argument("--min", dest="minutes", type=int, required=True, help="Actual minutes of the final session")
    run.add_argument("--self", dest="state", required=True, help="the tutor's rating after the why-questions, 1-4")
    run.add_argument("--help-level", "--help", dest="help_level", default="none", choices=sorted(HELP))
    run.add_argument("--note", default="")
    board = commands.add_parser("dashboard", help="Write progress/dashboard.md")
    board.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "add" and args.task in checked_projects(ROOT):
        raise SystemExit(f"{args.task} is passed by its check.py, not by one record: "
                         f"tools/progress.py check {args.task} --min ... --self ...")
    if args.command == "check":
        return run_check(ROOT, args)
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
    elif args.command == "next":
        result = next_step(ROOT, args.date)
        sys.stdout.reconfigure(encoding="utf-8")  # read by /study through a pipe, whatever the console code page
        print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else describe(result))
    else:
        markdown = render_dashboard(read_log(ROOT))
        target = ROOT / DASHBOARD
        if args.check:
            if not target.is_file() or target.read_text(encoding="utf-8") != markdown:
                raise SystemExit("Dashboard is stale. Run: python tools/progress.py dashboard")
        else:
            target.write_text(markdown, encoding="utf-8", newline="\n")
            # The progress map on course/START.md comes from the same journal (R05).
            if (ROOT / ROUTE).is_file():
                spec = importlib.util.spec_from_file_location("course_progress_map", Path(__file__).with_name("progress_map.py"))
                progress_map = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(progress_map)
                progress_map.write_start(ROOT)
        print(json.dumps({"dashboard": DASHBOARD, "attempts": len(read_log(ROOT))}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
