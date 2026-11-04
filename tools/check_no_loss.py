"""Snapshot course content and fail when a later state loses any of it.

snapshot: record IDs (modules, lessons, pilot sessions, homework), task counts
A/B/C, per-file word counts of course and solution Markdown, file lists of the
content trees and the number of collected tests.
compare: exit 1 when an ID, a file, a count or a per-file metric shrinks.

Renaming headings is allowed by the improvement plan, so a lower total word
count with unchanged body text is reported as a warning, not as a loss.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 1
TREES = ("course", "solutions", "exercises", "labs", "notebooks", "projects", "src",
         "tests", "data", "metadata", "progress", "quality", "archive")
MARKDOWN_TREES = ("course", "solutions")
SKIP_DIRS = {"__pycache__", ".ipynb_checkpoints", ".pytest_cache"}
# The guard's own snapshots are working data, not course content.
SKIP_PREFIXES = ("quality/snapshots/",)
IMAGE_SUFFIXES = {".png", ".svg", ".jpg", ".jpeg", ".gif", ".webp"}
WORD = re.compile(r"\w+")
HEADING = re.compile(r"(?m)^#{1,6} .*$")
SECTION = re.compile(r"(?m)^#{2,6} ")
TASK_A = re.compile(r"(?m)^#{2,6} .*[Сс]амостоятельн")
# B/C variants are written in several styles: "Вариант B", "**M07-L01-B**",
# "**C — через неделю**", "Повтор B:", "Отложенная задача C", "C через", "… B: условие".
TASK_BC = re.compile(
    r"(?:(?:[Вв]ариант|[Пп]овтор|[Оо]тложенн\w*(?:\s+задача)?|[Чч]ерез неделю)\s*\**\s*"
    r"(?:[MP]\d{2}(?:-L\d{2}|-HW\d{2})?-)?(?P<a>[BC])\b)"
    r"|(?m:^\**(?P<b>[BC])\s*(?:—|:))"
    r"|(?:(?<![\w,\[/])(?P<c>[BC])(?: через\b|:\s))")
IMAGE_REF = re.compile(r"!\[[^\]]*\]\(")
MERMAID = re.compile(r"(?m)^```mermaid\b")
# Total words may fall when a heading is renamed; body text must not.
SOFT_METRICS = {"words"}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def content_files(root: Path, tree: str) -> list[str]:
    """Files a learner or author owns; generated caches and lab outputs are skipped."""
    base = root / tree
    if not base.is_dir():
        return []
    found = []
    for path in base.rglob("*"):
        relative = path.relative_to(root)
        if (not path.is_file() or SKIP_DIRS.intersection(relative.parts) or path.suffix == ".pyc"
                or relative.as_posix().startswith(SKIP_PREFIXES)):
            continue
        if relative.parts[0] == "labs" and len(relative.parts) > 2 and relative.parts[2] == "output":
            continue
        found.append(relative.as_posix())
    return sorted(found)


def markdown_metrics(text: str) -> dict:
    body = HEADING.sub("", text)
    # Markers are counted in body text only, so a renamed heading cannot hide or fake a task.
    tasks = [m.group("a") or m.group("b") or m.group("c") for m in TASK_BC.finditer(body)]
    return {
        "words": len(WORD.findall(text)),
        "body_words": len(WORD.findall(body)),
        "headings": len(SECTION.findall(text)),
        "tasks_a": len(TASK_A.findall(text)),
        "tasks_b": tasks.count("B"),
        "tasks_c": tasks.count("C"),
        "images": len(IMAGE_REF.findall(text)),
        "mermaid": len(MERMAID.findall(text)),
    }


def notebook_metrics(path: Path) -> dict:
    cells = read_json(path).get("cells", [])
    source = lambda c: "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
    return {"cells": len(cells),
            "code_cells": sum(c["cell_type"] == "code" for c in cells),
            "markdown_words": sum(len(WORD.findall(source(c))) for c in cells if c["cell_type"] == "markdown")}


def collected_tests(root: Path) -> int:
    result = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"],
                            cwd=root, capture_output=True, text=True, encoding="utf-8", errors="replace")
    match = re.search(r"(\d+) tests? collected", result.stdout)
    if result.returncode != 0 or not match:
        raise RuntimeError(f"pytest --collect-only failed:\n{result.stdout[-2000:]}{result.stderr[-2000:]}")
    return int(match.group(1))


def git_commit(root: Path) -> str | None:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def course_version(root: Path) -> str | None:
    pyproject = root / "pyproject.toml"
    match = re.search(r'(?m)^version = "([^"]+)"', pyproject.read_text(encoding="utf-8")) if pyproject.is_file() else None
    return match.group(1) if match else None


def snapshot(root: Path = ROOT, include_tests: bool = True) -> dict:
    root = root.resolve()
    curriculum = read_json(root / "curriculum.json") if (root / "curriculum.json").is_file() else {"modules": []}
    modules = curriculum.get("modules", [])
    trees = {tree: content_files(root, tree) for tree in TREES}
    files = {}
    for tree in MARKDOWN_TREES:
        for relative in trees[tree]:
            if relative.endswith(".md"):
                files[relative] = markdown_metrics((root / relative).read_text(encoding="utf-8"))
    notebooks = [p for tree in ("notebooks", "solutions") for p in trees[tree] if p.endswith(".ipynb")]
    for relative in notebooks:
        files[relative] = notebook_metrics(root / relative)

    def total(prefix: str, metric: str) -> int:
        return sum(m.get(metric, 0) for path, m in files.items() if path.startswith(prefix))

    def matching(tree: str, pattern: str) -> int:
        return sum(bool(re.fullmatch(pattern, path)) for path in trees[tree])

    ids = {
        "modules": [m["id"] for m in modules],
        "lessons": [lesson["id"] for m in modules for lesson in m.get("lessons", [])],
        "homework": [m["homework"]["id"] for m in modules if m.get("homework")],
        "pilot_sessions": sorted(Path(p).stem for p in trees["course"] if re.fullmatch(r"course/pilot/P\d{2}\.md", p)),
    }
    counts = {
        **{key: len(value) for key, value in ids.items()},
        "lesson_files": matching("course", r"course/lessons/[^/]+\.md"),
        "module_pages": matching("course", r"course/modules/M\d{2}\.md"),
        "course_markdown_files": sum(p.endswith(".md") for p in trees["course"]),
        "course_body_words": total("course/", "body_words"),
        "solution_body_words": total("solutions/", "body_words"),
        "tasks_a": total("course/", "tasks_a"),
        "tasks_b": total("course/", "tasks_b"),
        "tasks_c": total("course/", "tasks_c"),
        "solution_files": len(trees["solutions"]),
        "solution_sections": total("solutions/", "headings"),
        "lab_files": len(trees["labs"]),
        "lab_demos": matching("labs", r"labs/M\d{2}/demo\.py"),
        "exercise_files": len(trees["exercises"]),
        "exercise_starters": matching("exercises", r"exercises/M\d{2}/starter\.py"),
        "exercise_checkers": matching("exercises", r"exercises/M\d{2}/check\.py"),
        "exercise_solutions": matching("solutions", r"solutions/exercises/M\d{2}/solution\.py"),
        "notebooks": len(notebooks),
        "test_files": matching("tests", r"tests/test_[^/]+\.py"),
        "figure_files": sum(p.startswith("course/assets/") and Path(p).suffix.lower() in IMAGE_SUFFIXES
                            for p in trees["course"]),
        "figure_refs": total("course/", "images"),
        "mermaid_blocks": total("course/", "mermaid"),
        "interactives": matching("course", r"course/visuals/[^/]+\.html"),
        "quality_files": len(trees["quality"]),
    }
    if include_tests:
        counts["tests"] = collected_tests(root)
    return {
        "schema": SCHEMA, "tool": "tools/check_no_loss.py",
        "created": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "course_version": course_version(root), "git_commit": git_commit(root),
        "tests_collected": include_tests,
        "counts": counts, "ids": ids,
        "trees": trees, "files": files,
    }


def compare(before: dict, after: dict) -> dict:
    losses, warnings, skipped = [], [], []
    for key, old in before["counts"].items():
        new = after["counts"].get(key)
        if new is None:
            if key == "tests" and not after.get("tests_collected", True):
                skipped.append("tests: not collected in the later snapshot")
            else:
                losses.append(f"count {key}: missing (was {old})")
        elif new < old:
            losses.append(f"count {key}: {old} -> {new}")
    for key, old_ids in before["ids"].items():
        gone = [item for item in old_ids if item not in set(after["ids"].get(key, []))]
        if gone:
            losses.append(f"ids {key}: removed {', '.join(gone)}")
    for tree, old_files in before["trees"].items():
        current = set(after["trees"].get(tree, []))
        gone = [path for path in old_files if path not in current]
        if gone:
            losses.append(f"files in {tree}/: removed {', '.join(gone)}")
    for path, old_metrics in before["files"].items():
        new_metrics = after["files"].get(path)
        if new_metrics is None:
            continue  # Already reported as a removed file.
        for metric, old in old_metrics.items():
            new = new_metrics.get(metric, 0)
            if new < old:
                message = f"{path}: {metric} {old} -> {new}"
                (warnings if metric in SOFT_METRICS else losses).append(message)
    growth = {key: after["counts"][key] - old for key, old in before["counts"].items()
              if key in after["counts"] and after["counts"][key] != old}
    return {"losses": losses, "warnings": warnings, "skipped": skipped, "growth": growth,
            "status": "lost_content" if losses else "no_loss"}


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("snapshot", help="Record the current content state")
    make.add_argument("--out", type=Path, required=True)
    make.add_argument("--no-tests", action="store_true", help="Skip pytest --collect-only")
    check = commands.add_parser("compare", help="Fail if content shrank since an earlier snapshot")
    check.add_argument("--before", type=Path, required=True)
    check.add_argument("--after", type=Path, help="Later snapshot; defaults to the current working tree")
    check.add_argument("--no-tests", action="store_true", help="Skip pytest --collect-only for the current tree")
    args = parser.parse_args(argv)
    if args.command == "snapshot":
        state = snapshot(ROOT, include_tests=not args.no_tests)
        write_json(args.out, state)
        print(json.dumps({"written": args.out.as_posix(), "counts": state["counts"]}, ensure_ascii=False, indent=2))
        return 0
    before = read_json(args.before)
    after = read_json(args.after) if args.after else snapshot(ROOT, include_tests=not args.no_tests)
    result = compare(before, after)
    print(json.dumps({"before": args.before.as_posix(), **result}, ensure_ascii=False, indent=2))
    return 1 if result["losses"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
