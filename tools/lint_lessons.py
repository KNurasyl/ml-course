"""Check every lesson against the 1.2.0 lesson template (improvement plan, section 3.3).

Each lesson has the canonical second-level headings below, in this order. A
heading may add a clarification after " — " ("Самостоятельная задача A —
M02-L01-A01"), and one heading may cover two sections joined by "; " when their
text cannot be separated without editing it. The bold **Результат:** paragraph
before the first heading also counts as "Результат и входные знания": the
reader shows it as the lesson's result. Documented gaps live in
metadata/lesson-template-exceptions.yaml.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
EXCEPTIONS = "metadata/lesson-template-exceptions.yaml"
CANONICAL = [
    "Результат и входные знания",
    "Объяснение",
    "Разобранный пример",
    "Практика с поддержкой",
    "Самостоятельная задача A",
    "Подсказки по очереди",
    "Критерии и типичные ошибки",
    "Новая и отложенная проверка",
    "Источники и углубление",
]
# Hints and criteria share a rank: some lessons state criteria right after the task.
RANK = {"Результат и входные знания": 0, "Объяснение": 1, "Разобранный пример": 2, "Практика с поддержкой": 3,
        "Самостоятельная задача A": 4, "Подсказки по очереди": 5, "Критерии и типичные ошибки": 5,
        "Новая и отложенная проверка": 6, "Источники и углубление": 7}
OPTIONAL = {"Проверь себя сначала", "Аналогия", "Схема", "Карточка урока"}
RESULT_BLOCK = re.compile(r"(?m)^\*\*Результат:?\*\*")
# T27: practice in a lesson with code starts in the "predict, then run" mode (ICAP, constructive).
PREDICT = "**Предскажи до запуска.**"
PRACTICE = re.compile(r"(?ms)^## Практика с поддержкой\n(.*?)(?=^## |\Z)")
# T20: an analogy sits between the explanation and the worked example and always says where it breaks.
ANALOGY = re.compile(r"(?ms)^## (Аналогия(?::[^\n]*)?)\n(.*?)(?=^## |\Z)")
ANALOGY_PARTS = ("| В уроке | В аналогии |", "**Где аналогия ломается:**")
# T22: the lesson card has four fields and stands after the delayed check, before the sources.
CARD = re.compile(r"(?ms)^## Карточка урока\n(.*?)(?=^## |\Z)")
CARD_FIELDS = ("Правило", "Смысл величин", "Когда применять", "Главная ошибка")
# T21: a "Схема" section holds one Mermaid diagram after the worked example and before practice,
# with at most nine blocks, a highlighted main path (classDef main) and a caption next to it (Mayer).
SCHEMA = re.compile(r"(?ms)^## Схема\n(.*?)(?=^## |\Z)")
MERMAID = re.compile(r"(?ms)^```mermaid\n(.*?)^```")
MAX_BLOCKS = 9


def diagram_blocks(source: str) -> int:
    """Approximate node count of a flowchart, or participant count of a sequence diagram."""
    lines = [line.strip() for line in source.splitlines()[1:] if line.strip() and not line.strip().startswith("%%")]
    if source.lstrip().startswith("sequenceDiagram"):
        return len({line.split()[1] for line in lines if line.split()[0] in ("participant", "actor")})
    nodes = set()
    for line in lines:
        if line.split()[0] in ("classDef", "class", "style", "linkStyle", "subgraph", "end", "direction"):
            continue
        line = re.sub(r"\|[^|]*\|", " ", line)  # Edge labels are not nodes.
        line = re.sub(r'"[^"]*"', "", line)
        nodes.update(re.findall(r"(?:^|[\s&>])([A-Za-z_]\w*)(?=\s*(?:[\[({>]|-|=|&|$))", line))
    return len(nodes)


# T23: at least two distinct external sources in the sources section.
SOURCES = re.compile(r"(?ms)^## Источники и углубление\n(.*?)(?=^## |\Z)")


def canonical_parts(heading: str) -> list[str]:
    """Canonical names a heading covers; an empty list means an authored heading."""
    parts = []
    for part in heading.split("; "):
        name = next((n for n in CANONICAL if part.lower() == n.lower() or part.lower().startswith(n.lower() + " — ")), None)
        if name is None:
            return []
        parts.append(name)
    return parts


def load_template(root: Path) -> dict:
    import yaml

    path = root / EXCEPTIONS
    return (yaml.safe_load(path.read_text(encoding="utf-8")) or {}) if path.is_file() else {}


def external_links(section: str) -> set[str]:
    return set(re.findall(r"\]\((https?://[^)\s]+)\)", section))


def lint(root: Path = ROOT) -> dict:
    template = load_template(root)
    exceptions, rollout = template.get("lessons", {}), template.get("rollout", {})
    errors = []
    lessons = sorted((root / "course/lessons").glob("*.md"))
    for lesson_id in sorted(set(exceptions) - {p.stem for p in lessons}):
        errors.append(f"{lesson_id}: exception for a lesson that does not exist")
    for lesson_id, allowed in sorted(exceptions.items()):
        if not str(allowed.get("reason", "")).strip():
            errors.append(f"{lesson_id}: exception needs a reason")
    for path in lessons:
        lesson_id, text = path.stem, path.read_text(encoding="utf-8")
        allowed = exceptions.get(lesson_id, {})
        first_heading = re.search(r"(?m)^## ", text)
        found = []
        if RESULT_BLOCK.search(text[:first_heading.start() if first_heading else len(text)]):
            found.append("Результат и входные знания")
        for heading in re.findall(r"(?m)^## (.+?)\s*$", text):
            parts = canonical_parts(heading)
            if not parts and heading not in OPTIONAL and heading.lower().startswith(tuple(n.lower() for n in CANONICAL)):
                errors.append(f"{lesson_id}: heading '{heading}' looks canonical but is not; use the exact name or add ' — ...'")
            found.extend(parts)
        for name in CANONICAL:
            count = found.count(name)
            if count > 1 and not (name == "Результат и входные знания" and count == 2):
                errors.append(f"{lesson_id}: section '{name}' appears {count} times")
            if count == 0 and name not in allowed.get("missing", []):
                errors.append(f"{lesson_id}: missing section '{name}'")
            if count and name in allowed.get("missing", []):
                errors.append(f"{lesson_id}: exception for '{name}' is stale, the section exists")
        for analogy in ANALOGY.finditer(text):
            if not all(part in analogy.group(2) for part in ANALOGY_PARTS):
                errors.append(f"{lesson_id}: analogy needs the mapping table and '**Где аналогия ломается:**'")
            before = [name for heading in re.findall(r"(?m)^## (.+?)\s*$", text[:analogy.start()])
                      for name in canonical_parts(heading)]
            if "Объяснение" not in before or "Разобранный пример" in before:
                errors.append(f"{lesson_id}: analogy must stand after the explanation and before the worked example")
        card = CARD.search(text)
        if card:
            rows = {line.strip("| ").split(" | ")[0] for line in card.group(1).splitlines() if line.startswith("| ")}
            if not set(CARD_FIELDS) <= rows:
                errors.append(f"{lesson_id}: lesson card needs the rows {', '.join(CARD_FIELDS)}")
            before = [name for heading in re.findall(r"(?m)^## (.+?)\s*$", text[:card.start()])
                      for name in canonical_parts(heading)]
            if "Новая и отложенная проверка" not in before or "Источники и углубление" in before:
                errors.append(f"{lesson_id}: lesson card must stand after the delayed check and before the sources")
        for schema in SCHEMA.finditer(text):
            diagram = MERMAID.search(schema.group(1))
            caption = MERMAID.sub("", schema.group(1)).strip()
            if not diagram:
                errors.append(f"{lesson_id}: schema section needs a mermaid diagram")
            else:
                if diagram_blocks(diagram.group(1)) > MAX_BLOCKS:
                    errors.append(f"{lesson_id}: schema has more than {MAX_BLOCKS} blocks")
                if "classDef main" not in diagram.group(1) or not re.search(r"(?m)(:::main\b|^\s*class .*\bmain\s*$)", diagram.group(1)):
                    errors.append(f"{lesson_id}: schema must highlight the main path with classDef main")
            if not caption:
                errors.append(f"{lesson_id}: schema needs a caption next to the diagram")
            before = [name for heading in re.findall(r"(?m)^## (.+?)\s*$", text[:schema.start()])
                      for name in canonical_parts(heading)]
            if "Разобранный пример" not in before or "Практика с поддержкой" in before:
                errors.append(f"{lesson_id}: schema must stand after the worked example and before practice")
        module = lesson_id[:3]
        if module in rollout.get("analogy", []) and not ANALOGY.search(text):
            errors.append(f"{lesson_id}: {module} is rolled out for T20 but the lesson has no analogy")
        if module in rollout.get("card", []) and not card:
            errors.append(f"{lesson_id}: {module} is rolled out for T22 but the lesson has no card")
        sources = SOURCES.search(text)
        if module in rollout.get("sources", []) and len(external_links(sources.group(1) if sources else "")) < 2:
            errors.append(f"{lesson_id}: {module} is rolled out for T23 but sources have fewer than two external links")
        practice = PRACTICE.search(text)
        # A Mermaid diagram is not code to run; only other fenced blocks call for a prediction.
        runnable = any(lang != "mermaid" for lang in re.findall(r"(?ms)^```(\w*)\n.*?^```", text))
        if runnable and practice and PREDICT not in practice.group(1):
            errors.append(f"{lesson_id}: practice of a lesson with code lacks the '{PREDICT}' line")
        ranks = [RANK[name] for name in dict.fromkeys(found)]
        if ranks != sorted(ranks) and not allowed.get("order_free"):
            errors.append(f"{lesson_id}: canonical sections are out of order: {' > '.join(dict.fromkeys(found))}")
    return {"lessons": len(lessons), "exceptions": sorted(exceptions), "errors": errors}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args(argv)
    result = lint(ROOT)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if result["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
