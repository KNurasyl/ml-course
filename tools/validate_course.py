"""Check curriculum integrity and links to local, actually present course assets."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def validate(root: Path = ROOT, require_pages: bool = False) -> dict:
    errors = []
    curriculum = json.loads((root / "curriculum.json").read_text(encoding="utf-8"))
    modules = curriculum["modules"]
    ids = [m["id"] for m in modules]
    if ids != [f"M{i:02}" for i in range(17)]:
        errors.append("Expected unique ordered modules M00..M16")
    lesson_ids = [lesson["id"] for m in modules for lesson in m["lessons"]]
    if len(set(lesson_ids)) != len(lesson_ids):
        errors.append("Duplicate lesson IDs")
    graph = {}
    for m in modules:
        for dep in m["prerequisites"]:
            if dep not in ids or ids.index(dep) >= ids.index(m["id"]):
                errors.append(f"Invalid module dependency: {m['id']} -> {dep}")
        outcomes = {o["id"] for o in m["outcomes"]}
        coverage = set()
        for lesson in m["lessons"]:
            graph[lesson["id"]] = lesson["prerequisites"]
            coverage.update(lesson["outcome_ids"])
            if not set(lesson["outcome_ids"]).issubset(outcomes):
                errors.append(f"Unknown outcome in {lesson['id']}")
            for field in ("practice", "assessment", "title"):
                if not lesson.get(field):
                    errors.append(f"Missing {field}: {lesson['id']}")
        if outcomes - coverage:
            errors.append(f"Outcomes without lessons: {m['id']}")
        for field in ("rubric", "critical_errors", "retry", "delayed_check"):
            if not m["homework"].get(field):
                errors.append(f"Missing homework {field}: {m['id']}")
    visited, active = set(), set()

    def walk(node):
        if node in active:
            errors.append(f"Dependency cycle at {node}")
            return
        if node in visited:
            return
        active.add(node)
        for dep in graph.get(node, []):
            if dep not in graph:
                errors.append(f"Unknown lesson dependency {dep}")
            else:
                walk(dep)
        active.remove(node)
        visited.add(node)

    for node in graph:
        walk(node)
    for number in range(1, 11):
        for folder in ("notebooks/pilot", "solutions/pilot"):
            p = root / folder / f"P{number:02}.ipynb"
            if not p.is_file():
                errors.append(f"Missing notebook: {p.relative_to(root)}")
            else:
                nb = json.loads(p.read_text(encoding="utf-8"))
                if not any(c["cell_type"] == "code" for c in nb["cells"]):
                    errors.append(f"No executable example: {p.relative_to(root)}")
    manifest_path = root / "course-manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["pilot"]["learner_status"] != "awaiting_learner":
            errors.append("Initial release must not claim the learner completed the pilot")
        for item in manifest.get("pilot", {}).get("sessions", []):
            for key in ("starter", "solution"):
                if not (root / item[key]).is_file():
                    errors.append(f"Missing mapped {key}: {item['id']}")
        if require_pages:
            required = ["HOME", "PROGRESS", "DIAGNOSTIC", "MENTOR", "MENTOR-KEYS", "GLOSSARY", "PROJECT", *ids,
                        *[f"P{i:02}" for i in range(1, 11)]]
            for key in required:
                value = manifest.get("pages", {}).get(key, {})
                if not value.get("url", "").startswith("https://chatgpt.com/space/page_"):
                    errors.append(f"Missing real Page URL: {key}")
                if value.get("readback_verified") is not True:
                    errors.append(f"Page not read back: {key}")
    elif require_pages:
        errors.append("Missing course-manifest.json")
    result = {"modules": len(modules), "lessons": len(lesson_ids), "errors": errors,
              "learner_mastery": "not_assessed"}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-pages", action="store_true")
    args = parser.parse_args()
    result = validate(require_pages=args.require_pages)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return bool(result["errors"])


if __name__ == "__main__":
    sys.exit(main())
