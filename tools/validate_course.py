"""Validate the curriculum and the offline reading/assignment route."""
from __future__ import annotations
import argparse
from html.parser import HTMLParser
import importlib.util
import json
from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
VERSION = "1.1.0"
MODULE_IDS = [f"M{i:02}" for i in range(17)]
PILOT_IDS = [f"P{i:02}" for i in range(1, 11)]
REQUIRED_MATERIALS = ["HOME", "PROGRESS", "DIAGNOSTIC", "MENTOR", "MENTOR-KEYS",
                      "GLOSSARY", "PROJECT", *MODULE_IDS, *PILOT_IDS]
REMOTE_PAGE = re.compile(r"https?://(?:www\.)?chatgpt\.com/space(?:/|\b)", re.I)


class HtmlLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.references = []
        self.ids = set()
        self.metadata = {}

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "meta" and attrs.get("name"):
            self.metadata[attrs["name"]] = attrs.get("content", "")
        if attrs.get("id"):
            self.ids.add(attrs["id"])
        if tag == "a" and attrs.get("name"):
            self.ids.add(attrs["name"])
        for attr in ("href", "src"):
            if attrs.get(attr):
                self.references.append((tag, attr, attrs[attr]))


def read_json(path, errors):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        errors.append(f"Cannot read {path.name}: {error}")
        return None


def reader_fingerprint(root):
    spec = importlib.util.spec_from_file_location("course_reader_fingerprint", Path(__file__).with_name("build_reader.py"))
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)
    return reader.source_fingerprint(root)


def validate(root: Path = ROOT, require_local: bool = False) -> dict:
    root = root.resolve()
    errors = []
    curriculum = read_json(root / "curriculum.json", errors)
    if curriculum is None:
        return {"modules": 0, "lessons": 0, "errors": errors, "learner_mastery": "not_assessed"}
    modules = curriculum["modules"]
    ids = [m["id"] for m in modules]
    if ids != MODULE_IDS:
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
    for session_id, dependencies in curriculum.get("pilot_session_prerequisites", {}).items():
        if session_id not in PILOT_IDS:
            errors.append(f"Unknown pilot session: {session_id}")
            continue
        for dep in dependencies:
            if dep not in PILOT_IDS or PILOT_IDS.index(dep) >= PILOT_IDS.index(session_id):
                errors.append(f"Invalid pilot dependency: {session_id} -> {dep}")

    documents = {}
    for number in range(1, 11):
        for folder in ("notebooks/pilot", "solutions/pilot"):
            p = root / folder / f"P{number:02}.ipynb"
            nb = read_json(p, errors)
            if nb is None:
                continue
            if not any(c["cell_type"] == "code" for c in nb["cells"]):
                errors.append(f"No executable example: {p.relative_to(root)}")
            documents[p] = "\n\n".join(
                "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
                for c in nb["cells"] if c["cell_type"] == "markdown")

    if require_local:
        _validate_local(root, curriculum, documents, errors)
    return {"modules": len(modules), "lessons": len(lesson_ids), "errors": errors,
            "learner_mastery": "not_assessed"}


def _validate_local(root, curriculum, documents, errors):
    import mistune

    registry = read_json(root / "metadata/material-registry.json", errors)
    manifest = read_json(root / "course-manifest.json", errors)
    if not isinstance(registry, dict):
        errors.append("Missing local material registry")
        registry = {}
    if not isinstance(manifest, dict):
        errors.append("Missing local course manifest")
        manifest = {}
    if manifest.get("course_version") != VERSION:
        errors.append(f"Manifest version must be {VERSION}")
    if manifest.get("canonical_content") != "local_markdown":
        errors.append("Manifest must use local Markdown as canonical content")
    if manifest.get("full_course_ready") is not False:
        errors.append("Learner-calibrated release requires real pilot evidence; materials alone are insufficient")
    if manifest.get("progress_source") != "progress/progress.md":
        errors.append("Manifest must point at the local progress log")
    if REMOTE_PAGE.search(json.dumps(manifest)):
        errors.append("Remote Page dependency in course-manifest.json")
    if manifest.get("pages") or manifest.get("publication"):
        errors.append("Manifest still uses remote publication metadata")
    sessions = manifest.get("pilot", {}).get("sessions", [])
    if [s.get("id") for s in sessions] != PILOT_IDS:
        errors.append("Manifest must contain P01..P10 exactly once in order")
    mapped_lessons = manifest.get("lessons", [])
    expected_lessons = {lesson["id"] for m in curriculum["modules"] for lesson in m["lessons"]}
    if {lesson.get("id") for lesson in mapped_lessons} != expected_lessons:
        errors.append("Manifest lesson mapping differs from the curriculum")

    def local_file(relative, label):
        if not isinstance(relative, str) or not relative or "://" in relative:
            errors.append(f"Missing local path: {label}")
            return None
        target = (root / relative).resolve()
        if not target.is_relative_to(root):
            errors.append(f"Path leaves course repository: {label}")
            return None
        if not target.is_file() or not target.stat().st_size:
            errors.append(f"Missing or empty local file: {relative} ({label})")
            return None
        return target

    for key in dict.fromkeys([*REQUIRED_MATERIALS, *registry]):
        item = registry.get(key, {})
        if item.get("id") != key or not item.get("title") or not item.get("status"):
            errors.append(f"Incomplete material registry entry: {key}")
        if item.get("version") != VERSION:
            errors.append(f"Wrong material version: {key}")
        target = local_file(item.get("path"), key)
        mapped = manifest.get("materials", {}).get(key, {})
        if mapped.get("path") != item.get("path") or mapped.get("version") != item.get("version"):
            errors.append(f"Manifest material mismatch: {key}")
        if target:
            content = target.read_text(encoding="utf-8")
            documents[target] = content
            if key in PILOT_IDS and f"{key}-HW01" not in content:
                errors.append(f"Missing independent assignment {key}-HW01")
    for session in sessions:
        for field in ("path", "starter", "solution"):
            local_file(session.get(field), f"{session.get('id')} {field}")
        if session.get("path") != registry.get(session.get("id"), {}).get("path"):
            errors.append(f"Pilot material mapping mismatch: {session.get('id')}")
    if any(key in registry for key in expected_lessons):
        for module in curriculum["modules"]:
            mid = module["id"]
            for lesson in module["lessons"]:
                lid = lesson["id"]
                path = local_file(registry.get(lid, {}).get("path"), lid)
                if path:
                    content = path.read_text(encoding="utf-8")
                    if len(content) < 2500 or "![" not in content or not re.search("самостоятельн", content, re.I):
                        errors.append(f"Lesson package lacks explanation, visual or independent task: {lid}")
            local_file(registry.get(f"{mid}-SOLUTIONS", {}).get("path"), f"{mid} solutions")
            for relative in (f"labs/{mid}/demo.py", f"labs/{mid}/README.md", f"course/visuals/{mid}.html"):
                local_file(relative, mid)
        if not manifest.get("full_materials_ready"):
            errors.append("Full local route is registered but its mandatory packages are incomplete")
    for name in ("README.md", "AGENTS.md"):
        path = local_file(name, name)
        if path:
            documents[path] = path.read_text(encoding="utf-8")
    book = local_file("START_HERE.html", "offline reader")
    if book:
        documents[book] = book.read_text(encoding="utf-8")

    html_cache = {}

    def parse_html(path, content=None):
        if path not in html_cache:
            parser = HtmlLinks()
            parser.feed(content if content is not None else path.read_text(encoding="utf-8"))
            html_cache[path] = parser
        return html_cache[path]

    def check_reference(source, reference, tag="a", attr="href"):
        if REMOTE_PAGE.search(reference):
            return  # A single content-level error is already reported below.
        parsed = urlsplit(reference)
        if parsed.scheme in ("http", "https", "mailto"):
            if tag in ("script", "link", "img", "iframe") or attr == "src":
                errors.append(f"Online asset in offline reader: {reference}")
            return
        if parsed.scheme == "data" and tag == "img":
            return
        if parsed.scheme:
            errors.append(f"Unsupported link scheme in {source.relative_to(root)}: {reference}")
            return
        path_part = unquote(parsed.path)
        target = (source.parent / path_part).resolve() if path_part else source
        if not target.is_relative_to(root):
            errors.append(f"Link leaves repository in {source.relative_to(root)}: {reference}")
        elif not target.exists():
            errors.append(f"Broken local link in {source.relative_to(root)}: {reference}")
        elif parsed.fragment and target.suffix.lower() in (".html", ".htm"):
            if unquote(parsed.fragment) not in parse_html(target).ids:
                errors.append(f"Missing HTML anchor in {source.relative_to(root)}: {reference}")

    markdown = mistune.create_markdown(renderer="ast", plugins=["table"])

    def walk_tokens(source, tokens):
        for token in tokens:
            if token["type"] in ("link", "image"):
                check_reference(source, token["attrs"]["url"], "img" if token["type"] == "image" else "a")
            if token["type"] in ("inline_html", "block_html"):
                parser = HtmlLinks()
                parser.feed(token.get("raw", ""))
                for tag, attr, reference in parser.references:
                    check_reference(source, reference, tag, attr)
            walk_tokens(source, token.get("children", []))

    for source, content in documents.items():
        if REMOTE_PAGE.search(content):
            errors.append(f"Remote Page dependency in {source.relative_to(root)}")
        if source.suffix.lower() in (".html", ".htm"):
            for tag, attr, reference in parse_html(source, content).references:
                check_reference(source, reference, tag, attr)
        else:
            walk_tokens(source, markdown(content))
    if book:
        try:
            expected = reader_fingerprint(root)
        except (OSError, ValueError, KeyError) as error:
            errors.append(f"Cannot verify reader sources: {error}")
        else:
            observed = parse_html(book).metadata.get("course-source-sha256")
            if observed != expected:
                errors.append("Offline reader is stale: rebuild with python tools/build_reader.py")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-local", action="store_true",
                        help="Also validate the complete offline reading and assignment route")
    args = parser.parse_args()
    result = validate(require_local=args.require_local)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return bool(result["errors"])


if __name__ == "__main__":
    sys.exit(main())
