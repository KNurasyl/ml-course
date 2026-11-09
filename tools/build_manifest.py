"""Build local course navigation without changing the learner's progress log.

curriculum.json is the canonical source of modules, lessons and their keys; the
registry mirrors those entries (--sync-registry copies them) and adds the other
materials. The manifest is generated only and must not be edited by hand.
"""
from __future__ import annotations
import argparse
import json
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = "1.2.0"
CREATED = "2026-11-03"
PILOT_IDS = [f"P{i:02}" for i in range(1, 11)]
MODULE_IDS = [f"M{i:02}" for i in range(17)]
REQUIRED_MATERIALS = ["HOME", "PROGRESS", "DIAGNOSTIC", "MENTOR", "MENTOR-KEYS",
                      "GLOSSARY", "PROJECT", *MODULE_IDS, *PILOT_IDS]


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def curriculum_entries(curriculum: dict) -> dict:
    """Registry entries that curriculum.json defines; it is the canonical source for them."""
    entries = {}
    for module in curriculum["modules"]:
        mid = module["id"]
        entries[mid] = {"id": mid, "title": f"{mid} {module['title']}", "path": f"course/modules/{mid}.md",
                        "version": VERSION, "status": "ready"}
        for lesson in module["lessons"]:
            lid = lesson["id"]
            entries[lid] = {"id": lid, "title": f"{lid}. {lesson['title']}", "path": f"course/lessons/{lid}.md",
                            "version": VERSION, "status": "ready"}
        entries[f"{mid}-SOLUTIONS"] = {"id": f"{mid}-SOLUTIONS", "title": f"{mid}. Решения для самопроверки",
                                       "path": f"solutions/modules/{mid}.md", "version": VERSION, "status": "ready"}
    return entries


def registry_drift(registry: dict, curriculum: dict) -> list[str]:
    """Curriculum items whose registry entry differs from what curriculum.json defines."""
    return [key for key, entry in curriculum_entries(curriculum).items() if key in registry and registry[key] != entry]


def extra_exercises(root: Path, module_id: str) -> list[dict]:
    """Extra exercises found on disk: exNN_*.py starter, check_exNN.py, reference and hidden cases."""
    found = []
    for checker in sorted((root / "exercises" / module_id).glob("check_ex*.py")):
        number = checker.stem.removeprefix("check_")
        starter = next((root / "exercises" / module_id).glob(f"{number}_*.py"), None)
        if starter is None:
            continue
        found.append({"id": f"{module_id}-{number}", "starter": f"exercises/{module_id}/{starter.name}",
                      "checker": f"exercises/{module_id}/{checker.name}",
                      "solution": f"solutions/exercises/{module_id}/{starter.name}",
                      "hidden_cases": f"tests/hidden/{module_id}_{number}.json"})
    return found


def sync_registry(root: Path = ROOT) -> list[str]:
    """Rewrite curriculum items in the registry and add those whose file exists; return changed keys."""
    path = root / "metadata" / "material-registry.json"
    registry = read_json(path)
    changed = []
    for key, entry in curriculum_entries(read_json(root / "curriculum.json")).items():
        if registry.get(key) != entry and (key in registry or (root / entry["path"]).is_file()):
            registry[key] = entry
            changed.append(key)
    if changed:
        path.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return changed


def local_asset(root: Path, relative: str) -> str:
    """Require a real nonempty file inside the repository, never a remote URL."""
    if not relative or "://" in relative:
        raise ValueError(f"Expected a local course path: {relative}")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file() or not path.stat().st_size:
        raise ValueError(f"Missing, empty or external course asset: {relative}")
    return path.relative_to(root.resolve()).as_posix()


def build_manifest(root: Path = ROOT) -> dict:
    root = root.resolve()
    curriculum = read_json(root / "curriculum.json")
    registry = read_json(root / "metadata" / "material-registry.json")
    inventory = read_json(root / "metadata" / "pilot-lab-inventory.json")
    if not isinstance(registry, dict):
        raise ValueError("material-registry.json must map course IDs to local materials")
    modules = curriculum["modules"]
    if [module["id"] for module in modules] != MODULE_IDS:
        raise ValueError("Expected ordered modules M00..M16")
    lesson_ids = [lesson["id"] for module in modules for lesson in module["lessons"]]
    full_route = any(key in registry for key in lesson_ids)
    required = [*REQUIRED_MATERIALS, *lesson_ids, *[f"{m}-SOLUTIONS" for m in MODULE_IDS]] if full_route else REQUIRED_MATERIALS
    if len(set(lesson_ids)) != len(lesson_ids):
        raise ValueError("Duplicate curriculum lesson IDs")
    if len(inventory) != 10 or {entry["id"] for entry in inventory} != set(PILOT_IDS):
        raise ValueError("Pilot inventory must contain P01..P10 exactly once")
    pilot_map = curriculum["pilot_lesson_map"]
    if set(pilot_map) != set(PILOT_IDS):
        raise ValueError("Expected a curriculum mapping for every pilot session")
    for session_id, linked_lessons in pilot_map.items():
        if not set(linked_lessons).issubset(lesson_ids):
            raise ValueError(f"Unknown curriculum lesson in {session_id}")

    canonical = curriculum_entries(curriculum)
    materials, missing = {}, []
    for key in dict.fromkeys([*required, *registry]):
        item = registry.get(key)
        if isinstance(item, dict) and key in canonical:
            item = canonical[key]  # Titles and paths of modules, lessons and keys come from curriculum.json.
        if not isinstance(item, dict) or item.get("id") != key:
            missing.append(key)
            continue
        if item.get("version") != VERSION:
            raise ValueError(f"Unexpected material version: {key}")
        try:
            path = local_asset(root, item.get("path", ""))
        except ValueError:
            missing.append(key)
            continue
        materials[key] = {"id": key, "title": item["title"], "path": path,
                          "version": VERSION, "status": item["status"]}
    reader_exists = (root / "START_HERE.html").is_file()
    reader_current = False
    if reader_exists and not missing:
        spec = importlib.util.spec_from_file_location("manifest_reader", Path(__file__).with_name("build_reader.py"))
        renderer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(renderer)
        fingerprint = renderer.source_fingerprint(root)
        reader_current = f'name="course-source-sha256" content="{fingerprint}"' in (root / "START_HERE.html").read_text(encoding="utf-8")
    delivery_ready = not missing and reader_exists and reader_current

    module_entries, lesson_entries = [], []
    for module in modules:
        module_id = module["id"]
        path = materials.get(module_id, {}).get("path")
        lesson_paths = [materials.get(lesson["id"], {}).get("path") for lesson in module["lessons"]]
        solution = materials.get(f"{module_id}-SOLUTIONS", {}).get("path")
        lab = f"labs/{module_id}/demo.py"
        visual = f"course/visuals/{module_id}.html"
        exercise = {"instructions": f"exercises/{module_id}/README.md", "starter": f"exercises/{module_id}/starter.py",
                    "checker": f"exercises/{module_id}/check.py", "solution": f"solutions/exercises/{module_id}/solution.py"}
        module_ready = bool(path and all(lesson_paths) and solution and (root/lab).is_file() and (root/visual).is_file()
                            and (root/f"labs/{module_id}/README.md").is_file() and all((root/p).is_file() for p in exercise.values()))
        module_entries.append({
            "id": module_id, "title": module["title"], "version": VERSION,
            "material_key": module_id, "path": path,
            "delivery_status": "local" if path else "missing",
            "design_status": "ready", "material_status": "ready" if module_ready else "incomplete",
            "full_module_ready": module_ready, "solution": solution, "lab": lab, "visual": visual,
            "lesson_ids": [lesson["id"] for lesson in module["lessons"]],
            "homework_id": module["homework"]["id"],
            "exercise": exercise, "extra_exercises": extra_exercises(root, module_id),
            "assessment_policy": "course/assessment.md",
        })
        for lesson in module["lessons"]:
            lesson_entries.append({
                "id": lesson["id"], "title": lesson["title"], "version": VERSION,
                "module_id": module_id, "material_key": lesson["id"], "path": materials.get(lesson["id"], {}).get("path"),
                "scope": "complete_lesson", "material_status": "ready" if materials.get(lesson["id"]) else "missing",
                "prerequisites": lesson["prerequisites"], "outcome_ids": lesson["outcome_ids"],
                "workload": lesson.get("workload", {}),
                "pilot_sessions": [key for key, value in pilot_map.items() if lesson["id"] in value],
            })

    inventory_by_id = {entry["id"]: entry for entry in inventory}
    sessions = []
    for session_id in PILOT_IDS:
        item = inventory_by_id[session_id]
        path = materials.get(session_id, {}).get("path")
        sessions.append({
            "id": session_id, "version": VERSION, "material_status": "ready" if path else "missing",
            "material_key": session_id, "path": path,
            "starter": local_asset(root, item["notebook"]),
            "solution": local_asset(root, item["solution"]),
            "curriculum_lesson_ids": pilot_map[session_id],
            "prerequisites": curriculum["pilot_session_prerequisites"][session_id],
            "hardware": {"execution": "CPU", "python": "3.11", "gpu_required": False},
            "paid_dependency": False, "internet_after_setup_required": False,
            "optional_source": item["source"],
        })

    names = ["Competency map", "Full curriculum design", "Local course delivery",
             "Diagnostics and pilot materials", "Learner pilot", "Personal calibration",
             "Remaining mandatory materials", "Full course release"]
    full_materials = all(m["full_module_ready"] for m in module_entries)
    delivery_ready = delivery_ready and (full_materials or not full_route)
    return {
        "schema_version": 3, "course_version": VERSION, "reader_design_version": "2.0", "created": CREATED,
        "language": "ru", "course_status": "complete_self_study_materials" if full_materials else "incomplete_materials",
        "full_materials_ready": full_materials,
        "full_course_ready": False,
        "technical_evidence": {"release": "quality/release-1.2.0.json", "previous_release": "quality/release-1.1.0.json",
                               "audit_resolution": "quality/audit-resolution-1.1.0.md",
                               "design_revision": "quality/design-v2/report.json",
                               "policy": "File completeness alone is not verification or learner mastery; inspect the release evidence and its scope."},
        "entrypoint": "START_HERE.html", "canonical_content": "local_markdown",
        "learning_mode": "self_study_with_optional_mentor",
        "progress_source": "progress/progress.md",
        "inputs": {"curriculum": "curriculum.json", "material_registry": "metadata/material-registry.json",
                   "pilot_inventory": "metadata/pilot-lab-inventory.json"},
        "materials": materials,
        "delivery": {"status": "local_ready" if delivery_ready else "incomplete",
                     "required_material_keys": required, "missing_materials": missing,
                     "reader_present": reader_exists, "reader_current": reader_current, "required_online_accounts": False},
        "stages": [{"id": i, "name": name,
                    "status": ("ready" if i <= 2 or (i <= 4 and delivery_ready) else
                               "awaiting_learner" if i in (5,6) else "ready" if i == 7 and full_materials else "awaiting_learner_calibration")}
                   for i, name in enumerate(names, 1)],
        "modules": module_entries, "lessons": lesson_entries,
        "pilot": {"material_status": "ready" if delivery_ready else "incomplete",
                  "learner_status": "see_progress_log",
                  "session_count": len(sessions), "sessions": sessions,
                  "mapping_policy": curriculum["pilot_mapping_policy"],
                  "technical_validation_report": "artifacts/notebook-validation.json"},
        "environment": {"profile": curriculum["profile"], "pilot_execution": "CPU",
                        "required_paid_services": False, "dependency_lock": "requirements.lock.txt"},
        "status_policy": {
            "module_materials": "Full materials are tracked separately from a real learner pilot. full_course_ready includes learner calibration and remains false until evidence exists.",
            "pilot_mapping": "Ready pilot materials do not mark their linked curriculum lessons complete.",
            "progress": "This generated manifest does not change or certify learner progress; use the local log.",
            "mastery": "Self-checks and technical execution are evidence, not automatic professional grading.",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Inspect without replacing the manifest")
    parser.add_argument("--sync-registry", action="store_true",
                        help="First copy module, lesson and key entries from curriculum.json into the registry")
    parser.add_argument("--sync-registry-only", action="store_true",
                        help="Only synchronise the registry with curriculum.json; do not build the manifest")
    args = parser.parse_args()
    if args.sync_registry or args.sync_registry_only:
        print(json.dumps({"registry_synced_from_curriculum": sync_registry()}, ensure_ascii=True))
        if args.sync_registry_only:
            return 0
    manifest = build_manifest()
    if args.check and (not (ROOT / "course-manifest.json").is_file() or read_json(ROOT / "course-manifest.json") != manifest):
        raise SystemExit("Manifest is stale. Run: python tools/build_manifest.py")
    if not args.check:
        (ROOT / "course-manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"written": not args.check, "modules": len(manifest["modules"]),
                      "lessons": len(manifest["lessons"]), "pilot_sessions": len(manifest["pilot"]["sessions"]),
                      "delivery_status": manifest["delivery"]["status"], "missing": manifest["delivery"]["missing_materials"],
                      "full_materials_ready": manifest["full_materials_ready"], "full_course_ready": False}, ensure_ascii=True, indent=2))
    return 0 if manifest["delivery"]["status"] == "local_ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
