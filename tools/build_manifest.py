"""Build navigation metadata without equating course design with mastery.

The Page registry is evidence supplied by the publishing workflow. Missing or
unverified entries are preserved as incomplete delivery, never invented.
Use --check to inspect the planned manifest without replacing its current file.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VERSION = "0.1.0"
CREATED = "2026-11-03"
PILOT_IDS = [f"P{i:02}" for i in range(1, 11)]
MODULE_IDS = [f"M{i:02}" for i in range(17)]
REQUIRED_PAGES = ["HOME", "PROGRESS", "DIAGNOSTIC", "MENTOR", "MENTOR-KEYS", "GLOSSARY", "PROJECT",
                  *MODULE_IDS, *PILOT_IDS]


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def page_state(registry: dict, key: str) -> str:
    item = registry.get(key, {})
    if not isinstance(item, dict):
        raise ValueError(f"Page registry entry must be an object: {key}")
    if not item.get("url", "").startswith("https://chatgpt.com/space/page_"):
        return "missing"
    return "verified" if item.get("readback_verified") is True else "awaiting_readback"


def local_asset(root: Path, relative: str) -> str:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"Missing or external course asset: {relative}")
    return path.relative_to(root.resolve()).as_posix()


def build_manifest(root: Path = ROOT) -> dict:
    root = root.resolve()
    curriculum = read_json(root / "curriculum.json")
    registry = read_json(root / "metadata" / "page-registry.json")
    inventory = read_json(root / "metadata" / "pilot-lab-inventory.json")
    if not isinstance(registry, dict):
        raise ValueError("page-registry.json must map stable course IDs to Page objects")
    modules = curriculum["modules"]
    if [module["id"] for module in modules] != MODULE_IDS:
        raise ValueError("Expected ordered modules M00..M16")
    lesson_ids = [lesson["id"] for module in modules for lesson in module["lessons"]]
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

    publication_states = {key: page_state(registry, key) for key in REQUIRED_PAGES}
    unpublished = [key for key, value in publication_states.items() if value == "missing"]
    unread = [key for key, value in publication_states.items() if value == "awaiting_readback"]
    module_entries = []
    lesson_entries = []
    for module in modules:
        module_id = module["id"]
        module_entries.append({
            "id": module_id, "title": module["title"], "version": VERSION,
            "page_key": module_id, "page_url": registry.get(module_id, {}).get("url"),
            "delivery_status": publication_states[module_id],
            "design_status": "ready", "material_status": module.get("material_status", "introductory_draft"),
            "full_module_ready": False, "learner_status": "not_assessed",
            "lesson_ids": [lesson["id"] for lesson in module["lessons"]],
            "homework_id": module["homework"]["id"],
        })
        for lesson in module["lessons"]:
            lesson_entries.append({
                "id": lesson["id"], "title": lesson["title"], "version": VERSION,
                "module_id": module_id, "page_key": module_id,
                "page_url": registry.get(module_id, {}).get("url"),
                "page_scope": "module_design", "material_status": "designed",
                "prerequisites": lesson["prerequisites"], "outcome_ids": lesson["outcome_ids"],
                "pilot_sessions": [key for key, value in pilot_map.items() if lesson["id"] in value],
                "learner_status": "not_assessed",
            })

    inventory_by_id = {entry["id"]: entry for entry in inventory}
    sessions = []
    for session_id in PILOT_IDS:
        item = inventory_by_id[session_id]
        sessions.append({
            "id": session_id, "version": VERSION, "material_status": "ready",
            "learner_status": "not_assessed", "page_key": session_id,
            "page_url": registry.get(session_id, {}).get("url"),
            "delivery_status": publication_states[session_id],
            "starter": local_asset(root, item["notebook"]),
            "solution": local_asset(root, item["solution"]),
            "curriculum_lesson_ids": pilot_map[session_id],
            "prerequisites": curriculum["pilot_session_prerequisites"][session_id],
            "hardware": {"execution": "CPU", "python": "3.11", "gpu_required": False},
            "paid_dependency": False, "network_after_setup_required": False,
            "source": item["source"],
        })

    names = ["Competency map", "Full curriculum design", "Course infrastructure",
             "Diagnostics and pilot materials", "Learner pilot", "Personal calibration",
             "Remaining mandatory materials", "Full course release"]
    return {
        "schema_version": 1, "course_version": VERSION, "created": CREATED,
        "language": "ru", "course_status": "pilot_release",
        "full_course_ready": False, "learner_mastery": "not_assessed",
        "career_readiness": "not_assessed",
        "inputs": {"curriculum": "curriculum.json", "page_registry": "metadata/page-registry.json",
                   "pilot_inventory": "metadata/pilot-lab-inventory.json"},
        "pages": registry,
        "publication": {"status": "verified" if not unpublished and not unread else "incomplete",
                        "required_page_keys": REQUIRED_PAGES,
                        "missing_pages": unpublished, "awaiting_readback": unread},
        "stages": [{"id": i, "name": name,
                    "status": "ready" if i <= 4 else "awaiting_learner" if i == 5 else "pending"}
                   for i, name in enumerate(names, 1)],
        "modules": module_entries, "lessons": lesson_entries,
        "pilot": {"material_status": "ready", "learner_status": "awaiting_learner",
                  "session_count": len(sessions), "sessions": sessions,
                  "mapping_policy": curriculum["pilot_mapping_policy"],
                  "technical_validation_report": "artifacts/notebook-validation.json"},
        "environment": {"profile": curriculum["profile"], "pilot_execution": "CPU",
                        "required_paid_services": False,
                        "dependency_lock": "requirements.lock.txt"},
        "status_policy": {
            "module_pages": "Module Pages contain curriculum design and introductions, not complete lesson packages.",
            "pilot_mapping": "A ready pilot session does not mark its linked full curriculum lessons or outcomes complete.",
            "publication": "Page delivery is verified only when every required URL is present and readback_verified is true.",
            "mastery": "Technical execution does not assess learner mastery or professional grade.",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Validate inputs and print a summary without writing")
    args = parser.parse_args()
    manifest = build_manifest()
    if not args.check:
        destination = ROOT / "course-manifest.json"
        destination.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"written": not args.check, "modules": len(manifest["modules"]),
                      "lessons": len(manifest["lessons"]), "pilot_sessions": len(manifest["pilot"]["sessions"]),
                      "publication": manifest["publication"], "full_course_ready": False,
                      "learner_status": manifest["pilot"]["learner_status"]}, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
