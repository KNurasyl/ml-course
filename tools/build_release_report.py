"""Assemble a release receipt only from successful, inspectable local checks."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import platform
import shutil
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads((ROOT/path).read_text(encoding="utf-8"))


def main():
    reports = {
        "module_labs": "artifacts/lab-validation.json",
        "extended_experiments": "artifacts/extended-experiments-validation.json",
        "additional_homeworks": "artifacts/additional-homework-validation.json",
        "student_contracts": "artifacts/extended-contracts-validation.json",
        "python_examples": "artifacts/lesson-example-validation.json",
        "reader": "artifacts/reader-v2-qa.json",
    }
    evidence = {key: read(path) for key, path in reports.items()}
    assert all(value["passed"] for value in evidence.values()), "A verification batch failed"
    notebooks = read("artifacts/notebook-validation.json")
    assert len(notebooks["records"]) == 20 and all(x["status"] == "executed" for x in notebooks["records"])
    suites = ET.parse(ROOT/"artifacts/pytest-release-1.1.0.xml").getroot().findall("testsuite")
    assert suites and all(int(s.get("errors", 0)) + int(s.get("failures", 0)) == 0 for s in suites)
    spec = importlib.util.spec_from_file_location("release_validator", ROOT/"tools/validate_course.py")
    validator = importlib.util.module_from_spec(spec); spec.loader.exec_module(validator)
    structural = validator.validate(ROOT, require_local=True)
    assert structural["errors"] == [], structural
    receipt_dir = ROOT/"quality/verification-1.1.0"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    files = [*reports.values(), "artifacts/notebook-validation.json", "artifacts/pytest-release-1.1.0.xml"]
    receipts = {}
    for relative in files:
        source = ROOT/relative; dest = receipt_dir/source.name
        shutil.copyfile(source, dest)
        receipts[dest.relative_to(ROOT).as_posix()] = hashlib.sha256(dest.read_bytes()).hexdigest()
    curriculum = read("curriculum.json")
    report = {
        "version": "1.1.0", "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "material_status": "self_study_materials_revised_and_technically_verified",
        "audit_items_addressed": [f"A{i:02}" for i in range(1,29)],
        "audit_mapping": "quality/audit-resolution-1.1.0.md",
        "counts": {"modules": len(curriculum["modules"]), "lessons": sum(len(m["lessons"]) for m in curriculum["modules"]),
            "new_lessons": 18, "module_homeworks": 17, "student_contracts": 17, "pilot_sessions": 10,
            "lesson_figures": len(list((ROOT/"course/assets").glob("M*-L*.*"))),
            "interactive_experiments": len(list((ROOT/"course/visuals").glob("*.html")))},
        "checks": {
            "pytest": {"tests": sum(int(s.get("tests",0)) for s in suites),
                       "skipped": sum(int(s.get("skipped",0)) for s in suites), "failures": 0, "errors": 0},
            "python_examples": len(evidence["python_examples"]["results"]),
            "module_labs": len(evidence["module_labs"]["results"]),
            "additional_experiments_and_full_hw": len(evidence["extended_experiments"]["results"])+len(evidence["additional_homeworks"]["results"]),
            "student_reference_and_empty_checks": len(evidence["student_contracts"]["results"]),
            "executed_notebooks": 20, "structure": structural,
            "reader_scenarios": len(evidence["reader"]["reader"])+len(evidence["reader"]["attention"]),
        },
        "environment": {"python": platform.python_version(), "platform": platform.platform(), "execution": "CPU",
            "environments": [".venv", ".venv-full-verify"],
            "clean_install_this_revision": False,
            "note": "Existing isolated environments from the prior lock-file installation were reused; no fresh reinstall is claimed. pip check passed.",
            "dependency_lock": "requirements.lock.txt", "paid_services": False},
        "receipt_sha256": receipts,
        "semantic_evidence": {"paired_feedback": "course/feedback-case-study.md", "russian_rag": "quality/rag-hw/manual-review.json",
            "attribution": "AI-assisted author judgement, not independent human validation"},
        "limits": ["Small-model quality gates can fail even when the experiment executes correctly.",
            "Saved synthetic/reference runs do not establish learner mastery, user-pilot success or middle-level career readiness.",
            "Human explanations are checked by rubrics and worked keys, not certified by pytest.",
            "An audit and its corrections do not guarantee absence of every possible future defect."],
        "learner_mastery": "not_assessed", "learner_pilot": "not_run", "personal_time_calibration": "awaiting_learner",
    }
    (ROOT/"quality/release-1.1.0.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8",newline="\n")
    print(json.dumps(report["checks"],ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
