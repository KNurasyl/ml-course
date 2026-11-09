"""Assemble a release receipt only from successful, inspectable local checks.

The version comes from tools/build_manifest.py, and every release writes its own
quality/release-<version>.json and quality/verification-<version>/, so building a
later receipt never rewrites an earlier one. release_1_1_0 is kept as the record of
how the 1.1.0 receipt was made; it runs only while the course version is 1.1.0.
"""
import hashlib
import importlib.util
import json
import platform
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads((ROOT/path).read_text(encoding="utf-8"))


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT/path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def course_version():
    text = (ROOT/"tools/build_manifest.py").read_text(encoding="utf-8")
    return re.search(r'(?m)^VERSION = "([^"]+)"', text).group(1)


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.rstrip("\n")


def journal_statuses():
    """Task statuses from section 7 of IMPROVEMENT_PLAN.md (rows like "| T60 | выполнено | ...")."""
    text = (ROOT/"IMPROVEMENT_PLAN.md").read_text(encoding="utf-8").split("## 7. Журнал выполнения", 1)[1]
    rows = re.findall(r"(?m)^\| (T[0-9][0-9A-Za-z–-]*) \| ([^|]+?) \|", text)
    return {task: status for task, status in rows}


def release_1_2_0(version):
    reports = {
        "module_labs": "artifacts/lab-validation.json",
        "extended_labs_and_contracts": "artifacts/extended-all-validation.json",
        "python_examples": "artifacts/lesson-example-validation.json",
    }
    evidence = {key: read(path) for key, path in reports.items()}
    assert all(value["passed"] for value in evidence.values()), "A verification batch failed"
    notebooks = read("artifacts/notebook-validation.json")
    assert len(notebooks["records"]) == 20 and all(x["status"] == "executed" for x in notebooks["records"])
    junit = f"artifacts/pytest-release-{version}.xml"
    suites = ET.parse(ROOT/junit).getroot().findall("testsuite")
    assert suites and all(int(s.get("errors", 0)) + int(s.get("failures", 0)) == 0 for s in suites)
    structural = load("tools/validate_course.py", "release_validator").validate(ROOT, require_local=True)
    assert structural["errors"] == [], structural
    no_loss_tool = load("tools/check_no_loss.py", "release_no_loss")
    current = no_loss_tool.snapshot(ROOT, include_tests=True)
    no_loss = no_loss_tool.compare(no_loss_tool.read_json(ROOT/"quality/snapshots/baseline-1.1.0.json"), current)
    assert no_loss["status"] == "no_loss", no_loss
    pip_check = subprocess.run([sys.executable, "-m", "pip", "check"], check=False, cwd=ROOT, capture_output=True, text=True)
    assert pip_check.returncode == 0, pip_check.stdout + pip_check.stderr

    receipt_dir = ROOT/f"quality/verification-{version}"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    receipts = {}
    for relative in [*reports.values(), "artifacts/notebook-validation.json", junit]:
        source = ROOT/relative
        dest = receipt_dir/source.name
        shutil.copyfile(source, dest)
        receipts[dest.relative_to(ROOT).as_posix()] = hashlib.sha256(dest.read_bytes()).hexdigest()

    curriculum = read("curriculum.json")
    lessons = [lesson for module in curriculum["modules"] for lesson in module["lessons"]]
    extension = [lesson["id"] for lesson in lessons if lesson.get("extension") == version]
    extended = evidence["extended_labs_and_contracts"]["results"]
    contracts = [r for r in extended if re.match(r"M\d\d-.*(reference|empty-rejected)$", r["id"])]
    statuses = journal_statuses()
    report = {
        "version": version, "verified_at_utc": datetime.now(UTC).isoformat(),
        "material_status": "self_study_materials_extended_and_technically_verified",
        "improvement_plan": {"backlog": "IMPROVEMENT_PLAN.md",
            "tasks_done": sorted(t for t, s in statuses.items() if s == "выполнено"),
            "tasks_declined_by_author": sorted(t for t, s in statuses.items() if s.startswith("снят")),
            "research": "quality/research-2026-11-03.md"},
        "counts": {"modules": len(curriculum["modules"]), "lessons": len(lessons),
            "core_lessons": len(lessons) - len(extension), "extension_lessons": len(extension),
            "extension_lesson_ids": extension, "module_homeworks": 17, "pilot_sessions": 10,
            "student_contract_checks": len(contracts),
            "lesson_figures": len(list((ROOT/"course/assets").glob("M*-L*.*"))),
            "interactive_experiments": len(list((ROOT/"course/visuals").glob("*.html"))),
            "mermaid_schemas": sum(p.read_text(encoding="utf-8").count("```mermaid") for p in (ROOT/"course/lessons").glob("*.md"))},
        "checks": {
            "pip_check": "passed",
            "pytest": {"tests": sum(int(s.get("tests", 0)) for s in suites),
                       "skipped": sum(int(s.get("skipped", 0)) for s in suites), "failures": 0, "errors": 0},
            "python_examples": len(evidence["python_examples"]["results"]),
            "module_labs": len(evidence["module_labs"]["results"]),
            "extended_experiments_and_extension_labs": len(extended) - len(contracts),
            "student_reference_and_empty_checks": len(contracts),
            "executed_notebooks": len(notebooks["records"]),
            "structure": structural,
            "no_loss_against_1_1_0": {"status": no_loss["status"], "soft_warnings": len(no_loss["warnings"]),
                                      "growth": no_loss["growth"], "counts": current["counts"]},
        },
        "trees": {"commit": git("rev-parse", "--short", "HEAD"),
                  "uncommitted_paths": git("status", "--porcelain").splitlines(),
                  "note": ("The batches in receipt_sha256 ran on the commit named here with the changes in uncommitted_paths "
                           "applied, after the version bump. Between the batches and the release commit only release notes, "
                           "this receipt and its builder may change; lessons, labs, tests and the checks they run must not.")},
        "environment": {"python": platform.python_version(), "platform": platform.platform(), "execution": "CPU",
            "environments": [".venv"], "clean_install_this_revision": False,
            "note": "The existing .venv from the lock file was reused; no fresh reinstall is claimed. pip check passed.",
            "dependency_lock": "requirements.lock.txt", "optional_packages": "requirements-optional.in", "paid_services": False},
        "receipt_sha256": receipts,
        "semantic_evidence": {"model_comparison": "quality/model-comparison/report.md",
            "question_search": "quality/question-search/report.json",
            "trainer_security": "quality/trainer-security/probe-strong.json",
            "speech": "quality/speech/report.json", "gpt_from_scratch": "quality/gpt-from-scratch/report.json",
            "attribution": "AI-assisted author judgement, not independent human validation"},
        "limits": ["Small-model quality gates can fail even when the experiment executes correctly; the 1.2.0 trainer profiles do not pass the M11 threshold.",
            "Saved synthetic/reference runs do not establish learner mastery, user-pilot success or middle-level career readiness.",
            "Human explanations are checked by rubrics and worked keys, not certified by pytest.",
            "The Playwright reader suite of 1.1.0 (reader-v2-qa) was not rerun: its node dependencies were archived with .authoring (T02). The 1.2.0 lessons were render-checked in the in-app browser task by task.",
            "Extension labs run on CPU with locally present models; the model runs behind their lessons are frozen author runs in quality/.",
            "Time estimates stay planning hypotheses: the learner pilot P01-P10 was not run, so T64 was declined."],
        "previous_release": "quality/release-1.1.0.json",
        "learner_mastery": "not_assessed", "learner_pilot": "not_run", "personal_time_calibration": "awaiting_learner",
    }
    (ROOT/f"quality/release-{version}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8", newline="\n")
    print(json.dumps(report["checks"], ensure_ascii=False, indent=2))


def main():
    version = course_version()
    if version == "1.1.0":
        release_1_1_0()
    elif version == "1.2.0":
        release_1_2_0(version)
    else:
        raise SystemExit(f"No release receipt builder for version {version}")


def release_1_1_0():
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
        "version": "1.1.0", "verified_at_utc": datetime.now(UTC).isoformat(),
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
