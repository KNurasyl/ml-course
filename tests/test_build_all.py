"""The single build entry point keeps a fixed, complete and safe order."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[1]


def load_tool(name):
    spec = importlib.util.spec_from_file_location(name, SOURCE / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build_all = load_tool("build_all")


def test_default_order_is_metadata_cheatsheet_visuals_figures_reader_manifest():
    scripts = [script for _, script in build_all.planned_steps(build_all.DEFAULT)]
    assert scripts == ["build_manifest.py --sync-registry-only", "build_cheatsheet.py", "progress.py dashboard",
                       "build_visuals.py", "build_module_visuals.py", "build_product_visuals.py",
                       "build_foundation_figures.py", "build_quantitative_figures.py", "build_responsive_figures.py",
                       "build_reader.py", "build_manifest.py"]
    assert all((SOURCE / "tools" / script.split()[0]).is_file() for _, scripts in build_all.STAGES for script in scripts)


def test_text_only_and_reports_modes(capsys):
    assert build_all.main(["--dry-run", "--text-only"]) == 0
    assert json.loads(capsys.readouterr().out) == ["tools/build_manifest.py --sync-registry-only",
                                                   "tools/build_cheatsheet.py", "tools/progress.py dashboard",
                                                   "tools/build_reader.py",
                                                   "tools/build_manifest.py"]
    assert build_all.main(["--dry-run", "--reports"]) == 0
    planned = json.loads(capsys.readouterr().out)
    # Evidence is rewritten only on request and only after the reader and manifest are current.
    assert planned[-2:] == ["tools/build_release_report.py", "tools/build_design_report.py"]
    assert "tools/build_release_report.py" not in [f"tools/{s}" for _, s in build_all.planned_steps(build_all.DEFAULT)]


def test_one_off_integration_scripts_are_quarantined():
    for name in ("integrate_upgrade.py", "register_full_course.py"):
        assert not (SOURCE / "tools" / name).exists()
        assert (SOURCE / "tools/legacy" / name).is_file()
    assert "не запускать" in (SOURCE / "tools/legacy/README.md").read_text(encoding="utf-8")
