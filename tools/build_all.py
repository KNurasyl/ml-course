"""Rebuild every generated course file in one fixed order.

metadata (registry entries copied from curriculum.json, the canonical source)
-> cheatsheet (lesson cards) -> visuals (interactive experiments) -> figures (lesson SVG) -> reader -> manifest
Each generator is deterministic for unchanged sources, so a full run leaves the
working tree untouched unless a source changed. Release reports are opt-in:
they rewrite committed evidence from artifacts of a full verification run.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
STAGES = [
    ("metadata", ["build_manifest.py --sync-registry-only"]),
    ("cheatsheet", ["build_cheatsheet.py"]),
    ("visuals", ["build_visuals.py", "build_module_visuals.py", "build_product_visuals.py"]),
    ("figures", ["build_foundation_figures.py", "build_quantitative_figures.py", "build_responsive_figures.py"]),
    ("reader", ["build_reader.py"]),
    ("manifest", ["build_manifest.py"]),
    ("reports", ["build_release_report.py", "build_design_report.py"]),
]
DEFAULT = ("metadata", "cheatsheet", "visuals", "figures", "reader", "manifest")
TEXT_ONLY = ("metadata", "cheatsheet", "reader", "manifest")


def planned_steps(stages: tuple[str, ...]) -> list[tuple[str, str]]:
    return [(stage, script) for stage, scripts in STAGES if stage in stages for script in scripts]


def run(steps: list[tuple[str, str]]) -> list[dict]:
    env = dict(os.environ, PYTHONUTF8="1", MPLBACKEND="Agg")
    results = []
    for stage, script in steps:
        started = time.perf_counter()
        name, *options = script.split()
        process = subprocess.run([sys.executable, "-X", "utf8", str(ROOT / "tools" / name), *options], cwd=ROOT, env=env,
                                 capture_output=True, text=True, encoding="utf-8", errors="replace")
        results.append({"stage": stage, "script": f"tools/{script}", "exit_code": process.returncode,
                        "seconds": round(time.perf_counter() - started, 2)})
        if process.returncode != 0:
            sys.stderr.write(process.stdout[-4000:] + process.stderr[-4000:])
            break
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--text-only", action="store_true",
                        help="Only reader and manifest: enough after Markdown or progress edits")
    parser.add_argument("--reports", action="store_true",
                        help="Also rewrite release evidence in quality/ from artifacts/ of a full verification run")
    parser.add_argument("--dry-run", action="store_true", help="Print the planned steps without running them")
    args = parser.parse_args(argv)
    stages = (TEXT_ONLY if args.text_only else DEFAULT) + (("reports",) if args.reports else ())
    steps = planned_steps(stages)
    if args.dry_run:
        print(json.dumps([f"tools/{script}" for _, script in steps], indent=2))
        return 0
    results = run(steps)
    passed = len(results) == len(steps) and all(r["exit_code"] == 0 for r in results)
    print(json.dumps({"steps": results, "passed": passed, "learner_mastery": "not_assessed"}, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
