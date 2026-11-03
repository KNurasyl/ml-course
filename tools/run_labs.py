"""Execute every module's reference lab, preserving output and exit status."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modules", nargs="*", default=[f"M{i:02}" for i in range(17)])
    args = parser.parse_args()
    folder = ROOT / "artifacts/lab-runs"
    folder.mkdir(parents=True, exist_ok=True)
    results = []
    for mid in args.modules:
        if mid not in [f"M{i:02}" for i in range(17)]:
            parser.error(f"Unknown module {mid}")
        started = time.perf_counter()
        env = dict(os.environ, PYTHONUTF8="1", MPLBACKEND="Agg")
        try:
            process = subprocess.run([sys.executable, "-X", "utf8", str(ROOT / f"labs/{mid}/demo.py")],
                cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", timeout=600)
            code, output = process.returncode, process.stdout + process.stderr
        except subprocess.TimeoutExpired as error:
            code, output = 124, f"Timed out after {error.timeout} seconds"
        (folder / f"{mid}.log").write_text(output, encoding="utf-8", newline="\n")
        results.append({"module":mid,"exit_code":code,"seconds":round(time.perf_counter()-started,3),"log":f"artifacts/lab-runs/{mid}.log"})
        print(json.dumps(results[-1]), flush=True)
    report = {"results":results,"passed":all(r["exit_code"]==0 for r in results),"learner_mastery":"not_assessed"}
    (ROOT / "artifacts/lab-validation.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8",newline="\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
