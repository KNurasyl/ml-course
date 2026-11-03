"""Execute authored Python examples in isolated processes; never grade learners."""
from pathlib import Path
import json
import os
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT / "artifacts/lesson-examples"
    output.mkdir(parents=True, exist_ok=True)
    paths = sorted((ROOT / "course/lessons").glob("*.md"))
    paths += sorted((ROOT / "solutions/modules").glob("*.md"))
    results = []
    for path in paths:
        blocks = re.findall(r"```python\s*\n(.*?)```", path.read_text(encoding="utf-8"), re.S)
        for index, block in enumerate(blocks, 1):
            name = f"{path.parent.name}-{path.stem}-{index}"
            script = output / f"{name}.py"
            script.write_text("import sys\nsys.path.insert(0, " + repr(str(ROOT)) + ")\n" + block, encoding="utf-8")
            started = time.perf_counter()
            try:
                p = subprocess.run([sys.executable, "-X", "utf8", str(script)], cwd=ROOT,
                    env=dict(os.environ, PYTHONUTF8="1", MPLBACKEND="Agg", HF_HUB_OFFLINE="1"),
                    capture_output=True, text=True, encoding="utf-8", timeout=180)
                code, text = p.returncode, p.stdout + p.stderr
            except subprocess.TimeoutExpired:
                code, text = 124, "Example exceeded 180 seconds"
            (output / f"{name}.log").write_text(text, encoding="utf-8")
            results.append({"path": path.relative_to(ROOT).as_posix(), "block": index,
                "exit_code": code, "seconds": round(time.perf_counter() - started, 3),
                "log": f"artifacts/lesson-examples/{name}.log"})
            if code:
                print(json.dumps(results[-1]), flush=True)
    report = {"results": results, "passed": all(r["exit_code"] == 0 for r in results),
        "learner_mastery": "not_assessed", "scope": "All fenced Python examples; conceptual answers use authored keys and rubrics."}
    (ROOT / "artifacts/lesson-example-validation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"examples": len(results), "passed": report["passed"]}))
    return int(not report["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
