"""Run additional reference experiments and check every student contract.

Starter failures are expected evidence that an empty solution is not accepted.
Commands come from local reviewed authoring metadata and run without a shell.
"""
from pathlib import Path
import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
# The longest optional lab, labs/M10/rerank.py, scores 4608 cross-encoder pairs in about 25 minutes on 4 CPU
# threads (lesson M10-L09X); the former 900 s limit cut it off, so a run gets up to 40 minutes.
TIMEOUT_SECONDS = 2400


def command_args(command):
    if isinstance(command, list):
        args = list(command)
    else:
        args = shlex.split(command, posix=False)
    if args and ("python" in args[0].lower()):
        args.pop(0)
    if not args or not (ROOT / args[0].replace("\\", "/")).is_file():
        raise ValueError(f"Unknown script: {command}")
    return [sys.executable, "-X", "utf8", *args]


def run_command(invocation):
    """Run one lab; on timeout stop its whole process tree. A lab may start llama-server, and killing only the
    Python child left the server holding port 18080, which then failed the next lab and a live test (1.2.0)."""
    process = subprocess.Popen(invocation, cwd=ROOT, env=dict(os.environ, PYTHONUTF8="1", MPLBACKEND="Agg", HF_HUB_OFFLINE="1"),
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8",
                               start_new_session=os.name != "nt")
    try:
        out, err = process.communicate(timeout=TIMEOUT_SECONDS)
        return process.returncode, out + err
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(process.pid)], capture_output=True, check=False)
        else:
            os.killpg(process.pid, signal.SIGKILL)
        process.communicate()
        return 124, f"Timeout after {TIMEOUT_SECONDS} seconds"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contracts-only", action="store_true")
    parser.add_argument("--experiments-only", action="store_true")
    args = parser.parse_args()
    tasks = []
    if not args.experiments_only:
        for i in range(17):
            mid = f"M{i:02}"
            tasks += [(f"{mid}-reference", [f"exercises/{mid}/check.py", f"solutions/exercises/{mid}/solution.py"], True),
                (f"{mid}-empty-rejected", [f"exercises/{mid}/check.py", f"exercises/{mid}/starter.py"], False)]
            # Extra exercises (plan 3.4): exNN_*.py starter, check_exNN.py and a reference of the same name.
            for checker in sorted((ROOT / "exercises" / mid).glob("check_ex*.py")):
                number = checker.stem.removeprefix("check_")
                starter = next((ROOT / "exercises" / mid).glob(f"{number}_*.py"))
                checker_path, starter_path = f"exercises/{mid}/{checker.name}", f"exercises/{mid}/{starter.name}"
                tasks += [(f"{mid}-{number}-reference", [checker_path, f"solutions/exercises/{mid}/{starter.name}"], True),
                          (f"{mid}-{number}-empty-rejected", [checker_path, starter_path], False)]
    if not args.contracts_only:
        for name in ("foundations", "llm", "product", "root", "extensions"):  # extensions: optional 1.2.0 labs
            meta = json.loads((ROOT / f"metadata/upgrade-{name}.json").read_text(encoding="utf-8"))
            commands = meta.get("new_lab_commands", meta.get("lab_commands", []))
            for i, command in enumerate(commands, 1):
                tasks.append((f"{name}-{i:02}", command, True))
    folder = ROOT / "artifacts/extended-labs"
    folder.mkdir(parents=True, exist_ok=True)
    results = []
    for name, command, expected_success in tasks:
        started = time.perf_counter()
        invocation = command_args(command)
        code, output = run_command(invocation)
        (folder / f"{name}.log").write_text(output, encoding="utf-8")
        passed = code == 0 if expected_success else code not in (0, 124)
        result = {"id": name, "command": invocation[3:], "expected_success": expected_success,
            "exit_code": code, "passed": passed, "seconds": round(time.perf_counter()-started, 3),
            "log": f"artifacts/extended-labs/{name}.log"}
        results.append(result)
        print(json.dumps(result), flush=True)
    suffix = "contracts" if args.contracts_only else "experiments" if args.experiments_only else "all"
    report = {"results": results, "passed": all(r["passed"] for r in results), "learner_mastery": "not_assessed"}
    (ROOT / f"artifacts/extended-{suffix}-validation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return int(not report["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
