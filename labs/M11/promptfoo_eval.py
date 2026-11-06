"""promptfoo as an option over the course's own harness (extension 1.2.0, T38, lesson M11-L06).

One test set runs on two local profiles: the 24 interview answers with gold scores of M09-L08X. The profiles are tiny
(Qwen2.5-0.5B Q4_K_M) and strong (Qwen3.5-4B Q4_K_M), each behind its own llama-server on 127.0.0.1. The YAML config and
the JS checks live in labs/M11/promptfoo. prompt.json, cases.jsonl and feedback.schema.json there are generated from
labs/M09/structured_output.py, so the two harnesses read one source. promptfoo runs through npx at a pinned version.
Telemetry and update checks are off, and its database stays in artifacts/. Nothing is installed globally.

After the run, the course's own Python checks recompute every promptfoo verdict on the same outputs: the JSON Schema,
demo.validate_feedback and the gold score. Every disagreement is listed. Outputs are also compared with the T31
prompt_only run, which sent the same request without promptfoo. The own harness stays primary, and promptfoo is a
second way to run and view the same set.

    python labs/M11/promptfoo_eval.py sync      # regenerate the three generated files
    python labs/M11/promptfoo_eval.py run       # both servers, npx promptfoo eval, comparison
    python labs/M11/promptfoo_eval.py compare   # comparison only, from an existing results.json
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import importlib.util
import json
import os
import shutil
import statistics
import subprocess
import time
from pathlib import Path

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "labs/M11/promptfoo"
OUT = ROOT / "artifacts/M11/promptfoo"
T31_REPORT = ROOT / "artifacts/M09/structured/report.json"
PROMPTFOO = "0.124.1"
PROFILES = {"tiny": ("qwen2.5-0.5b-instruct-q4_k_m", 18080), "strong": ("qwen3.5-4b-q4_k_m", 18081)}
USER = "Вопрос: {{question}}\nОтвет кандидата: {{answer}}"
METRICS = ("schema", "contract", "score")


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


so = load("labs/M09/structured_output.py", "m09_structured_output")
demo = so.load("demo")


def generated():
    """The three files promptfoo reads, rendered from the Python source of M09-L08X."""
    cases = []
    for i, (question, answer, gold) in enumerate(so.CASES, 1):
        # promptfoo renders the variables into a JSON chat prompt with Nunjucks and then parses it: a quote,
        # a backslash or a control character in a variable would break that JSON.
        if any(c in '"\\' or ord(c) < 32 for c in question + answer):
            raise ValueError(f"case {i}: quotes, backslashes and control characters are not allowed in variables")
        cases.append({"description": f"c{i:02d} gold {gold}",
                      "vars": {"case_id": f"c{i:02d}", "question": question, "answer": answer, "gold": gold}})
    prompt = [{"role": "system", "content": so.SYSTEM}, {"role": "user", "content": USER}]
    return {"prompt.json": json.dumps(prompt, ensure_ascii=False, indent=2) + "\n",
            "cases.jsonl": "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in cases),
            "feedback.schema.json": json.dumps(so.SCHEMA, ensure_ascii=False, indent=2) + "\n"}


def sync(folder=CONFIG_DIR, write=True):
    """Names of generated files that differ from the Python source; rewrites them unless write is False."""
    files = generated()
    stale = [name for name, text in files.items()
             if not (folder / name).exists() or (folder / name).read_bytes().decode("utf-8") != text]
    if write:
        for name in stale:
            (folder / name).write_text(files[name], encoding="utf-8", newline="\n")
    return stale


def expected_messages(question, answer):
    return [{"role": "system", "content": so.SYSTEM},
            {"role": "user", "content": USER.replace("{{question}}", question).replace("{{answer}}", answer)}]


def own_verdicts(output, gold):
    """The same three questions answered by the course's Python code."""
    try:
        value, parsed = json.loads(output), True
    except (json.JSONDecodeError, TypeError):
        value, parsed = None, False
    response = demo.feedback_or_abstain(output)
    return {"schema": bool(parsed and Draft202012Validator(so.SCHEMA).is_valid(value)),
            "contract": response["status"] == "ok", "score": response["score"] == gold}, response


def shown(path):
    path = Path(path).resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)


def t31_texts(path=T31_REPORT):
    """First-attempt prompt_only outputs of the T31 run, keyed by model, question and answer; None without the report."""
    if not Path(path).exists():
        return None
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    return {(key.split("/")[0], r["question"], r["answer"]): r["attempts"][0]["text"]
            for key, result in report["results"].items() if key.endswith("/prompt_only") for r in result["rows"]}


def compare(results_path, t31_path=T31_REPORT):
    data = json.loads(Path(results_path).read_text(encoding="utf-8"))
    t31 = t31_texts(t31_path)
    rows = collections.defaultdict(list)
    for row in data["results"]["results"]:
        label = row["provider"].get("label") or row["provider"]["id"]
        rows[label].append(row)
    profiles = {}
    for label, items in rows.items():
        model = next((name for key, (name, _) in PROFILES.items() if key == label), items[0]["provider"]["id"].split(":")[-1])
        summary = {"model": model, "n": len(items), "errors": 0, "test_pass": 0, "prompt_as_expected": 0,
                   "promptfoo": dict.fromkeys(METRICS, 0), "own": dict.fromkeys(METRICS, 0), "agree": dict.fromkeys(METRICS, 0),
                   "same_text_as_t31_prompt_only": None if t31 is None else 0, "scores_by_gold": {}, "disagreements": []}
        by_gold = collections.defaultdict(collections.Counter)
        latency = []
        for row in items:
            v = row["vars"]
            output = (row.get("response") or {}).get("output")
            output = output if isinstance(output, str) else ""
            summary["errors"] += row.get("failureReason") == 2  # 1: an assertion failed, 2: the call itself failed
            summary["test_pass"] += bool(row.get("success"))
            if row.get("latencyMs") is not None:
                latency.append(row["latencyMs"])
            try:
                summary["prompt_as_expected"] += json.loads(row["prompt"]["raw"]) == expected_messages(v["question"], v["answer"])
            except (KeyError, TypeError, json.JSONDecodeError):
                pass
            components = (row.get("gradingResult") or {}).get("componentResults") or []
            pf = {c["assertion"]["metric"]: bool(c["pass"]) for c in components if (c.get("assertion") or {}).get("metric")}
            own, response = own_verdicts(output, v["gold"])
            by_gold[str(v["gold"])][str(response["score"])] += 1
            for metric in METRICS:
                summary["promptfoo"][metric] += pf.get(metric, False)
                summary["own"][metric] += own[metric]
                if pf.get(metric) == own[metric]:
                    summary["agree"][metric] += 1
                else:
                    summary["disagreements"].append({"case_id": v["case_id"], "metric": metric, "promptfoo": pf.get(metric),
                                                     "own": own[metric], "output": output})
            if t31 is not None:
                summary["same_text_as_t31_prompt_only"] += t31.get((model, v["question"], v["answer"])) == output
        summary["scores_by_gold"] = {g: dict(by_gold[g]) for g in sorted(by_gold)}
        summary["latency_ms_median"] = round(statistics.median(latency)) if latency else None
        profiles[label] = summary
    return {"task": "T38", "promptfoo": PROMPTFOO, "config": "labs/M11/promptfoo/promptfooconfig.yaml",
            "cases": len(so.CASES), "metrics": {"schema": "is-json with feedback.schema.json",
                                                "contract": "checks.js:contract = demo.validate_feedback",
                                                "score": "checks.js:score = contract and score == gold"},
            "t31_report": None if t31 is None else shown(t31_path), "profiles": profiles}


def npx():
    exe = shutil.which("npx")
    if not exe:
        raise FileNotFoundError("Node.js 22.22+ with npx is needed for promptfoo; the own harness works without it")
    return exe


def run(profiles=tuple(PROFILES), threads=so.THREADS, out_dir=OUT, first=None):
    local, serving = so.load("llama_local"), so.load("llama_serving")
    sync()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results, status = out / "results.json", {}
    with contextlib.ExitStack() as stack:
        for label in profiles:
            name, port = PROFILES[label]
            entry = serving.gguf_entry(name)
            path = ROOT / entry["path"] / entry["file"]
            if not path.exists():
                status[label] = "not_downloaded"
                continue
            stack.enter_context(local.Server(path, threads=threads, ctx=4096, alias=name, port=port, log=out / f"{label}.server.log"))
            status[label] = "served"
        served = [label for label in profiles if status[label] == "served"]
        if not served:
            raise SystemExit(f"no GGUF model found: {status}; see M09-L07X for the download")
        env = {**os.environ, "PROMPTFOO_DISABLE_TELEMETRY": "1", "PROMPTFOO_DISABLE_UPDATE": "1",
               "PROMPTFOO_CONFIG_DIR": str(out / "home")}
        command = [npx(), "--yes", f"promptfoo@{PROMPTFOO}", "eval", "-c", str(CONFIG_DIR / "promptfooconfig.yaml"),
                   "-o", str(results), "--no-cache", "--no-share", "--no-progress-bar", "-j", "1"]
        if len(served) < len(PROFILES):
            command += ["--filter-providers", "|".join(served)]
        if first:
            command += ["--filter-first-n", str(first)]
        results.unlink(missing_ok=True)
        started = time.perf_counter()
        # cwd is the config folder: promptfoo reads a .env file from its working directory, and there is none here
        with (out / "promptfoo.log").open("w", encoding="utf-8") as log:
            code = subprocess.run(command, check=False, cwd=CONFIG_DIR, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT).returncode
        seconds = time.perf_counter() - started
    if code not in (0, 100) or not results.exists():  # 100 is promptfoo's exit code when a test failed
        raise RuntimeError(f"promptfoo exited with {code}, see {out / 'promptfoo.log'}")
    report = compare(results)
    report.update({"status": status, "exit_code": code, "eval_seconds": round(seconds, 1), "threads": threads,
                   "llama_cpp": local.RELEASE["tag"]})
    (out / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    return report


def table_text(report):
    lines = ["| Профиль | Случаев | Схема pf / свой | Контракт pf / свой | Оценка pf / свой | Расхождений | Как в T31 |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for label, s in report["profiles"].items():
        cells = [f"{s['promptfoo'][m]} / {s['own'][m]}" for m in METRICS]
        same = "—" if s["same_text_as_t31_prompt_only"] is None else f"{s['same_text_as_t31_prompt_only']}/{s['n']}"
        lines.append(f"| {label} | {s['n']} | {' | '.join(cells)} | {len(s['disagreements'])} | {same} |")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["sync", "run", "compare"])
    parser.add_argument("--profiles", nargs="+", choices=list(PROFILES), default=list(PROFILES))
    parser.add_argument("--first", type=int, help="first N cases only, for a quick check")
    parser.add_argument("--threads", type=int, default=so.THREADS)
    args = parser.parse_args()
    if args.command == "sync":
        print("rewritten:", sync() or "nothing, files match the Python source")
    else:
        report = run(tuple(args.profiles), args.threads, first=args.first) if args.command == "run" else compare(OUT / "results.json")
        print(table_text(report))
