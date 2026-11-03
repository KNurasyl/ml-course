"""A real paired evaluation: collect immutable outputs, annotate, then score.

Model generation and judgement are deliberately separate. No expected label is
passed to the interviewer; missing judgement is never replaced by a zero.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
CONFIGS = ("baseline", "candidate")
RUBRIC = "feedback-grounding-usefulness-v2"


def digest(value):
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def validate_cases(cases):
    if not cases or len({c["case_id"] for c in cases}) != len(cases):
        raise ValueError("cases must be nonempty with unique case_id")
    for row in cases:
        if row["track"] not in ("hr", "backend") or row["split"] not in ("development", "check"):
            raise ValueError("unknown track or split")
        if any(not isinstance(row.get(k), str) or not row[k].strip() for k in ("question", "answer", "scenario_id", "origin")):
            raise ValueError("case content/provenance missing")
    dev = {(c["track"], c["scenario_id"]) for c in cases if c["split"] == "development"}
    check = {(c["track"], c["scenario_id"]) for c in cases if c["split"] == "check"}
    if dev & check:
        raise ValueError("scenario leakage across development/check")


def collect(cases, model, out_dir):
    from projects.interview_trainer.core import retrieve
    from projects.interview_trainer.feedback import evaluate_answer
    validate_cases(cases)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    output_path = out / "outputs.jsonl"
    if output_path.exists():
        raise FileExistsError("Use a NEW --out directory: measured outputs must not be silently overwritten")
    frozen_files = [ROOT / "projects/interview_trainer" / name for name in ("feedback.py", "core.py", "llm.py")]
    code = {p.name: sha256(p.read_bytes()).hexdigest() for p in frozen_files}
    run = {"record_schema": 2, "rubric_version": RUBRIC, "case_digest": digest(cases), "code_sha256": code,
           "model": "local Qwen2.5-0.5B-Instruct", "revision": "7ae557604adf67be50417f59c2c2f167def9a775",
           "configs": list(CONFIGS), "case_count": len(cases),
           "policy": "Outputs are measurements; annotations are separately attributed judgement.",
           "cases": cases, "learner_mastery": "not_assessed"}
    (out / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = []
    # One loaded model, sequential calls. Alternate order to expose timing bias.
    for index, case in enumerate(cases):
        configs = CONFIGS if index % 2 == 0 else CONFIGS[::-1]
        sources = retrieve(case["question"]) if case["track"] == "backend" else []
        for config in configs:
            start = time.perf_counter()
            result = evaluate_answer(model, mode=case["track"], question=case["question"],
                                     answer=case["answer"], sources=sources, config=config)
            row = {"case_id": case["case_id"], "track": case["track"], "split": case["split"],
                   "scenario_id": case["scenario_id"], "config": config, "sources": sources,
                   "result": result, "seconds": round(time.perf_counter() - start, 6)}
            row["output_sha256"] = digest(result)
            row["case_sha256"] = digest(case)
            row["record_sha256"] = digest(row)
            with output_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows.append(row)
            print(json.dumps({"case_id": case["case_id"], "config": config, "status": result["status"], "seconds": row["seconds"]}), flush=True)
    validate_outputs(rows, cases)
    annotations = [{"case_id": r["case_id"], "config": r["config"], "output_sha256": r["output_sha256"],
                    "rubric_version": RUBRIC, "grounded": None, "useful": None,
                    "critical_error": None, "evidence": "", "reason": "", "rater": ""} for r in rows]
    write_jsonl(out / "annotations-template.jsonl", annotations)
    return rows


def validate_outputs(outputs, cases):
    expected = {(c["case_id"], config) for c in cases for config in CONFIGS}
    keys = [(r["case_id"], r["config"]) for r in outputs]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("outputs require exactly one row per case/config; no dropped or duplicated cases")
    case_by_id = {c["case_id"]: c for c in cases}
    for row in outputs:
        case = case_by_id[row["case_id"]]
        if any(row.get(k) != case[k] for k in ("track", "split", "scenario_id")):
            raise ValueError("output metadata does not match the collected case")
        if row["output_sha256"] != digest(row["result"]):
            raise ValueError("output content changed after collection")
        if "record_sha256" in row:
            if row.get("case_sha256") != digest(case):
                raise ValueError("input case changed after collection")
            if row["record_sha256"] != digest({k: v for k, v in row.items() if k != "record_sha256"}):
                raise ValueError("output provenance or source record changed")


def evaluate_annotations(cases, outputs, annotations):
    """Validate paired judgement; return separate track/split reports."""
    validate_cases(cases); validate_outputs(outputs, cases)
    measured = {(r["case_id"], r["config"]): r for r in outputs}
    keys = [(a["case_id"], a["config"]) for a in annotations]
    if len(keys) != len(set(keys)) or set(keys) != set(measured):
        raise ValueError("annotations must cover every measured pair exactly once")
    judged = {}
    for a in annotations:
        key = (a["case_id"], a["config"])
        if a.get("output_sha256") != measured[key]["output_sha256"]:
            raise ValueError("annotation is attached to another output")
        if a.get("rubric_version") != RUBRIC:
            raise ValueError("mixed rubric versions; re-annotate explicitly")
        if any(type(a.get(k)) is not int or not 0 <= a[k] <= 2 for k in ("grounded", "useful")):
            raise ValueError("ratings must be explicit integers 0..2; missing is not zero")
        if type(a.get("critical_error")) is not bool:
            raise ValueError("critical judgement must be explicit")
        if any(not isinstance(a.get(k), str) or not a[k].strip() for k in ("rater", "evidence", "reason")):
            raise ValueError("judgement needs attributed evidence and reasoning")
        # Evidence is an exact excerpt of the displayed response; reasoning can disagree with it.
        if a["evidence"] not in measured[key]["result"]["display_text"]:
            raise ValueError("annotation evidence is not in the displayed response")
        judged[key] = a
    import numpy as np
    report = {"rubric_version": RUBRIC, "tracks": {}, "rater_counts": dict(Counter(a["rater"] for a in annotations)),
              "disclaimer": "Judgement is not automated ground truth. Source quotes require semantic review.",
              "learner_mastery": "not_assessed"}
    for track in ("hr", "backend"):
        report["tracks"][track] = {}
        for split in ("development", "check"):
            subset = [c for c in cases if c["track"] == track and c["split"] == split]
            if not subset:
                continue
            per_config = {}
            for config in CONFIGS:
                aa = [judged[c["case_id"], config] for c in subset]
                rr = [measured[c["case_id"], config] for c in subset]
                per_config[config] = {"n": len(aa), "grounded_mean": float(np.mean([a["grounded"] for a in aa])),
                    "useful_fraction": float(np.mean([a["useful"] >= 1 for a in aa])),
                    "critical_count": sum(a["critical_error"] for a in aa),
                    "fallback_fraction": float(np.mean([r["result"]["fallback_used"] for r in rr])),
                    "p50_seconds": float(np.median([r["seconds"] for r in rr]))}
            delta = np.array([judged[c["case_id"], "candidate"]["useful"] - judged[c["case_id"], "baseline"]["useful"] for c in subset])
            grouped = {}
            for case, change in zip(subset, delta):
                grouped.setdefault(case["scenario_id"], []).append(float(change))
            scenario_changes = np.array([np.mean(v) for v in grouped.values()])
            rng = np.random.default_rng(1106)
            boot = scenario_changes[rng.integers(0, len(scenario_changes), (4000, len(scenario_changes)))].mean(1)
            regressions = [c["case_id"] for c in subset if judged[c["case_id"], "candidate"]["critical_error"] and not judged[c["case_id"], "baseline"]["critical_error"]]
            candidate = per_config["candidate"]
            # Policy chosen before measurement. Failure must remain visible.
            reasons = []
            if candidate["critical_count"]: reasons.append("candidate_has_critical_errors")
            if candidate["grounded_mean"] < 1.5: reasons.append("insufficient_grounding")
            if candidate["useful_fraction"] < .75: reasons.append("insufficient_usefulness")
            if candidate["fallback_fraction"] > .5: reasons.append("model_observation_coverage_below_half")
            report["tracks"][track][split] = {"configs": per_config, "case_ids": [c["case_id"] for c in subset],
                "paired_usefulness_delta": float(scenario_changes.mean()), "case_weighted_delta": float(delta.mean()),
                "scenario_count": len(grouped), "bootstrap_95": [float(v) for v in np.quantile(boot, [.025, .975])],
                "bootstrap_unit": "scenario mean; every scenario has equal weight", "bootstrap_seed": 1106,
                "critical_regressions": regressions, "release_allowed": not reasons, "release_blockers": reasons,
                "scope": "Small authored examples. This interval does not establish broad user benefit."}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    collect_p = sub.add_parser("collect")
    collect_p.add_argument("--cases", type=Path, default=Path(__file__).with_name("cases-v2.jsonl"))
    collect_p.add_argument("--out", type=Path, required=True)
    score_p = sub.add_parser("score")
    score_p.add_argument("--run", type=Path, required=True)
    score_p.add_argument("--annotations", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "collect":
        from projects.interview_trainer.llm import LocalLLM
        collect(read_jsonl(args.cases), LocalLLM(), args.out)
    else:
        run = json.loads((args.run / "run.json").read_text(encoding="utf-8"))
        if digest(run["cases"]) != run["case_digest"]:
            raise ValueError("case set changed after collection")
        outputs = read_jsonl(args.run / "outputs.jsonl")
        if run.get("record_schema") == 2 and any("record_sha256" not in row for row in outputs):
            raise ValueError("record schema 2 requires full provenance hashes")
        integrity_path = args.run / "integrity.json"
        if integrity_path.exists():
            integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
            for name, expected in integrity["files"].items():
                if sha256((args.run / name).read_bytes()).hexdigest() != expected:
                    raise ValueError(f"archived evidence changed: {name}")
        report = evaluate_annotations(run["cases"], outputs, read_jsonl(args.annotations))
        (args.run / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
