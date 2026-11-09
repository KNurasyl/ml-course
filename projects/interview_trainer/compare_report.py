"""Tables of T51 from the two collect.json files and the annotations of the shown feedback (no model needed).

Annotations follow the M11 rubric feedback-grounding-usefulness-v2 and are attributed in their `rater` field; the author's
AI-assisted review is not independent human validation. A judge verdict passes when useful >= 1 and there is no critical
error; for the comparison with the judge the annotation is reduced the same way (the judge does not rate grounding).
"""
from __future__ import annotations

import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / "quality/model-comparison"
RUBRIC = "feedback-grounding-usefulness-v2"


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load(folder=FROZEN):
    collected = {p: json.loads((Path(folder) / f"collect-{p}.json").read_text(encoding="utf-8")) for p in ("tiny", "strong")}
    annotations = read_jsonl(Path(folder) / "annotations.jsonl") if (Path(folder) / "annotations.jsonl").exists() else []
    return collected, annotations


def check_annotations(collected, annotations):
    """Every shown feedback has exactly one annotation tied to its hash; evidence is a quote of the shown text."""
    shown = {(p, r["case_id"]): r["display_text"] for p, c in collected.items() for r in c["rubric_outputs"]}
    keys = [(a["profile"], a["case_id"]) for a in annotations]
    if len(keys) != len(set(keys)) or set(keys) != set(shown):
        raise ValueError("annotations must cover every shown feedback exactly once")
    for a in annotations:
        text = shown[(a["profile"], a["case_id"])]
        if a["output_sha256"] != sha(text) or a["rubric_version"] != RUBRIC:
            raise ValueError(f"annotation of {a['profile']}/{a['case_id']} is attached to another output or rubric")
        if any(type(a[k]) is not int or not 0 <= a[k] <= 2 for k in ("grounded", "useful")) or type(a["critical_error"]) is not bool:
            raise ValueError("ratings must be explicit")
        if a["evidence"] not in text or not a["reason"].strip() or not a["rater"].strip():
            raise ValueError("evidence must quote the shown feedback, with a reason and a rater")
    return True


def passes(label):
    return label["grounded"] >= 1 and label["useful"] >= 1 and not label["critical_error"]


def rubric_table(collected, annotations):
    by_key = {(a["profile"], a["case_id"]): a for a in annotations}
    out = {}
    for profile, data in collected.items():
        rows = data["rubric_outputs"]
        labels = [by_key[(profile, r["case_id"])] for r in rows]
        out[profile] = {"n": len(rows), "pass": sum(passes(a) for a in labels), "grounded_mean": round(statistics.mean(a["grounded"] for a in labels), 2),
                        "useful_at_least_1": sum(a["useful"] >= 1 for a in labels), "critical": sum(a["critical_error"] for a in labels),
                        "model_fallback": sum(r["status"] != "grounded_candidate" for r in rows),
                        "by_track": {t: sum(passes(by_key[(profile, r["case_id"])]) for r in rows if r["track"] == t) for t in ("hr", "backend")},
                        "codes": dict(Counter(code for a in labels for code in a.get("codes", [])))}
    return out


def kappa(pairs):
    n = len(pairs)
    observed = sum(a == b for a, b in pairs) / n
    p_label, p_judge = sum(a for a, _ in pairs) / n, sum(b for _, b in pairs) / n
    expected = p_label * p_judge + (1 - p_label) * (1 - p_judge)
    return round((observed - expected) / (1 - expected), 3) if expected < 1 else None


def judge_table(collected):
    out = {}
    for profile, data in collected.items():
        rows = data["judge"]
        valid = [r for r in rows if r["verdict"] is not None]
        pairs = [(r["label"]["useful"] >= 1 and not r["label"]["critical_error"], r["verdict"]["useful"] >= 1 and not r["verdict"]["critical_error"]) for r in valid]
        out[profile] = {"n": len(rows), "valid": len(valid), "accuracy": round(sum(a == b for a, b in pairs) / len(pairs), 3), "kappa": kappa(pairs),
                        "false_approvals": sum(b and not a for a, b in pairs), "false_rejections": sum(a and not b for a, b in pairs),
                        "label_pass": sum(a for a, _ in pairs), "judge_pass": sum(b for _, b in pairs),
                        "critical_found": sum(r["label"]["critical_error"] and r["verdict"]["critical_error"] for r in valid),
                        "critical_labelled": sum(r["label"]["critical_error"] for r in valid),
                        "useful_exact": sum(r["label"]["useful"] == r["verdict"]["useful"] for r in valid),
                        "verdicts": dict(Counter(r["raw"] for r in rows))}
    return out


def json_table(collected):
    out = {}
    for profile, data in collected.items():
        out[profile] = {}
        for way in ("free_text", "json_mode", "bounded_choice"):
            rows = [r for r in data["json"] if r["way"] == way]
            out[profile][way] = {"n": len(rows), "parsed": sum(r["parsed"] for r in rows), "schema_valid": sum(r["schema_valid"] for r in rows),
                                 "truncated": sum(bool(r["truncated"]) for r in rows), "median_seconds": round(statistics.median(r["seconds"] for r in rows), 2)}
    return out


def cost_table(collected):
    return {p: {"speed": d["speed"], "peak_mb_llm_phase": d["peak_mb_llm_phase"], "peak_mb_end": d["peak_mb_end"], "dialogue": d["dialogue"],
                "voice": {k: {"p50": v["p50"], "p95": v["p95"], "llm_p50": v["llm_p50"],
                              "asr_p50": round(statistics.median(t["asr"] for t in v["turns"]), 2),
                              "tts_first_p50": round(statistics.median(t["tts_first"] for t in v["turns"]), 2)} for k, v in d["voice"].items()},
                "open_seconds": d["open_seconds"]} for p, d in collected.items()}


def build(folder=FROZEN):
    collected, annotations = load(folder)
    result = {"task": "T51", "pass_rule": collected["tiny"]["pass_rule"], "judge": judge_table(collected), "json": json_table(collected),
              "cost": cost_table(collected)}
    if annotations:
        check_annotations(collected, annotations)
        result["rubric"] = rubric_table(collected, annotations)
        result["annotation_rater"] = sorted({a["rater"] for a in annotations})
    return result
