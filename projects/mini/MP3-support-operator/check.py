"""MP3 check: two runs of the support operator over the MP2 set, metrics recomputed from the run files.

    python projects/mini/MP3-support-operator/check.py [--dir submissions/MP3] [--cases submissions/MP2/cases.jsonl]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import SUBMISSIONS, Report, kb_documents, read_jsonl  # noqa: E402

RUNS = ("baseline", "improved")
METRICS = ("citation_valid", "source_hit", "refusal_recall", "false_refusal", "retrieval_recall")
HIGHER_IS_BETTER = {"citation_valid": True, "source_hit": True, "refusal_recall": True, "false_refusal": False,
                    "retrieval_recall": True}


def metrics(cases: list[dict], rows: list[dict], kb: dict) -> dict:
    by_id = {r.get("case_id"): r for r in rows}
    answerable = [c for c in cases if c.get("answerable")]
    unanswerable = [c for c in cases if c.get("answerable") is False]
    share = lambda hits, total: round(hits / total, 4) if total else 0.0
    return {
        "citation_valid": share(sum(all(s in kb for s in r.get("cited") or []) for r in rows), len(rows)),
        "source_hit": share(sum(bool(set(by_id.get(c["id"], {}).get("cited") or []) & set(c["expected_sources"]))
                                for c in answerable), len(answerable)),
        "refusal_recall": share(sum(bool(by_id.get(c["id"], {}).get("refused")) for c in unanswerable), len(unanswerable)),
        "false_refusal": share(sum(bool(by_id.get(c["id"], {}).get("refused")) for c in answerable), len(answerable)),
        "retrieval_recall": share(sum(bool(set(by_id.get(c["id"], {}).get("retrieved") or []) & set(c["expected_sources"]))
                                      for c in answerable), len(answerable)),
    }


def check(folder: Path, cases_path: Path) -> Report:
    report = Report(f"MP3 «Оператор поддержки»: проверка {folder}")
    kb = kb_documents()
    cases = read_jsonl(cases_path) if cases_path.is_file() else []
    report.check(len(cases) >= 40 and sum(c.get("answerable") is False for c in cases) >= 8,
                 "набор MP2 на месте: от 40 случаев, от 8 без ответа", str(cases_path))
    stated = json.loads((folder / "metrics.json").read_text(encoding="utf-8")) if (folder / "metrics.json").is_file() else {}
    computed = {}
    for run in RUNS:
        path = folder / "runs" / f"{run}.jsonl"
        rows = read_jsonl(path) if path.is_file() else []
        covered = {r.get("case_id") for r in rows} >= {c["id"] for c in cases} and bool(cases)
        report.check(covered, f"{run}: ответ на каждый случай MP2", f"ответов {len(rows)} из {len(cases)}")
        computed[run] = metrics(cases, rows, kb)
        mismatch = [m for m in METRICS
                    if not isinstance((stated.get(run) or {}).get(m), (int, float)) or abs(stated[run][m] - computed[run][m]) > 0.01]
        report.check(not mismatch, f"{run}: метрики в metrics.json совпадают с пересчётом",
                     ", ".join(f"{m}: пересчёт {computed[run][m]}" for m in mismatch) if mismatch else
                     ", ".join(f"{m} {computed[run][m]}" for m in METRICS))
    report.check(computed["improved"]["citation_valid"] == 1.0, "в улучшенной версии все ссылки существуют в базе")
    changes = str(stated.get("changes") or "")
    report.check(len(changes.split()) >= 20, "изменения между версиями описаны (от 20 слов)")
    worse = [m for m in METRICS if (computed["improved"][m] - computed["baseline"][m]) * (1 if HIGHER_IS_BETTER[m] else -1) < 0]
    report.check(all(m in changes for m in worse), "ухудшение метрик не скрыто: каждая упавшая метрика названа в changes",
                 f"упали: {worse}" if worse else "ни одна метрика не упала")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=SUBMISSIONS / "MP3")
    parser.add_argument("--cases", type=Path, default=SUBMISSIONS / "MP2/cases.jsonl")
    args = parser.parse_args()
    return check(args.dir, args.cases).finish()


if __name__ == "__main__":
    sys.exit(main())
