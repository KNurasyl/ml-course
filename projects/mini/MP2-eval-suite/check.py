"""MP2 check: the evaluation set, rubric, annotations, agreement and regression tests in submissions/MP2/.

    python projects/mini/MP2-eval-suite/check.py [--dir submissions/MP2]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import SUBMISSIONS, Report, cohen_kappa, kb_documents, read_jsonl  # noqa: E402


def check(folder: Path, run_tests: bool = True) -> Report:
    report = Report(f"MP2 «Набор проверок»: проверка {folder}")
    kb = kb_documents()
    cases = read_jsonl(folder / "cases.jsonl") if (folder / "cases.jsonl").is_file() else []
    ids = [c.get("id") for c in cases]
    report.check(len(cases) >= 40 and len(set(ids)) == len(ids), "не меньше 40 случаев с уникальными id", f"случаев {len(cases)}")
    no_answer = [c for c in cases if c.get("answerable") is False]
    report.check(len(no_answer) >= 8, "не меньше 8 случаев без ответа в базе", f"таких {len(no_answer)}")
    unknown = sorted({s for c in cases for s in c.get("expected_sources") or [] if s not in kb})
    report.check(not unknown, "все expected_sources есть в базе знаний", f"нет в базе: {unknown[:5]}" if unknown else "")
    incomplete = [c.get("id") for c in cases if c.get("answerable") and (not c.get("expected_sources") or not c.get("expected_facts"))]
    report.check(not incomplete, "у случаев с ответом есть источник и ожидаемый факт", f"без них: {incomplete[:5]}" if incomplete else "")
    leaked = [c.get("id") for c in no_answer if c.get("expected_sources")]
    report.check(not leaked, "у случаев без ответа нет источников", f"с источником: {leaked[:5]}" if leaked else "")
    rubric = (folder / "rubric.md").read_text(encoding="utf-8") if (folder / "rubric.md").is_file() else ""
    criteria = re.findall(r"(?m)^##\s+\S.*$", rubric)
    report.check(len(criteria) >= 3 and "..." not in rubric and any("отказ" in c.lower() for c in criteria),
                 "в рубрике не меньше трёх заполненных критериев, один про отказ", f"критериев {len(criteria)}")
    rows = read_jsonl(folder / "annotations.jsonl") if (folder / "annotations.jsonl").is_file() else []
    paired = [r for r in rows if r.get("manual") is not None and r.get("auto") is not None]
    report.check(len(paired) >= 20, "не меньше 20 ответов размечены вручную и автоматически", f"размечено {len(paired)}")
    agreement = json.loads((folder / "agreement.json").read_text(encoding="utf-8")) if (folder / "agreement.json").is_file() else {}
    if len(paired) >= 2:
        kappa = cohen_kappa([r["manual"] for r in paired], [r["auto"] for r in paired])
        stated = agreement.get("kappa")
        report.check(isinstance(stated, (int, float)) and abs(stated - kappa) <= 0.01 and agreement.get("n") == len(paired),
                     "каппа Коэна в agreement.json совпадает с пересчётом", f"пересчёт {kappa:.3f} на {len(paired)}, указано {stated}")
    else:
        report.check(False, "каппа Коэна в agreement.json совпадает с пересчётом", "мало размеченных пар")
    note = str(agreement.get("note") or "")
    report.check(len(note.split()) >= 10 and "TODO" not in note, "вывод о согласии записан (note, от 10 слов)")
    tests = folder / "test_regression.py"
    text = tests.read_text(encoding="utf-8") if tests.is_file() else ""
    report.check(len(re.findall(r"(?m)^def test_", text)) >= 2, "в test_regression.py не меньше двух тестов")
    if run_tests and text:
        done = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(tests)],
                              capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=folder)
        last = (done.stdout.strip().splitlines() or [""])[-1]
        report.check(done.returncode == 0, "регрессионные тесты проходят", last)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=SUBMISSIONS / "MP2")
    return check(parser.parse_args().dir).finish()


if __name__ == "__main__":
    sys.exit(main())
