"""F4: design 3.0 acceptance — token contrast, a green report that matches its evidence, the release link."""
from __future__ import annotations

import importlib.util
import json
import subprocess
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("design_v3_report", ROOT / "tools/build_design_v3_report.py")
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)
REPORT = json.loads((ROOT / "quality/design-v3/report.json").read_text(encoding="utf-8"))
CSS = (ROOT / "tools/reader.css").read_text(encoding="utf-8")


def test_text_and_background_token_pairs_meet_4_5_in_every_palette():
    pairs = builder.token_contrast(CSS)
    assert len(pairs) == 3 * len(builder.TEXT_PAIRS) == 126
    assert [pair for pair in pairs if not pair["pass"]] == []
    # The light palette is the print palette.
    assert builder.token_sets(CSS)["light"] == builder.token_sets(CSS)["print"]


def test_browser_check_and_report_builder_use_the_same_pairs():
    script = ("const c=require('./tools/check_reader_design.cjs');"
              "console.log(JSON.stringify({pairs:c.TEXT_PAIRS,rows:c.tokenContrast(require('fs').readFileSync('tools/reader.css','utf8'))}))")
    result = json.loads(subprocess.run(["node", "-e", script], cwd=ROOT, capture_output=True, text=True, check=True).stdout)
    assert [tuple(pair) for pair in result["pairs"]] == builder.TEXT_PAIRS
    assert result["rows"] == builder.token_contrast(CSS)


def test_report_is_green_and_matches_the_current_tokens_and_its_evidence():
    assert REPORT["status"] == "green" and REPORT["kind"] == "design_acceptance_not_learner_study"
    assert REPORT["reader_checks"]["failed"] == 0 and REPORT["experiment_checks"] == {"cases": 52, "passed": 58, "failed": 0}
    acceptance = REPORT["acceptance"]
    assert acceptance["external_requests"] == 0 and acceptance["colour_literals_outside_root"] == 0
    assert acceptance["fonts"]["golos_text_faces"] == 6 and acceptance["fonts"]["jetbrains_mono_faces"] == 4
    assert set(builder.REQUIRED_CHECKS) <= set(REPORT["reader_checks"]["names"])
    # A palette change after the acceptance run needs a new run and a new report.
    assert REPORT["token_contrast"] == builder.token_contrast(CSS)
    for item in REPORT["screenshots"] + REPORT["print"]:
        assert sha256((ROOT / item["path"]).read_bytes()).hexdigest() == item["sha256"], item["path"]
    themes = {Path(item["path"]).stem.rsplit("-", 1)[-1] for item in REPORT["screenshots"]}
    assert {"dark", "light"} <= themes and len(REPORT["print"]) == 2


def test_release_status_links_the_design_report():
    status = (ROOT / "RELEASE_STATUS.md").read_text(encoding="utf-8")
    assert "## Оформление 3.0" in status and "](quality/design-v3/report.json)" in status
