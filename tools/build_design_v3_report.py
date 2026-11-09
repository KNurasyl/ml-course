"""Archive the design 3.0 acceptance (phase 6, F4) into quality/design-v3/report.json.

Run after `node tools/run_design_checks.cjs` and after looking at its screenshots. Like
build_design_report.py for design 2.0, this records observed evidence and refuses a run with
failures or one made on another reader build. It does not start a browser and does not assess
a learner. The token contrast is computed here again from tools/reader.css, independently of
the browser check, and both must agree.
"""
from __future__ import annotations

import json
import re
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "quality/design-v3"
TEXTS = ["ink", "muted", "accent", "act", "hint", "ok", "warn", "error"]
GROUNDS = ["bg", "surface", "paper", "code-bg"]
TEXT_PAIRS = [(text, ground) for text in TEXTS for ground in GROUNDS] + [
    ("accent", "accent-soft"), ("act", "act-soft"), ("hint", "hint-soft"), ("ok", "ok-soft"), ("warn", "warn-soft"),
    ("ink", "accent-soft"), ("ink", "act-soft"), ("ink", "hint-soft"), ("mark-fg", "mark-bg"), ("bg", "accent")]
REQUIRED_CHECKS = [
    "F2 no requests outside file:, data:, about: and blob:",
    "F2 embedded Golos Text is used for text",
    "F2 embedded JetBrains Mono is used for code",
    "D1 colour literals only inside :root token blocks",
    "F4 text/background token pairs meet 4.5:1 in dark, light and print",
]


def read(relative: str):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def token_sets(css: str) -> dict[str, dict[str, str]]:
    patterns = {"dark": r":root\{color-scheme:dark;([^}]*)\}",
                "light": r':root\[data-theme="light"\]\{color-scheme:light;([^}]*)\}',
                "print": r"@media print\{:root\{color-scheme:light;([^}]*)\}\}"}
    return {theme: dict(re.findall(r"--([\w-]+):(#[0-9a-fA-F]{6})", re.search(pattern, css).group(1)))
            for theme, pattern in patterns.items()}


def contrast(a: str, b: str) -> float:
    def luminance(colour: str) -> float:
        channels = [int(colour[i:i + 2], 16) / 255 for i in (1, 3, 5)]
        channels = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
        return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]
    high, low = sorted((luminance(a), luminance(b)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def token_contrast(css: str) -> list[dict]:
    return [{"theme": theme, "text": text, "ground": ground, "ratio": round(contrast(tokens[text], tokens[ground]), 3),
             "pass": contrast(tokens[text], tokens[ground]) >= 4.5}
            for theme, tokens in token_sets(css).items() for text, ground in TEXT_PAIRS]


def evidence(folder: Path, pattern: str) -> list[dict]:
    return [{"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path.read_bytes()).hexdigest()}
            for path in sorted(folder.rglob(pattern))]


def detail(checks: dict, name: str) -> dict:
    return checks[name]["detail"]


def main() -> dict:
    reader = read("artifacts/design-v3/reader/qa-results.json")
    frames = read("artifacts/design-v3/integration/qa-results.json")
    reader_hash = sha256((ROOT / "START_HERE.html").read_bytes()).hexdigest()
    for name, data in (("reader", reader), ("experiments", frames)):
        if data["failed"] or data["errors"]:
            raise SystemExit(f"{name} checks contain failures: rerun and fix before archiving")
        if data.get("reader_sha256") != reader_hash:
            raise SystemExit(f"{name} checks ran on another reader build: rerun tools/run_design_checks.cjs")
    checks = {check["name"]: check for check in reader["checks"]}
    missing = [name for name in REQUIRED_CHECKS if name not in checks]
    if missing:
        raise SystemExit(f"required checks missing: {missing}")
    if len(frames["cases"]) != 52 or frames["passed"] != 58:
        raise SystemExit("experiment checks must cover 26 experiments at two widths and 6 behaviours")
    pairs = token_contrast((ROOT / "tools/reader.css").read_text(encoding="utf-8"))
    if pairs != reader["measurements"]["tokenContrast"]:
        raise SystemExit("token contrast differs from the browser check: rerun after the last stylesheet change")
    weak = [pair for pair in pairs if not pair["pass"]]
    if weak:
        raise SystemExit(f"token pairs below 4.5:1: {weak}")
    screenshots = evidence(OUT / "screenshots", "*.png")
    theme_shots = [shot for shot in screenshots if shot["path"].count("/") == 3]
    if not any(s["path"].endswith("-dark.png") for s in theme_shots) or not any(s["path"].endswith("-light.png") for s in theme_shots):
        raise SystemExit("screenshots of both themes are required")
    by_theme = {theme: min(p["ratio"] for p in pairs if p["theme"] == theme) for theme in ("dark", "light", "print")}
    fonts = detail(checks, "F2 embedded Golos Text is used for text")
    report = {
        "kind": "design_acceptance_not_learner_study",
        "design": "3.0 (phase 6: dark theme and fonts)",
        "status": "green",
        "date": reader["date"],
        "reader_sha256": reader_hash,
        "acceptance": {
            "external_requests": len(detail(checks, "F2 no requests outside file:, data:, about: and blob:")["external"]),
            "requests_seen": detail(checks, "F2 no requests outside file:, data:, about: and blob:")["requests"],
            "fonts": {"golos_text_faces": fonts["declared"], "golos_text_loaded": fonts["loaded"],
                      "jetbrains_mono_faces": detail(checks, "F2 embedded JetBrains Mono is used for code")["declared"]},
            "colour_literals_outside_root": detail(checks, "D1 colour literals only inside :root token blocks")["count"],
            "token_pairs": {"count": len(pairs), "threshold": 4.5, "min_ratio_by_theme": by_theme},
        },
        "reader_checks": {"passed": reader["passed"], "failed": reader["failed"], "names": [check["name"] for check in reader["checks"]]},
        "experiment_checks": {"cases": len(frames["cases"]), "passed": frames["passed"], "failed": frames["failed"]},
        "token_contrast": pairs,
        "screenshots": screenshots,
        "print": evidence(OUT / "print", "*.pdf"),
        "how_to_reproduce": ["node tools/run_design_checks.cjs (PW_PATH points at a node_modules folder with playwright)",
                             "look at quality/design-v3/screenshots", "python tools/build_design_v3_report.py"],
        "limits": [
            "Contrast is checked for text/background token pairs, not for every rendered element; matplotlib light files keep the 1.1.0 palette.",
            "Print was checked through Edge's PDF output (page.pdf) and its built-in viewer, not through a printer.",
            "Browser checks ran in Microsoft Edge only; Firefox and Safari were not run.",
            "These checks describe the textbook build; they do not assess a learner.",
        ],
    }
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return report


if __name__ == "__main__":
    result = main()
    print(json.dumps({"status": result["status"], "reader_checks": result["reader_checks"]["passed"],
                      "token_pairs": result["acceptance"]["token_pairs"], "screenshots": len(result["screenshots"])}, ensure_ascii=False))
