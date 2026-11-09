"""Render the 17 parameter experiments; each HTML's CONFIG remains its data source.

Shared module-experiments.js contains calculations, visual-style.css the design.
This tool never changes lesson Markdown or learner state.
"""
from pathlib import Path
from html import escape
import json

ROOT = Path(__file__).resolve().parents[1]


def build() -> None:
    for path in sorted((ROOT / "course/visuals").glob("M??.html")):
        config = json.JSONDecoder().raw_decode(path.read_text(encoding="utf-8").split("const CONFIG=", 1)[1])[0]
        mid, title, label, minimum, maximum, step, value, question, limitation = config
        output = f'''<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escape(mid)} · {escape(title)}</title><link rel="stylesheet" href="theme-tokens.css"><link rel="stylesheet" href="visual-style.css"></head>
<body><main><span class="tag">{escape(mid)} · Численный эксперимент</span>
<h1>{escape(title)}</h1><p class="question">{escape(question)}</p>
<div class="control"><label for="parameter"><span>{escape(label)}</span><output id="value" for="parameter">{value}</output></label>
<input id="parameter" aria-label="{escape(label, quote=True)}" type="range" min="{minimum}" max="{maximum}" step="{step}" value="{value}"></div>
<div class="figure" id="figure"></div><p class="result" id="result" role="status" aria-live="polite"></p>
<button id="reset" type="button">Исходное значение</button><p class="limit"><strong>Граница модели.</strong> {escape(limitation)}</p>
</main><script>const CONFIG={json.dumps(config, ensure_ascii=False)};</script>
<script src="module-experiments.js"></script><script src="visual-frame.js"></script></body></html>
'''
        path.write_text(output, encoding="utf-8", newline="\n")
        print(path.name)


if __name__ == "__main__":
    build()
