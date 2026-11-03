"""Build an offline reading copy from the course's canonical Markdown files.

The reader is one static HTML file. It makes no network requests, stores no
learning state and does not execute notebook code. Edit Markdown, then rebuild.
"""
from __future__ import annotations

import argparse
from html import escape, unescape
from hashlib import sha256
import json
from pathlib import Path
import re
from urllib.parse import quote, unquote, urlsplit

import mistune


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = "START_HERE.html"
VERSION = "1.0.0"
PILOT_IDS = [f"P{i:02}" for i in range(1, 11)]
MODULE_IDS = [f"M{i:02}" for i in range(17)]
FOUNDATION_IDS = ["HOME", "DIAGNOSTIC", "GLOSSARY", "PROJECT", "MENTOR", "PROGRESS"]


def text_only(value: str) -> str:
    return unescape(re.sub(r"<[^>]+>", "", value))


def slug(value: str) -> str:
    value = text_only(value).strip().lower()
    return re.sub(r"[^\w-]+", "-", value, flags=re.UNICODE).strip("-") or "section"


def checked_file(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root.resolve()) or not candidate.is_file():
        raise ValueError(f"Missing or external course file: {relative}")
    return candidate


def source_fingerprint(root: Path) -> str:
    """Fingerprint every canonical text, its registry and the rendering code."""
    root = root.resolve()
    registry = json.loads((root / "metadata/material-registry.json").read_text(encoding="utf-8"))
    paths = {item["path"] for item in registry.values()}
    paths.update({"metadata/material-registry.json", "tools/build_reader.py"})
    for folder in ("course/assets", "course/visuals"):
        paths.update(path.relative_to(root).as_posix() for path in (root / folder).glob("*") if path.is_file())
    digests = {}
    for relative in paths:
        source = checked_file(root, relative)
        digests[source.relative_to(root).as_posix()] = sha256(source.read_bytes()).hexdigest()
    serialized = json.dumps(digests, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(serialized).hexdigest()


class CourseRenderer(mistune.HTMLRenderer):
    def __init__(self, root: Path, source: Path, material_id: str, path_ids: dict[Path, str]):
        super().__init__(escape=True)
        self.root = root
        self.source = source
        self.material_id = material_id
        self.path_ids = path_ids
        self.heading_counts: dict[str, int] = {}

    def destination(self, url: str) -> tuple[str | None, bool]:
        parsed = urlsplit(url)
        if parsed.scheme in {"http", "https"}:
            return url, True
        if parsed.scheme or parsed.netloc:
            return None, False
        if not parsed.path:
            return f"#{self.material_id}--{unquote(parsed.fragment)}" if parsed.fragment else f"#{self.material_id}", False
        local = (self.source.parent / unquote(parsed.path)).resolve()
        if not local.is_relative_to(self.root):
            raise ValueError(f"Link leaves repository in {self.source}: {url}")
        if not local.exists():
            raise ValueError(f"Broken link in {self.source.relative_to(self.root)}: {url}")
        if local in self.path_ids:
            material = self.path_ids[local]
            return f"#{material}" + (f"--{unquote(parsed.fragment)}" if parsed.fragment else ""), False
        target = quote(local.relative_to(self.root).as_posix(), safe="/:@")
        if parsed.fragment:
            target += "#" + parsed.fragment
        return target, False

    def link(self, text: str, url: str, title: str | None = None) -> str:
        target, external = self.destination(url)
        if target is None:
            return f'<span class="unsupported-link">{text}</span>'
        attrs = f' href="{escape(target, quote=True)}"'
        if title:
            attrs += f' title="{escape(title, quote=True)}"'
        if external:
            attrs += ' target="_blank" rel="noopener noreferrer" class="external-link"'
        elif target.endswith(".ipynb"):
            attrs += ' class="notebook-link" title="Практический файл: открой его в Jupyter"'
        return f"<a{attrs}>{text}</a>"

    def image(self, text: str, url: str, title: str | None = None) -> str:
        target, external = self.destination(url)
        if target is None:
            return f"<span>{escape(text)}</span>"
        if external:
            return self.link(f"Изображение: {escape(text)}", url, title)
        return f'<img src="{escape(target, quote=True)}" alt="{escape(text, quote=True)}" loading="lazy">'

    def heading(self, text: str, level: int, **attrs) -> str:
        base = slug(text)
        count = self.heading_counts.get(base, 0)
        self.heading_counts[base] = count + 1
        anchor = f"{self.material_id}--{base}" + (f"-{count}" if count else "")
        level = max(2, level)
        return f'<h{level} id="{escape(anchor, quote=True)}">{text}</h{level}>\n'


def without_repeated_title(markdown: str, title: str) -> str:
    lines = markdown.splitlines()
    first = next((i for i, line in enumerate(lines) if line.strip()), None)
    if first is not None and lines[first].startswith("# ") and slug(lines[first][2:]) == slug(title):
        del lines[first]
    return "\n".join(lines)


def fold_hints(content: str) -> str:
    """Keep support available, without showing it before the learner's attempt."""
    pattern = re.compile(r'(<h2\b[^>]*>)(.*?)(</h2>)(.*?)(?=<h2\b|\Z)', re.DOTALL)

    def replace(match: re.Match) -> str:
        title = text_only(match.group(2))
        if "подсказ" not in title.lower() and "ошибки и критерии самопроверки" not in title.lower():
            return match.group(0)
        anchor = re.search(r'\bid="([^"]+)"', match.group(1))
        identifier = f' id="{anchor.group(1)}"' if anchor else ""
        return '<details class="hints"' + identifier + '><summary>' + escape(title) + '</summary><div>' + match.group(4) + '</div></details>\n'

    return pattern.sub(replace, content)


def ordered_materials(registry: dict) -> list[str]:
    route = [key for module in MODULE_IDS for key in [module, *[f"{module}-L{i:02}" for i in range(1, 6)]]]
    primary = ["HOME", "DIAGNOSTIC", *PILOT_IDS, *route, "GLOSSARY", "PROJECT", "MENTOR", "PROGRESS", "MENTOR-KEYS", *[f"{m}-SOLUTIONS" for m in MODULE_IDS]]
    return [key for key in primary if key in registry] + [key for key in registry if key not in primary]


def display_status(key: str) -> tuple[str, str]:
    if key in PILOT_IDS:
        return "Готовое занятие", "ready"
    if key in MODULE_IDS:
        return "Модуль и домашняя работа", "ready"
    if re.fullmatch(r"M\d{2}-L\d{2}", key):
        return "Урок · Теория и практика", "ready"
    if key == "MENTOR-KEYS" or key.endswith("-SOLUTIONS"):
        return "После своей попытки", "answer"
    return "Материал курса", "reference"


def display_id(key: str) -> str:
    return {"HOME": "Путеводитель", "DIAGNOSTIC": "Начальная проверка", "GLOSSARY": "Справочник",
            "PROJECT": "Сквозной проект", "MENTOR": "Как учиться", "PROGRESS": "Запись результатов",
            "MENTOR-KEYS": "Самопроверка", "REF-CATALOG": "Дополнительные курсы",
            "REF-METHOD": "Методика обучения"}.get(key, key)


def nav_item(key: str, item: dict) -> str:
    short = re.sub(rf"^{re.escape(key)}\s*[:.·—–-]?\s*", "", item["title"]).strip()
    marker = key if key in PILOT_IDS or key in MODULE_IDS else {"HOME": "01", "DIAGNOSTIC": "02"}.get(key, "•")
    return f'<li data-key="{escape(key)}"><a href="#{escape(key)}" data-material="{escape(key)}"><span class="nav-marker">{escape(marker)}</span><span>{escape(short)}</span></a></li>'


CSS = r"""
:root{color-scheme:light;--ink:#23312e;--muted:#62716c;--paper:#f5f4ef;--white:#fffefa;--line:#dddeda;--accent:#236653;--soft:#e7f0e8;--blue:#285b83;--sidebar:292px}
*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:96px}body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.75 'Segoe UI',Arial,sans-serif}a{color:var(--blue);text-underline-offset:3px}a:hover{color:var(--accent)}button,input{font:inherit}button,a,input,summary{-webkit-tap-highlight-color:transparent}button:focus-visible,a:focus-visible,input:focus-visible,summary:focus-visible{outline:3px solid #d59f43;outline-offset:4px}button{cursor:pointer}button:disabled{opacity:.45;cursor:default}[hidden]{display:none!important}
.sidebar{position:fixed;inset:0 auto 0 0;width:var(--sidebar);background:#eceee7;border-right:1px solid var(--line);overflow:auto;z-index:30;padding:28px 21px 36px}.brand{display:block;text-decoration:none;color:var(--ink);margin-bottom:20px}.brand-symbol{display:inline-flex;align-items:center;justify-content:center;vertical-align:middle;width:38px;height:38px;background:var(--accent);color:white;border-radius:11px;font-weight:700;letter-spacing:-1px;margin-right:9px}.brand-name{font-size:18px;font-weight:700}.brand-caption{display:block;color:var(--muted);font-size:12px;line-height:1.45;letter-spacing:.02em;margin:10px 0 0 1px}.search{display:block;margin:18px 0 7px}.search span{font-size:12px;color:var(--muted);font-weight:600}.search input{display:block;width:100%;border:1px solid #cdd3cb;border-radius:8px;background:var(--white);padding:9px 11px;margin-top:5px;line-height:1.4;min-width:0}.search-status{font-size:12px;line-height:1.4;color:var(--muted);min-height:18px;margin:6px 0 19px}.nav-group{margin:20px 0}.nav-module>summary{cursor:pointer;font-size:12px;font-weight:600;line-height:1.5;color:#42604a;padding:7px 10px}.nav-module[open]>summary{margin-bottom:5px}.nav-group h2{font-size:10px;text-transform:uppercase;letter-spacing:.13em;line-height:1.5;color:#6b786e;margin:0 10px 9px}.nav-group ul{list-style:none;padding:0;margin:0}.nav-group a{display:flex;gap:10px;align-items:flex-start;text-decoration:none;color:#43524a;font-size:13px;line-height:1.45;padding:8px 10px;border-radius:7px}.nav-group a:hover{background:#e1e7dd}.nav-group a[aria-current="page"]{background:var(--white);color:var(--accent);box-shadow:0 1px 5px #23312e0a;font-weight:600}.nav-marker{font-size:10px;font-weight:700;color:#7a897d;flex:0 0 27px;padding-top:2px}.nav-group a[aria-current="page"] .nav-marker{color:var(--accent)}.nav-foot{font-size:11px;color:var(--muted);line-height:1.6;padding:17px 10px 0;border-top:1px solid #d8ddd3}
.layout{margin-left:var(--sidebar)}.topbar{position:sticky;top:0;z-index:20;background:#f5f4eff2;backdrop-filter:blur(12px);border-bottom:1px solid var(--line);min-height:68px;padding:14px 44px;display:flex;justify-content:space-between;gap:18px;align-items:center}.top-label{font-size:12px;color:var(--muted);letter-spacing:.02em}.top-actions{display:flex;gap:9px;align-items:center}.small-button{border:1px solid #d4d9d0;background:var(--white);border-radius:7px;padding:6px 12px;color:#43524a;font-size:12px;line-height:1.5}.small-button:hover{background:var(--soft)}.menu-toggle{display:none}.reader{max-width:1090px;margin:0 auto;padding:42px 54px 80px}.book-label{display:flex;align-items:center;gap:8px;margin-bottom:21px;font-size:11px;font-weight:600;letter-spacing:.11em;text-transform:uppercase;color:var(--accent)}.book-label:before{content:"";width:7px;height:7px;border-radius:50%;background:#70a37f}.material{background:var(--white);border:1px solid var(--line);border-radius:13px;box-shadow:0 3px 18px #23312e03;margin:0 0 34px;padding:37px 43px 28px;overflow-wrap:anywhere;scroll-margin-top:90px}.material-head{padding-bottom:24px;border-bottom:1px solid #e7e8e1;margin-bottom:25px}.eyebrow{display:flex;gap:10px;align-items:center;flex-wrap:wrap;font-size:11px;color:var(--muted);margin-bottom:12px;letter-spacing:.025em}.material-id{font-weight:700;letter-spacing:.08em}.badge{border:1px solid #d9dfd2;padding:2px 8px;border-radius:100px;font-size:10px;background:#f4f6f0;line-height:1.6}.badge.ready{background:#e9f4e9;border-color:#d4e6d4;color:#316b46}.badge.draft{background:#f7f1df;border-color:#eadfbb;color:#76672c}.badge.answer{background:#f1ecf5;border-color:#ded4e7;color:#715b7d}h1{font-size:clamp(27px,3.1vw,38px);line-height:1.2;letter-spacing:-.025em;margin:0;font-weight:650;max-width:780px}.body-content{font-size:16px}.body-content>p:first-child{color:#4b5d53}.body-content p{margin:0 0 19px}.body-content h2{font-size:23px;line-height:1.4;letter-spacing:-.015em;margin:35px 0 16px;padding-top:4px}.body-content h3{font-size:19px;line-height:1.45;margin:28px 0 13px}.body-content h4{font-size:17px;margin:23px 0 12px}.body-content ul,.body-content ol{padding-left:25px;margin:12px 0 22px}.body-content li{padding-left:3px;margin:7px 0}.body-content li>p{margin-bottom:7px}.body-content blockquote{margin:23px 0;background:#f2f6ed;border-left:3px solid #7a9c78;padding:16px 20px;color:#456048}.body-content blockquote p:last-child{margin-bottom:0}.body-content code{font-family:Consolas,'Cascadia Code',monospace;background:#f1f2eb;padding:2px 5px;border-radius:4px;font-size:.88em;overflow-wrap:anywhere}.body-content pre{background:#263731;color:#f3f5ed;border:1px solid #263731;border-radius:9px;padding:18px 21px;line-height:1.65;overflow:auto;max-width:100%;font-size:14px;margin:23px 0}.body-content pre code{background:none;color:inherit;padding:0;font-size:inherit;white-space:pre;overflow-wrap:normal}.table-wrap{overflow:auto;max-width:100%;border:1px solid #dfe3d8;border-radius:8px;margin:21px 0 25px}.body-content table{border-collapse:collapse;width:100%;font-size:13px;line-height:1.6;min-width:400px}.body-content th{text-align:left;background:#f0f3ea;font-weight:600;color:#40553e}.body-content th,.body-content td{padding:12px 15px;border-bottom:1px solid #e5e8df;vertical-align:top}.body-content tbody tr:last-child td{border-bottom:0}.body-content tr:nth-child(even) td{background:#fafbf6}.body-content img{max-width:100%;height:auto}.experiment{width:100%;height:660px;border:1px solid #d8e2d0;border-radius:10px;background:#f5f7ee}.body-content hr{border:0;border-top:1px solid var(--line);margin:30px 0}.hints,.answer-key{border:1px solid #dce3d7;border-radius:9px;background:#f5f7f0;margin:25px 0;overflow:hidden}.hints summary,.answer-key>summary{font-weight:600;color:#43613f;cursor:pointer;padding:16px 19px;font-size:15px}.hints>div,.answer-key>div{padding:6px 20px 8px}.answer-key{background:#faf7fb;border-color:#e4dce8}.answer-key>summary{color:#70597c}.answer-note{color:#715d78;font-size:14px;line-height:1.7}.practice-note{padding:13px 16px;background:#f4f6ee;border-radius:7px;font-size:13px;line-height:1.65;color:#596e50;margin:0 0 24px}.source-note{font-size:11px;color:#7b897e;border-top:1px solid #e7e8e1;margin-top:30px;padding-top:14px}.source-note code{font-size:11px}.reading-nav{display:flex;gap:16px;justify-content:space-between;margin:22px 0 0}.reading-nav a{text-decoration:none;color:var(--accent);font-size:13px;line-height:1.6;background:#eef3e8;border:1px solid #dce5d2;border-radius:7px;padding:11px 16px;max-width:48%}.reading-nav a:last-child{margin-left:auto;text-align:right}.reading-nav small{display:block;font-size:10px;color:#7a8871;text-transform:uppercase;letter-spacing:.06em}.empty-search{display:none;color:#6d7768;font-size:13px;padding:9px 10px}.empty-search.is-visible{display:block}.no-script{background:#fff4d8;padding:12px 18px;color:#5e4b24;border-radius:6px}.backdrop{display:none}.screen-reader-only{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap;border:0}
@media(min-width:1500px){.reader{max-width:1140px;padding-top:52px}.material{padding:42px 55px}}
@media(max-width:1120px){:root{--sidebar:255px}.sidebar{padding:24px 14px}.reader{padding:31px 27px 60px}.material{padding:30px}.topbar{padding:14px 27px}.body-content{font-size:15px}}
@media(max-width:800px){.layout{margin-left:0}.sidebar{transform:translateX(-101%);width:290px;transition:transform .15s ease}.nav-open .sidebar{transform:translateX(0)}.nav-open .backdrop{display:block;position:fixed;inset:0;background:#12201966;z-index:25;border:0}.menu-toggle{display:inline-block}.topbar{padding:12px 18px;min-height:62px}.top-label{font-size:11px}.reader{padding:24px 17px 42px}.material{padding:26px 22px;border-radius:10px}.book-label{margin-left:4px;font-size:10px}.material-head{padding-bottom:19px}.body-content h2{font-size:21px}.reading-nav a{padding:10px 11px;font-size:12px}.body-content pre{padding:15px;font-size:12px}.body-content table{font-size:12px}.body-content th,.body-content td{padding:10px 12px}.source-note{font-size:10px}}
@media(max-width:430px){.top-label{max-width:142px;line-height:1.4}.print-button{padding:6px 9px;font-size:11px}.material{padding:23px 18px}.reader{padding:21px 11px 35px}.body-content{font-size:15px}.body-content h2{font-size:20px}.reading-nav{gap:9px}.practice-note{padding:12px;font-size:12px}h1{font-size:27px}}
@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}.sidebar{transition:none}}
@media print{body{background:white;font-size:11pt}.sidebar,.topbar,.book-label,.reading-nav,.source-note,.backdrop,.no-script{display:none!important}.layout{margin:0}.reader{max-width:none;margin:0;padding:0}.material{border:0;box-shadow:none;margin:0;padding:0;background:white}.material-head{break-after:avoid}.body-content{font-size:11pt}.body-content h2{break-after:avoid;font-size:15pt}.body-content h3{break-after:avoid;font-size:13pt}.body-content pre{white-space:pre-wrap;overflow-wrap:anywhere;color:black;background:#f3f3f3;border:1px solid #ddd;break-inside:avoid}.body-content pre code{white-space:pre-wrap}.body-content table{font-size:9pt;min-width:0}.table-wrap{overflow:visible}.body-content a{color:#194b6f}.hints,.answer-key{break-inside:avoid}.practice-note{font-size:10pt}@page{size:A4;margin:18mm}}
"""


SCRIPT = r"""
(() => {
  'use strict';
  const articles = Array.from(document.querySelectorAll('.material'));
  const links = Array.from(document.querySelectorAll('[data-material]'));
  const search = document.getElementById('course-search');
  const status = document.getElementById('search-status');
  const menu = document.getElementById('menu-toggle');
  const groups = Array.from(document.querySelectorAll('.nav-group'));
  const byId = new Map(articles.map(article => [article.id, article]));
  const normalized = text => text.toLocaleLowerCase('ru').replace(/ё/g, 'е');
  const corpus = new Map(articles.map(article => [article.id, normalized(article.textContent)]));
  document.body.classList.add('js');

  function closeMenu() {
    document.body.classList.remove('nav-open');
    menu.setAttribute('aria-expanded', 'false');
  }
  function select() {
    let fragment;
    try { fragment = decodeURIComponent(location.hash.slice(1)); } catch (_) { fragment = 'HOME'; }
    const id = fragment.split('--')[0] || 'HOME';
    const current = byId.get(id) || byId.get('HOME') || articles[0];
    articles.forEach(article => { article.hidden = article !== current; });
    links.forEach(link => {
      if (link.dataset.material === current.id) { link.setAttribute('aria-current', 'page'); const parent = link.closest('details.nav-module'); if(parent) parent.open = true; }
      else link.removeAttribute('aria-current');
    });
    document.title = current.dataset.title + ' · Курс ML и LLM';
    closeMenu();
    const heading = fragment.includes('--') ? document.getElementById(fragment) : null;
    requestAnimationFrame(() => {
      if (heading) heading.scrollIntoView({block:'start'});
      else window.scrollTo({top:0, behavior:'instant'});
    });
  }
  function filter() {
    const terms = normalized(search.value.trim()).split(/\s+/).filter(Boolean);
    let found = 0;
    document.querySelectorAll('.nav-group li[data-key]').forEach(item => {
      const text = corpus.get(item.dataset.key) || '';
      const match = terms.every(term => text.includes(term));
      item.hidden = !match;
      if (match) found += 1;
    });
    groups.forEach(group => { group.hidden = !Array.from(group.querySelectorAll('li')).some(item => !item.hidden); });
    if(terms.length) document.querySelectorAll('details.nav-module').forEach(group => { group.open = !group.parentElement.hidden; });
    status.textContent = terms.length ? `Найдено разделов: ${found}` : 'Поиск по названиям и тексту';
    document.getElementById('empty-search').classList.toggle('is-visible', found === 0);
  }
  menu.addEventListener('click', () => {
    const open = document.body.classList.toggle('nav-open');
    menu.setAttribute('aria-expanded', String(open));
  });
  document.getElementById('backdrop').addEventListener('click', closeMenu);
  document.getElementById('print-section').addEventListener('click', () => window.print());
  search.addEventListener('input', filter);
  window.addEventListener('hashchange', select);
  document.addEventListener('keydown', event => {
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'k') {
      event.preventDefault(); document.body.classList.add('nav-open');
      menu.setAttribute('aria-expanded','true'); search.focus();
    }
    if (event.key === 'Escape') closeMenu();
  });
  document.addEventListener('click', event => {
    const link = event.target.closest('a[data-material]');
    if (link && link.hash === location.hash) { closeMenu(); window.scrollTo({top:0, behavior:'instant'}); }
  });
  filter(); select();
})();
"""


def build(root: Path) -> tuple[str, int]:
    root = root.resolve()
    registry = json.loads((root / "metadata/material-registry.json").read_text(encoding="utf-8"))
    if not isinstance(registry, dict) or "HOME" not in registry:
        raise ValueError("Material registry must map IDs to local documents and include HOME")
    order = ordered_materials(registry)
    files = {key: checked_file(root, registry[key]["path"]) for key in order}
    if len(set(files.values())) != len(files):
        raise ValueError("Different materials must not silently share one Markdown source")
    path_ids = {path: key for key, path in files.items()}
    sections = []
    for key in order:
        item, source = registry[key], files[key]
        title = item["title"]
        raw = source.read_text(encoding="utf-8")
        if "chatgpt.com/space/" in raw:
            raise ValueError(f"Page dependency remains in {item['path']}")
        renderer = CourseRenderer(root, source, key, path_ids)
        markdown = mistune.create_markdown(renderer=renderer, plugins=["table", "strikethrough", "task_lists"])
        body = markdown(without_repeated_title(raw, title))
        body = body.replace("<table>", '<div class="table-wrap"><table>').replace("</table>", "</table></div>")
        body = fold_hints(body)
        label, kind = display_status(key)
        if key == "MENTOR-KEYS" or key.endswith("-SOLUTIONS"):
            body = '<p class="answer-note">Сначала сохрани собственный ответ. После просмотра решения выполни новый вариант задачи и отдельно отметь, какая помощь понадобилась.</p><details class="answer-key"><summary>Открыть ответы и разборы</summary><div>' + body + '</div></details>'
        if key in PILOT_IDS:
            body = '<div class="practice-note">Практика: открой <strong>START_PRACTICE.cmd</strong> в папке курса, затем файл <code>notebooks/pilot/' + key + '.ipynb</code> в Jupyter. Для своих ответов используй копию в <code>submissions</code>.</div>' + body
        module_id = key.split("-")[0]
        if key in MODULE_IDS and (root / f"course/visuals/{key}.html").exists():
            body += f'<h2>Попробуй изменить параметры</h2><iframe class="experiment" src="course/visuals/{key}.html" title="Интерактивная демонстрация {key}" loading="lazy" sandbox="allow-scripts"></iframe><p><a href="course/visuals/{key}.html" target="_blank">Открыть демонстрацию отдельно</a></p>'
        if re.fullmatch(r"M\d{2}-L\d{2}", key):
            body += f'<div class="practice-note"><a href="#{module_id}">Модуль, домашняя работа и интерактивная демонстрация</a> · <a href="#{module_id}-SOLUTIONS">Решения для самопроверки</a></div>'
        position = order.index(key)
        previous = order[position - 1] if position else None
        following = order[position + 1] if position + 1 < len(order) else None
        neighbors = '<nav class="reading-nav" aria-label="Переход между разделами">'
        if previous:
            neighbors += f'<a href="#{previous}"><small>Предыдущий раздел</small>← {escape(registry[previous]["title"])}</a>'
        if following:
            neighbors += f'<a href="#{following}"><small>Следующий раздел</small>{escape(registry[following]["title"])} →</a>'
        neighbors += '</nav>'
        sections.append(f'<article class="material" id="{escape(key)}" data-title="{escape(title, quote=True)}"><header class="material-head"><div class="eyebrow"><span class="material-id">{escape(display_id(key))}</span><span class="badge {kind}">{label}</span></div><h1>{escape(title)}</h1></header><div class="body-content">{body}</div><footer class="source-note">Редактируемый текст: <code>{escape(item["path"])}</code>. Локальная версия {VERSION}.</footer>{neighbors}</article>')
    groups = [("Начать и ориентироваться", FOUNDATION_IDS), ("Первые десять занятий", PILOT_IDS)]
    groups.extend((registry[m]["title"], [m, *[f"{m}-L{i:02}" for i in range(1,6)]]) for m in MODULE_IDS if m in registry)
    groups.append(("Ответы и самопроверка", ["MENTOR-KEYS", *[f"{m}-SOLUTIONS" for m in MODULE_IDS]]))
    known = {key for _, keys in groups for key in keys}
    extras = [key for key in order if key not in known]
    if extras:
        groups.append(("Дополнительно", extras))
    nav_groups = []
    for title, keys in groups:
        items = '<ul>' + ''.join(nav_item(key, registry[key]) for key in keys if key in registry) + '</ul>'
        if keys and keys[0] in MODULE_IDS:
            content = '<details class="nav-module"><summary>' + escape(title) + '</summary>' + items + '</details>'
        else:
            content = '<h2>' + escape(title) + '</h2>' + items
        nav_groups.append('<section class="nav-group">' + content + '</section>')
    navigation = '\n'.join(nav_groups)
    html = f'''<!doctype html>
<html lang="ru">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><meta name="color-scheme" content="light"><meta name="course-source-sha256" content="{source_fingerprint(root)}"><meta name="description" content="Локальный самоучитель ML и LLM: объяснения, самостоятельные задачи, ноутбуки и самопроверка."><title>Курс ML и LLM · Локальный самоучитель</title><style>{CSS}</style></head>
<body>
<aside class="sidebar" id="course-menu" aria-label="Оглавление курса"><a class="brand" href="#HOME"><span class="brand-symbol">ML</span><span class="brand-name">ML и LLM</span><span class="brand-caption">Локальный самоучитель<br>Читай · Выполняй · Проверяй</span></a><label class="search"><span>Найти тему</span><input id="course-search" type="search" placeholder="Например, градиент" autocomplete="off" spellcheck="false" aria-describedby="search-status"></label><p class="search-status" id="search-status" role="status">Поиск по названиям и тексту</p><nav aria-label="Разделы учебника">{navigation}</nav><p class="empty-search" id="empty-search">Совпадений нет. Попробуй другое слово.</p><p class="nav-foot">Версия {VERSION} · Самостоятельное обучение<br>Прогресс записывай в progress/progress.md.<br>Чтение и примеры доступны локально.</p></aside>
<button class="backdrop" id="backdrop" aria-label="Закрыть оглавление" tabindex="-1"></button>
<div class="layout"><header class="topbar"><span class="top-label">От основ моделей к ML и LLM-системам</span><div class="top-actions"><button class="small-button menu-toggle" id="menu-toggle" aria-controls="course-menu" aria-expanded="false">Оглавление</button><button class="small-button print-button" id="print-section">Печать раздела</button></div></header><main class="reader"><div class="book-label">Учебные материалы и практика</div><noscript><p class="no-script">JavaScript выключен: все разделы показаны ниже подряд. Ссылки оглавления продолжают работать.</p></noscript>{''.join(sections)}</main></div>
<script>{SCRIPT}</script>
</body></html>'''
    return html, len(sections)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check that the HTML matches current Markdown without changing it")
    args = parser.parse_args()
    html, count = build(ROOT)
    target = ROOT / OUTPUT
    if args.check:
        if not target.is_file() or target.read_text(encoding="utf-8") != html:
            raise SystemExit("Reader is missing or stale. Run: python tools/build_reader.py")
    else:
        target.write_text(html, encoding="utf-8", newline="\n")
    print(json.dumps({"output": OUTPUT, "materials": count, "mode": "check" if args.check else "build", "network_required": False, "learner_mastery": "not_assessed"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
