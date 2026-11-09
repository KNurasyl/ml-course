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
VERSION = "1.2.0"
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
    for relative in ("curriculum.json", "metadata/lesson-visuals.json", "metadata/responsive-figures.json",
                     "tools/reader.css", "tools/reader.js"):
        if (root / relative).is_file():
            paths.add(relative)
    for folder in ("course/assets", "course/visuals"):
        paths.update(path.relative_to(root).as_posix() for path in (root / folder).rglob("*") if path.is_file())
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

    def block_code(self, code: str, info: str | None = None) -> str:
        # Mermaid source stays readable text; reader.js draws it with the local library when visible.
        if info and info.split()[0] == "mermaid":
            return f'<figure class="diagram"><pre class="mermaid">{escape(code)}</pre></figure>\n'
        return super().block_code(code, info)

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
            if material.endswith("-SOLUTIONS") and not parsed.fragment and re.fullmatch(r"M\d{2}-L\d{2}X?", self.material_id):
                heading = next((line[3:].strip() for line in local.read_text(encoding="utf-8").splitlines()
                                if line.startswith("## ") and self.material_id in line), None)
                if heading:
                    return f"#{material}--{slug(heading)}", False
            return f"#{material}" + (f"--{unquote(parsed.fragment)}" if parsed.fragment else ""), False
        target = quote(local.relative_to(self.root).as_posix(), safe="/:@")
        if parsed.fragment:
            target += "#" + parsed.fragment
        return target, False

    def link(self, text: str, url: str, title: str | None = None) -> str:
        target, external = self.destination(url)
        # A retry link must not silently return to answer A when the source has
        # a distinct B/C key. Explicit fragments remain authoritative.
        parsed = urlsplit(url)
        if (target and '-SOLUTIONS--' in target and not parsed.fragment
                and re.search(r'B\s*(?:/|и)\s*C', text_only(text), re.I)):
            local = (self.source.parent / unquote(parsed.path)).resolve()
            heading = next((line[3:].strip() for line in local.read_text(encoding='utf-8').splitlines()
                            if line.startswith('## ') and self.material_id in line
                            and re.search(r'B.*C', line)), None)
            if heading:
                target = target.split('--')[0] + '--' + slug(heading)
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
        mapping_path = self.root / "metadata/responsive-figures.json"
        mapping = json.loads(mapping_path.read_text(encoding="utf-8")) if mapping_path.exists() else {}
        alternatives = mapping.get(unquote(target), {})
        desktop = alternatives.get("desktop", target)
        mobile = alternatives.get("mobile")
        checked_file(self.root, unquote(desktop))
        picture = ''
        if mobile:
            checked_file(self.root, mobile)
            picture = f'<source media="(max-width: 600px)" srcset="{escape(mobile, quote=True)}">'
        url = escape(desktop, quote=True)
        return (f'<a class="figure-expand" href="{url}" target="_blank" data-figure="{url}" '
                f'aria-label="Увеличить рисунок: {escape(text, quote=True)}">'
                f'<picture>{picture}<img src="{url}" alt="{escape(text, quote=True)}" loading="lazy"></picture></a>'
                f'<span class="figure-hint">{escape(alternatives.get("caption", text))} · Открыть крупнее ↗</span>')

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


def lesson_order(root: Path, registry: dict) -> dict[str, list[str]]:
    path = root / "curriculum.json"
    if path.exists():
        curriculum = json.loads(path.read_text(encoding="utf-8"))
        return {m["id"]: [l["id"] for l in m["lessons"] if l["id"] in registry] for m in curriculum["modules"]}
    return {mid: sorted(k for k in registry if re.fullmatch(mid + r"-L\d{2}X?", k)) for mid in MODULE_IDS}


def ordered_materials(registry: dict, root: Path = ROOT) -> list[str]:
    lessons = lesson_order(root, registry)
    route = [key for module in MODULE_IDS for key in [module, *lessons.get(module, [])]]
    primary = ["HOME", "DIAGNOSTIC", *PILOT_IDS, *route, "GLOSSARY", "PROJECT", "MENTOR", "PROGRESS", "MENTOR-KEYS", *[f"{m}-SOLUTIONS" for m in MODULE_IDS]]
    return [key for key in primary if key in registry] + [key for key in registry if key not in primary]


def fold_answers(content: str) -> str:
    """One answer per disclosure; opening one task never reveals its neighbours."""
    pattern = re.compile(r'<h2\b([^>]*)>(.*?)</h2>(.*?)(?=<h2\b|\Z)', re.DOTALL)
    def replace(match):
        identifier = re.search(r'\bid="([^"]+)"', match.group(1))
        anchor = f' id="{identifier.group(1)}"' if identifier else ""
        if not text_only(match.group(3)).strip():
            return f'<h2{anchor}>{match.group(2)}</h2>'
        task = re.search(r'M\d{2}-L\d{2}X?|M\d{2}-HW\d{2}|P\d{2}(?:-HW\d{2})?|D00', text_only(match.group(2)))
        actions = ''
        if task:
            key = task.group(0)
            key = key.split('-HW')[0]
            key = 'DIAGNOSTIC' if key == 'D00' else key
            actions = (f'<nav class="answer-actions" aria-label="После самопроверки">'
                       f'<a href="#{key}" data-return-task="{key}">← Вернуться к задаче</a>'
                       f'<a href="#{key}" data-return-retry="{key}">Новый вариант B / C →</a></nav>')
        return (f'<details class="answer-key"{anchor}><summary>Ответ: {escape(text_only(match.group(2)))}</summary>'
                f'<div>{actions}{match.group(3)}{actions}</div></details>')
    return pattern.sub(replace, content)


def experiment_frame(root: Path, relative: str, title: str) -> str:
    """Bundle local frame dependencies; opaque file:// sandboxes block sidecar files."""
    source = checked_file(root, relative)
    content = source.read_text(encoding="utf-8")
    def dependency(url: str) -> str:
        parsed = urlsplit(url)
        if parsed.scheme or parsed.netloc:
            raise ValueError(f'Online dependency in offline experiment: {url}')
        path = (source.parent / unquote(parsed.path)).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError(f'Experiment dependency leaves repository: {url}')
        return checked_file(root, path.relative_to(root).as_posix()).read_text(encoding='utf-8')
    content = re.sub(r'<link\b[^>]*rel="stylesheet"[^>]*href="([^"]+)"[^>]*>',
                     lambda m: '<style>' + dependency(m.group(1)) + '</style>', content)
    content = re.sub(r'<script\b[^>]*src="([^"]+)"[^>]*>\s*</script>',
                     lambda m: '<script>' + dependency(m.group(1)) + '</script>', content)
    return (f'<iframe class="experiment" src="{escape(relative)}" title="{escape(title)}" '
            f'loading="lazy" sandbox="allow-scripts" srcdoc="{escape(content, quote=True)}"></iframe>')


def inline_experiment(root: Path, key: str, body: str) -> str:
    mapping_path = root / "metadata/lesson-visuals.json"
    mapping = json.loads(mapping_path.read_text(encoding="utf-8")) if mapping_path.is_file() else {}
    relative = mapping.get(key)
    if relative is None and (root / f"course/visuals/{key}.html").is_file():
        relative = f"course/visuals/{key}.html"
    if relative is None:
        return body
    checked_file(root, relative)
    block = (f'<section class="lesson-experiment"><h2>Предскажи и проверь</h2>'
             '<p>Перед изменением параметра запиши прогноз. Затем объясни, почему результат совпал или разошёлся с ним.</p>'
             + experiment_frame(root, relative, f'Численный опыт {key}') +
             f'<p><a href="{escape(relative)}" target="_blank">Открыть опыт отдельно на весь экран</a></p></section>')
    # Place at the first mechanism illustration, before the worked example.
    marker = '</span>'
    hint = body.find('<span class="figure-hint">')
    if hint >= 0:
        end = body.find(marker, hint) + len(marker)
        # Markdown puts the image in a paragraph; insert after it, not inside.
        paragraph_end = body.find('</p>', end)
        if paragraph_end >= 0 and paragraph_end - end < 100:
            end = paragraph_end + 4
        return body[:end] + block + body[end:]
    return body + block


def display_status(key: str) -> tuple[str, str]:
    if key in PILOT_IDS:
        return "Готовое занятие", "ready"
    if key in MODULE_IDS:
        return "Модуль и домашняя работа", "ready"
    if re.fullmatch(r"M\d{2}-L\d{2}X?", key):
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
    marker = key if key in PILOT_IDS or key in MODULE_IDS else (key.split('-')[-1] if re.fullmatch(r'M\d{2}-L\d{2}X?', key) else {"HOME": "01", "DIAGNOSTIC": "02"}.get(key, "•"))
    return f'<li data-key="{escape(key)}"><a href="#{escape(key)}" data-material="{escape(key)}"><span class="nav-marker">{escape(marker)}</span><span>{escape(short)}</span></a></li>'


def lesson_presentation(body: str, key: str) -> tuple[str, str, str]:
    """Separate the outcome, local map and task roles without rewriting source prose."""
    outcome = ''
    for label, class_name in (("Результат", "outcome"), ("До урока", "prerequisites")):
        pattern = rf'<p><strong>{label}:?</strong>(.*?)</p>'
        found = re.search(pattern, body, re.S)
        if not found:
            continue
        if class_name == 'outcome':
            outcome += f'<p class="outcome"><strong>После урока:</strong>{found.group(1)}</p>'
        else:
            outcome += f'<details class="prerequisites"><summary>До урока и ориентир времени</summary><div>{found.group(1)}</div></details>'
        body = body[:found.start()] + body[found.end():]
    headings = re.findall(r'<h2\b[^>]*id="([^"]+)"[^>]*>(.*?)</h2>', body, re.S)
    outline = ''
    if headings:
        items = ''.join(f'<li><a href="#{escape(anchor)}">{escape(text_only(title))}</a></li>' for anchor, title in headings)
        outline = f'<aside class="lesson-outline" aria-label="Карта урока"><details><summary>В этом уроке</summary><ol>{items}</ol></details></aside>'
    body = fold_hints(body)
    pattern = re.compile(r'(<h2\b[^>]*>)(.*?)(</h2>)(.*?)(?=<h2\b|<details class="hints"|\Z)', re.S)
    def stage(match):
        title = text_only(match.group(2)).lower()
        role = next((role for terms, role in [
            (("разобран", "численный пример"), 'example'),
            (("практика с поддержкой", "упражнение с поддержкой"), 'practice'),
            (("самостоятельн",), 'task'),
            (("повтор", "перенос", "отложенн"), 'retry'),
        ] if any(term in title for term in terms)), None)
        if not role:
            return match.group(0)
        label = {'example':'Разбираем вместе', 'practice':'Пробуем с поддержкой', 'task':'Твоя попытка', 'retry':'Проверяем перенос'}[role]
        return f'<section class="lesson-stage" data-stage="{role}"><span class="stage-label">{label}</span>{match.group(0)}</section>'
    body = pattern.sub(stage, body)
    body = re.sub(r'<p><code>([^<]+)</code></p>', r'<p class="formula"><code>\1</code></p>', body)
    return body, outcome, outline


def home_presentation(body: str, registry: dict) -> str:
    """Navigation only; the editable introduction still comes from course/README.md."""
    routes = [
        ('P01', '01 / ПЕРВЫЕ ШАГИ', 'Модель, ошибка, обучение', 'Десять занятий: от первого предсказания до своего эксперимента.'),
        ('M00', '02 / ФУНДАМЕНТ · M00–M06', 'Математика и ML-системы', 'Данные, оптимизация, статистика и честное сравнение моделей.'),
        ('M07', '03 / НЕЙРОСЕТИ · M07–M12', 'От PyTorch к LLM и RAG', 'Пойми вычисления, запусти модель, собери поиск и проверь качество.'),
        ('M13', '04 / ПРОДУКТ · M13–M16', 'Голосовой AI-интервьюер', 'Эксплуатация, инструменты, голосовой диалог и защита проекта.'),
    ]
    cards = ''.join(f'<a class="route-step" href="#{key}"><small>{eyebrow}</small><strong>{title}</strong><span>{text}</span></a>'
                    for key, eyebrow, title, text in routes if key in registry)
    # The source introduction contains its own quick-start links. Place those after
    # the compact route; keep environment instructions in an explicit disclosure.
    body = re.sub(r'(<h2\b[^>]*>Чтение и практика</h2>)(.*?)(?=<h2\b|\Z)',
                  r'<details class="setup-details"><summary>Как открыть Python и выполнять задания</summary><div>\2</div></details>', body, flags=re.S)
    intro = re.search(r'<p>(.*?)</p>', body, re.S)
    start = ''
    if intro:
        start = '<p class="home-intro">' + intro.group(1) + '</p>'
        body = body[:intro.start()] + body[intro.end():]
    start += '<div class="home-actions">'
    if 'DIAGNOSTIC' in registry:
        start += '<a class="primary-action" href="#DIAGNOSTIC">Начать диагностику →</a>'
    if 'MENTOR' in registry:
        start += '<a class="secondary-action" href="#MENTOR">Как устроено обучение</a>'
    start += '</div><p class="home-caption">Сначала сохрани свою попытку, затем открывай решение. Освоение проверяется новой задачей.</p>'
    return start + '<h2 id="HOME--course-route">Твой маршрут</h2><div class="course-route">' + cards + '</div>' + body


CSS = Path(__file__).with_name("reader.css").read_text(encoding="utf-8")
SCRIPT = Path(__file__).with_name("reader.js").read_text(encoding="utf-8")


def build(root: Path) -> tuple[str, int]:
    root = root.resolve()
    registry = json.loads((root / "metadata/material-registry.json").read_text(encoding="utf-8"))
    if not isinstance(registry, dict) or "HOME" not in registry:
        raise ValueError("Material registry must map IDs to local documents and include HOME")
    order = ordered_materials(registry, root)
    lessons = lesson_order(root, registry)
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
        body = re.sub(r'<p><code>([^<]+)</code></p>', r'<p class="formula"><code>\1</code></p>', body)
        body = body.replace("<table>", '<div class="table-wrap"><table>').replace("</table>", "</table></div>")
        intro, outline = '', ''
        is_lesson = bool(re.fullmatch(r"M\d{2}-L\d{2}X?", key))
        if is_lesson or key in PILOT_IDS:
            body, intro, outline = lesson_presentation(body, key)
        else:
            body = fold_hints(body)
        label, kind = display_status(key)
        if key == "MENTOR-KEYS" or key.endswith("-SOLUTIONS"):
            body = '<p class="answer-note">Открывай только нужную задачу после своей попытки. Остальные ответы остаются закрыты. После полного разбора выполни новый вариант.</p>' + fold_answers(body)
        if key == "HOME":
            body = home_presentation(body, registry)
        if key in PILOT_IDS:
            body = '<div class="practice-note">Практика: открой <strong>START_PRACTICE.cmd</strong> в папке курса, затем файл <code>notebooks/pilot/' + key + '.ipynb</code> в Jupyter. Для своих ответов используй копию в <code>submissions</code>.</div>' + body
        module_id = key.split("-")[0]
        if key in MODULE_IDS and (root / f"course/visuals/{key}.html").exists():
            body += f'<h2>Попробуй изменить параметры</h2>' + experiment_frame(root, f'course/visuals/{key}.html', f'Интерактивная демонстрация {key}') + f'<p><a href="course/visuals/{key}.html" target="_blank">Открыть демонстрацию отдельно</a></p>'
        if re.fullmatch(r"M\d{2}-L\d{2}X?", key):
            body = inline_experiment(root, key, body)
            solution_target, _ = renderer.destination(f"../../solutions/modules/{module_id}.md")
            body += f'<div class="practice-note"><a href="#{module_id}">Модуль и домашняя работа</a> · <a href="{solution_target}">Решение этой задачи</a> · <a href="#MENTOR">Правила самопроверки</a></div>'
        position = order.index(key)
        previous = order[position - 1] if position else None
        following = order[position + 1] if position + 1 < len(order) else None
        neighbors = '<nav class="reading-nav" aria-label="Переход между разделами">'
        if previous:
            neighbors += f'<a href="#{previous}"><small>Предыдущий раздел</small>← {escape(registry[previous]["title"])}</a>'
        if following:
            neighbors += f'<a href="#{following}"><small>Следующий раздел</small>{escape(registry[following]["title"])} →</a>'
        neighbors += '</nav>'
        if kind == 'answer':
            back = module_id if module_id in registry else 'DIAGNOSTIC'
            neighbors = (f'<nav class="reading-nav" aria-label="После самопроверки"><a href="#{back}"><small>Вернуться к практике</small>Задания и новые варианты →</a></nav>')
        short_title = re.sub(rf'^{re.escape(key)}\s*[:.·—–-]?\s*', '', title).strip()
        if key == 'HOME':
            short_title = 'ML и LLM. От модели к своему продукту'
        context = module_id if module_id in MODULE_IDS else ''
        sections.append(f'<article class="material" id="{escape(key)}" data-title="{escape(title, quote=True)}" data-short-title="{escape(short_title, quote=True)}" data-kind="{kind}" data-module="{context}"><header class="material-head"><div class="eyebrow"><span class="material-id">{escape(display_id(key))}</span><span class="badge {kind}">{label}</span></div><h1>{escape(short_title)}</h1>{intro}</header><div class="article-layout{" has-outline" if outline else ""}">{outline}<div class="body-content{" home-content" if key == "HOME" else ""}">{body}</div></div><footer class="source-note">Исходный текст: <code>{escape(item["path"])}</code> · Материалы {VERSION} · Оформление 2.0</footer>{neighbors}</article>')
    groups = [("Начать и ориентироваться", FOUNDATION_IDS), ("Первые десять занятий", PILOT_IDS)]
    groups.extend((registry[m]["title"], [m, *lessons.get(m, [])]) for m in MODULE_IDS if m in registry)
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
<body><a class="skip-link" href="#main-content">К материалу</a>
<aside class="sidebar" id="course-menu" aria-label="Оглавление курса"><div class="sidebar-head"><a class="brand" href="#HOME"><span class="brand-symbol">ML</span><span class="brand-name">ML и LLM</span><span class="brand-caption">Технический учебник · Практика</span></a><button id="menu-close" class="small-button menu-close" aria-label="Закрыть оглавление">×</button><label class="search"><span class="screen-reader-only">Найти тему</span><span class="search-row"><input id="course-search" type="search" placeholder="Найти тему…" autocomplete="off" spellcheck="false" aria-describedby="search-status"><button id="search-clear" class="small-button search-clear" type="button" aria-label="Очистить поиск" hidden>×</button></span></label><p class="search-status" id="search-status" role="status">Поиск по тексту · Ctrl K</p></div><div class="sidebar-scroll"><nav id="book-navigation" aria-label="Разделы учебника">{navigation}</nav><ol class="search-results" id="search-results" aria-label="Результаты поиска"></ol><p class="empty-search" id="empty-search" hidden>Попробуй другое слово или его основу.</p><p class="nav-foot">Материалы {VERSION} · Чтение без интернета<br><a href="#PROGRESS">Журнал самостоятельной работы</a></p></div></aside>
<button class="backdrop" id="backdrop" aria-label="Закрыть оглавление" tabindex="-1"></button>
<div class="layout"><header class="topbar"><nav class="breadcrumbs" id="breadcrumbs" aria-label="Текущее место"><a href="#HOME">Учебник</a></nav><div class="top-actions"><button class="small-button menu-toggle" id="menu-toggle" aria-controls="course-menu" aria-expanded="false">☰ Содержание</button><button class="small-button print-button" id="print-section">Печать</button></div></header><main class="reader" id="main-content" tabindex="-1"><noscript><p class="no-script">JavaScript выключен: все разделы показаны ниже подряд. Ссылки оглавления продолжают работать.</p></noscript>{''.join(sections)}</main></div>
<dialog id="figure-dialog" aria-labelledby="figure-title"><div class="figure-toolbar"><strong id="figure-title">Рисунок</strong><button id="figure-zoom" class="small-button">Исходный размер</button><button id="figure-close" class="small-button">Закрыть</button></div><div class="figure-scroll"><img id="figure-large" alt=""></div><a id="figure-original" target="_blank">Открыть исходный файл отдельно</a></dialog>
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
