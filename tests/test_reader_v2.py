"""Reader regressions: pedagogical order, answer privacy, routing and freshness.

These tests build a small isolated course, not the generated repository HTML.
The JavaScript navigation check runs the production select() function with a
minimal DOM contract; real layout/browser checks remain in reader-v2-qa.json.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess

import mistune
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("reader_v2_features", ROOT / "tools/build_reader.py")
reader = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reader)


def write(root: Path, relative: str, content: str) -> Path:
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return target


@pytest.fixture
def small_course(tmp_path):
    # Deliberately lexical/insertion order differs from the learning order.
    registry = {
        "HOME": {"path": "course/README.md", "title": "Начало"},
        "M04": {"path": "course/modules/M04.md", "title": "Статистика"},
        "M04-L01": {"path": "course/lessons/M04-L01.md", "title": "Вероятность"},
        "M04-L06": {"path": "course/lessons/M04-L06.md", "title": "Логарифм"},
        "M04-SOLUTIONS": {"path": "solutions/modules/M04.md", "title": "Ключи"},
    }
    for item in registry.values():
        write(tmp_path, item["path"], f"# {item['title']}\n\nТекст для самостоятельной работы.\n")
    write(tmp_path, "solutions/modules/M04.md", """# Ключи

## M04-L01-A

Ответ про вероятность.

## M04-L06-A

Ответ про логарифм.

### Почему это работает

Объяснение именно логарифма.

## M04-HW01

Ответ домашней работы.
""")
    write(tmp_path, "metadata/material-registry.json", json.dumps(registry, ensure_ascii=False))
    write(tmp_path, "curriculum.json", json.dumps({"modules": [
        {"id": "M04", "lessons": [{"id": "M04-L06"}, {"id": "M04-L01"}]}
    ]}))
    write(tmp_path, "metadata/lesson-visuals.json", json.dumps({"M04-L06": "course/visuals/logarithm.html"}))
    write(tmp_path, "course/visuals/logarithm.html", "<p>Интерактивный логарифм</p>")
    write(tmp_path, "course/visuals/other.html", "<p>Другой опыт</p>")
    write(tmp_path, "tools/build_reader.py", Path(reader.__file__).read_text(encoding="utf-8"))
    return tmp_path, registry


def article(html: str, key: str) -> str:
    found = re.search(r'<article\b[^>]*\bid="' + re.escape(key) + r'"[^>]*>(.*?)</article>', html, re.S)
    assert found, f"Missing article {key}"
    return found.group(1)


def disclosures(html: str) -> dict[str, tuple[str, str]]:
    return {
        match.group(2): (match.group(1), match.group(3))
        for match in re.finditer(r'(<details\b[^>]*\bid="([^"]+)"[^>]*>)(.*?)</details>', html, re.S)
    }


def test_curriculum_order_drives_articles_navigation_and_reading_neighbors(small_course):
    root, registry = small_course
    html, _ = reader.build(root)
    assert reader.ordered_materials(registry, root).index("M04-L06") < reader.ordered_materials(registry, root).index("M04-L01")
    assert html.index('<article class="material" id="M04-L06"') < html.index('<article class="material" id="M04-L01"')
    # The next reading link must follow prerequisites rather than numeric IDs.
    lesson = article(html, "M04-L06")
    assert re.search(r'href="#M04-L01"><small>Следующий раздел</small>', lesson)
    sidebar = html.split('</aside>', 1)[0]
    assert sidebar.index('data-material="M04-L06"') < sidebar.index('data-material="M04-L01"')


def test_real_mathematical_bridge_precedes_probability():
    registry = json.loads((ROOT / "metadata/material-registry.json").read_text(encoding="utf-8"))
    order = reader.ordered_materials(registry, ROOT)
    assert order.index("M04-L06") < order.index("M04-L01")
    assert order.index("M04-L02") < order.index("M04-L07") < order.index("M04-L03")


def test_lesson_link_targets_its_answer_and_preserves_explicit_other_task(small_course):
    root, registry = small_course
    paths = {(root / item["path"]).resolve(): key for key, item in registry.items()}
    renderer = reader.CourseRenderer(root, root / "course/lessons/M04-L06.md", "M04-L06", paths)
    auto, external = renderer.destination("../../solutions/modules/M04.md")
    assert auto == "#M04-SOLUTIONS--m04-l06-a"
    assert external is False
    explicit, _ = renderer.destination("../../solutions/modules/M04.md#m04-hw01")
    assert explicit == "#M04-SOLUTIONS--m04-hw01"
    html, _ = reader.build(root)
    assert f'href="{auto}">Решение этой задачи</a>' in article(html, "M04-L06")
    assert auto[1:] in disclosures(article(html, "M04-SOLUTIONS"))


def test_fold_answers_keeps_each_answer_closed_and_outside_its_neighbors(small_course):
    root, registry = small_course
    source = root / "solutions/modules/M04.md"
    renderer = reader.CourseRenderer(root, source, "M04-SOLUTIONS", {})
    rendered = mistune.create_markdown(renderer=renderer)(source.read_text(encoding="utf-8"))
    folded = reader.fold_answers(rendered)
    sections = disclosures(folded)
    target_opening, target_body = sections["M04-SOLUTIONS--m04-l06-a"]
    assert not re.search(r'\bopen(?:\s|=|>)', target_opening)
    assert "Ответ про логарифм" in target_body
    assert "Объяснение именно логарифма" in target_body
    assert "Ответ про вероятность" not in target_body
    assert "Ответ домашней работы" not in target_body
    assert "<details" not in target_body, "Sibling answers must not be nested inside the target"
    for anchor in ("M04-SOLUTIONS--m04-l01-a", "M04-SOLUTIONS--m04-hw01"):
        opening, body = sections[anchor]
        assert "Ответ про логарифм" not in body
        assert not re.search(r'\bopen(?:\s|=|>)', opening)


def test_real_new_transformer_lesson_has_an_individual_disclosure():
    registry = json.loads((ROOT / "metadata/material-registry.json").read_text(encoding="utf-8"))
    paths = {(ROOT / item["path"]).resolve(): key for key, item in registry.items()}
    source = ROOT / registry["M08-L06"]["path"]
    renderer = reader.CourseRenderer(ROOT, source, "M08-L06", paths)
    target, _ = renderer.destination("../../solutions/modules/M08.md")
    assert target.startswith("#M08-SOLUTIONS--m08-l06"), "A lesson must not route to a whole solutions file"
    solution_path = ROOT / registry["M08-SOLUTIONS"]["path"]
    solution_renderer = reader.CourseRenderer(ROOT, solution_path, "M08-SOLUTIONS", paths)
    rendered = mistune.create_markdown(renderer=solution_renderer)(solution_path.read_text(encoding="utf-8"))
    keys = disclosures(reader.fold_answers(rendered))
    assert target[1:] in keys, "Deep-link must select a disclosure, not a heading buried with neighboring answers"
    assert "M08-L07" not in keys[target[1:]][1], "Opening residual must not reveal the next forward task"


def test_extension_lesson_ids_with_x_are_lessons_and_answers_return_to_them():
    folded = reader.fold_answers('<h2 id="M10-SOLUTIONS--m10-l07x-a">M10-L07X-A</h2><p>Ключ</p>')
    assert 'href="#M10-L07X"' in folded and 'href="#M10-L07"' not in folded
    assert reader.nav_item("M10-L07X", {"title": "M10-L07X. Эмбеддеры"}).count(">L07X<") == 1
    order = reader.lesson_order(ROOT, json.loads((ROOT / "metadata/material-registry.json").read_text(encoding="utf-8")))
    assert order["M10"][-1] == "M10-L07X"


@pytest.mark.parametrize("relative", ["metadata/lesson-visuals.json", "curriculum.json"])
def test_fingerprint_changes_when_mapping_or_pedagogical_order_changes(small_course, relative):
    root, _ = small_course
    path = root / relative
    original = path.read_bytes()
    before = reader.source_fingerprint(root)
    value = json.loads(original)
    if relative.endswith("lesson-visuals.json"):
        value["M04-L06"] = "course/visuals/other.html"
    else:
        value["modules"][0]["lessons"].reverse()
    path.write_text(json.dumps(value), encoding="utf-8")
    assert reader.source_fingerprint(root) != before
    path.write_bytes(original)
    assert reader.source_fingerprint(root) == before, "Restoring exact sources must restore freshness identity"


def test_inline_experiment_and_separate_link_belong_only_to_the_selected_lesson(small_course):
    root, _ = small_course
    html, _ = reader.build(root)
    assigned = article(html, "M04-L06")
    assert '<iframe class="experiment" src="course/visuals/logarithm.html"' in assigned
    assert 'href="course/visuals/logarithm.html" target="_blank"' in assigned
    for key in ("HOME", "M04", "M04-L01", "M04-SOLUTIONS"):
        other = article(html, key)
        assert "logarithm.html" not in other
        assert 'class="lesson-experiment"' not in other
    # The ID-named fallback is a separate supported route; mapping takes precedence.
    write(root, "course/visuals/M04-L01.html", "<p>Вероятности</p>")
    write(root, "course/visuals/M04-L06.html", "<p>Не должно перекрыть mapping</p>")
    rebuilt, _ = reader.build(root)
    assert 'src="course/visuals/M04-L01.html"' in article(rebuilt, "M04-L01")
    assert 'src="course/visuals/logarithm.html"' in article(rebuilt, "M04-L06")
    assert 'src="course/visuals/M04-L06.html"' not in article(rebuilt, "M04-L06")


def test_missing_mapped_visual_fails_instead_of_silently_shipping_a_broken_experiment(small_course):
    root, _ = small_course
    write(root, "metadata/lesson-visuals.json", json.dumps({"M04-L06": "course/visuals/missing.html"}))
    with pytest.raises(ValueError, match="Missing or external course file"):
        reader.build(root)


def test_sandbox_experiment_bundles_local_dependencies_and_rejects_external_paths(small_course):
    root, _ = small_course
    write(root, 'course/visuals/frame.css', 'body{color:#182b3a}')
    write(root, 'course/visuals/frame.js', 'parent.postMessage({type:"course:resize",height:100},"*");')
    write(root, 'course/visuals/example.html', '<link rel="stylesheet" href="frame.css"><p>Опыт</p><script src="frame.js"></script>')
    frame = reader.experiment_frame(root, 'course/visuals/example.html', 'Опыт')
    from html import unescape
    content = unescape(re.search(r'srcdoc="(.*?)"></iframe>', frame, re.S).group(1))
    assert '<style>body{color:#182b3a}</style>' in content
    assert 'parent.postMessage' in content and '<script src=' not in content
    assert 'sandbox="allow-scripts"' in frame
    write(root, 'course/visuals/example.html', '<script src="https://example.org/remote.js"></script>')
    with pytest.raises(ValueError, match='Online dependency'):
        reader.experiment_frame(root, 'course/visuals/example.html', 'Опыт')
    write(root, 'course/visuals/example.html', '<script src="../../../outside.js"></script>')
    with pytest.raises(ValueError, match='leaves repository'):
        reader.experiment_frame(root, 'course/visuals/example.html', 'Опыт')


def test_retry_link_opens_bc_key_while_first_attempt_keeps_a(small_course):
    root, registry = small_course
    with (root/'solutions/modules/M04.md').open('a', encoding='utf-8') as stream:
        stream.write('\n## M04-L06: B и C\n\nДругие условия.\n')
    paths = {(root/item['path']).resolve():key for key,item in registry.items()}
    renderer = reader.CourseRenderer(root, root/'course/lessons/M04-L06.md', 'M04-L06', paths)
    assert '#M04-SOLUTIONS--m04-l06-b-и-c' in renderer.link('ключи B/C', '../../solutions/modules/M04.md')
    assert '#M04-SOLUTIONS--m04-l06-a' in renderer.link('Решение A', '../../solutions/modules/M04.md')


def test_production_hash_navigation_opens_target_without_revealing_adjacent_answers():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is only needed to execute the navigation JavaScript; Python reader tests still run")
    start = reader.SCRIPT.index("  function select() {")
    end = reader.SCRIPT.index("  function filter() {", start)
    select_function = reader.SCRIPT[start:end]
    # Minimal DOM roles used by production select(); no reimplementation of select.
    harness = r"""
const assert = require('node:assert/strict');
const home={id:'HOME',dataset:{title:'Home'},hidden:false};
const lesson={id:'M04-L06',dataset:{title:'Log'},hidden:false};
const answers={id:'M04-SOLUTIONS',dataset:{title:'Answers'},hidden:true};
const articles=[home,lesson,answers];
const byId=new Map(articles.map(x=>[x.id,x]));
const links=[];
let lastMaterial='';
const updateContext=()=>{};
const revealActive=()=>{};
for(const a of articles){a.contains=node=>node===target;a.querySelector=()=>({setAttribute(){},focus(){}});}
const previous={tagName:'DETAILS',open:false,parentElement:answers};
const next={tagName:'DETAILS',open:false,parentElement:answers};
const target={tagName:'DETAILS',open:false,parentElement:answers,scrolled:false,
  scrollIntoView(){this.scrolled=true;},setAttribute(){},focus(){}};
const document={title:'',getElementById(id){return id==='M04-SOLUTIONS--m04-l06-a'?target:null;}};
let closed=0,topScroll=0;
const closeMenu=()=>closed++;
const requestAnimationFrame=fn=>fn();
const window={scrollTo(){topScroll++;}};
const location={hash:'#M04-SOLUTIONS--m04-l06-a'};
""" + select_function + r"""
select();
assert.equal(answers.hidden,false);
assert.equal(home.hidden,true);
assert.equal(lesson.hidden,true);
assert.equal(target.open,true);
assert.equal(target.scrolled,true);
assert.equal(previous.open,false);
assert.equal(next.open,false);
assert.equal(topScroll,0);
assert.equal(closed,1);
location.hash='#M04-L06'; select();
assert.equal(lesson.hidden,false);
assert.equal(answers.hidden,true);
assert.equal(previous.open,false);
assert.equal(next.open,false);
location.hash='#unknown'; select();
assert.equal(home.hidden,false);
console.log('target-only navigation passed');
"""
    result = subprocess.run([node, "-"], input=harness, text=True, capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert "target-only navigation passed" in result.stdout
