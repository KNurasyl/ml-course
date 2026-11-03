"""Archive observed design QA, rejecting failures or a different reader build.

Run only after the two Playwright checkers and inspect their screenshots. This
tool records their evidence; it does not execute a browser or assess a learner.
"""
from datetime import datetime, timezone
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'quality/design-v2'


def read(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8'))


def main():
    OUT.mkdir(exist_ok=True, parents=True)
    reader = read('artifacts/design-redesign/reader/qa-results.json')
    frames = read('artifacts/design-redesign/integration/qa-results.json')
    html_hash = sha256((ROOT/'START_HERE.html').read_bytes()).hexdigest()
    for name, data in [('reader', reader), ('frames', frames)]:
        if data['failed'] or data['errors']:
            raise ValueError(f'{name} QA contains failures')
        if data.get('reader_sha256') != html_hash:
            raise ValueError(f'{name} QA used another reader build; rerun the checker')
    assert reader['passed'] >= 36
    assert len(frames['cases']) == 52 and frames['passed'] == 58
    product = read('artifacts/design-redesign/product/qa-results.json')
    assert len(product['checks']) == 60 and all(row['passed'] for row in product['checks']) and not product['errors']
    layouts = read('artifacts/design-redesign/interactives/layout-checks.json')
    assert len(layouts) == 60 and all(row['scroll'] <= row['w'] and not row['svgClips'] for row in layouts)
    parameters = read('artifacts/design-redesign/interactives/parameter-checks.json')
    assert len(parameters) == 17 and all(all(row[k] for k in ('changed','graphicChanged','reset','noNaN')) for row in parameters)
    bounds = read('artifacts/redesign-figures/final-bounds.json')
    assert len(bounds) == 208 and all(not row['overflow'] and not row['overlaps'] for row in bounds)
    mapping = read('metadata/responsive-figures.json')
    assert len(mapping) == 104
    for pair in mapping.values():
        assert all((ROOT/pair[layout]).is_file() for layout in ('desktop','mobile'))
    receipts = {
        'original-findings.json': 'artifacts/design-audit-2026-11-03/findings.json',
        'reader-qa.json': 'artifacts/design-redesign/reader/qa-results.json',
        'integration-qa.json': 'artifacts/design-redesign/integration/qa-results.json',
        'product-qa.json': 'artifacts/design-redesign/product/qa-results.json',
        'interactive-layouts.json': 'artifacts/design-redesign/interactives/layout-checks.json',
        'interactive-parameters.json': 'artifacts/design-redesign/interactives/parameter-checks.json',
        'attention-qa.json': 'artifacts/design-redesign/interactives/attention-checks.json',
        'foundation-qa.json': 'artifacts/design-redesign/interactives/foundation-checks.json',
        'figures-review.json': 'artifacts/redesign-figures/report.json',
        'figures-bounds.json': 'artifacts/redesign-figures/final-bounds.json',
    }
    for target, source in receipts.items():
        shutil.copyfile(ROOT/source, OUT/target)
    shots = OUT/'screenshots'
    shots.mkdir(exist_ok=True)
    for name in reader['screenshots']:
        shutil.copyfile(ROOT/'artifacts/design-redesign/reader'/name, shots/name)
    for name in ('attention-1366.png','attention-320.png','voice-1366.png','voice-320.png'):
        shutil.copyfile(ROOT/'artifacts/design-redesign/integration'/name, shots/name)
    spec=importlib.util.spec_from_file_location('reader_evidence',ROOT/'tools/build_reader.py')
    renderer=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(renderer)
    fingerprint=renderer.source_fingerprint(ROOT)
    assert f'name="course-source-sha256" content="{fingerprint}"' in (ROOT/'START_HERE.html').read_text(encoding='utf-8')
    report={
        'created_utc':datetime.now(timezone.utc).isoformat(),
        'design_version':'2.0','material_version':'1.1.0','audit_base_commit':'305e9a4',
        'scope':'offline textbook visual design and interaction; not learner mastery',
        'resolved_findings':[f'D{i:02}' for i in range(1,14)],
        'reader_sha256':html_hash,'course_source_sha256':fingerprint,
        'counts':{'materials':161,'lessons':103,'figures':104,'svg_layouts':208,'interactive_experiments':26},
        'browser':{'engine':'Microsoft Edge / Playwright','offline':True,
                   'reader_checks_passed':reader['passed'],'embedded_checks_passed':frames['passed'],
                   'embedded_layouts_checked':len(frames['cases']),'all_materials_reflow_width':320,
                   'standalone_product_checks_passed':len(product['checks']),
                   'standalone_other_layout_cases':len(layouts),'sliders_checked':len(parameters),
                   'svg_bounds_checked':len(bounds),'visual_figure_sample':34},
        'tests':read('quality/design-v2/python-tests.json'),
        'limits':['Mobile layouts were tested at browser viewports, not on a physical phone.',
                  'All SVG bounds were checked; visual figure review sampled34 figures.',
                  'Not a complete WCAG conformance assessment or a learner pilot.',
                  'Training, model quality and live voice were not re-benchmarked by this visual revision.'],
        'evidence_sha256':{p.relative_to(OUT).as_posix():sha256(p.read_bytes()).hexdigest()
                           for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='report.json'},
    }
    (OUT/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({'report':'quality/design-v2/report.json','resolved':13,'reader_checks':reader['passed'],'embedded_checks':frames['passed']},ensure_ascii=False))


if __name__ == '__main__':
    main()
