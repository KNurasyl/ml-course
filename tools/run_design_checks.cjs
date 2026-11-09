/* Design 3.0 acceptance run (phase 6, F4): reader and experiment checks, then theme screenshots.

   Playwright is not a project dependency and nothing is installed into the course. Point PW_PATH at a
   node_modules folder that contains playwright (for example the npx cache after `npx -y playwright@1.64.1
   --version`) and run, from the repository root:
     PowerShell: $env:PW_PATH = "<...>\node_modules"; node tools/run_design_checks.cjs
   The browser is Microsoft Edge (PW_CHANNEL=msedge, default) or another Playwright channel; PW_CHANNEL=""
   uses Playwright's own Chromium. Results go to artifacts/design-v3 (ignored by git); screenshots go to
   quality/design-v3/screenshots. Then look at the screenshots and run tools/build_design_v3_report.py.
   SKIP_QA=1 takes only the screenshots. Nothing here assesses a learner. */
const path = require('node:path');
const fs = require('node:fs/promises');
const {pathToFileURL} = require('node:url');

const repo = path.resolve(__dirname, '..');
const {chromium} = require(process.env.PW_PATH ? path.join(process.env.PW_PATH, 'playwright') : 'playwright');
// [material id, file name, element to bring to the top, wait for its Mermaid svg, experiment frame to wait for]
const PAGES = [
  ['HOME', 'home', null, false, null],
  ['M10-L06', 'lesson', null, false, null],
  ['M10-L06', 'code', '#M10-L06 .body-content pre:has(> code)', false, null],
  ['M00-L01', 'python-code', '#M00-L01 .body-content pre:has(> code.language-python)', false, null],
  ['M08-L03', 'mermaid', '#M08-L03 pre.mermaid', true, null],
  ['M05-L04', 'mermaid-M05-L04', '#M05-L04 pre.mermaid', true, null],
  ['M14-L06', 'mermaid-M14-L06', '#M14-L06 pre.mermaid', true, null],
  ['M00-L02', 'experiment', '#M00-L02 iframe.experiment', false, '#M00-L02 iframe.experiment'],
  ['M16-L06', 'product-experiment', '#M16-L06 iframe.experiment', false, '#M16-L06 iframe.experiment'],
];

async function screenshots(browser) {
  const out = path.join(repo, 'quality/design-v3/screenshots');
  await fs.mkdir(out, {recursive: true});
  const url = pathToFileURL(path.join(repo, 'START_HERE.html')).href;
  const shots = [], errors = [];
  for (const theme of ['dark', 'light']) {
    for (const [width, height] of [[1440, 1000], [390, 844]]) {
      for (const [id, name, selector, mermaid, frame] of PAGES) {
        // A fresh context per shot: the menu shows only the module of this page, whatever was opened before.
        const context = await browser.newContext({viewport: {width, height}, offline: true, deviceScaleFactor: 1});
        if (theme === 'light') await context.addInitScript(() => { try { localStorage.setItem('ml-llm-course-theme', 'light'); } catch {} });
        const page = await context.newPage();
        page.on('pageerror', error => errors.push(`${theme}/${width}/${name}: ${error.message}`));
        await page.goto(url + '#' + id);
        await page.locator('#' + id).waitFor({state: 'visible'});
        if (selector) {
          await page.locator(selector).first().scrollIntoViewIfNeeded();
          if (mermaid) await page.waitForFunction(s => document.querySelector(s + ' svg'), selector);
          if (frame) {
            await page.frameLocator(frame).locator(`html[data-theme="${theme}"]`).waitFor({state: 'attached'});
            await page.waitForFunction(s => parseFloat(document.querySelector(s).style.height) > 0, frame);
          }
          await page.evaluate(s => window.scrollBy(0, document.querySelector(s).getBoundingClientRect().top - 90), selector);
        }
        await page.evaluate(() => document.fonts.ready.then(() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))));
        const actual = await page.evaluate(() => document.documentElement.dataset.theme);
        if (actual !== theme) errors.push(`${name}: expected ${theme}, page has ${actual}`);
        const file = `${name}-${width}-${theme}.png`;
        await page.screenshot({path: path.join(out, file), scale: 'css', animations: 'disabled'});
        shots.push(file);
        await context.close();
      }
    }
  }
  return {shots: shots.length, errors};
}

(async () => {
  const channel = process.env.PW_CHANNEL ?? 'msedge';
  const browser = await chromium.launch({...(channel ? {channel} : {}), headless: true});
  const summary = {};
  try {
    if (process.env.SKIP_QA !== '1') {
      const reader = await require('./check_reader_design.cjs').run(browser, repo, {out: path.join(repo, 'artifacts/design-v3/reader')});
      summary.reader = {passed: reader.passed, failed: reader.failed, errors: reader.errors, failedChecks: reader.checks.filter(c => !c.pass).map(c => c.name)};
      const frames = await require('./check_embedded_visuals.cjs').run(browser, repo, {out: path.join(repo, 'artifacts/design-v3/integration')});
      summary.frames = {cases: frames.cases.length, passed: frames.passed, failed: frames.failed, errors: frames.errors};
    }
    summary.screenshots = await screenshots(browser);
  } finally {
    await browser.close();
  }
  console.log(JSON.stringify(summary, null, 1));
  const failed = (summary.reader?.failed || 0) + (summary.frames?.failed || 0) + summary.screenshots.errors.length;
  process.exitCode = failed ? 1 : 0;
})().catch(error => { console.error(error); process.exit(1); });
