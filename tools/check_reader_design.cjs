/* Run with an existing Playwright browser: await require(thisFile).run(browser, repo).
   Browser ownership stays with the caller. All interactions use the actual reader. */
const fs = require('node:fs/promises');
const path = require('node:path');
const {pathToFileURL} = require('node:url');

// D1: colour literals are allowed only in theme token blocks, i.e. rules whose every selector starts
// with :root (at top level or inside @media). Comments and strings are ignored; offsets keep lines.
const NAMED_COLORS = new Set(('aliceblue antiquewhite aqua aquamarine azure beige bisque black blanchedalmond blue blueviolet brown '+
  'burlywood cadetblue chartreuse chocolate coral cornflowerblue cornsilk crimson cyan darkblue darkcyan darkgoldenrod darkgray '+
  'darkgreen darkgrey darkkhaki darkmagenta darkolivegreen darkorange darkorchid darkred darksalmon darkseagreen darkslateblue '+
  'darkslategray darkslategrey darkturquoise darkviolet deeppink deepskyblue dimgray dimgrey dodgerblue firebrick floralwhite '+
  'forestgreen fuchsia gainsboro ghostwhite gold goldenrod gray green greenyellow grey honeydew hotpink indianred indigo ivory '+
  'khaki lavender lavenderblush lawngreen lemonchiffon lightblue lightcoral lightcyan lightgoldenrodyellow lightgray lightgreen '+
  'lightgrey lightpink lightsalmon lightseagreen lightskyblue lightslategray lightslategrey lightsteelblue lightyellow lime '+
  'limegreen linen magenta maroon mediumaquamarine mediumblue mediumorchid mediumpurple mediumseagreen mediumslateblue '+
  'mediumspringgreen mediumturquoise mediumvioletred midnightblue mintcream mistyrose moccasin navajowhite navy oldlace olive '+
  'olivedrab orange orangered orchid palegoldenrod palegreen paleturquoise palevioletred papayawhip peachpuff peru pink plum '+
  'powderblue purple rebeccapurple red rosybrown royalblue saddlebrown salmon sandybrown seagreen seashell sienna silver skyblue '+
  'slateblue slategray slategrey snow springgreen steelblue tan teal thistle tomato turquoise violet wheat white whitesmoke '+
  'yellow yellowgreen').split(' '));
function colorLiteralsOutsideRoot(css) {
  const blank = text => text.replace(/[^\n]/g, ' ');
  const text = css.replace(/\/\*[\s\S]*?\*\//g, blank).replace(/(["'])(?:\\.|(?!\1)[^\\\n])*\1/g, blank);
  const lineAt = offset => text.slice(0, offset).split('\n').length;
  const found = [], stack = [];
  let last = 0;
  const scan = (selector, start, body) => {
    let offset = start;
    for (const declaration of body.split(';')) {
      const colon = declaration.indexOf(':');
      if (colon > 0) {
        const value = declaration.slice(colon + 1), base = offset + colon + 1;
        const add = (at, literal) => found.push({line:lineAt(base + at), selector, literal});
        for (const m of value.matchAll(/#[0-9a-f]{3,8}\b/gi)) add(m.index, m[0]);
        for (const m of value.matchAll(/\b(?:rgba?|hsla?|hwb|lab|lch|oklab|oklch|color)\(/gi)) add(m.index, m[0]);
        for (const m of value.matchAll(/(?<![-\w#])[a-z]+(?![-\w(])/gi)) if (NAMED_COLORS.has(m[0].toLowerCase())) add(m.index, m[0]);
      }
      offset += declaration.length + 1;
    }
  };
  for (let i = 0; i < text.length; i++) {
    if (text[i] === '{') {
      if (stack.length) stack.at(-1).nested = true;
      stack.push({selector:text.slice(last, i).trim().replace(/\s+/g, ' '), start:i + 1, nested:false});
      last = i + 1;
    } else if (text[i] === '}') {
      const block = stack.pop();
      if (!block) throw new Error(`Unbalanced } at line ${lineAt(i)}`);
      const token = !block.selector.startsWith('@') && block.selector.split(',').every(s => s.trim().startsWith(':root'));
      if (!block.nested && !token) scan(block.selector, block.start, text.slice(block.start, i));
      last = i + 1;
    }
  }
  if (stack.length) throw new Error('Unclosed { in CSS');
  return found;
}

// F4: text/background token pairs meet 4.5:1 (WCAG 2.1 AA, body text) in the dark, light and print token sets.
// Every text token on every page surface, each hue on its soft fill, ink on soft fills, marks and text on accent.
const TEXT_PAIRS = [...['ink','muted','accent','act','hint','ok','warn','error'].flatMap(text => ['bg','surface','paper','code-bg'].map(ground => [text, ground])),
  ['accent','accent-soft'], ['act','act-soft'], ['hint','hint-soft'], ['ok','ok-soft'], ['warn','warn-soft'],
  ['ink','accent-soft'], ['ink','act-soft'], ['ink','hint-soft'], ['mark-fg','mark-bg'], ['bg','accent']];
function tokenSets(css) {
  const read = pattern => Object.fromEntries([...css.match(pattern)[1].matchAll(/--([\w-]+):(#[0-9a-fA-F]{6})/g)].map(m => [m[1], m[2]]));
  return {dark:read(/:root\{color-scheme:dark;([^}]*)\}/), light:read(/:root\[data-theme="light"\]\{color-scheme:light;([^}]*)\}/),
    print:read(/@media print\{:root\{color-scheme:light;([^}]*)\}\}/)};
}
function contrastRatio(a, b) {
  const lum = hex => {const v = [1, 3, 5].map(i => parseInt(hex.slice(i, i + 2), 16) / 255).map(c => c <= .04045 ? c / 12.92 : ((c + .055) / 1.055) ** 2.4);
    return .2126 * v[0] + .7152 * v[1] + .0722 * v[2];};
  const [high, low] = [lum(a), lum(b)].sort((x, y) => y - x);
  return (high + .05) / (low + .05);
}
function tokenContrast(css) {
  return Object.entries(tokenSets(css)).flatMap(([theme, t]) => TEXT_PAIRS.map(([text, ground]) => {
    const ratio = contrastRatio(t[text], t[ground]);
    return {theme, text, ground, ratio:Math.round(ratio * 1000) / 1000, pass:ratio >= 4.5};
  }));
}

async function run(browser, repo, options = {}) {
  const out = options.out || path.join(repo, 'artifacts/design-redesign/reader');
  await fs.mkdir(out, {recursive:true});
  const url = pathToFileURL(path.join(repo,'START_HERE.html')).href;
  const html=await fs.readFile(path.join(repo,'START_HERE.html'));
  const report = {date:new Date().toISOString(),reader_sha256:require('node:crypto').createHash('sha256').update(html).digest('hex'),kind:'browser_design_verification_not_learner_study',checks:[],errors:[],measurements:{},screenshots:[]};
  const context = await browser.newContext({viewport:{width:1440,height:1000},offline:true,deviceScaleFactor:1});
  const page = await context.newPage();
  const requests = [];context.on('request', request => requests.push(request.url()));
  page.on('pageerror',error=>report.errors.push(error.message));
  const check = (name, pass, detail={}) => {report.checks.push({name,pass:!!pass,detail});};
  const settle = async()=>page.evaluate(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve))));
  const go = async id => {await page.goto(url+'#'+id);await page.locator('#'+id).waitFor({state:'visible'});await settle();};
  const snap = async name => {await page.screenshot({path:path.join(out,name+'.png'),scale:'css',animations:'disabled'});report.screenshots.push(name+'.png');};
  try {
    await go('HOME');
    const outside=colorLiteralsOutsideRoot(await fs.readFile(path.join(repo,'tools/reader.css'),'utf8'));
    check('D1 colour literals only inside :root token blocks',outside.length===0,{count:outside.length,outside:outside.slice(0,20)});
    const tokenPairs=tokenContrast(await fs.readFile(path.join(repo,'tools/reader.css'),'utf8'));report.measurements.tokenContrast=tokenPairs;
    check('F4 text/background token pairs meet 4.5:1 in dark, light and print',tokenPairs.length===3*TEXT_PAIRS.length&&tokenPairs.every(pair=>pair.pass),
      {pairs:tokenPairs.length,min:Math.min(...tokenPairs.map(pair=>pair.ratio)),weak:tokenPairs.filter(pair=>!pair.pass)});
    const themeState=()=>page.evaluate(()=>({theme:document.documentElement.dataset.theme,meta:document.querySelector('meta[name="color-scheme"]')?.content,
      bg:getComputedStyle(document.body).backgroundColor,label:document.getElementById('theme-toggle').textContent,
      stored:(()=>{try{return localStorage.getItem('ml-llm-course-theme');}catch{return 'unavailable';}})()}));
    const first=await themeState();
    check('D1 first open is dark',first.theme==='dark'&&first.meta==='dark light'&&first.bg==='rgb(21, 23, 28)'&&first.stored===null,first);
    await page.locator('#theme-toggle').click();const light=await themeState();
    check('D1 theme switch turns light and saves the choice',light.theme==='light'&&light.bg==='rgb(246, 245, 241)'&&light.stored==='light'&&light.label.includes('светлая'),light);
    await page.reload();await page.locator('#HOME').waitFor({state:'visible'});await settle();const kept=await themeState();
    check('D1 saved light choice survives reload',kept.theme==='light'&&kept.bg==='rgb(246, 245, 241)'&&kept.label.includes('светлая'),kept);
    await page.locator('#theme-toggle').click();const back=await themeState();
    check('D1 theme switch returns to dark',back.theme==='dark'&&back.bg==='rgb(21, 23, 28)'&&back.stored==='dark',back);
    // F2: fonts come from data: URLs in the reader. check() alone is true for unknown families, so loaded faces are counted too.
    const fontState=family=>page.evaluate(async family=>{await document.fonts.ready;const faces=[...document.fonts].filter(f=>f.family.replace(/["']/g,'')===family);
      return{declared:faces.length,loaded:faces.filter(f=>f.status==='loaded').length,latin:document.fonts.check(`17px "${family}"`,'Rag 0l1'),cyrillic:document.fonts.check(`17px "${family}"`,'Ёжик')};},family);
    const sans=await fontState('Golos Text');sans.body=await page.evaluate(()=>getComputedStyle(document.body).fontFamily);
    check('F2 embedded Golos Text is used for text',sans.declared===6&&sans.loaded>=2&&sans.latin&&sans.cyrillic&&sans.body.startsWith('"Golos Text"'),sans);
    const cta=page.locator('.primary-action');const box=await cta.boundingBox();
    check('D01 prominent start action in first desktop viewport',box&&box.y+box.height<1000);
    await snap('home-desktop');
    await cta.click();await page.locator('#DIAGNOSTIC').waitFor({state:'visible'});
    check('D01 start action opens diagnostic',await page.locator('#DIAGNOSTIC').isVisible());
    await go('M10-L06');
    const orientation=await page.evaluate(()=>{const a=document.querySelector('[data-material="M10-L06"]'),s=document.querySelector('.sidebar-scroll'),ar=a.getBoundingClientRect(),sr=s.getBoundingClientRect();return{active:a.getAttribute('aria-current'),top:ar.top,bottom:ar.bottom,scrollTop:sr.top,scrollBottom:sr.bottom,breadcrumb:document.querySelector('#breadcrumbs').textContent};});
    check('D02 active lesson visible and module in breadcrumb',orientation.active==='page'&&orientation.top>=orientation.scrollTop&&orientation.bottom<=orientation.scrollBottom&&orientation.breadcrumb.includes('M10'),orientation);
    await snap('lesson-desktop');
    await go('M08-L03');
    const body=await page.locator('#M08-L03 .body-content').boundingBox();
    check('D07 prose has bounded measure',body.width<=740,{width:body.width});
    check('D06 distinct task roles and local map',(await page.locator('#M08-L03 .lesson-stage').count())>=4&&(await page.locator('#M08-L03 .lesson-outline a').count())>=7);
    await page.locator('#M08-L03 pre code').first().scrollIntoViewIfNeeded();
    const mono=await fontState('JetBrains Mono');mono.code=await page.locator('#M08-L03 pre code').first().evaluate(el=>getComputedStyle(el).fontFamily);
    check('F2 embedded JetBrains Mono is used for code',mono.declared===4&&mono.loaded>=1&&mono.latin&&mono.code.startsWith('"JetBrains Mono"'),mono);
    // Code shows every character as typed: JetBrains Mono ligatures (== drawn as one sign) are off in blocks, inline code and source notes.
    const ligatures=await page.evaluate(()=>Object.fromEntries([['block','#M08-L03 pre code'],['inline','.body-content p code'],['source','.source-note code']]
      .map(([name,selector])=>[name,getComputedStyle(document.querySelector(selector)).fontVariantLigatures])));
    check('F2 code ligatures are off',Object.values(ligatures).every(value=>value==='none'),ligatures);
    // D6: every Python example carries build-time token classes in theme colours, in both themes.
    const codeColours=()=>page.evaluate(()=>{const blocks=Array.from(document.querySelectorAll('pre code.language-python'));
      const colour=kind=>{const el=document.querySelector('pre code.language-python .'+kind);return el&&getComputedStyle(el).color;};
      return{blocks:blocks.length,highlighted:blocks.filter(block=>block.querySelector('.k,.s,.n,.c')).length,k:colour('k'),s:colour('s'),n:colour('n'),c:colour('c')};});
    const darkCode=await codeColours();await page.locator('#theme-toggle').click();const lightCode=await codeColours();await page.locator('#theme-toggle').click();
    check('D6 Python examples are highlighted with theme tokens',darkCode.blocks===142&&darkCode.highlighted===142
      &&darkCode.k==='rgb(88, 196, 198)'&&darkCode.s==='rgb(230, 181, 102)'&&darkCode.n==='rgb(134, 170, 240)'&&darkCode.c==='rgb(163, 169, 180)'
      &&lightCode.k==='rgb(11, 116, 120)'&&lightCode.s==='rgb(149, 96, 15)'&&lightCode.n==='rgb(31, 85, 176)'&&lightCode.c==='rgb(90, 98, 112)',{dark:darkCode,light:lightCode});
    await page.evaluate(()=>window.scrollTo(0,0));await settle();
    check('D07 formula blocks and shape table',await page.locator('#M08-L03 .formula').count()>=4&&await page.locator('#M08-L03 table').count()>=1);
    await snap('attention-desktop');
    const diagram=page.locator('#M08-L03 pre.mermaid').first();await diagram.scrollIntoViewIfNeeded();
    await page.waitForFunction(()=>document.querySelector('#M08-L03 pre.mermaid svg, #M08-L03 pre.mermaid.diagram-unavailable'));
    // D5: the diagram is drawn in the page theme and its styles name the tokens, so a switch and print need no redraw.
    const diagramLook=()=>diagram.evaluate(el=>{const svg=el.querySelector('svg'),main=svg?.querySelector('.node.main rect, .node.main polygon, .node.main path'),edge=svg?.querySelector('path.flowchart-link');
      return{island:el.dataset.theme||null,svg:svg?.id||null,bg:getComputedStyle(el).backgroundColor,main:main&&getComputedStyle(main).fill,edge:edge&&getComputedStyle(edge).stroke};});
    const darkDiagram=await diagramLook();
    await page.locator('#theme-toggle').click();const lightDiagram=await diagramLook();await page.locator('#theme-toggle').click();
    check('D5 Mermaid follows the page theme without a redraw',darkDiagram.island===null&&darkDiagram.svg&&darkDiagram.bg==='rgb(32, 36, 44)'&&darkDiagram.main==='rgb(31, 42, 68)'&&darkDiagram.edge==='rgb(163, 169, 180)'
      &&lightDiagram.svg===darkDiagram.svg&&lightDiagram.bg==='rgb(255, 255, 255)'&&lightDiagram.main==='rgb(228, 236, 251)'&&lightDiagram.edge==='rgb(90, 98, 112)',{dark:darkDiagram,light:lightDiagram});
    await page.emulateMedia({media:'print'});const printedDiagram=await diagramLook();await page.emulateMedia({media:'screen'});
    check('D5 Mermaid prints light while the dark theme is chosen',printedDiagram.bg==='rgb(255, 255, 255)'&&printedDiagram.main==='rgb(228, 236, 251)'&&printedDiagram.edge==='rgb(90, 98, 112)',printedDiagram);
    // Headings follow the 1.2.0 vocabulary (T13): "Самостоятельная задача A", "Новая и отложенная проверка".
    await page.locator('#M08-L03 .lesson-outline a').filter({hasText:'Самостоятельная задача A'}).click();await settle();
    const taskY=await page.locator('#M08-L03 [data-stage="task"] h2').evaluate(el=>el.getBoundingClientRect().top);
    check('D06 local map moves to task',taskY>=60&&taskY<180,{top:taskY});await snap('assignment-desktop');
    await page.locator('#M08-L03 .practice-note a').filter({hasText:'Решение этой задачи'}).click();await settle();
    const answer=page.locator('#M08-SOLUTIONS .answer-key[open]').first();
    check('D11 lesson opens its own answer',await answer.locator('summary').innerText()==='Ответ: M08-L03-A');
    check('D11 neighboring answers remain closed',await page.locator('#M08-SOLUTIONS .answer-key[open]').count()===1);await snap('answer-return');
    await answer.locator('[data-return-task]').first().click();await settle();
    check('D11 return lands on assignment',decodeURIComponent(new URL(page.url()).hash).includes('самостоятельная-задача-a'));
    await page.locator('#M08-L03 .practice-note a').filter({hasText:'Решение этой задачи'}).click();await settle();
    await page.locator('#M08-SOLUTIONS .answer-key[open] [data-return-retry]').first().click();await settle();
    check('D11 new attempt lands on B/C',decodeURIComponent(new URL(page.url()).hash).includes('новая-и-отложенная-проверка'));
    await page.locator('#course-search').fill('обратное давление');
    check('D10 result explains hidden answer context',(await page.locator('#search-results').innerText()).includes('Решение — после своей попытки'));
    await page.locator('#search-results a').first().click();await settle();
    const mark=page.locator('article:visible mark.search-hit').first();const mr=await mark.boundingBox();
    check('D10 exact match visible inside opened answer',await mark.isVisible()&&mr.y>=60&&mr.y<600&&await page.locator('#M10-SOLUTIONS .answer-key[open]').count()===1,{top:mr.y});await snap('search-answer');
    await page.locator('#course-search').fill('зюзюбрамбам-невозможный-запрос');
    check('Search empty state is explicit',await page.locator('#empty-search').isVisible()&&await page.locator('#search-results li').count()===0);
    await page.locator('#search-clear').click();
    check('Clear restores navigation',await page.locator('#book-navigation').isVisible());
    await page.keyboard.press('Control+k');check('Keyboard shortcut focuses search',await page.locator('#course-search').evaluate(el=>el===document.activeElement));
    // D3: M04-L06 is an inline figure; the dialog shows a copy of the shown layout, its ids renamed.
    await go('M04-L06');await page.locator('#M04-L06 a[data-figure]').first().click();
    check('Figure zoom opens dialog',await page.locator('#figure-dialog').isVisible());
    const zoomed=await page.evaluate(()=>{const box=document.getElementById('figure-large-svg'),svg=box.querySelector('svg');
      const ids=Array.from(document.querySelectorAll('.figure-variant svg [id], #figure-large-svg [id]'),el=>el.id);
      return{svg:!box.hidden&&!!svg,img:!document.getElementById('figure-large').hidden,labelledby:svg?.getAttribute('aria-labelledby'),ids:ids.length,duplicates:ids.length-new Set(ids).size,title:document.getElementById('figure-title').textContent};});
    check('D3 zoom shows the inline figure with unique ids',zoomed.svg&&!zoomed.img&&zoomed.labelledby==='M04-L06-desktop-title-zoom M04-L06-desktop-desc-zoom'&&zoomed.duplicates===0&&zoomed.title.length>0,zoomed);
    await page.locator('#figure-zoom').click();
    const native=await page.evaluate(()=>{const svg=document.querySelector('#figure-large-svg svg');return{native:document.querySelector('.figure-scroll').classList.contains('native'),width:svg.getBoundingClientRect().width,natural:Number(svg.getAttribute('width'))};});
    check('Native size control works',native.native&&Math.abs(native.width-native.natural)<1,native);
    await page.keyboard.press('Escape');check('Dialog closes by Escape',!await page.locator('#figure-dialog').isVisible());
    // D3: inline figures take the page theme. Every painted colour of every layout (hidden ones too) is a figure token
    // of the current theme; the card is --paper with no white backing; print is light while the dark theme is chosen.
    const figurePaint=()=>page.evaluate(()=>{const style=getComputedStyle(document.documentElement);
      const rgb=hex=>'rgb('+[1,3,5].map(i=>parseInt(hex.trim().slice(i,i+2),16)).join(', ')+')';
      const tokens=new Map(['--ink','--muted','--accent','--act','--hint','--line','--accent-soft','--act-soft','--hint-soft','--bg','--surface'].map(name=>[rgb(style.getPropertyValue(name)),name]));
      const issues=new Set(),used=new Set();let painted=0;
      const note=(value,where)=>{if(value==='none')return;painted++;const name=tokens.get(value);if(name)used.add(name);else issues.add(where+' '+value);};
      for(const el of document.querySelectorAll('.figure-variant svg *')){const tag=el.tagName.toLowerCase(),s=getComputedStyle(el);
        if(/^(rect|path|circle|text)$/.test(tag))note(s.fill,tag+' fill');
        if(/^(rect|path|circle)$/.test(tag))note(s.stroke,tag+' stroke');}
      const figure=document.querySelector('#M04-L06 .figure-inline');
      return{theme:document.documentElement.dataset.theme,figures:document.querySelectorAll('.figure-inline').length,painted,used:[...used].sort(),issues:[...issues].slice(0,10),
        card:getComputedStyle(figure).backgroundColor,ink:getComputedStyle(figure.querySelector('.figure-desktop text')).fill,
        layout:Array.from(figure.querySelectorAll('.figure-variant')).filter(v=>v.getClientRects().length).map(v=>v.classList[1]).join(',')};});
    const darkFigures=await figurePaint();await page.locator('#theme-toggle').click();const lightFigures=await figurePaint();await page.locator('#theme-toggle').click();
    check('D3 inline figures paint only with theme tokens in both themes',darkFigures.figures>=60&&darkFigures.painted>1000&&!darkFigures.issues.length&&!lightFigures.issues.length
      &&darkFigures.theme==='dark'&&darkFigures.card==='rgb(32, 36, 44)'&&darkFigures.ink==='rgb(231, 229, 222)'&&darkFigures.layout==='figure-desktop'
      &&lightFigures.theme==='light'&&lightFigures.card==='rgb(255, 255, 255)'&&lightFigures.ink==='rgb(29, 36, 48)',{dark:darkFigures,light:lightFigures});
    await page.emulateMedia({media:'print'});const printedFigures=await figurePaint();await page.emulateMedia({media:'screen'});
    check('D3 inline figures print light while the dark theme is chosen',printedFigures.theme==='dark'&&printedFigures.ink==='rgb(29, 36, 48)'&&!printedFigures.issues.length,printedFigures);
    // D4, D7: an image figure carries a light and a dark picture; the shown one follows the theme. Print shows the light one
    // at once: it is loaded eagerly, because a print does not wait for a picture to switch or load its source.
    const shownImage=id=>page.evaluate(id=>Array.from(document.querySelectorAll(`#${id} .figure-themed img`)).filter(img=>img.getClientRects().length)
      .map(img=>img.complete&&img.naturalWidth?img.currentSrc.split('/').pop():'not loaded').join(','),id);
    const awaitShown=(id,ending)=>page.waitForFunction(([id,e])=>Array.from(document.querySelectorAll(`#${id} .figure-themed img`))
      .some(img=>img.getClientRects().length&&img.complete&&img.naturalWidth&&img.currentSrc.endsWith(e)),[id,ending]).catch(()=>{});
    const showImage=async(id,ending)=>{await go(id);await page.locator(`#${id} .figure-themed`).first().scrollIntoViewIfNeeded();await awaitShown(id,ending);};
    await showImage('M02-L02','-dark.svg');const imageDark=await shownImage('M02-L02');
    await page.locator('#theme-toggle').click();await awaitShown('M02-L02','-desktop.svg');const imageLight=await shownImage('M02-L02');
    await page.locator('#theme-toggle').click();await awaitShown('M02-L02','-dark.svg');
    await page.emulateMedia({media:'print'});const imagePrint=await shownImage('M02-L02');await page.emulateMedia({media:'screen'});
    check('D4 image figure follows the theme and prints light',imageDark==='M02-L02-desktop-dark.svg'&&imageLight==='M02-L02-desktop.svg'&&imagePrint==='M02-L02-desktop.svg',{imageDark,imageLight,imagePrint});
    await showImage('M08-L08X','-dark.svg');
    check('D4 hand-drawn figure shows its dark copy',await shownImage('M08-L08X')==='M08-L08X-dark.svg');
    const pictures=await page.evaluate(()=>{const all=Array.from(document.querySelectorAll('.figure-expand:not(.figure-inline)'));
      return{figures:all.length,themed:all.filter(f=>f.matches('.figure-themed')&&f.querySelector('picture.figure-light img[loading="eager"]')&&f.querySelector('picture.figure-dark img')).length,
        mobileDark:all.filter(f=>f.querySelector('picture.figure-dark source[data-layout="mobile"]')).length};});
    check('D4 every image figure offers a dark copy',pictures.figures===50&&pictures.themed===50&&pictures.mobileDark===36,pictures);
    // D7: a lesson draws its diagrams when it is opened, so a lesson printed right away prints them drawn.
    await go('M14-L08X');const diagramTop=await page.locator('#M14-L08X pre.mermaid').evaluate(el=>Math.round(el.getBoundingClientRect().top));
    const drawnOnOpen=await page.waitForFunction(()=>document.querySelector('#M14-L08X pre.mermaid svg'),null,{timeout:8000}).then(()=>true,()=>false);
    check('D7 a lesson draws its diagrams on opening, before they scroll into view',drawnOnOpen&&diagramTop>1400,{diagramTop,drawnOnOpen});
    // Contrast of the exact utility roles from the audit, composited over solid surfaces.
    const contrast=await page.evaluate(()=>{
      const lum=rgb=>{const v=rgb.slice(0,3).map(v=>{v/=255;return v<=.04045?v/12.92:((v+.055)/1.055)**2.4;});return .2126*v[0]+.7152*v[1]+.0722*v[2];};
      return ['.brand-caption','.search-status','.nav-group h2','.nav-marker','.nav-foot','.source-note'].map(selector=>{
        const el=Array.from(document.querySelectorAll(selector)).find(el=>el.getClientRects().length);const color=getComputedStyle(el).color.match(/[\d.]+/g).map(Number);
        let parent=el,bg;while(parent){const value=getComputedStyle(parent).backgroundColor.match(/[\d.]+/g).map(Number);if(value.length===3||value[3]===1){bg=value;break;}parent=parent.parentElement;}bg??=[255,255,255];const a=lum(color),b=lum(bg);return{selector,ratio:(Math.max(a,b)+.05)/(Math.min(a,b)+.05),font:getComputedStyle(el).fontSize};
      });
    });report.measurements.utilityContrast=contrast;check('D12 six audited utility roles meet 4.5 contrast',contrast.every(x=>x.ratio>=4.5),contrast);
    await page.setViewportSize({width:390,height:844});await go('HOME');
    const mobileCTA=await page.locator('.primary-action').boundingBox();check('D01 mobile start action in first viewport',mobileCTA.y+mobileCTA.height<844);await snap('home-mobile');
    const themeBox=await page.locator('#theme-toggle').boundingBox();
    check('D1 theme switch visible in the mobile top bar',await page.locator('#theme-toggle').isVisible()&&themeBox.x+themeBox.width<=390,{rect:themeBox});
    await go('M10-L06');const mobileH2=await page.locator('#M10-L06 .body-content h2').first().boundingBox();
    check('D13 first theory starts before 500px at390',mobileH2.y<500,{top:mobileH2.y,previousAudit:622});await snap('lesson-mobile');
    await page.locator('#menu-toggle').click();await settle();
    check('D02 mobile menu exposes active lesson',await page.locator('[data-material="M10-L06"]').isInViewport?.()??await page.locator('[data-material="M10-L06"]').isVisible());
    const mobileActive=await page.locator('[data-material="M10-L06"]').boundingBox();check('D02 current mobile link within screen',mobileActive.y>100&&mobileActive.y+mobileActive.height<844,{rect:mobileActive});await snap('menu-mobile');
    await page.keyboard.press('Escape');check('Mobile escape restores menu button focus',await page.locator('#menu-toggle').evaluate(el=>el===document.activeElement));
    await page.locator('#M10-L06 .lesson-outline summary').click();await page.locator('#M10-L06 .lesson-outline a').filter({hasText:'Самостоятельная'}).click();await settle();
    check('D06 mobile map closes after selection',!await page.locator('#M10-L06 .lesson-outline details').evaluate(el=>el.open));await snap('assignment-mobile');
    // D3: both layouts are inline; a container query on the figure shows the one that fits the column.
    const shownLayout=key=>page.locator('#'+key+' a[data-figure]').first().evaluate(el=>Array.from(el.querySelectorAll('.figure-variant')).filter(v=>v.getClientRects().length).map(v=>v.classList[1]).join(','));
    await go('M04-L06');const figure=page.locator('#M04-L06 a[data-figure]').first();await figure.scrollIntoViewIfNeeded();
    check('D03 mobile-specific figure selected',await shownLayout('M04-L06')==='figure-mobile');await snap('log-figure-mobile');
    await page.setViewportSize({width:820,height:1000});await go('M04-L06');await page.locator('#M04-L06 a[data-figure]').first().scrollIntoViewIfNeeded();
    const tablet={layout:await shownLayout('M04-L06'),width:await page.locator('#M04-L06 .figure-expand').evaluate(el=>el.clientWidth)};
    check('D03 tablet selects figure by actual column width',tablet.layout==='figure-mobile'&&tablet.width<600,tablet);await snap('log-figure-tablet');
    await page.setViewportSize({width:390,height:844});
    await showImage('M02-L02','-mobile-dark.svg');
    check('D4 phone shows the mobile dark copy',await shownImage('M02-L02')==='M02-L02-mobile-dark.svg');
    for(const key of ['M08-L03','M08-L07','M15-L06']){await go(key);const fig=page.locator('#'+key+' a[data-figure]').first();await fig.scrollIntoViewIfNeeded();await settle();await snap(key+'-figure-mobile');}
    // At320, walk every document (161 in 1.1.0, more since 1.2.0). Tables and code may scroll internally; the page may not.
    await page.setViewportSize({width:320,height:844});
    const ids=await page.locator('article').evaluateAll(nodes=>nodes.map(el=>el.id));const overflows=[];
    for(const id of ids){await go(id);const measure=await page.evaluate(()=>({width:innerWidth,scroll:document.documentElement.scrollWidth}));if(measure.scroll>measure.width+1)overflows.push({id,...measure});}
    check('All161 sections reflow at320 without page overflow',ids.length>=161&&overflows.length===0,{sections:ids.length,overflows});
    // Malformed and unknown deep links should recover to HOME without errors.
    await page.goto(url+'#M99-missing');await settle();check('Unknown fragment recovers',await page.locator('#HOME').isVisible());
    await page.goto(url+'#%E0%A4%A');await settle();check('Malformed encoded fragment recovers',await page.locator('#HOME').isVisible());
    await go('M08-L03');await page.keyboard.press('Control+k');await page.locator('#course-search').fill('softmax');
    check('Mobile keyboard search opens menu',await page.locator('#menu-toggle').getAttribute('aria-expanded')==='true');await snap('search-mobile320');
    await page.keyboard.press('Escape');
    await page.setViewportSize({width:1440,height:1000});await go('M08-L03');
    await page.locator('#M08-L03 [data-stage="retry"] a').filter({hasText:'ключи B/C'}).click();await settle();
    check('D11 B/C link opens B/C instead of A',(await page.locator('#M08-SOLUTIONS .answer-key[open]').last().locator('summary').innerText()).includes('B и C'));
    await page.emulateMedia({media:'print'});
    check('Print excludes navigation and unopened answers',!await page.locator('.sidebar').isVisible()&&!await page.locator('#M08-SOLUTIONS .answer-key:not([open])').first().isVisible());
    const printed=await page.evaluate(()=>({theme:document.documentElement.dataset.theme,bg:getComputedStyle(document.documentElement).getPropertyValue('--bg').trim(),ink:getComputedStyle(document.body).color}));
    check('D1 print uses light tokens while the dark theme is chosen',printed.theme==='dark'&&printed.bg==='#f6f5f1'&&printed.ink==='rgb(29, 36, 48)',printed);
    // D7: print has no theme switch and shows the address of every external link in brackets.
    const printView=await page.evaluate(()=>{const link=document.querySelector('#M08-L03 .external-link');
      return{toggle:document.getElementById('theme-toggle').getClientRects().length,after:getComputedStyle(link,'::after').content,href:link.getAttribute('href')};});
    check('D7 print hides the theme switch and shows external link addresses',printView.toggle===0&&/attr\(href\)|https?:/.test(printView.after),printView);
    await page.emulateMedia({media:'screen'});
    const screenAfter=await page.evaluate(()=>getComputedStyle(document.querySelector('#M08-L03 .external-link'),'::after').content);
    check('D7 link addresses appear only in print',screenAfter==='none',{screenAfter});
    // D5: every diagram is drawn, keeps labels inside their shapes and paints only with diagram tokens, in both themes.
    const diagramPaint=pre=>{const root=document.documentElement,style=getComputedStyle(root),svg=pre.querySelector('svg');
      if(!svg)return{theme:root.dataset.theme,issues:['not drawn']};
      const rgb=hex=>'rgb('+[1,3,5].map(i=>parseInt(hex.trim().slice(i,i+2),16)).join(', ')+')';
      const tokens=new Set(['--paper','--ink','--muted','--accent','--accent-soft','--hint','--hint-soft'].map(name=>rgb(style.getPropertyValue(name)))),issues=new Set();
      const paint=(value,where)=>{if(!value||value==='none'||value.startsWith('url('))return;
        const srgb=value.match(/^color\(srgb ([\d.]+) ([\d.]+) ([\d.]+)(?: \/ ([\d.]+))?\)$/),plain=value.match(/^rgba?\((\d+), (\d+), (\d+)(?:, ([\d.]+))?\)$/);
        const parts=srgb?[...srgb.slice(1,4).map(v=>Math.round(v*255)),srgb[4]??1]:plain?[...plain.slice(1,4).map(Number),plain[4]??1]:null;
        if(parts&&Number(parts[3])===0)return;
        if(!parts||!tokens.has(`rgb(${parts.slice(0,3).join(', ')})`))issues.add(where+' '+value);};
      for(const el of svg.querySelectorAll('*')){
        if(!el.closest('marker')&&!el.getClientRects().length)continue;
        const s=getComputedStyle(el),tag=el.tagName.toLowerCase();
        if(/^(rect|path|polygon|circle|ellipse|line|polyline)$/.test(tag)){paint(s.fill,tag+' fill');if(parseFloat(s.strokeWidth)>0)paint(s.stroke,tag+' stroke');}
        if(Array.from(el.childNodes).some(n=>n.nodeType===3&&n.textContent.trim()))paint(tag==='text'||tag==='tspan'?s.fill:s.color,tag+' text');
        if(el.namespaceURI==='http://www.w3.org/1999/xhtml')paint(s.backgroundColor,tag+' background');
        const label=tag==='foreignobject'&&el.firstElementChild;
        if(label&&(label.scrollWidth>parseFloat(el.getAttribute('width'))+1||label.scrollHeight>parseFloat(el.getAttribute('height'))+1))issues.add('label overflow: '+el.textContent.trim().slice(0,40));}
      return{theme:root.dataset.theme,issues:[...issues]};};
    const diagramIds=await page.evaluate(()=>Array.from(new Set(Array.from(document.querySelectorAll('pre.mermaid'),pre=>pre.closest('article').id))));
    const diagramIssues=[];let diagramCount=0;
    for(const id of diagramIds){await go(id);
      for(const pre of await page.locator('#'+id+' pre.mermaid').all()){
        await pre.scrollIntoViewIfNeeded();await page.waitForFunction(el=>el.querySelector('svg')||el.classList.contains('diagram-unavailable'),await pre.elementHandle());
        const dark=await pre.evaluate(diagramPaint);await page.evaluate(()=>document.getElementById('theme-toggle').click());
        const light=await pre.evaluate(diagramPaint);await page.evaluate(()=>document.getElementById('theme-toggle').click());
        diagramCount++;
        if(dark.theme!=='dark'||light.theme!=='light'||dark.issues.length||light.issues.length)diagramIssues.push({id,dark,light});}}
    check('D5 all diagrams paint only with theme tokens in both themes',diagramCount>=33&&diagramIssues.length===0,{diagrams:diagramCount,issues:diagramIssues.slice(0,10)});
    const anchorIssues=await page.evaluate(()=>Array.from(document.querySelectorAll('a[href^="#"]')).filter(a=>{let id;try{id=decodeURIComponent(a.hash.slice(1));}catch{return true;}return !document.getElementById(id);}).slice(0,30).map(a=>a.getAttribute('href')));
    check('All generated fragment links resolve',anchorIssues.length===0,{issues:anchorIssues});
    const external=requests.filter(url=>!/^(file|data|about|blob):/.test(url));
    check('F2 no requests outside file:, data:, about: and blob:',external.length===0,{requests:requests.length,external:external.slice(0,20)});
    check('No browser JavaScript errors',report.errors.length===0,{errors:report.errors});
  } catch(error) {report.errors.push(error.stack);report.checks.push({name:'QA script completed',pass:false,detail:error.message});}
  finally {await context.close();}
  report.passed=report.checks.filter(x=>x.pass).length;report.failed=report.checks.filter(x=>!x.pass).length;
  await fs.writeFile(path.join(out,'qa-results.json'),JSON.stringify(report,null,2));
  return report;
}
module.exports={run,colorLiteralsOutsideRoot,tokenContrast,TEXT_PAIRS};

// Without a browser: node tools/check_reader_design.cjs [file.css] runs only the static colour check.
if (require.main === module) {
  const file = process.argv[2] || path.join(__dirname, 'reader.css');
  const css = require('node:fs').readFileSync(file, 'utf8'), outside = colorLiteralsOutsideRoot(css);
  console.log(JSON.stringify({check:'colour literals only inside :root token blocks', file:path.relative(process.cwd(), file), count:outside.length, outside}, null, 2));
  // The token contrast needs the reader's token blocks; another stylesheet gets only the literal check.
  const weak = /:root\{color-scheme:dark;/.test(css) ? tokenContrast(css).filter(pair => !pair.pass) : [];
  if (weak.length) console.log(JSON.stringify({check:'text/background token pairs meet 4.5:1', weak}, null, 2));
  process.exitCode = outside.length || weak.length ? 1 : 0;
}
