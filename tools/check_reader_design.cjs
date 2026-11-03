/* Run with an existing Playwright browser: await require(thisFile).run(browser, repo).
   Browser ownership stays with the caller. All interactions use the actual reader. */
const fs = require('node:fs/promises');
const path = require('node:path');
const {pathToFileURL} = require('node:url');

async function run(browser, repo) {
  const out = path.join(repo, 'artifacts/design-redesign/reader');
  await fs.mkdir(out, {recursive:true});
  const url = pathToFileURL(path.join(repo,'START_HERE.html')).href;
  const html=await fs.readFile(path.join(repo,'START_HERE.html'));
  const report = {date:new Date().toISOString(),reader_sha256:require('node:crypto').createHash('sha256').update(html).digest('hex'),kind:'browser_design_verification_not_learner_study',checks:[],errors:[],measurements:{},screenshots:[]};
  const context = await browser.newContext({viewport:{width:1440,height:1000},offline:true,deviceScaleFactor:1});
  const page = await context.newPage();
  page.on('pageerror',error=>report.errors.push(error.message));
  const check = (name, pass, detail={}) => {report.checks.push({name,pass:!!pass,detail});};
  const settle = async()=>page.evaluate(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve))));
  const go = async id => {await page.goto(url+'#'+id);await page.locator('#'+id).waitFor({state:'visible'});await settle();};
  const snap = async name => {await page.screenshot({path:path.join(out,name+'.png'),scale:'css',animations:'disabled'});report.screenshots.push(name+'.png');};
  try {
    await go('HOME');
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
    check('D07 formula blocks and shape table',await page.locator('#M08-L03 .formula').count()>=4&&await page.locator('#M08-L03 table').count()>=1);
    await snap('attention-desktop');
    await page.locator('#M08-L03 .lesson-outline a').filter({hasText:'Самостоятельная работа'}).click();await settle();
    const taskY=await page.locator('#M08-L03 [data-stage="task"] h2').evaluate(el=>el.getBoundingClientRect().top);
    check('D06 local map moves to task',taskY>=60&&taskY<180,{top:taskY});await snap('assignment-desktop');
    await page.locator('#M08-L03 .practice-note a').filter({hasText:'Решение этой задачи'}).click();await settle();
    const answer=page.locator('#M08-SOLUTIONS .answer-key[open]').first();
    check('D11 lesson opens its own answer',await answer.locator('summary').innerText()==='Ответ: M08-L03-A');
    check('D11 neighboring answers remain closed',await page.locator('#M08-SOLUTIONS .answer-key[open]').count()===1);await snap('answer-return');
    await answer.locator('[data-return-task]').first().click();await settle();
    check('D11 return lands on assignment',decodeURIComponent(new URL(page.url()).hash).includes('самостоятельная-работа'));
    await page.locator('#M08-L03 .practice-note a').filter({hasText:'Решение этой задачи'}).click();await settle();
    await page.locator('#M08-SOLUTIONS .answer-key[open] [data-return-retry]').first().click();await settle();
    check('D11 new attempt lands on B/C',decodeURIComponent(new URL(page.url()).hash).includes('повтор-и-перенос'));
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
    await go('M04-L06');await page.locator('#M04-L06 a[data-figure]').first().click();
    check('Figure zoom opens dialog',await page.locator('#figure-dialog').isVisible());
    await page.locator('#figure-zoom').click();check('Native size control works',await page.locator('#figure-large').evaluate(el=>el.classList.contains('native')));
    await page.keyboard.press('Escape');check('Dialog closes by Escape',!await page.locator('#figure-dialog').isVisible());
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
    await go('M10-L06');const mobileH2=await page.locator('#M10-L06 .body-content h2').first().boundingBox();
    check('D13 first theory starts before 500px at390',mobileH2.y<500,{top:mobileH2.y,previousAudit:622});await snap('lesson-mobile');
    await page.locator('#menu-toggle').click();await settle();
    check('D02 mobile menu exposes active lesson',await page.locator('[data-material="M10-L06"]').isInViewport?.()??await page.locator('[data-material="M10-L06"]').isVisible());
    const mobileActive=await page.locator('[data-material="M10-L06"]').boundingBox();check('D02 current mobile link within screen',mobileActive.y>100&&mobileActive.y+mobileActive.height<844,{rect:mobileActive});await snap('menu-mobile');
    await page.keyboard.press('Escape');check('Mobile escape restores menu button focus',await page.locator('#menu-toggle').evaluate(el=>el===document.activeElement));
    await page.locator('#M10-L06 .lesson-outline summary').click();await page.locator('#M10-L06 .lesson-outline a').filter({hasText:'Самостоятельная'}).click();await settle();
    check('D06 mobile map closes after selection',!await page.locator('#M10-L06 .lesson-outline details').evaluate(el=>el.open));await snap('assignment-mobile');
    await go('M04-L06');const figure=page.locator('#M04-L06 a[data-figure]').first();await figure.scrollIntoViewIfNeeded();
    await page.locator('#M04-L06 a[data-figure] img').first().evaluate(el=>el.decode());
    check('D03 mobile-specific figure selected',await figure.locator('img').evaluate(el=>el.currentSrc.includes('mobile')));await snap('log-figure-mobile');
    await page.setViewportSize({width:820,height:1000});await go('M04-L06');await page.locator('#M04-L06 a[data-figure]').first().scrollIntoViewIfNeeded();
    await page.waitForFunction(()=>document.querySelector('#M04-L06 .figure-expand img').currentSrc.includes('mobile'));
    check('D03 tablet selects figure by actual column width',await page.locator('#M04-L06 .figure-expand').evaluate(el=>el.clientWidth<600));await snap('log-figure-tablet');
    await page.setViewportSize({width:390,height:844});
    for(const key of ['M08-L03','M08-L07','M15-L06']){await go(key);const fig=page.locator('#'+key+' a[data-figure]').first();await fig.scrollIntoViewIfNeeded();await fig.locator('img').evaluate(el=>el.decode());await snap(key+'-figure-mobile');}
    // At320, walk all161 documents. Tables and code may scroll internally; the page may not.
    await page.setViewportSize({width:320,height:844});
    const ids=await page.locator('article').evaluateAll(nodes=>nodes.map(el=>el.id));const overflows=[];
    for(const id of ids){await go(id);const measure=await page.evaluate(()=>({width:innerWidth,scroll:document.documentElement.scrollWidth}));if(measure.scroll>measure.width+1)overflows.push({id,...measure});}
    check('All161 sections reflow at320 without page overflow',ids.length===161&&overflows.length===0,{sections:ids.length,overflows});
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
    await page.emulateMedia({media:'screen'});
    const anchorIssues=await page.evaluate(()=>Array.from(document.querySelectorAll('a[href^="#"]')).filter(a=>{let id;try{id=decodeURIComponent(a.hash.slice(1));}catch{return true;}return !document.getElementById(id);}).slice(0,30).map(a=>a.getAttribute('href')));
    check('All generated fragment links resolve',anchorIssues.length===0,{issues:anchorIssues});
    check('No browser JavaScript errors',report.errors.length===0,{errors:report.errors});
  } catch(error) {report.errors.push(error.stack);report.checks.push({name:'QA script completed',pass:false,detail:error.message});}
  finally {await context.close();}
  report.passed=report.checks.filter(x=>x.pass).length;report.failed=report.checks.filter(x=>!x.pass).length;
  await fs.writeFile(path.join(out,'qa-results.json'),JSON.stringify(report,null,2));
  return report;
}
module.exports={run};
