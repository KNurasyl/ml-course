/* Integration contract: every distinct experiment works inside the sandboxed reader. */
const fs=require('node:fs/promises'),path=require('node:path'),{pathToFileURL}=require('node:url');
async function run(browser,repo,options={}){
 const url=pathToFileURL(path.join(repo,'START_HERE.html')).href,out=options.out||path.join(repo,'artifacts/design-redesign/integration');await fs.mkdir(out,{recursive:true});
 const report={date:new Date().toISOString(),reader_sha256:require('node:crypto').createHash('sha256').update(await fs.readFile(path.join(repo,'START_HERE.html'))).digest('hex'),cases:[],errors:[],behavior:[]};
 for(const width of [1366,320]){
  const context=await browser.newContext({viewport:{width,height:844},offline:true,deviceScaleFactor:1}),page=await context.newPage();context.setDefaultTimeout(5000);page.on('pageerror',e=>report.errors.push(e.message));
  try{
   await page.goto(url+'#HOME');
   const entries=await page.locator('iframe.experiment').evaluateAll(frames=>Array.from(new Map(frames.map(f=>[f.getAttribute('src'),{src:f.getAttribute('src'),id:f.closest('article').id}])).values()));
   for(const entry of entries){
    await page.goto(url+'#'+entry.id);await page.locator('#'+entry.id).waitFor({state:'visible'});
    const selector='#'+entry.id+' iframe.experiment[src="'+entry.src+'"]';const frame=page.frameLocator(selector);
    await page.locator(selector).scrollIntoViewIfNeeded();
    await page.waitForFunction(selector=>parseFloat(document.querySelector(selector).style.height)>0,selector);
    // An off-screen cross-origin frame gets no animation frames, so a late scroll would hang an unbounded wait.
    await page.locator(selector).scrollIntoViewIfNeeded();
    await Promise.race([frame.locator('body').evaluate(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)))),page.waitForTimeout(3000)]);
    const measure=await frame.locator('body').evaluate(body=>({innerWidth,innerHeight,scrollWidth:document.documentElement.scrollWidth,scrollHeight:document.documentElement.scrollHeight,font:getComputedStyle(body).fontFamily,controls:Array.from(document.querySelectorAll('input,select,button')).filter(el=>el.getClientRects().length).map(el=>{const r=el.getBoundingClientRect();return{id:el.id,left:r.left,right:r.right};})}));
    report.cases.push({width,...entry,...measure,pass:measure.scrollWidth<=measure.innerWidth+1&&measure.scrollHeight<=measure.innerHeight+2&&measure.font.includes('Segoe UI')&&measure.controls.every(c=>c.left>=0&&c.right<=measure.innerWidth+1)});
    // D2: the frame follows the reader theme live and shows no bright block in the dark theme.
    // Folded into the case so the case count stays the one archived for design-v2.
    const look=()=>frame.locator('html').evaluate(html=>{const rgb=value=>value.match(/[\d.]+/g).map(Number),bright=[];
     for(const el of document.querySelectorAll('body *')){const v=rgb(getComputedStyle(el).backgroundColor);
      if((v.length===3||v[3]>.5)&&Math.min(v[0],v[1],v[2])>200&&el.getClientRects().length)bright.push(el.tagName.toLowerCase()+(el.getAttribute('class')?'.'+el.getAttribute('class').split(' ')[0]:''));}
     // The canvas shows the html background, or the body's when html has none (CSS background propagation).
     const own=getComputedStyle(html).backgroundColor,page=rgb(own).length===4&&rgb(own)[3]===0?getComputedStyle(document.body).backgroundColor:own;
     return{theme:html.dataset.theme,page,bright:bright.slice(0,8)};});
    const switchTo=async theme=>{await page.locator('#theme-toggle').click();await frame.locator(`html[data-theme="${theme}"]`).waitFor({state:'attached'});};
    const dark=await look();await switchTo('light');const light=await look();await switchTo('dark');
    const themed=dark.theme==='dark'&&dark.page==='rgb(21, 23, 28)'&&!dark.bright.length&&light.theme==='light'&&light.page==='rgb(246, 245, 241)';
    Object.assign(report.cases.at(-1),{theme:{dark,light},pass:report.cases.at(-1).pass&&themed});
    await fs.writeFile(path.join(out,'partial-results.json'),JSON.stringify(report,null,2));
   }
   // Both expansion and shrinking are required; using scrollHeight would hide the latter bug.
   await page.goto(url+'#M08-L03');await page.locator('#M08-L03').waitFor({state:'visible'});const iframe=page.locator('#M08-L03 iframe'),attention=page.frameLocator('#M08-L03 iframe');await iframe.scrollIntoViewIfNeeded();
   const initial=(await iframe.boundingBox()).height;await attention.locator('#advanced summary').click();await page.waitForFunction(h=>parseFloat(document.querySelector('#M08-L03 iframe').style.height)>h,initial);const expanded=(await iframe.boundingBox()).height;
   await attention.locator('#advanced summary').click();await page.waitForFunction(h=>parseFloat(document.querySelector('#M08-L03 iframe').style.height)<h,expanded);const collapsed=(await iframe.boundingBox()).height;
   report.behavior.push({name:'iframe expands and shrinks',width,initial,expanded,collapsed,pass:expanded>initial&&Math.abs(initial-collapsed)<3});
   await attention.locator('#advanced summary').click();await attention.locator('#axis').selectOption('col');await attention.locator('#step').selectOption('4');const warn=await attention.locator('#invariant').innerText();report.behavior.push({name:'wrong axis diagnostic inside iframe',width,pass:warn.includes('Неправильная ось')});
   await attention.locator('#reset').click();await attention.locator('#step').selectOption('5');await iframe.scrollIntoViewIfNeeded();await page.screenshot({path:path.join(out,'attention-'+width+'.png'),scale:'css'});
   await page.goto(url+'#M15-L06');await page.locator('#M15-L06').waitFor({state:'visible'});const voice=page.frameLocator('#M15-L06 iframe');await page.locator('#M15-L06 iframe').scrollIntoViewIfNeeded();
   const slider=voice.locator('input[type=range]').first();await slider.focus();await slider.press('Home');const before=await voice.locator('body').innerText();await slider.press('End');const after=await voice.locator('body').innerText();report.behavior.push({name:'voice threshold changes segmentation in embedded experiment',width,pass:before!==after});await page.screenshot({path:path.join(out,'voice-'+width+'.png'),scale:'css'});
  }catch(error){report.errors.push(error.stack);}finally{await context.close();}
 }
 report.passed=report.cases.filter(x=>x.pass).length+report.behavior.filter(x=>x.pass).length;report.failed=report.cases.filter(x=>!x.pass).length+report.behavior.filter(x=>!x.pass).length+report.errors.length;await fs.writeFile(path.join(out,'qa-results.json'),JSON.stringify(report,null,2));return report;
}
module.exports={run};
