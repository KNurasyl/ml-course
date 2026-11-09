/* Navigation, fragment search and sizing for the offline textbook. No network or grades. */
(() => {
  'use strict';
  const articles = Array.from(document.querySelectorAll('.material'));
  const links = Array.from(document.querySelectorAll('[data-material]'));
  const byId = new Map(articles.map(article => [article.id, article]));
  const search = document.getElementById('course-search');
  const status = document.getElementById('search-status');
  const results = document.getElementById('search-results');
  const menu = document.getElementById('menu-toggle');
  const sidebar = document.getElementById('course-menu');
  const sidebarScroll = sidebar.querySelector('.sidebar-scroll');
  const normalized = text => text.toLocaleLowerCase('ru').replace(/ё/g, 'е');
  const narrow = matchMedia('(max-width: 760px)');
  const outlineWide = matchMedia('(min-width: 1280px)');
  const entries = [];
  let lastMaterial = '';
  document.body.classList.add('js');

  // Every result points to one stable block, including a block in a closed answer.
  articles.forEach(article => {
    const title = article.querySelector('h1');
    const nodes = [title, ...article.querySelectorAll('.tool-versions > div, .body-content h2, .body-content h3, .body-content p, .body-content li, .body-content pre, .body-content td, .body-content summary')];
    nodes.forEach((node, index) => {
      if (!node || node.closest('nav') || (node.tagName === 'LI' && node.querySelector('p, li'))) return;
      const text = node.textContent.replace(/\s+/g, ' ').trim();
      if (!text) return;
      if (!node.id) node.id = `${article.id}--fragment-${index}`;
      const answer = !!node.closest('.answer-key') || article.dataset.kind === 'answer';
      entries.push({article, node, text, normalized:normalized(text), answer});
    });
  });

  function closeMenu(restoreFocus = false) {
    const wasOpen = document.body.classList.contains('nav-open');
    document.body.classList.remove('nav-open');
    menu.setAttribute('aria-expanded', 'false');
    document.querySelector('.layout').inert = false;
    if (restoreFocus && wasOpen) menu.focus();
  }
  function revealActive() {
    if (search.value.trim()) return;
    const active = sidebar.querySelector('[aria-current="page"]');
    if (!active) return;
    const parent = active.closest('details');
    if (parent) parent.open = true;
    const a = active.getBoundingClientRect(), box = sidebarScroll.getBoundingClientRect();
    const moduleBox = parent?.getBoundingClientRect();
    if (moduleBox && moduleBox.height <= box.height - 24) {
      if (moduleBox.top < box.top + 12 || moduleBox.bottom > box.bottom - 12)
        sidebarScroll.scrollTop += moduleBox.top - box.top - 12;
    } else if (a.top < box.top + 12 || a.bottom > box.bottom - 12) {
      sidebarScroll.scrollTop += a.top - box.top - Math.min(120, box.height / 3);
    }
  }
  function openMenu(focusSearch = false) {
    document.body.classList.add('nav-open');
    menu.setAttribute('aria-expanded', 'true');
    if (narrow.matches) document.querySelector('.layout').inert = true;
    revealActive();
    (focusSearch ? search : document.getElementById('menu-close')).focus();
  }
  function updateContext(current) {
    const breadcrumbs = document.getElementById('breadcrumbs');
    breadcrumbs.replaceChildren();
    const addLink = (label, target) => {
      const a = document.createElement('a'); a.href = '#' + target; a.textContent = label;
      breadcrumbs.append(a);
    };
    const divider = () => { const span = document.createElement('span'); span.textContent = '/'; span.className = 'divider'; span.ariaHidden = 'true'; breadcrumbs.append(span); };
    addLink('Учебник', 'HOME');
    const module = current.dataset.module;
    if (module && module !== current.id && byId.has(module)) { divider(); addLink(module, module); }
    if (current.id !== 'HOME') {
      divider(); const span = document.createElement('span'); span.textContent = current.dataset.shortTitle || current.dataset.title; span.title = current.dataset.title; breadcrumbs.append(span);
    }
    document.querySelectorAll('.lesson-outline details').forEach(details => { details.open = outlineWide.matches; });
    current.querySelectorAll('iframe.experiment').forEach(frame => { frame.contentWindow?.postMessage({type:'course:measure'}, '*'); });
    revealActive();
  }
  function clearHighlights() {
    document.querySelectorAll('mark.search-hit').forEach(mark => {
      const parent = mark.parentNode; mark.replaceWith(document.createTextNode(mark.textContent)); parent.normalize();
    });
  }
  function highlight(node, terms) {
    if (!terms.length) return;
    const walker = document.createTreeWalker(node, NodeFilter.SHOW_TEXT);
    const texts = []; while (walker.nextNode()) if (!walker.currentNode.parentElement.closest('script, style, button')) texts.push(walker.currentNode);
    texts.forEach(textNode => {
      const value = textNode.textContent, lower = normalized(value), ranges = [];
      for (const term of terms) {
        let position = lower.indexOf(term);
        while (position >= 0) { ranges.push([position, position + term.length]); position = lower.indexOf(term, position + term.length); }
      }
      ranges.sort((a,b)=>a[0]-b[0]);
      const merged = []; for (const range of ranges) { const previous = merged.at(-1); if (previous && range[0] <= previous[1]) previous[1] = Math.max(previous[1], range[1]); else merged.push(range); }
      if (!merged.length) return;
      const fragment = document.createDocumentFragment(); let cursor = 0;
      for (const [start,end] of merged) { fragment.append(value.slice(cursor,start)); const mark=document.createElement('mark');mark.className='search-hit';mark.textContent=value.slice(start,end);fragment.append(mark);cursor=end; }
      fragment.append(value.slice(cursor)); textNode.replaceWith(fragment);
    });
  }
  function select() {
    let fragment;
    try { fragment = decodeURIComponent(location.hash.slice(1)); } catch (_) { fragment = 'HOME'; }
    if (fragment === 'main-content') {document.getElementById('main-content').focus();return;}
    const id = fragment.split('--')[0] || 'HOME';
    const current = byId.get(id) || byId.get('HOME') || articles[0];
    articles.forEach(article => { article.hidden = article !== current; });
    links.forEach(link => {
      if (link.dataset.material === current.id) { link.setAttribute('aria-current', 'page'); const parent = link.closest('details.nav-module'); if(parent) parent.open = true; }
      else link.removeAttribute('aria-current');
    });
    document.title = current.dataset.title + ' · Курс ML и LLM';
    closeMenu();
    updateContext(current);
    const heading = fragment.includes('--') ? document.getElementById(fragment) : null;
    requestAnimationFrame(() => {
      if (heading && current.contains(heading)) {
        if (heading.tagName === 'DETAILS') heading.open = true;
        let parent = heading.parentElement;
        while (parent) { if(parent.tagName === 'DETAILS') parent.open = true; parent=parent.parentElement; }
        heading.scrollIntoView({block:'start',behavior:'instant'});
        heading.setAttribute('tabindex','-1'); heading.focus({preventScroll:true});
      }
      else { window.scrollTo({top:0, behavior:'instant'}); if(lastMaterial) {current.querySelector('h1').setAttribute('tabindex','-1');current.querySelector('h1').focus({preventScroll:true});} }
      lastMaterial = current.id;
      revealActive();
    });
  }
  function filter() {
    const terms = normalized(search.value.trim()).split(/\s+/).filter(Boolean);
    results.replaceChildren();
    document.getElementById('book-navigation').hidden = terms.length > 0;
    document.getElementById('search-clear').hidden = !terms.length;
    clearHighlights();
    if (!terms.length) { status.textContent = 'Поиск по тексту · Ctrl K'; document.getElementById('empty-search').hidden = true; revealActive(); return; }
    const matched = entries.filter(entry => terms.every(term => entry.normalized.includes(term)));
    status.textContent = matched.length ? `Фрагментов: ${matched.length}${matched.length > 60 ? ' · показаны первые 60' : ''}` : 'Совпадений нет';
    document.getElementById('empty-search').hidden = matched.length !== 0;
    matched.slice(0,60).forEach(entry => {
      const li=document.createElement('li'), a=document.createElement('a'), title=document.createElement('strong'), meta=document.createElement('span'), snippet=document.createElement('span');
      a.href = '#' + entry.node.id;
      title.textContent = entry.article.dataset.shortTitle || entry.article.dataset.title;
      meta.className='result-meta'; meta.textContent = entry.article.id + (entry.answer ? ' · Решение — после своей попытки' : ' · Фрагмент урока');
      snippet.className='result-snippet';
      if (entry.answer) snippet.textContent = `Совпадение «${search.value.trim()}» внутри ответа. Открыть этот фрагмент →`;
      else {const position=Math.min(...terms.map(term=>entry.normalized.indexOf(term)));const start=Math.max(0,position-45);snippet.textContent=(start?'…':'')+entry.text.slice(start,start+160)+(entry.text.length>start+160?'…':'');highlight(snippet,terms);}
      a.append(title,meta,snippet); li.append(a); results.append(li);
      a.addEventListener('click',()=>{clearHighlights();highlight(entry.node,terms);if(location.hash === a.hash) select();});
    });
    sidebarScroll.scrollTop=0;
  }
  // file:// sandbox frames have an opaque origin; trust only the exact known WindowProxy.
  window.addEventListener('message', event => {
    if (event.data?.type !== 'course:resize' || !Number.isFinite(event.data.height)) return;
    const frame = Array.from(document.querySelectorAll('iframe.experiment')).find(frame => frame.contentWindow === event.source);
    if (!frame || event.data.height < 60 || event.data.height > 24000) return;
    frame.style.height = Math.ceil(event.data.height + 2) + 'px';
  });
  menu.addEventListener('click',()=>document.body.classList.contains('nav-open')?closeMenu(true):openMenu());
  document.getElementById('menu-close').addEventListener('click',()=>closeMenu(true));
  document.getElementById('backdrop').addEventListener('click',()=>closeMenu(true));
  document.getElementById('print-section').addEventListener('click',()=>window.print());
  // Dark by default. The <head> boot script restores a saved choice before paint; storage may be
  // unavailable on file:// or in private windows, so the switch still works without saving.
  const themeKey='ml-llm-course-theme', themeButton=document.getElementById('theme-toggle');
  function applyTheme(theme, save) {
    const light=theme==='light';
    document.documentElement.dataset.theme=light?'light':'dark';
    themeButton.querySelector('.theme-state').textContent=light?': светлая':': тёмная';
    themeButton.setAttribute('aria-label',light?'Тема: светлая. Включить тёмную':'Тема: тёмная. Включить светлую');
    if(save){try{localStorage.setItem(themeKey,light?'light':'dark');}catch(_){}}
  }
  applyTheme(document.documentElement.dataset.theme,false);themeButton.hidden=false;
  themeButton.addEventListener('click',()=>applyTheme(document.documentElement.dataset.theme==='light'?'dark':'light',true));
  document.getElementById('search-clear').addEventListener('click',()=>{search.value='';filter();search.focus();});
  search.addEventListener('input', filter);
  window.addEventListener('hashchange', select);
  outlineWide.addEventListener('change',()=>document.querySelectorAll('.lesson-outline details').forEach(details=>{details.open=outlineWide.matches;}));
  narrow.addEventListener('change',()=>closeMenu());
  // Tool-version notes stay open on wide screens and fold on phones, where theory must start in the first screen.
  const foldVersions=()=>document.querySelectorAll('.tool-versions').forEach(details=>{details.open=!narrow.matches;});
  foldVersions();narrow.addEventListener('change',foldVersions);
  let outlinePending=false;
  function markOutline(){
    outlinePending=false;
    const article=articles.find(article=>!article.hidden);if(!article)return;
    const items=Array.from(article.querySelectorAll('.lesson-outline a'));
    let active=items[0];
    for(const item of items){const target=document.getElementById(decodeURIComponent(item.hash.slice(1)));if(target?.getBoundingClientRect().top<=150)active=item;}
    items.forEach(item=>{if(item===active)item.setAttribute('aria-current','location');else item.removeAttribute('aria-current');});
  }
  window.addEventListener('scroll',()=>{if(!outlinePending){outlinePending=true;requestAnimationFrame(markOutline);}},{passive:true});
  document.addEventListener('keydown', event => {
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'k') {event.preventDefault();if(narrow.matches)openMenu(true);else search.focus();}
    if (event.key === 'Escape') closeMenu(true);
    if (event.key === 'Tab' && narrow.matches && document.body.classList.contains('nav-open')) {
      const focusable=Array.from(sidebar.querySelectorAll('a,button,input,summary')).filter(el=>el.getClientRects().length&&!el.disabled&&!el.hidden);
      const first=focusable[0], last=focusable.at(-1);
      if(event.shiftKey && document.activeElement===first){event.preventDefault();last.focus();}
      else if(!event.shiftKey && document.activeElement===last){event.preventDefault();first.focus();}
    }
  });
  document.querySelectorAll('[data-return-task],[data-return-retry]').forEach(link=>{
    const key=link.dataset.returnTask||link.dataset.returnRetry, target=byId.get(key);
    if(!target)return;
    const stage=target.querySelector(link.dataset.returnRetry?'[data-stage="retry"] h2':'[data-stage="task"] h2');
    if(stage)link.href='#'+stage.id;
  });
  // A tablet can have a narrow reading column beside the navigation. Select the
  // diagram by its actual available width, not only by the browser viewport.
  const figureObserver=new ResizeObserver(entries=>{
    for(const entry of entries){
      if(entry.contentRect.width<=0)continue;
      const source=entry.target.querySelector('source');if(!source)continue;
      const media=entry.contentRect.width<600?'(min-width: 0px)':'(max-width: 0px)';
      if(source.media!==media)source.media=media;
    }
  });
  document.querySelectorAll('.figure-expand').forEach(figure=>figureObserver.observe(figure));
  document.addEventListener('click', event => {
    const figure=event.target.closest('a[data-figure]');
    if(figure){const dialog=document.getElementById('figure-dialog');if(typeof dialog.showModal==='function'){
      event.preventDefault();const large=document.getElementById('figure-large');large.src=figure.querySelector('img').currentSrc||figure.dataset.figure;large.alt=figure.querySelector('img').alt;large.classList.remove('native');
      document.getElementById('figure-title').textContent=large.alt;document.getElementById('figure-original').href=figure.href;document.getElementById('figure-zoom').textContent='Исходный размер';dialog.showModal();}return;}
    const link=event.target.closest('a[href^="#"]');
    if(link && link.hash===location.hash && !link.closest('#search-results'))select();
    if(link?.closest('.lesson-outline') && !outlineWide.matches)link.closest('details').open=false;
  });
  document.getElementById('figure-close').addEventListener('click',()=>document.getElementById('figure-dialog').close());
  document.getElementById('figure-zoom').addEventListener('click',event=>{const native=document.getElementById('figure-large').classList.toggle('native');event.target.textContent=native?'Уместить на экране':'Исходный размер';});
  // Clipboard is optional on file://; a failed copy is reported, never silently claimed.
  document.querySelectorAll('.body-content pre').forEach(pre=>{
    const code=pre.querySelector('code'); if(!code)return;
    const button=document.createElement('button');button.className='copy-code';button.type='button';button.textContent='Копировать';button.ariaLabel='Копировать код';
    button.addEventListener('click',async()=>{try{await navigator.clipboard.writeText(code.textContent);button.textContent='Скопировано';}catch{button.textContent='Выдели код вручную';}setTimeout(()=>{button.textContent='Копировать';},2400);});pre.prepend(button);
  });
  // Mermaid is a local file loaded on the first visible diagram; hidden diagrams would get zero sizes.
  const diagrams=document.querySelectorAll('pre.mermaid');
  if(diagrams.length&&'IntersectionObserver' in window){
    let library=null;
    const load=()=>library||(library=new Promise((resolve,reject)=>{
      const script=document.createElement('script');script.src='tools/vendor/mermaid/mermaid.min.js';
      script.onload=()=>{window.mermaid.initialize({startOnLoad:false,securityLevel:'strict',theme:'neutral',fontFamily:'Segoe UI, system-ui, sans-serif'});resolve(window.mermaid);};
      script.onerror=reject;document.head.appendChild(script);}));
    const observer=new IntersectionObserver(entries=>{for(const entry of entries){
      if(!entry.isIntersecting)continue;observer.unobserve(entry.target);
      load().then(mermaid=>mermaid.run({nodes:[entry.target]})).catch(()=>entry.target.classList.add('diagram-unavailable'));
    }},{rootMargin:'200px'});
    diagrams.forEach(diagram=>observer.observe(diagram));
  }
  filter();select();markOutline();
})();
