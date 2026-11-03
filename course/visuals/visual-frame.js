/* Offline iframe sizing. Parent must validate event.source against its frame. */
(() => {
  'use strict';
  if (window.parent === window) return;
  let scheduled = false, last = 0;
  function measure() {
    scheduled = false;
    // body has natural height, unlike documentElement.scrollHeight which includes
    // the old iframe viewport and would prevent shrinking after a step change.
    const body = document.body;
    if (!body) return;
    const style = getComputedStyle(body);
    const height = Math.ceil(body.getBoundingClientRect().height +
      parseFloat(style.marginTop || 0) + parseFloat(style.marginBottom || 0));
    if (height > 0 && height !== last) {
      last = height;
      parent.postMessage({type: 'course:resize', height}, '*');
    }
  }
  function schedule() { if (!scheduled) { scheduled = true; requestAnimationFrame(measure); } }
  function start() {
    new ResizeObserver(schedule).observe(document.body);
    new MutationObserver(schedule).observe(document.body, {childList:true,subtree:true,characterData:true});
    schedule();
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start, {once:true}); else start();
  addEventListener('load', schedule); addEventListener('resize', schedule);
  document.addEventListener('input', schedule); document.addEventListener('change', schedule);
  document.addEventListener('toggle', schedule, true);
  addEventListener('message', event => {
    if (event.source === parent && event.data?.type === 'course:measure') {
      last = 0; schedule();
    }
  });
})();
