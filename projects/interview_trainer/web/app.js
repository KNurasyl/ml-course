'use strict';
const $ = id => document.getElementById(id);
let session = null, playback = null, audioURL = null, recording = null, busy = false, speechEpoch = 0;
function status(text) { $('status').textContent = text; }
function setBusy(value) { busy = value; ['start','submit','speak','record','recover'].forEach(id => $(id).disabled = value || !!recording); }
async function api(path, value, binary=false) {
  const response = await fetch(path,{method:'POST',headers:{'Content-Type':binary?'audio/wav':'application/json'},body:binary?value:JSON.stringify(value)});
  if (!response.ok) {const result = await response.json(); throw new Error(result.error || response.status);}
  return path === '/tts' ? response.blob() : response.json();
}
function silence() { speechEpoch++; if(playback) playback.pause(); playback=null; if(audioURL) URL.revokeObjectURL(audioURL); audioURL=null; }
async function speak(text) {
  silence(); const epoch = speechEpoch;
  setBusy(true); status('Синтез речи…');
  try { const blob=await api('/tts',{text}); if(epoch!==speechEpoch)return; audioURL=URL.createObjectURL(blob); playback=new Audio(audioURL); await playback.play(); status('Можно прервать голос кнопкой.'); }
  catch(error){status(error.message);} finally{setBusy(false);}
}
function render() {
  $('interview').hidden = session.finished; $('export').hidden=false;
  $('counter').textContent=`Вопрос ${session.turn+1} / 4`;
  $('question').textContent=session.question || 'Интервью завершено';
  $('results').replaceChildren();
  for(const turn of session.history){
    const card=document.createElement('article'); card.className='result';
    for(const [tag,text,cls] of [['h3',turn.question,''],['p',turn.answer,'candidate'],['div',turn.feedback,'feedback'],['p',`Генерация и загрузка: ${turn.seconds.toFixed(2)} с. Источники поиска ниже; их наличие не гарантирует, что модель использовала их правильно.`,'muted']]){const element=document.createElement(tag);element.textContent=text;element.className=cls;card.append(element);}
    for(const source of turn.sources){const element=document.createElement(source.url?'a':'p');element.textContent=`[${source.id}] ${source.title}`;if(source.url){element.href=source.url;element.target='_blank';element.rel='noopener noreferrer';}card.append(element,document.createElement('br'));}
    if(turn.generation?.truncated){const note=document.createElement('p');note.className='muted';note.textContent='Достигнут лимит генерации: разбор может быть незавершённым. Учти это при оценке качества.';card.append(note);}
    const listen=document.createElement('button');listen.className='secondary';listen.textContent='Озвучить разбор';listen.onclick=()=>{if(!busy&&!recording)speak(turn.feedback);};card.append(listen);
    $('results').append(card);
  }
}
$('start').onclick=async()=>{silence();setBusy(true);try{session=await api('/start',{mode:$('mode').value});sessionStorage.setItem('course-interview-id',session.id);$('answer').value='';render();status('Ответь на первый вопрос.');}catch(e){status(e.message);}finally{setBusy(false);}};
$('submit').onclick=async()=>{if(!session||busy)return;silence();setBusy(true);status('Модель разбирает ответ. Первый запуск дольше последующих.');try{session=await api('/answer',{id:session.id,turn:session.turn,answer:$('answer').value});$('answer').value='';render();status(session.finished?'Готово. Проверь обратную связь по рубрике и сохрани сессию.':'Разбор готов. Перейди к следующему вопросу.');}catch(e){status(e.message);}finally{setBusy(false);}};
$('speak').onclick=()=>speak(session.question); $('silence').onclick=()=>{silence();status('Озвучивание остановлено.');};
function wav(chunks, rate) {
  const size=chunks.reduce((n,x)=>n+x.length,0), input=new Float32Array(size);let offset=0;for(const c of chunks){input.set(c,offset);offset+=c.length;}
  // Linear resampling is a transparent baseline, not a production anti-alias filter.
  const count=Math.min(120*16000,Math.floor(size*16000/rate)), buffer=new ArrayBuffer(44+count*2), view=new DataView(buffer);
  const str=(pos,text)=>[...text].forEach((c,i)=>view.setUint8(pos+i,c.charCodeAt(0)));
  str(0,'RIFF');view.setUint32(4,36+count*2,true);str(8,'WAVE');str(12,'fmt ');view.setUint32(16,16,true);view.setUint16(20,1,true);view.setUint16(22,1,true);view.setUint32(24,16000,true);view.setUint32(28,32000,true);view.setUint16(32,2,true);view.setUint16(34,16,true);str(36,'data');view.setUint32(40,count*2,true);
  for(let i=0;i<count;i++){let p=i*rate/16000,j=Math.floor(p),f=p-j;let value=(input[j]||0)*(1-f)+(input[j+1]||0)*f;value=Math.max(-1,Math.min(1,value));view.setInt16(44+i*2,value<0?value*32768:value*32767,true);}
  return new Blob([buffer],{type:'audio/wav'});
}
async function finishRecording() {
  if(!recording)return; const r=recording; recording=null; clearTimeout(r.timer); $('stop').disabled=true;r.node.disconnect();r.source.disconnect();r.stream.getTracks().forEach(t=>t.stop());await r.context.close();setBusy(true);status('Распознавание…');
  try{const result=await api('/asr',wav(r.chunks,r.rate),true);$('answer').value=result.text;status(`Распознано за ${result.seconds.toFixed(2)} с. Исправь текст перед отправкой.`);}catch(e){status(e.message);}finally{setBusy(false);}
}
$('record').onclick=async()=>{
  silence();setBusy(true);let stream=null,context=null;
  try{stream=await navigator.mediaDevices.getUserMedia({audio:{channelCount:1,echoCancellation:true,noiseSuppression:true}});context=new AudioContext();await context.audioWorklet.addModule('/pcm-worklet.js');const source=context.createMediaStreamSource(stream),node=new AudioWorkletNode(context,'pcm-recorder'),chunks=[];node.port.onmessage=e=>chunks.push(e.data);source.connect(node);node.connect(context.destination);recording={stream,context,source,node,chunks,rate:context.sampleRate,timer:setTimeout(finishRecording,120000)};setBusy(false);$('stop').disabled=false;status('Идёт запись. После ответа нажми «Закончить запись».');}
  catch(e){stream?.getTracks().forEach(t=>t.stop());if(context)await context.close();setBusy(false);status(`Микрофон: ${e.message}. Можно ввести ответ текстом.`);}
};
$('stop').onclick=finishRecording;
$('export').onclick=()=>{const url=URL.createObjectURL(new Blob([JSON.stringify(session,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download=`interview-${session.id}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
window.addEventListener('beforeunload',()=>{recording?.stream.getTracks().forEach(t=>t.stop());silence();});
async function recover() {
  const id=session?.id || sessionStorage.getItem('course-interview-id');
  if(!id){status('Сохранённой сессии в этой вкладке нет. Начни новое интервью.');return;}
  silence();setBusy(true);const previousTurn=session?.turn;
  try{const response=await fetch('/session/'+encodeURIComponent(id));const state=await response.json();if(!response.ok){if(response.status===404)sessionStorage.removeItem('course-interview-id');throw new Error(state.error);}session=state;if(session.turn!==previousTurn)$('answer').value='';render();status('Состояние восстановлено. Уже принятый ответ повторно не отправляй.');}
  catch(e){status(e.message);}finally{setBusy(false);}
}
$('recover').onclick=recover;
if(sessionStorage.getItem('course-interview-id')) recover();
