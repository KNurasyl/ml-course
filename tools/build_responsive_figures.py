"""Responsive technical diagrams for the offline textbook.

All authored SVGs use one visual vocabulary: blue for data, teal for an
operation/result, amber for a constraint or rejected path. A label always
carries the meaning as well as colour. The mobile drawings reflow the content;
they are not thumbnails of the desktop drawing.
"""
from pathlib import Path
from html import escape
import json, math, re, textwrap
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'course/assets/responsive';OUT.mkdir(parents=True,exist_ok=True)
NS={'s':'http://www.w3.org/2000/svg'}
# Colours are theme tokens (6.0, D3); old literals -> tokens: ink 182b3a, muted 526474, blue 245bba -> accent,
# teal 087e82 -> act, amber a65c12 -> hint, line ccd6df, soft eef4fc -> accent-soft, op edf8f7 -> act-soft,
# warn fff3e5 -> hint-soft, white text on blue -> bg, empty cell f6f7f9 -> surface; the white backing rect is gone.
# The reader inlines these SVGs so they take the page theme. A standalone file falls back to the light values,
# read from the light token block of tools/reader.css.
def light_tokens():
 css=(ROOT/'tools/reader.css').read_text(encoding='utf-8')
 block=re.search(r':root\[data-theme="light"\]\{color-scheme:light;(.*?)\}',css,re.DOTALL).group(1)
 return dict(re.findall(r'--([\w-]+):(#[0-9a-fA-F]{6})',block))
LIGHT=light_tokens()
INK='ink';MUTED='muted';BLUE='accent';TEAL='act';AMBER='hint'
LINE='line';SOFT='accent-soft';OP='act-soft';WARN='hint-soft';ON_BLUE='bg';EMPTY='surface'
def paint(prop,token):return f'{prop}:var(--{token},{LIGHT[token]})'
MAP={};OWN=set()

class SVG:
 def __init__(self,w,h,title):
  self.w,self.h=w,h
  self.s=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" aria-labelledby="title desc"><title id="title">{escape(title)}</title><desc id="desc">{escape(title)}. Подписи и числа поясняют значения цвета.</desc><defs><marker id="arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7" style="{paint("fill",TEAL)}"/></marker><marker id="blue" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7" style="{paint("fill",BLUE)}"/></marker></defs>']
 def text(self,x,y,t,size=17,color=INK,weight=400,anchor='start'):
  self.s.append(f'<text x="{x}" y="{y}" style="{paint("fill",color)}" font-family="Segoe UI,Arial,sans-serif" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}">{escape(str(t))}</text>')
 def lines(self,x,y,t,n=32,size=17,color=INK,weight=400,line=24):
  strings=[]
  for para in str(t).split('\n'):strings.extend(textwrap.wrap(para,width=n,break_long_words=False,break_on_hyphens=False) or [''])
  for k,t in enumerate(strings):self.text(x,y+k*line,t,size,color,weight)
  return y+len(strings)*line
 def rect(self,x,y,w,h,fill=SOFT,stroke='none',radius=6):
  style=paint('fill',fill)+('' if stroke=='none' else ';'+paint('stroke',stroke))
  self.s.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" style="{style}"/>')
 def path(self,d,color=TEAL,width=2,arrow=False,dash=False):
  self.s.append(f'<path d="{d}" fill="none" style="{paint("stroke",color)}" stroke-width="{width}"'+(' stroke-dasharray="6 5"' if dash else '')+(f' marker-end="url(#{"blue" if color==BLUE else "arrow"})"' if arrow else '')+'/>')
 def dot(self,x,y,r=4,color=AMBER):self.s.append(f'<circle cx="{x}" cy="{y}" r="{r}" style="{paint("fill",color)}"/>')
 def save(self,name,layout):
  # Ids carry the file name: the reader inlines many SVGs into one document.
  doc,prefix='\n'.join(self.s+['</svg>']),f'{name}-{layout}'
  doc=doc.replace('aria-labelledby="title desc"',f'aria-labelledby="{prefix}-title {prefix}-desc"')
  for key in ('title','desc','arrow','blue'):doc=doc.replace(f'id="{key}"',f'id="{prefix}-{key}"').replace(f'url(#{key})',f'url(#{prefix}-{key})')
  target=OUT/f'{name}-{layout}.svg';target.write_text(doc+'\n',encoding='utf-8',newline='\n')
  relative=target.relative_to(ROOT).as_posix();OWN.add(relative)
  return relative

def bind(name,ext,desktop,mobile,caption=''):
 MAP[f'course/assets/{name}.{ext}']={'desktop':desktop,'mobile':mobile,'caption':caption}

def pairs(name,title,rows,note='',ext='svg'):
 paths={}
 for mob in [False,True]:
  w=360 if mob else 760;size=18 if mob else 18
  titlelines=textwrap.wrap(title,27 if mob else 55,break_long_words=False)
  y=28+len(titlelines)*27
  heights=[]
  for a,b in rows:
   if mob:heights.append(18+24*len(textwrap.wrap(a,30,break_long_words=False))+24*len(textwrap.wrap(b,29,break_long_words=False))+20)
   else:heights.append(max(24*len(textwrap.wrap(a,23,break_long_words=False)),24*len(textwrap.wrap(b,48,break_long_words=False)))+30)
  footerlines=textwrap.wrap(note,31 if mob else 77,break_long_words=False)
  s=SVG(w,y+sum(heights)+28+len(footerlines)*22,title)
  for i,t in enumerate(titlelines):s.text(16,28+i*27,t,21 if mob else 24,weight=650)
  for i,((a,b),h) in enumerate(zip(rows,heights)):
   s.path(f'M16,{y+h-8} H{w-16}',LINE,1)
   if mob:
    s.rect(16,y+8,3,h-24,BLUE,radius=0)
    yy=s.lines(30,y+24,a,30,size,BLUE,600)
    s.lines(30,yy+4,b,29,size,INK)
   else:
    s.lines(18,y+24,a,23,size,BLUE,600)
    s.lines(280,y+24,b,48,size,INK)
   y+=h
  s.lines(16,y+20,note,31 if mob else 77,15,MUTED,line=22)
  paths['mobile' if mob else 'desktop']=s.save(name,'mobile' if mob else 'desktop')
 bind(name,ext,paths['desktop'],paths['mobile'],note)

def attention():
 q=[1,0,-1];v=[10,20,30]
 weights=[]
 for qi in q:
  e=[math.exp(qi*k) for k in q];weights.append([x/sum(e) for x in e])
 for mob in [False,True]:
  s=SVG(360 if mob else 760,695 if mob else 350,'Attention: каждая строка Q даёт свой выход O')
  s.text(16,30,'Attention: от весов к выходу',21 if mob else 25,weight=650)
  s.text(16,60,'Q = K = [1, 0, −1]; dₖ = 1',17,MUTED)
  s.text(16,86,'V = [10, 20, 30]',17,BLUE)
  if not mob:
   s.text(20,125,'Запрос',16,MUTED)
   for j in range(3):s.text(160+j*92,125,f'K{j+1}',16,MUTED,anchor='middle')
   s.text(445,125,'O = Σ A · V',17,TEAL,weight=600)
  for i,row in enumerate(weights):
   y=125+i*180 if mob else 145+i*57
   value=sum(a*b for a,b in zip(row,v))
   if mob:
    s.text(16,y,f'Q{i+1} → строка весов A{i+1}',18,BLUE,600)
    for j,a in enumerate(row):
     x=16+j*111;s.rect(x,y+34,104,46,BLUE if a>.5 else SOFT)
     s.text(x+52,y+25,f'K{j+1}',15,MUTED,anchor='middle')
     s.text(x+52,y+64,f'{a:.3f}',18,ON_BLUE if a>.5 else INK,600,'middle')
    s.text(16,y+111,f'O{i+1} = {value:.3f}',21,TEAL,650)
    s.text(16,y+139,'Сумма весов = 1 (до округления)',16,MUTED)
   else:
    s.text(20,y+27,f'Q{i+1}',19,BLUE,600)
    for j,a in enumerate(row):
     x=118+j*92;s.rect(x,y,84,42,BLUE if a>.5 else SOFT)
     s.text(x+42,y+27,f'{a:.3f}',18,ON_BLUE if a>.5 else INK,600,'middle')
    s.path(f'M400,{y+21} H430',arrow=True)
    s.rect(445,y,260*value/30,42,OP)
    s.text(456,y+27,f'O{i+1} = {value:.3f}',19,TEAL,600)
  s.text(16,677 if mob else 335,'Softmax нормирует ключи, не запросы.',16,MUTED)
  layout='mobile' if mob else 'desktop'
  MAP.setdefault('course/assets/M08-L03.svg',{})[layout]=s.save('M08-L03',layout)
 MAP['course/assets/M08-L03.svg']['caption']='Строки Q1, Q2, Q3 и соответствующие выходы O1, O2, O3 идут в одном порядке. Числа округлены только для подписи.'

def transformer():
 for mob in [False,True]:
  w=360 if mob else 760
  s=SVG(w,1120 if mob else 750,'Pre-norm Transformer: вход, операции, residual и выход')
  s.text(16,30,'Pre-norm Transformer',23 if mob else 26,weight=650)
  s.text(16,58,'Одна учебная attention-head',17,MUTED)
  if mob:
   def block(y,label,sub='',kind='op'):
    s.rect(42,y,230,64,SOFT if kind=='data' else OP,BLUE if kind=='data' else TEAL)
    s.text(57,y+27,label,19,BLUE if kind=='data' else TEAL,600)
    if sub:s.text(57,y+50,sub,16,INK)
   block(84,'X','[B, T, D]','data')
   block(185,'LayerNorm₁','форма сохраняется')
   block(286,'Causal attention','[B,T,T] @ [B,T,D]')
   for a,b in [(148,185),(249,286),(350,390),(430,468),(532,570),(634,672),(736,774),(814,852)]:s.path(f'M157,{a} V{b-5}',arrow=True)
   s.path('M272,116 H320 V410 H181',BLUE,2,True,True)
   s.text(290,275,'X',18,BLUE,600)
   s.dot(157,410,21,OP);s.text(157,417,'+',27,TEAL,600,'middle')
   block(468,'U = X + Attention(…)','[B, T, D]','data')
   block(570,'LayerNorm₂','форма сохраняется')
   block(672,'MLP: Linear → GELU','D → H → D')
   s.path('M272,500 H320 V794 H181',BLUE,2,True,True)
   s.text(290,660,'U',18,BLUE,600)
   s.dot(157,794,21,OP);s.text(157,801,'+',27,TEAL,600,'middle')
   block(852,'Y = U + MLP(…)','[B, T, D]','data')
   s.lines(16,957,'Attention смешивает позиции. MLP меняет признаки каждой позиции.',31,17)
   s.path('M16,1030 H54',BLUE,2,False,True);s.text(65,1036,'Residual: исходные данные',16,BLUE)
   s.lines(16,1074,'Y поступает в следующий блок.',31,17,TEAL)
  else:
   def block(x,y,label,sub='',kind='op',ww=192):
    s.rect(x,y,ww,75,SOFT if kind=='data' else OP,BLUE if kind=='data' else TEAL)
    s.text(x+14,y+31,label,20,BLUE if kind=='data' else TEAL,600)
    if sub:s.text(x+14,y+57,sub,16,INK)
   block(16,102,'X','[B, T, D]','data',130)
   block(196,102,'LayerNorm₁','форма сохраняется',ww=185)
   block(430,102,'Causal attention','смешивает позиции',ww=205)
   for a,b in [(146,196),(381,430),(635,677)]:s.path(f'M{a},139 H{b-7}',arrow=True)
   s.dot(700,139,22,OP);s.text(700,148,'+',29,TEAL,600,'middle')
   s.path('M80,102 V79 H700 V112',BLUE,2,True,True)
   s.text(475,72,'Residual X',16,BLUE)
   s.path('M700,162 V238 H80 V278',arrow=True)
   s.text(200,208,'Q/K/V: [B,T,D] → scores: [B,T,T] → AV: [B,T,D]',17,MUTED)
   block(16,285,'U','[B, T, D]','data',130)
   block(196,285,'LayerNorm₂','форма сохраняется',ww=185)
   block(430,285,'MLP','Linear → GELU → Linear',ww=205)
   for a,b in [(146,196),(381,430),(635,677)]:s.path(f'M{a},322 H{b-7}',arrow=True)
   s.dot(700,322,22,OP);s.text(700,331,'+',29,TEAL,600,'middle')
   s.path('M80,360 V404 H700 V350',BLUE,2,True,True)
   s.text(320,398,'Residual U',16,BLUE)
   s.path('M722,322 H742 V455 H400 V486',arrow=True)
   block(280,494,'Y = U + MLP(…)','[B, T, D]','data',250)
   s.text(16,607,'MLP: [B,T,D] → [B,T,H] → [B,T,D]. Позиции не смешиваются.',19)
   s.path('M16,644 H62',BLUE,2,False,True);s.text(75,650,'Пунктир: прямой путь данных к сложению.',18,BLUE)
   s.text(16,700,'Y поступает в следующий блок; форма входа и выхода совпадает.',19,TEAL)
  layout='mobile' if mob else 'desktop';MAP.setdefault('course/assets/M08-L07.svg',{})[layout]=s.save('M08-L07',layout)
 MAP['course/assets/M08-L07.svg']['caption']='Синий обозначает данные и residual; бирюзовый — преобразование и сложение. Учебная схема одной головы, pre-norm.'

def streaming():
 for mob in [False,True]:
  s=SVG(360 if mob else 760,1110 if mob else 650,'Два времени: аудиосегмент и очередь отменяемых фрагментов')
  s.lines(16,30,'Аудио и очередь: два разных времени',29 if mob else 60,22,weight=650,line=28)
  if mob:
   s.text(16,108,'1. Время в аудиофайле',19,BLUE,650)
   events=[('0 мс','Речь начинается','partial может изменяться'),('400 мс','Пауза 350 мс','меньше порога 500 мс'),('750 мс','Речь продолжается','та же реплика'),('1350 мс','Начинается тишина','ожидаем 500 мс'),('1850 мс','endpoint_final','сегмент закрыт')]
   s.path('M26,140 V562',BLUE,2)
   for i,(at,a,b) in enumerate(events):
    y=155+i*88;s.dot(26,y-5,5,BLUE);s.text(44,y,at,16,BLUE,600);s.text(44,y+25,a,18,INK,600);s.text(44,y+48,b,16,MUTED)
   s.text(16,630,'2. Очередь: активен ход 5',19,TEAL,650)
   ev=[('receive(5, A)','A в очереди',TEAL),('cancel()','A удалён',AMBER),('receive(5, B)','B отклонён как поздний',AMBER),('begin(7) → receive(7, C)','C принят',TEAL),('pop()','Только C разрешён к выдаче',TEAL)]
   for i,(a,b,c) in enumerate(ev):
    y=670+i*67;s.path(f'M16,{y+47} H344',LINE,1);s.text(16,y,a,17,c,600);s.text(16,y+25,b,17)
   s.lines(16,1030,'Метки аудио не измеряют задержку ASR, LLM или TTS.',31,17,MUTED)
  else:
   s.text(16,83,'1. Аудиовремя, мс · порог паузы 500 мс',20,BLUE,650)
   x0=70;scale=.31
   for a,b,label,fill in [(0,400,'Речь',SOFT),(400,750,'Пауза',WARN),(750,1350,'Речь',SOFT),(1350,1850,'Тишина 500 мс',OP)]:
    x=x0+a*scale;s.rect(x,124,(b-a)*scale-2,49,fill);s.text(x+8,154,label,16)
   for t in [0,400,750,1350,1850]:
    x=x0+t*scale;s.path(f'M{x},178 V185',MUTED,1);s.text(x,209,t,15,MUTED,anchor='middle')
   s.path(f'M{x0+1850*scale},112 V238',TEAL,2,False,True)
   s.text(420,264,'endpoint_final: один сегмент',18,TEAL,600)
   s.text(70,240,'partial → обновляемая гипотеза',17,BLUE)
   s.text(16,327,'2. Очередь · активен ход 5; события без масштаба времени',19,TEAL,650)
   labels=[('receive(5,A)','A в очереди',TEAL),('cancel()','A удалён',AMBER),('late(5,B)','B отклонён',AMBER),('receive(7,C)','C принят',TEAL),('pop()','выдаёт C',TEAL)]
   for i,(a,b,c) in enumerate(labels):
    x=16+i*148;s.rect(x,365,138,100,WARN if c==AMBER else OP);s.text(x+8,400,a,16,c,600);s.text(x+8,434,b,16)
    if i<4:s.path(f'M{x+138},479 H{x+145}',TEAL,2,True)
   s.text(16,521,'Перед receive(7,C) начинается ход 7. Отмена очищает очередь.',18)
   s.text(16,585,'Время аудио и длительность вычисления записываются отдельно.',18,MUTED)
  layout='mobile' if mob else 'desktop';MAP.setdefault('course/assets/M15-L06.svg',{})[layout]=s.save('M15-L06',layout)
 MAP['course/assets/M15-L06.svg']['caption']='Верхняя шкала — миллисекунды аудио. Нижняя — порядок событий очереди; расстояния между событиями не означают задержку.'

def logarithms():
 for mob in [False,True]:
  w=360 if mob else 960;h=1005 if mob else 370;s=SVG(w,h,'exp, ln и потеря −ln(p): три связанные функции')
  panels=[('1. exp(x) = eˣ',(-2,2),(0,8),lambda x:math.exp(x),[(0,1)],'x — любой; exp(x) > 0'),('2. ln(x)',(.04,4),(-3.3,1.5),lambda x:math.log(x),[(1,0)],'Область: x > 0. ln(1) = 0'),('3. Потеря L = −ln(p)',(.01,1),(0,5),lambda x:-math.log(x),[(.01,-math.log(.01)),(.5,-math.log(.5)),(.9,-math.log(.9))],'0 < p ≤ 1. Чем ниже p, тем выше L')]
  for i,(title,xlim,ylim,fun,points,foot) in enumerate(panels):
   ox=0 if mob else i*320;oy=i*335 if mob else 0;pw=360 if mob else 320
   s.text(ox+16,oy+29,title,21,weight=650)
   left=ox+48;right=ox+pw-22;top=oy+60;bottom=oy+245
   xx=lambda x:left+(x-xlim[0])/(xlim[1]-xlim[0])*(right-left)
   yy=lambda y:bottom-(y-ylim[0])/(ylim[1]-ylim[0])*(bottom-top)
   for y in ([0,2,4,6,8] if i==0 else [-3,-2,-1,0,1] if i==1 else [0,1,2,3,4,5]):
    s.path(f'M{left},{yy(y)} H{right}',LINE,1);s.text(left-8,yy(y)+5,str(y),14,MUTED,anchor='end')
   ticks=[-2,-1,0,1,2] if i==0 else [1,2,3,4] if i==1 else [0,.5,1]
   for x in ticks:s.text(xx(x),bottom+23,str(x).replace('0.5','0,5'),14,MUTED,anchor='middle')
   s.path(f'M{left},{top} V{bottom} H{right}',MUTED,1)
   vals=[xlim[0]+k*(xlim[1]-xlim[0])/150 for k in range(151)]
   s.path(' '.join(('M' if k==0 else 'L')+f'{xx(x):.2f},{yy(fun(x)):.2f}' for k,x in enumerate(vals)),BLUE,2.5)
   for x,y in points:s.dot(xx(x),yy(y),4,AMBER)
   if i==0:s.text(xx(0)+10,yy(1)-14,'(0; 1)',16,AMBER,600)
   if i==1:s.text(xx(1)+10,yy(0)-14,'(1; 0)',16,AMBER,600)
   if i==2:
    s.text(left+12,top+20,'p=0,01 → L≈4,605',16,AMBER,600)
    s.text(left+70,top+72,'p=0,5 → L≈0,693',15,AMBER)
   s.text(right,oy+286,'p' if i==2 else 'x',16,MUTED,anchor='end')
   s.lines(ox+16,oy+308,foot,33 if mob else 30,16,MUTED,line=22)
  layout='mobile' if mob else 'desktop';MAP.setdefault('course/assets/M04-L06.png',{})[layout]=s.save('M04-L06',layout)
 MAP['course/assets/M04-L06.png']['caption']='На телефоне графики идут последовательно: экспонента, обратная функция ln и потеря вероятностного ответа. Точки сохраняют исходные значения.'

def generic_pairs():
 for path in sorted((ROOT/'course/assets').glob('*.svg')):
  root=ET.parse(path).getroot();vb=root.attrib.get('viewBox','')
  texts=root.findall('.//s:text',NS)
  title=root.find('s:title',NS)
  if vb.startswith('0 0 1040 ') and not root.findall('.//s:g[@id="figure_1"]',NS):
   rows=[]
   for t in texts:
    if t.get('x')=='44':
     other=next((x for x in texts if x.get('x')=='385' and x.get('y')==t.get('y')),None)
     if other is not None:rows.append((''.join(t.itertext()),''.join(other.itertext())))
   if rows:pairs(path.stem,''.join(title.itertext()) if title is not None else path.stem,rows,'Учебные числа и условия — в пояснении урока.')
  elif vb=='0 0 960 460':
   rows=[]
   for t in texts:
    if t.get('x')=='46':
     other=next((x for x in texts if x.get('x')=='440' and x.get('y')==t.get('y')),None)
     if other is not None:rows.append((''.join(t.itertext()),''.join(other.itertext())))
   if rows:pairs(path.stem,''.join(title.itertext()) if title is not None else path.stem,rows,'Сопоставление понятий; строки не означают поток выполнения.')

def special_pairs():
 pairs('M00-L04','Обучение и контекст меняют разные вещи',[('Обучение','Данные → оптимизация → новые параметры θ'),('Контекст запроса','История + найденные тексты → вход модели'),('Инференс','Вход + фиксированные параметры θ → ответ')],'Запись диалога сама по себе не выполняет шаг обучения.','png')
 pairs('M00-L05','Обучение передаёт сервису артефакт',[('Подготовка','Данные с метками → выбор параметров → артефакт с версией'),('Корректный запрос','x → проверка контракта → модель → ответ + версия'),('Неверный запрос','Проверка контракта → явная ошибка; модель не вызывается')],'Артефакт связывает обучение и обслуживание запроса.','png')
 pairs('M03-L02','Цепное правило: идём от потери назад',[('Прямой расчёт','w=1 → ŷ=wx+b=2 → e=ŷ−y=−3 → L=e²=9'),('Локальные производные','dL/de=2e=−6; de/dŷ=1; dŷ/dw=x=2'),('Обратный расчёт','dL/dw = (−6) × 1 × 2 = −12')],'Задано x=2, y=5, b=0.','png')
 pairs('M01-L01','Среднее по axis=0 сохраняет признаки',[('Объект 1','Признак 1: 1; признак 2: 10'),('Объект 2','Признак 1: 3; признак 2: 30'),('Объект 3','Признак 1: 5; признак 2: 50'),('Среднее по трём строкам','Признак 1: 3; признак 2: 30')],'Исходная форма [3,2] → результат [2].','png')
 pairs('M01-L02','Три проблемы данных — три проверки',[('a; seconds=10','Исходная запись'),('a; seconds=10','Точная копия исходной записи'),('b; seconds=NULL','Неизвестное значение'),('c; seconds=−3','Нарушение допустимого диапазона')],'NULL, дубликат и неверное значение не взаимозаменяемы.','png')
 pairs('M04-L01','Условие определяет знаменатель',[('Короткие: 100 ответов','80 корректных, 20 некорректных'),('Длинные: 50 ответов','30 корректных, 20 некорректных'),('Всего: 150 ответов','110 корректных, 40 некорректных')],'Например: P(корректно | длинный) = 30 / 50.','png')
 pairs('M05-L04','Каждый fold заново учит подготовку',[('Fold 1','Validation: часть 1. Train: части 2 и 3.'),('Fold 2','Validation: часть 2. Train: части 1 и 3.'),('Fold 3','Validation: часть 3. Train: части 1 и 2.'),('Test хранится отдельно','Не используется ни для fit, ни для выбора настроек.')],'fit(preprocessing + model) выполняется только на train данного fold.','png')
 pairs('M05-L06','Обучение, выбор и оценка разделены',[('Train: только fit','Медианы → масштаб → категории → веса. Один Pipeline для каждого C.'),('Validation: выбор','Стоимость (2FN + FP) / N. Фиксируем C и порог до test.'),('Test: только predict','Ошибки и ограничения выбранного решения. Обратной стрелки к выбору нет.')],'Порог и модель зафиксированы перед независимой оценкой.')
 pairs('M05-L07','Группа и время ограничивают split',[('Пользователь a: дни 1, 2','Train (day < 5)'),('Пользователь b: дни 3, 6','Пересекает train и validation → обе строки excluded'),('Пользователь c: дни 8, 9','Test (day ≥ 8)')],'Validation: дни 5…7. Политика исключения фиксируется до обучения.')
 pairs('M08-L06','LayerNorm: нормируем признаки строки',[('x = [1, 3, 5, 7]','μ=4; variance=5 → z≈[−1.342, −0.447, 0.447, 1.342]'),('x = [10, 10, 10, 10]','variance=0; ε>0 → z=[0, 0, 0, 0]'),('Обучаемое преобразование','y = γ · z + β по каждой координате'),('Residual — другая операция','[2, −1, 3] + [0.5, 1, −2] = [2.5, 0, 1]')],'Нормируются признаки D; позиции T не смешиваются.')
 pairs('M09-L06','Память: три разных измерения',[('Артефакт на диске','Размер сериализованных весов: float32 / packed int8'),('Работающий процесс','RSS после подготовки: модель + библиотеки'),('Наблюдаемый пик','Всё дерево процесса; шаг измерения 30 мс'),('Скорость — отдельно','40 токенов / 10 секунд = 4 token/s = 0.25 s/token')],'Не складывай и не подменяй эти измерения.')
 pairs('M10-L06','Найденный документ ≠ полный контекст',[('Исходное утверждение','Индекс ускоряет подходящие запросы, но не гарантирует ускорения выборки большей части таблицы.'),('Окно отрезало условие','«Индекс ускоряет подходящие запросы». Document Recall = 1, но оговорка потеряна.'),('Абзац сохранил условие','«… но не гарантирует …». Теперь проверяем утверждение ответа.')],'Трасса: версия → chunk → кандидаты → контекст → claim.')
 pairs('M12-L01','Вмешательство выбираем по причине ошибки',[('Нарушен формат','Инструкция + схема → проверить fallback'),('Нет факта в источнике','Обновить документы → проверить поиск'),('Нужен новый навык','Примеры + SFT → проверить регрессии')],'После изменения — тот же независимый набор и анализ новых ошибок.')
 pairs('M12-L06','Assistant-only loss: контекст виден, loss избирателен',[('input_ids','[11, 12, 13, 21, 22, 2]'),('labels','[−100, −100, −100, 21, 22, 2]'),('attention mask','[1, 1, 1, 1, 1, 1]'),('Роли токенов','11–13: system/user. 21,22,2: assistant + EOS.'),('Внутренний сдвиг loss','Logits позиции 2 (индекс с нуля) предсказывают token 21.')],'Prefix доступен attention; −100 исключает его из objective.')
 pairs('M12-L07','LoRA: сумма двух путей',[('Замороженный путь','X [N,in] → W [out,in] → XWᵀ + b'),('Обучаемый путь','X → A [rank,in] → B [out,rank] → XAᵀBᵀ · α/rank'),('Сложение','Y = XWᵀ + b + XAᵀBᵀ · α/rank')],'Base hash неизменен; ответы могут измениться и ухудшиться.')
 pairs('M16-L01','Две рубрики проверяют разные свойства',[('Содержание','HR: конкретное личное действие. Техническое: корректный факт.'),('Обоснование','HR: почему выбрано действие. Техническое: компромиссы и ограничения.'),('Проверка','HR: подтверждённый результат. Техническое: план измерения / тест.'),('Критическая ошибка','HR: приписанный опыт. Техническое: ложный факт или вывод.')],'Оценка относится к ответу по явной рубрике, не к личности.')

def remaining_pairs():
 pairs('M11-L06','Один случай, две версии, общая рубрика',[('Общий вход: case_id=p','Один вопрос и один ответ подаются версиям A и B.'),('Два сохранённых результата','Отдельно для A и B: raw + display + hash.'),('Общая разметка','Для каждого результата: балл + основание.'),('JOIN по case_id','Парная разность B − A. Отсутствующая оценка не равна 0.')],'Новая критическая ошибка блокирует выпуск независимо от среднего.')
 pairs('M12-L02','Разбиение целыми сессиями',[('Сессия A: A1, A2','Обе строки в train.'),('Сессия B: B1, B2','Обе строки в train.'),('Сессия C: C1, C2','Обе строки в check.'),('Группы не пересекаются','Train: {A,B}. Check: {C}. Пересечение ∅.')],'Строки одной сессии нельзя распределять по разным частям.')
 pairs('M12-L03','Биграмма видит предыдущий символ',[('Цепочка символов','^ → a → b → a → b → $'),('Пара a → b','Обучается вероятность b по строке W[a].'),('Алфавит: ^ a b c $','W имеет форму [5,5] = 25 весов.'),('Предел контекста','aba и aaaa оканчиваются на a: используется одна строка W[a].')],'Transformer получает доступный контекст через attention; биграмма его не видит.')
 pairs('M15-L04','Поздний ответ не оживляет отменённый ход',[('t0: начат ход 11','Вычисление долгое, результат ещё не готов.'),('t1: отменён ход 11','Старые данные уже не вправе звучать.'),('t2: начат ход 12','Вычисление 11 может продолжаться параллельно.'),('t3: результат 12 готов','Проверка идентификатора принимает его.'),('t4: результат 11 готов','Проверка идентификатора отклоняет его.')],'t0…t4 — порядок событий, не равные интервалы времени.')
 pairs('M15-L05','Удаление одного слова меняет смысл',[('Слово 1','Эталон «я» → гипотеза «я»'),('Слово 2: удалено отрицание','Эталон «не» → гипотеза ∅'),('Слово 3','Эталон «использовал» → гипотеза «использовал»'),('Слово 4','Эталон «docker» → гипотеза «docker»')],'WER = 1/4 = 0.25. Пропуск отрицания — отдельная критическая ошибка.')

def lora_matrix():
 for mob in [False,True]:
  s=SVG(360 if mob else 760,900 if mob else 480,'Низкий ранг: семь обучаемых чисел, двенадцать элементов')
  s.lines(16,30,'7 обучаемых чисел → 12 элементов',28 if mob else 60,22,weight=650,line=28)
  arrays=[[[1],[0],[-1],[2]],[[0,1,2]],[[0,1,2],[0,0,0],[0,-1,-2],[0,2,4]]]
  titles=['A: 4 × 1 · 4 параметра','B: 1 × 3 · 3 параметра','ΔW = A @ B: 4 × 3']
  positions=[(142,130),(78,465),(78,605)] if mob else [(35,140),(208,220),(490,140)]
  for a,title,(x,y) in zip(arrays,titles,positions):
   s.text(16 if mob else x-5,y-22,title,18,BLUE,600)
   for i,row in enumerate(a):
    for j,v in enumerate(row):
     fill=SOFT if v>0 else WARN if v<0 else EMPTY
     s.rect(x+j*66,y+i*56,62,52,fill,LINE,2);s.text(x+j*66+31,y+i*56+34,str(v),21,INK,500,'middle')
  if mob:
   s.lines(16,385,'Каждая строка ΔW кратна B. Ранг ΔW ≤ 1.',31,17,TEAL)
   s.lines(16,864,'Ранг не равен числу выученных фактов.',33,16,MUTED,line=22)
  else:
   s.text(154,260,'@',28,TEAL,600);s.text(425,260,'=',28,TEAL,600)
   s.text(16,415,'Каждая строка ΔW кратна B. Ранг ΔW ≤ 1.',19,TEAL)
   s.text(16,453,'Ранг не равен числу выученных фактов.',18,MUTED)
  layout='mobile' if mob else 'desktop';MAP.setdefault('course/assets/M12-L04.svg',{})[layout]=s.save('M12-L04',layout)
 MAP['course/assets/M12-L04.svg']['caption']='Одинаковый размер ячейки сохраняет смысл формы матриц. Содержимое ΔW вычислено из показанных A и B.'

def build():
 foundation=ROOT/'metadata/foundation-figures.json'
 if foundation.exists():MAP.update(json.loads(foundation.read_text(encoding='utf-8')))
 quantitative=ROOT/'metadata/quantitative-figures.json'
 if quantitative.exists():MAP.update(json.loads(quantitative.read_text(encoding='utf-8')))
 generic_pairs();special_pairs();remaining_pairs();attention();transformer();streaming();logarithms();lora_matrix()
 # Figures drawn here take theme tokens and are inlined by the reader; matplotlib figures stay images (D4).
 own=[key for key,value in MAP.items() if value.get('desktop') in OWN and value.get('mobile') in OWN]
 if {value for key in own for value in (MAP[key]['desktop'],MAP[key]['mobile'])}!=OWN:raise SystemExit('Every drawn SVG must belong to one figure')
 for key in own:MAP[key]['inline']=True
 (ROOT/'metadata/responsive-figures.json').write_text(json.dumps(MAP,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
 for path in OUT.glob('*.svg'):ET.parse(path)
 print(f'{len(MAP)} figures mapped, {len(own)} inline with theme tokens; responsive SVGs parsed successfully')

if __name__=='__main__':build()
