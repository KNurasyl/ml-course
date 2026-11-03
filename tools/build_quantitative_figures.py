"""Render the course's numerical diagrams with desktop and mobile layouts.
Fixed teaching examples and stored measurements are used; this does not retrain
models or replace historical measurement evidence.
"""
from pathlib import Path
import json,sys,textwrap,math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import binom
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'course/assets/responsive';OUT.mkdir(parents=True,exist_ok=True)
BLUE='#245bba';TEAL='#087e82';AMBER='#a65c12';GRAY='#526474'
plt.rcParams.update({'font.family':'Segoe UI','font.size':13,'svg.fonttype':'none','svg.hashsalt':'ml-llm-textbook' ,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','axes.facecolor':'white','text.color':'#182b3a','axes.labelcolor':'#182b3a','xtick.color':GRAY,'ytick.color':GRAY,'axes.prop_cycle':matplotlib.cycler(color=[BLUE,TEAL,AMBER])})
MAP={};MOBILE=False

def wrap(t,n=30):return '\n'.join(textwrap.fill(x,n,break_long_words=False,break_on_hyphens=False) for x in t.split('\n'))
def canvas(title,cols=1):
 fig,axes=plt.subplots(cols if MOBILE else 1,1 if MOBILE else cols,figsize=(3.8,4.6*cols+.8) if MOBILE else (10,4.8),layout='constrained')
 fig.suptitle(wrap(title,32 if MOBILE else 75),fontsize=15 if MOBILE else 18,fontweight='bold')
 return fig,axes

def save(fig,key,ext='svg'):
 for text in fig.findobj(matplotlib.text.Text):text.set_linespacing(1.4)
 for ax in fig.axes:
  ax.title.set_text(wrap(ax.title.get_text(),29 if MOBILE else 60));ax.title.set_fontsize(14)
  ax.xaxis.label.set_text(wrap(ax.xaxis.label.get_text(),33 if MOBILE else 80))
  ax.yaxis.label.set_text(wrap(ax.yaxis.label.get_text(),30 if MOBILE else 55))
  if MOBILE:
   for t in ax.texts:t.set_text(wrap(t.get_text(),28))
   legend=ax.get_legend()
   if legend:
    for t in legend.get_texts():t.set_text(wrap(t.get_text(),30));t.set_fontsize(12)
  ax.grid(axis='y',alpha=.13)
 layout='mobile' if MOBILE else 'desktop';path=OUT/f'{key}-{layout}.svg'
 fig.savefig(path,bbox_inches='tight',pad_inches=.16,metadata={'Date':None,'Title':key,'Description':'Responsive layout; source numbers unchanged.'});plt.close(fig)
 path.write_bytes(path.read_bytes().replace(b'\r\n', b'\n'))
 MAP.setdefault(f'course/assets/{key}.{ext}',{})[layout]=path.relative_to(ROOT).as_posix()

def build():
 global MOBILE
 for MOBILE in [False,True]:
  fig,axes=canvas('Четыре испытания: P(X≥3) = 5/16',2)
  seq={k:[] for k in range(5)}
  for n in range(16):
   binary=f'{n:04b}';seq[binary.count('1')].append(binary)
  axes[0].axis('off')
  for k in range(5):
   yy=.94-k*.19
   axes[0].text(.01,yy,f'X={k}',color=AMBER if k>=3 else BLUE,fontweight='bold',va='top')
   axes[0].text(.25,yy,' '.join(seq[k][:3])+('\n'+' '.join(seq[k][3:]) if len(seq[k])>3 else ''),fontfamily='Consolas',fontsize=13,va='top')
  axes[0].set_title('Все 16 равновероятных исходов')
  ax=axes[1]
  p=np.array([math.comb(4,k)/16 for k in range(5)])
  bars=ax.bar(range(5),p,color=[BLUE]*3+[AMBER]*2)
  ax.bar_label(bars,labels=[f'{math.comb(4,k)}/16' for k in range(5)],padding=5)
  ax.set(xlabel='Число успехов X',ylabel='Вероятность',xticks=range(5),ylim=(0,.46))
  save(fig,'M04-L07','png')

  fig,ax=canvas('Мощность зависит от эффекта и размера выборки')
  for p1,label in [(.7,'Эффект +0.10'),(.65,'Эффект +0.05')]:
   sizes=[50,100,200,400,800];power=[]
   for n in sizes:
    k=next(k for k in range(n+2) if binom.sf(k-1,n,.6)<=.05)
    power.append(float(binom.sf(k-1,n,p1)))
   ax.plot(sizes,power,marker='o',label=label,lw=2.5)
  ax.axhline(.8,color=AMBER,ls='--',label='Требование 0.8')
  ax.set(xlabel='Независимые наблюдения n',ylabel='Мощность',ylim=(0,1.04),title='p₀=0.6; exact test; α=0.05');ax.legend(loc='lower right')
  save(fig,'M04-L08','png')

  fig,ax=canvas('Очередь увеличивает задержку запроса')
  for i,a in enumerate([0,1,2,3]):
   start=2*i;ax.barh(i,start-a,left=a,color='#dce7f7',label='Ожидание' if i==0 else None)
   ax.barh(i,2,left=start,color=BLUE,label='Модель: 2 с' if i==0 else None)
   ax.text(start+1,i,f'{2+i} с',ha='center',va='center',color='white',fontsize=13)
  ax.set(yticks=range(4),yticklabels=['Запрос 1','Запрос 2','Запрос 3','Запрос 4'],xlabel='Секунды от начала');ax.invert_yaxis();ax.legend(loc='upper center',bbox_to_anchor=(.5,-.18),fontsize=12)
  save(fig,'M06-L03')

  fig,ax=canvas('Checkpoint выбирается по validation')
  ax.plot([1,2,3,4],[.7,.4,.2,.1],'o-',label='train',lw=2.5);ax.plot([1,2,3,4],[.72,.45,.42,.55],'o-',label='validation',lw=2.5)
  ax.axvline(3,color=AMBER,ls='--',label='Выбрана эпоха 3')
  ax.set(xlabel='Эпоха',ylabel='Loss',xticks=[1,2,3,4],ylim=(0,.82));ax.legend()
  save(fig,'M07-L04')

  fig,ax=canvas('Causal mask: будущие ключи запрещены')
  a=np.tril(np.ones((4,4)));a/=a.sum(1,keepdims=True)
  ax.imshow(a,cmap='Blues',vmin=0,vmax=1)
  for i in range(4):
   for j in range(4):ax.text(j,i,f'{a[i,j]:.2f}' if j<=i else '×',ha='center',va='center',fontsize=15,color='white' if a[i,j]>.6 else '#182b3a')
  ax.set(xticks=range(4),yticks=range(4),xticklabels=['K1','K2','K3','K4'],yticklabels=['Q1','Q2','Q3','Q4'],xlabel='Ключи · × означает вес 0',ylabel='Запросы')
  save(fig,'M08-L04')

  fig,ax=canvas('Контекст включает резерв ответа')
  parts=[60,180,200,40,20,64];names=['Правила','История','Источники','Вопрос','Маркеры','Ответ'];colors=[BLUE,TEAL,'#51799f','#a65c12','#667085','#7545a0']
  for i,vals in enumerate([parts,[60,100,200,40,20,64]]):
   left=0
   for value,name,color in zip(vals,names,colors):
    ax.barh(i,value,left=left,color=color,label=name if i==0 else None)
    left+=value
   ax.text(left+7,i,str(left),va='center',fontsize=13)
  ax.axvline(512,color=AMBER,ls='--');ax.set(xlim=(0,640),yticks=[0,1],yticklabels=['До','После'],xlabel='Токены · лимит 512');ax.invert_yaxis()
  ax.legend(loc='upper center',bbox_to_anchor=(.5,-.2),ncol=2 if MOBILE else 3,fontsize=11)
  save(fig,'M09-L02')

  fig,ax=canvas('200 млн параметров: только веса')
  bars=ax.bar(['float32','float16','int8'],[800,400,200],color=[BLUE,TEAL,GRAY]);ax.bar_label(bars,labels=['800 MB','400 MB','200 MB'],padding=6)
  ax.set(ylim=(0,980),ylabel='Миллионы байт',title='Без кэша, активаций и runtime')
  save(fig,'M09-L04')

  fig,ax=canvas('RRF: B получает два высоких ранга')
  vals=[1/12+1/11,1/11+1/13,1/12,1/13];bars=ax.barh(['B','A','D','C'],vals,color=[BLUE,BLUE,GRAY,GRAY]);ax.invert_yaxis()
  ax.bar_label(bars,labels=[f'{x:.5f}' for x in vals],padding=5);ax.set(xlim=(0,.24),xlabel='RRF score · c=10',title='Слияние [A,B,C] и [B,D,A]')
  save(fig,'M10-L03')

  fig,ax=canvas('Judge: 2 совпадения из 4')
  ax.imshow(np.ones((2,2)),cmap='Blues',vmin=0,vmax=2)
  for i,row in enumerate([['TN=1','FP=1'],['FN=1','TP=1']]):
   for j,t in enumerate(row):ax.text(j,i,t,ha='center',va='center',fontsize=20)
  ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['неверно','верно'],yticklabels=['неверно','верно'],xlabel='Оценка judge',ylabel='Ручная оценка');save(fig,'M11-L04')

  fig,axes=canvas('Обучаем 7 чисел, получаем 12 элементов',3)
  A=np.array([[1],[0],[-1],[2]]);B=np.array([[0,1,2]])
  for ax,a,title in zip(axes,[A,B,A@B],['A: 4×1; 4 параметра','B: 1×3; 3 параметра','ΔW=A@B: 4×3; ранг ≤1']):
   ax.imshow(a,cmap='RdBu',vmin=-4,vmax=4,aspect='auto')
   for (i,j),v in np.ndenumerate(a):ax.text(j,i,str(v),ha='center',va='center',fontsize=19,color='white' if abs(v)>2 else '#182b3a')
   ax.set(xticks=[],yticks=[],title=title)
  save(fig,'M12-L04')

  fig,ax=canvas('Адаптер улучшил новое и ухудшил старое')
  x=np.arange(2);w=.32
  for offset,vals,label,color in [(-w/2,[.166378,3.731763],'База',BLUE),(w/2,[11.989497,.566252],'Адаптер rank 1',AMBER)]:
   ax.bar(x+offset,vals,w,color=color,label=label)
   for xx,v in zip(x+offset,vals):ax.text(xx,v*1.16,f'{v:.3f}',ha='center',fontsize=12)
  ax.set_yscale('log');ax.set(ylim=(.1,25),xticks=x,xticklabels=['Старая\nab','Новая\nac'],ylabel='Cross entropy · log scale',title='CPU: seed 17 · PyTorch 2.15');ax.legend(loc='upper center',bbox_to_anchor=(.5,-.2))
  save(fig,'M12-L05')

  fig,ax=canvas('Очередь объясняет хвост задержки')
  x=np.arange(1,21);queue=np.array([0]*16+[900]*4)
  ax.bar(x,np.full(20,500),label='ASR + генерация: 500 мс',color=BLUE)
  ax.bar(x,queue,bottom=500,label='Очередь: 900 мс',color=AMBER)
  ax.axhline(680,color=TEAL,ls='--',label='Среднее 680 мс');ax.axhline(1400,color='#7045a0',ls=':',label='p95 = 1400 мс')
  ax.set(xlabel='Номер запроса',ylabel='Задержка, мс',ylim=(0,2350),xticks=[1,5,10,15,20]);ax.legend(fontsize=11)
  save(fig,'M13-L04')

  fig,axes=canvas('Дискретизация: амплитуда во времени',2)
  rate=16000;t=np.arange(320)/rate;y=.7*np.sin(2*np.pi*220*t)
  axes[0].plot(t*1000,y,color=BLUE);axes[0].set(title='Тон 220 Гц · первые 20 мс',xlabel='Время, мс',ylabel='Амплитуда',ylim=(-1,1))
  axes[1].stem(t[:20]*1000,y[:20],linefmt=AMBER,markerfmt='o',basefmt='k-');axes[1].set(title='Шаг: 1/16000 секунды',xlabel='Время, мс',ylabel='Амплитуда',ylim=(-1,1))
  save(fig,'M15-L01')

  data=json.loads((ROOT/'course/assets/data/M07-HW01-measurements.json').read_text(encoding='utf-8'))
  fig,axes=canvas('Checkpoint и ошибки: фактический запуск M07',2)
  for field,label,color in [('train_loss','train',BLUE),('val_loss','validation',TEAL)]:
   axes[0].plot([r['epoch'] for r in data['history']],[r[field] for r in data['history']],label=label,color=color,lw=2)
  axes[0].axvline(data['selected_epoch'],ls='--',color=AMBER,label='Выбрана эпоха 115')
  axes[0].set(xlabel='Эпоха',ylabel='Cross entropy',title='Выбор без доступа к test');axes[0].legend(fontsize=11)
  for cls,color in [(0,BLUE),(1,TEAL)]:
   rows=[r for r in data['test'] if r['target']==cls]
   axes[1].scatter([r['x'][0] for r in rows],[r['x'][1] for r in rows],s=18,c=color,alpha=.65,label=f'Класс {cls}')
  rows=[r for r in data['test'] if r['target']!=r['predicted']]
  axes[1].scatter([r['x'][0] for r in rows],[r['x'][1] for r in rows],s=70,facecolors='none',edgecolors=AMBER,label='Ошибка')
  for r in rows:
   if r['id'] in [101,141]:axes[1].annotate(f"id={r['id']}",r['x'],xytext=(8,18),textcoords='offset points',arrowprops={'arrowstyle':'->'},fontsize=12)
  axes[1].set(xlabel='Исходный признак x1',ylabel='Исходный признак x2',title='Test: реальные ошибки');axes[1].legend(fontsize=11)
  save(fig,'M07-HW01','png')

 (ROOT/'metadata/quantitative-figures.json').write_text(json.dumps(MAP,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
 print(f'{len(MAP)} quantitative figure pairs rendered')
 return MAP

if __name__=='__main__':build()
