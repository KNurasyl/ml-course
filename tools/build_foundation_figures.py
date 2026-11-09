"""Rebuild responsive foundation plots from the original course's fixed synthetic data.

This renderer contains its own numerical examples; it never imports or executes
archived course generators. Desktop and mobile layouts use identical numbers.
Run with the course virtual environment. The original PNGs remain for provenance.
"""
from pathlib import Path
from math import comb,sqrt
import contextlib,json,re,sys,textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle,FancyArrowPatch
from matplotlib.text import Text

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'course/assets/responsive'
OUT.mkdir(parents=True,exist_ok=True)
BLUE='#245bba';GREEN='#087e82';RED='#a65c12';PURPLE='#7045a0';GRAY='#526474'
# D4: every figure also gets a dark copy, <name>-dark.svg, drawn with the 6.0 dark tokens. The light pass keeps
# these literals, so the existing light files stay byte-identical. Dark: blue -> accent, green -> act, red -> hint,
# purple -> ink (no purple in 6.0, author's decision), gray -> muted, white backdrop -> paper, white text on a
# strong fill -> bg, dark text on a weak fill -> ink, Blues -> a paper..accent ramp.
PAPER='white';ON_STRONG='white';ON_WEAK='#102a43';CMAP='Blues';THEME='light'
# build_responsive_figures.py redraws these with theme tokens and the reader inlines them (D3): no dark copy.
DRAWN_INLINE={'M00-L04','M00-L05','M01-L01','M01-L02','M03-L02','M04-L01','M05-L04'}
MOBILE=False
MAPPING={}
plt.rcParams.update({'font.family':'Segoe UI','font.size':13,'axes.titlesize':14,'axes.labelsize':13,'figure.facecolor':'#ffffff','axes.facecolor':'#ffffff','axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.15,'savefig.facecolor':'#ffffff','svg.fonttype':'none','svg.hashsalt':'ml-llm-textbook','text.color':'#182b3a','axes.labelcolor':'#182b3a','xtick.color':'#526474','ytick.color':'#526474'})

def tokens(theme):
    """6.0 token values for one theme, read from the reader's stylesheet."""
    css=(ROOT/'tools/reader.css').read_text(encoding='utf-8')
    pattern=r':root\{color-scheme:dark;(.*?)\}' if theme=='dark' else r':root\[data-theme="light"\]\{color-scheme:light;(.*?)\}'
    return dict(re.findall(r'--([\w-]+):(#[0-9a-fA-F]{6})',re.search(pattern,css,re.DOTALL).group(1)))

def dark_rc(t):
    return {'figure.facecolor':t['paper'],'axes.facecolor':t['paper'],'savefig.facecolor':t['paper'],'savefig.edgecolor':t['paper'],
            'text.color':t['ink'],'axes.labelcolor':t['ink'],'axes.titlecolor':t['ink'],'xtick.color':t['muted'],'ytick.color':t['muted'],
            'axes.edgecolor':t['muted'],'grid.color':t['muted'],'patch.edgecolor':t['muted'],'legend.facecolor':t['paper'],
            'legend.edgecolor':t['line-strong'],'axes.prop_cycle':matplotlib.cycler(color=[t['accent'],t['act'],t['hint']])}

def wrap(value,n=30):
    return '\n'.join(textwrap.fill(line,n,break_long_words=False,break_on_hyphens=False) for line in value.split('\n'))

def canvas(title,cols=1):
    if MOBILE:
        fig,axes=plt.subplots(cols,1,figsize=(3.8,4.5*cols+.65),layout='constrained')
    else:
        fig,axes=plt.subplots(1,cols,figsize=(9,5),layout='constrained')
    fig.suptitle(wrap(title.split(' · ',1)[-1],29 if MOBILE else 64),fontweight='bold',fontsize=15 if MOBILE else 17)
    return fig,axes

def save(fig,key):
    for text in fig.findobj(matplotlib.text.Text):text.set_linespacing(1.4)
    if MOBILE:
        for ax in fig.axes:
            ax.title.set_text(wrap(ax.title.get_text(),30))
            ax.xaxis.label.set_text(wrap(ax.xaxis.label.get_text(),30))
            ax.yaxis.label.set_text(wrap(ax.yaxis.label.get_text(),28))
            for t in ax.texts:
                if t.get_text():
                    t.set_text(wrap(t.get_text(),29))
            leg=ax.get_legend()
            if leg:
                for t in leg.get_texts():
                    t.set_text(wrap(t.get_text(),26))
                    t.set_fontsize(12)
        suffix='mobile'
    else:suffix='desktop'
    if THEME=='dark' and key in DRAWN_INLINE:
        plt.close(fig);return
    path=OUT/(f'{key}-{suffix}-dark.svg' if THEME=='dark' else f'{key}-{suffix}.svg')
    fig.savefig(path,bbox_inches='tight',pad_inches=.16,metadata={'Date':None,'Title':key,'Description':'Identical fixed teaching data; responsive layout.'})
    path.write_bytes(path.read_bytes().replace(b'\r\n', b'\n'))
    plt.close(fig)
    if THEME=='dark':return
    entry=MAPPING.setdefault(f'course/assets/{key}.png',{})
    entry[suffix]=path.relative_to(ROOT).as_posix()

def bars(key,title,labels,values,ylabel,colors=None):
    fig,ax=canvas(title)
    bars=ax.bar(labels,values,color=colors or BLUE,width=.6)
    ax.bar_label(bars,labels=[f'{v:g}' for v in values],padding=7)
    ax.set_ylabel(ylabel);ax.set_ylim(min(0,min(values)*1.2),max(values)*1.25 or 1)
    save(fig,key)

def table(key,title,rows,cols,values,colors=None):
    fig,ax=canvas(title);ax.axis('off')
    # Mobile table is drawn as readable text pairs by the SVG layout renderer.
    t=ax.table(cellText=values,rowLabels=rows,colLabels=cols,cellLoc='center',loc='center',cellColours=colors)
    t.auto_set_font_size(False);t.set_fontsize(11 if MOBILE else 14);t.scale(.85 if MOBILE else 1,2.8)
    save(fig,key)

def arrow(ax,start,end,color=None,text=None,offset=(0,0)):
    color=color or BLUE
    ax.annotate('',xy=end,xytext=start,arrowprops={'arrowstyle':'->','lw':2.6,'color':color})
    if text:ax.text(end[0]+offset[0],end[1]+offset[1],text,color=color,ha='left',va='bottom',fontweight='bold')

def box(ax,x,y,label,color=None,w=2.3,h=.8):
    color=color or BLUE
    ax.add_patch(Rectangle((x-w/2,y-h/2),w,h,facecolor=color+'16',edgecolor=color,lw=1.8))
    ax.text(x,y,label,ha='center',va='center',fontsize=11)

def build(themes=('light','dark')):
    global THEME,BLUE,GREEN,RED,PURPLE,GRAY,PAPER,ON_STRONG,ON_WEAK,CMAP
    light=(BLUE,GREEN,RED,PURPLE,GRAY,PAPER,ON_STRONG,ON_WEAK,CMAP)
    for THEME in themes:
        if THEME=='dark':
            t=tokens('dark');context=plt.rc_context(dark_rc(t))
            BLUE,GREEN,RED,PURPLE,GRAY,PAPER,ON_STRONG,ON_WEAK=t['accent'],t['act'],t['hint'],t['ink'],t['muted'],t['paper'],t['bg'],t['ink']
            CMAP=LinearSegmentedColormap.from_list('paper-accent',[t['paper'],t['accent']])
        else:
            context=contextlib.nullcontext();BLUE,GREEN,RED,PURPLE,GRAY,PAPER,ON_STRONG,ON_WEAK,CMAP=light
        with context:draw()
    BLUE,GREEN,RED,PURPLE,GRAY,PAPER,ON_STRONG,ON_WEAK,CMAP=light
    return MAPPING

def draw():
    global MOBILE
    for MOBILE in (False,True):
        # M00: model, parameters, loss, training/context distinction, service boundary.
        fig,ax=canvas('M00-L01 · Предсказание — применение фиксированного правила')
        x=np.linspace(-1,6,100);ax.plot(x,2*x+1,color=BLUE,label='ŷ = 2x + 1')
        ax.scatter([0,2,5],[1,5,11],color=RED,zorder=3,s=60)
        for a,b in [(0,1),(2,5),(5,11)]:ax.annotate(f'({a}, {b})',(a,b),xytext=(8,6),textcoords='offset points')
        ax.set(xlabel='Вход x',ylabel='Предсказание ŷ');ax.legend();save(fig,'M00-L01')

        fig,ax=canvas('M00-L02 · Смещение двигает линию, вес меняет наклон')
        x=np.linspace(-1,4,100)
        for w,b,c,label in [(2,1,BLUE,'A: w=2, b=1'),(2,3,GREEN,'B: w=2, b=3'),(3,1,PURPLE,'C: w=3, b=1')]:ax.plot(x,w*x+b,color=c,label=label,lw=2.5)
        ax.scatter([2],[7],color=RED,zorder=4);ax.annotate('B и C совпадают только здесь',(2,7),xytext=(-.8,11),arrowprops={'arrowstyle':'->'})
        ax.set(xlabel='x',ylabel='ŷ');ax.legend(loc='lower right');save(fig,'M00-L02')
        bars('M00-L03','M00-L03 · Средний квадрат ошибки на одних данных',['A: 2x+1','B: x+1','C: 2x'],[0,14/3,1],'MSE',[GREEN,RED,BLUE])

        fig,ax=canvas('M00-L04 · Обучение меняет параметры; контекст меняет вход')
        ax.set(xlim=(0,12),ylim=(0,6));ax.axis('off')
        for a in [(1.7,4.8,'Данные\nобучения',GREEN),(5.7,4.8,'Оптимизация',GREEN),(10,4.8,'Параметры θ',GREEN),(1.7,1.2,'История +\nнайденные тексты',BLUE),(5.7,1.2,'Контекст\nтекущего запроса',BLUE),(10,1.2,'Инференс\nс фиксированными θ',BLUE)]:box(ax,*a)
        for y in [4.8,1.2]:
            arrow(ax,(2.9,y),(4.4,y),GREEN if y>3 else BLUE);arrow(ax,(6.9,y),(8.7,y),GREEN if y>3 else BLUE)
        arrow(ax,(10,4.3),(10,1.7),GRAY)
        ax.text(6,3,'Запись диалога сама по себе не выполняет шаг обучения',ha='center',fontsize=12)
        save(fig,'M00-L04')

        fig,ax=canvas('M00-L05 · Артефакт связывает обучение и путь запроса')
        ax.set(xlim=(0,13),ylim=(0,7));ax.axis('off')
        for a in [(1.6,5.7,'Данные\nс метками',GREEN),(5.7,5.7,'Выбор\nпараметров',GREEN),(10.8,5.7,'Артефакт\nс версией',GREEN),(1.6,2.5,'Запрос x',BLUE),(5.7,2.5,'Проверка\nконтракта',BLUE),(10.8,2.5,'Модель → ответ\n+ версия',BLUE),(5.7,.6,'Явная ошибка\nневерного входа',RED)]:box(ax,*a)
        for y in [5.7,2.5]:arrow(ax,(2.9,y),(4.4,y));arrow(ax,(7,y),(9.5,y))
        arrow(ax,(10.8,5.2),(10.8,3.0),GREEN);arrow(ax,(5.7,2),(5.7,1.1),RED)
        save(fig,'M00-L05')

        # M01: shapes, data quality, groups, cardinality, train-only transform.
        table('M01-L01','M01-L01 · axis=0 убирает строки, сохраняя два признака',['Объект 1','Объект 2','Объект 3','Среднее'],['Признак 1','Признак 2'],[['1','10'],['3','30'],['5','50'],['3','30']],[[BLUE+'18',GREEN+'18']]*3+[[BLUE+'50',GREEN+'50']])
        table('M01-L02','M01-L02 · Три разных проблемы требуют разных решений',['Строка 1','Строка 2','Строка 3','Строка 4'],['session','seconds','Проверка'],[['a','10','Исходная запись'],['a','10','Точная копия'],['b','NULL','Неизвестное значение'],['c','−3','Нарушение диапазона']],[[GREEN+'18']*3,[PURPLE+'28']*3,[BLUE+'28']*3,[RED+'28']*3])
        fig,ax=canvas('M01-L03 · Группа целиком находится в одной части')
        colors=[BLUE,BLUE,GREEN,GREEN,PURPLE,PURPLE]
        ax.barh(np.arange(6),np.ones(6),left=[0,0,0,0,1.3,1.3],color=colors,height=.65)
        ax.set_yticks(range(6),['A: сессия 1','A: сессия 2','B: сессия 1','B: сессия 2','C: сессия 1','C: сессия 2'])
        ax.set_xticks([.5,1.8],['TRAIN\n4 строки\n2 человека','TEST\n2 строки\n1 человек']);ax.set_xlim(0,2.4);ax.invert_yaxis();ax.grid(False)
        save(fig,'M01-L03')
        fig,axes=canvas('M01-L04 · JOIN меняет вес сессии при вычислении AVG',2)
        for ax,labels,values,colors,title,avg in [(axes[0],['a','b'],[10,30],[BLUE,GREEN],'По сессиям: n=2',20),(axes[1],['a₁','a₂','a₃','b'],[10,10,10,30],[BLUE]*3+[GREEN],'После JOIN: n=4',15)]:
            ax.bar(labels,values,color=colors);ax.axhline(avg,color=RED,ls='--',label=f'Среднее = {avg}');ax.set_title(title);ax.set_ylim(0,36);ax.legend();ax.set_ylabel('Секунды')
        save(fig,'M01-L04')
        fig,axes=canvas('M01-L05 · Один параметр μ_train=4 для всех новых объектов',2)
        for ax,vals,title in [(axes[0],[2,4,6],'TRAIN'),(axes[1],[10,14],'TEST')]:
            ax.scatter(vals,np.ones(len(vals)),color=BLUE,s=90,label='Исходные')
            ax.scatter(np.array(vals)-4,np.zeros(len(vals)),color=GREEN,s=90,label='После x−4')
            for v in vals:arrow(ax,(v,.9),(v-4,.1),GRAY)
            ax.axvline(0,color=GRAY,ls=':');ax.set_yticks([0,1],['Преобразованные','Исходные']);ax.set_title(title);ax.set_xlim(-4,16);ax.set_ylim(-.3,1.5);ax.set_xlabel('Значение признака')
        save(fig,'M01-L05')

        # M02: actual vector geometry rather than box chains.
        fig,ax=canvas('M02-L01 · Средний вектор лежит посередине между объектами')
        for v,c,t in [((2,1),BLUE,'u=(2,1)'),((4,3),GREEN,'v=(4,3)'),((3,2),PURPLE,'(u+v)/2=(3,2)')]:arrow(ax,(0,0),v,c,t,(.05,.05))
        ax.plot([2,4],[1,3],color=GRAY,ls='--');ax.set(xlim=(-.3,6),ylim=(-.3,4),xlabel='Координата 1',ylabel='Координата 2');ax.set_aspect('equal');save(fig,'M02-L01')
        fig,ax=canvas('M02-L02 · Dot учитывает длины и угол')
        arrow(ax,(0,0),(3,4),BLUE,'u=(3,4), ||u||=5');arrow(ax,(0,0),(4,0),GREEN,'v=(4,0), ||v||=4',(.1,.1))
        ax.plot([3,3],[0,4],color=GRAY,ls='--');ax.text(.3,3.4,'u·v = 12\ncos θ = 0.6\n||u−v|| = √17',bbox={'facecolor':PAPER,'alpha':.9,'edgecolor':'none'})
        ax.set(xlim=(-.3,6.5),ylim=(-.5,5),xlabel='Координата 1',ylabel='Координата 2');ax.set_aspect('equal');save(fig,'M02-L02')
        fig,ax=canvas('M02-L03 · Порядок преобразований меняет результат')
        x=np.array([1,2]);ax.scatter(*x,color=GRAY,s=90);ax.annotate('x=(1,2)',(1,2),xytext=(0,23),textcoords='offset points',ha='center')
        ax.plot([1,2,-2],[2,2,2],color=BLUE,marker='o',lw=2,label='A затем B: BAx=(−2,2)')
        ax.plot([1,-2,-4],[2,1,1],color=RED,marker='o',lw=2,label='B затем A: ABx=(−4,1)')
        ax.text(-3.8,3.4,'A: растяжение оси x в 2 раза\nB: поворот на 90° против часовой стрелки')
        ax.set(xlim=(-5,3),ylim=(-.5,4.5),xlabel='Координата 1',ylabel='Координата 2');ax.legend(loc='lower left');save(fig,'M02-L03')
        fig,axes=canvas('M02-L04 · Масштабирование меняет ближайшего соседа',2)
        for ax,points,title in [(axes[0],[(10,0),(0,2)],'Исходные: B ближе'),(axes[1],[(.1,0),(0,2)],'Первый признак /100: A ближе')]:
            ax.scatter([0],[0],color=GRAY,s=90,marker='x',label='Запрос')
            for (a,b),c,t in zip(points,[BLUE,GREEN],['A','B']):
                ax.scatter(a,b,color=c,s=90);ax.plot([0,a],[0,b],color=c,lw=2);ax.annotate(t,(a,b),xytext=(5,7),textcoords='offset points')
            ax.set_title(title);ax.set(xlim=(-.5,11 if ax==axes[0] else 2.5),ylim=(-.3,2.8),xlabel='Признак 1',ylabel='Признак 2')
        save(fig,'M02-L04')
        bars('M02-L05','M02-L05 · Точный поиск: оценки и первые два документа',['SQL · top1','API · top2','UI'],[1,.8,0],'Cosine сходство',[GREEN,BLUE,GRAY])

        # M03: derivatives, graph, aggregation, optimization, regularization.
        fig,ax=canvas('M03-L01 · Высота функции и наклон — разные числа')
        x=np.linspace(-1,5,200);ax.plot(x,x*x,color=BLUE,label='f(w)=w²');ax.plot(x,9+6*(x-3),color=RED,ls='--',label='Касательная: наклон 6')
        ax.scatter([3],[9],color=PURPLE,s=80,zorder=4);ax.annotate('f(3)=9',(3,9),xytext=(3.3,5),arrowprops={'arrowstyle':'->'})
        ax.set(xlabel='w',ylabel='f(w)',ylim=(-3,26));ax.legend();save(fig,'M03-L01')
        fig,ax=canvas('M03-L02 · Цепное правило: локальные чувствительности перемножаются')
        ax.set(xlim=(0,13),ylim=(0,5));ax.axis('off')
        for x,t in [(1.6,'w=1'),(5.5,'ŷ=wx+b=2'),(9,'e=ŷ−y=−3'),(11.7,'L=e²=9')]:box(ax,x,3,t,BLUE,w=2.2 if x!=11.7 else 2)
        for start,end,label in [((2.8,3),(4.3,3),'×x = 2'),((6.7,3),(7.8,3),'×1'),((10.2,3),(10.6,3),'×2e = −6')]:
            arrow(ax,start,end,BLUE);ax.text((start[0]+end[0])/2,2.1,label,ha='center',color=RED)
        ax.text(6.5,.8,'dL/dw = (−6) × 1 × 2 = −12      ·      x=2, y=5, b=0',ha='center',fontsize=14)
        save(fig,'M03-L02')
        fig,ax=canvas('M03-L03 · Градиент набора — среднее вкладов объектов')
        vals=[-4,-16,-10];b=ax.bar(['Объект 1\nx=1','Объект 2\nx=2','Среднее\ndw'],vals,color=[BLUE,BLUE,GREEN]);ax.bar_label(b,padding=5)
        ax.axhline(0,color=GRAY);ax.set_ylim(-20,3);ax.set_ylabel('Вклад 2·e·x');ax.text(.3,1,'x=[1,2], y=[2,4], w=b=0');save(fig,'M03-L03')
        fig,ax=canvas('M03-L04 · Шаг определяет сходимость даже для одного веса')
        for eta,c in [(.1,BLUE),(.75,GREEN),(1.1,RED)]:
            w=0.;values=[]
            for _ in range(13):values.append((w-3)**2);w-=eta*2*(w-3)
            ax.semilogy(range(13),values,marker='o',color=c,label=f'η={eta}')
        ax.set(xlabel='Шаг',ylabel='L=(w−3)² · логарифмическая шкала');ax.legend();save(fig,'M03-L04')
        fig,ax=canvas('M03-L05 · Регуляризация меняет оптимизируемую цель')
        w=np.linspace(-1,5,200);ax.plot(w,(w-4)**2,color=BLUE,label='Ошибка (w−4)²');ax.plot(w,w*w,color=RED,label='Штраф w²');ax.plot(w,(w-4)**2+w*w,color=GREEN,lw=3,label='Сумма J')
        ax.scatter([2],[8],color=PURPLE,s=90);ax.annotate('Минимум J: w=2, J=8',(2,8),xytext=(-.5,15),arrowprops={'arrowstyle':'->'})
        ax.set(xlabel='w',ylabel='Значение',ylim=(0,32));ax.legend(loc='upper center');save(fig,'M03-L05')

        # M04: counts/uncertainty/statistical procedure.
        table('M04-L01','M04-L01 · Условие определяет знаменатель',['Короткие','Длинные','Всего'],['Корректно','Некорректно','Всего'],[['80','20','100'],['30','20','50'],['110','40','150']],[[BLUE+'18']*3,[GREEN+'28']*3,[GRAY+'18']*3])
        fig,ax=canvas('M04-L02 · Разброс наблюдений не равен ошибке среднего')
        for i,v in enumerate([1,2,3,4]):ax.plot([v,2.5],[i,i],color=BLUE,lw=3);ax.scatter(v,i,color=BLUE,s=90)
        ax.axvline(2.5,color=RED,ls='--',label='Среднее 2.5');ax.set_yticks(range(4),['x₁=1','x₂=2','x₃=3','x₄=4']);ax.set(xlim=(.5,5),xlabel='Значение',ylim=(-.6,4.5));ax.text(.8,4,'s=1.291 · SE=0.6455 (при независимости)');ax.legend(loc='lower right');save(fig,'M04-L02')
        def wilson(k,n):
            z=1.959963984540054;p=k/n;den=1+z*z/n;c=(p+z*z/(2*n))/den;h=z*sqrt(p*(1-p)/n+z*z/(4*n*n))/den
            return c-h,c+h
        fig,ax=canvas('M04-L03 · 90% на десяти и тысяче наблюдений')
        for y,k,n,c in [(1,9,10,BLUE),(0,900,1000,GREEN)]:
            lo,hi=wilson(k,n);ax.plot([lo,hi],[y,y],color=c,lw=5);ax.scatter(k/n,y,color=c,s=100);ax.text(lo,y+.16,f'[{lo:.3f}; {hi:.3f}]',color=c)
        ax.set_yticks([0,1],['900 / 1000','9 / 10']);ax.set(xlim=(.5,1),ylim=(-.5,1.6),xlabel='Доля и 95%-интервал Уилсона');ax.axvline(.9,color=GRAY,ls=':');save(fig,'M04-L03')
        fig,ax=canvas('M04-L04 · Bootstrap пересэмплирует пять исходных пар')
        d=np.array([1.,0,0,0,1]);rng=np.random.default_rng(42);means=d[rng.integers(0,5,size=(20000,5))].mean(1)
        ax.hist(means,bins=np.arange(-.1,1.11,.2),color=BLUE,edgecolor=PAPER,rwidth=.9)
        for q in [0,.8]:ax.axvline(q,color=RED,ls='--',label='Квантили 2.5% и 97.5%' if q==0 else None)
        ax.axvline(.4,color=GREEN,lw=2,label='Исходное среднее 0.4');ax.set(xlabel='Средняя парная разность',ylabel='Число bootstrap-повторов');ax.legend();save(fig,'M04-L04')
        fig,ax=canvas('M04-L05 · Два хвоста при девяти победах из десяти')
        k=np.arange(11);p=np.array([comb(10,int(j))/2**10 for j in k]);cs=[RED if j in [0,1,9,10] else BLUE for j in k]
        ax.bar(k,p,color=cs);ax.set_xticks(k);ax.set(xlabel='Число побед B при H₀: p=0.5',ylabel='Вероятность');ax.text(.2,.235,'Сумма выделенных столбцов:\n22/1024 ≈ 0.02148',bbox={'facecolor':PAPER,'edgecolor':'none'});save(fig,'M04-L05')

        # M05: metrics, probabilities, tree function, CV, error slices.
        fig,ax=canvas('M05-L01 · Какие ошибки скрывает общая accuracy?')
        matrix=np.array([[4,1],[2,1]]);ax.imshow(matrix,cmap=CMAP,vmin=0,vmax=4)
        names=[['TN','FP'],['FN','TP']]
        for i in range(2):
            for j in range(2):ax.text(j,i,f'{names[i][j]}\n{matrix[i,j]}',ha='center',va='center',fontsize=24,color=ON_STRONG if matrix[i,j]>2 else ON_WEAK)
        ax.set_xticks([0,1],['Прогноз 0','Прогноз 1']);ax.set_yticks([0,1],['Истина 0','Истина 1']);ax.grid(False);save(fig,'M05-L01')
        fig,ax=canvas('M05-L02 · Вероятность и порог — разные части решения')
        z=np.linspace(-5,5,300);p=1/(1+np.exp(-z));ax.plot(z,p,color=BLUE,lw=3,label='σ(z)')
        for threshold,c in [(.5,GREEN),(.8,RED)]:ax.axhline(threshold,color=c,ls='--',label=f'Порог {threshold}')
        ax.scatter([-2,0,2],1/(1+np.exp(-np.array([-2,0,2]))),color=PURPLE,s=60)
        ax.set(xlabel='Линейный score z',ylabel='Вероятность p',ylim=(-.05,1.05));ax.legend();save(fig,'M05-L02')
        fig,ax=canvas('M05-L03 · Дерево даёт средние целевых значений в листьях')
        ax.scatter([0,1,2,3],[0,0,2,2],color=RED,s=90,zorder=4,label='Обучающие точки')
        ax.step([-1,1.5,5],[0,2,2],where='post',color=BLUE,lw=3,label='Дерево глубины 1')
        ax.axvline(1.5,color=GRAY,ls='--');ax.axhline(1,color=GREEN,ls=':',label='Постоянный baseline=1')
        ax.set(xlabel='x',ylabel='Предсказание',ylim=(-.5,3),xlim=(-.5,4.5));ax.legend(loc='upper left');save(fig,'M05-L03')
        fig,ax=canvas('M05-L04 · Подготовка обучается заново внутри каждого fold')
        for row in range(3):
            for col in range(3):
                validation=col==row
                ax.add_patch(Rectangle((col,row),.95,.7,facecolor=GREEN if validation else BLUE,alpha=.8))
                ax.text(col+.475,row+.35,'VALIDATION' if validation else 'TRAIN',ha='center',va='center',color=ON_STRONG,fontsize=11)
        ax.add_patch(Rectangle((3.35,0),.95,2.7,facecolor=GRAY,alpha=.25));ax.text(3.825,1.35,'TEST\nотдельно',ha='center',va='center')
        ax.set(xlim=(-.1,4.5),ylim=(-.2,3.5));ax.set_yticks([.35,1.35,2.35],['Fold 1','Fold 2','Fold 3']);ax.set_xticks([.475,1.475,2.475],['Часть 1','Часть 2','Часть 3']);ax.grid(False)
        ax.text(0,3.05,'fit(preprocessing + model) только на синих блоках',fontsize=12);save(fig,'M05-L04')
        fig,ax=canvas('M05-L05 · Общая метрика скрывает неодинаковые ошибки')
        b=ax.bar(['HR · n=2','technical · n=2'],[.5,3],color=[BLUE,RED],width=.55);ax.bar_label(b,padding=6)
        ax.axhline(1.75,color=GREEN,ls='--',label='Общая MAE=1.75');ax.set(ylabel='Средняя абсолютная ошибка',ylim=(0,3.7));ax.legend();save(fig,'M05-L05')
    return MAPPING

if __name__=='__main__':
    mapping=build(tuple(sys.argv[1:]) or ('light','dark'))
    (ROOT/'metadata/foundation-figures.json').write_text(json.dumps(mapping,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(f"Rendered {len(mapping)} quantitative figures in two layouts")
