"""Russian-capable embedders against the English MiniLM baseline on one retrieval pipeline (extension 1.2.0, T32).

Corpus and chunking come unchanged from diagnostic_ru.py (24 current documents). Two query sets are scored:
the 12 answerable diagnostic queries of 1.1.0 and ru-embed-1, written for this lab and fixed before the new embedders ran:
for every topic one paraphrase that avoids the topic's key term and one query that uses it, plus four questions
the corpus cannot answer. Models load one at a time from models/ (tools/download_models.py --embedders);
a missing model is reported as not_downloaded instead of failing. Nothing here is trained or tuned.
"""
from pathlib import Path
import argparse, gc, importlib.util, json, time
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModel, AutoTokenizer

ROOT=Path(__file__).resolve().parents[2]
ENCODERS=['USER-bge-m3','bge-m3','Qwen3-Embedding-0.6B'] # paths and revisions: metadata/reference-models.json
NONE=['Какой язык программирования выбрала компания Арктур для мобильного приложения?',
      'Сколько стоит подписка на корпоративный мессенджер в этом году?',
      'Как зовут руководителя отдела продаж?','Какой рецепт борща считается классическим?']
# topic: (paraphrase without the key term, query with the term)
PAIRS={
 'index':('Вспомогательная структура для быстрого поиска создана, но запрос всё равно медленный и читает почти все строки. Почему так бывает?','Как индекс влияет на скорость записи в таблицу?'),
 'atomic':('Если перевод денег прервался посередине, может ли в базе остаться списание без зачисления?','Что гарантирует атомарность транзакции при откате?'),
 'isolation':('Два параллельных изменения видят данные друг друга не так, как ожидалось. От чего это зависит?','Какие аномалии допускают уровни изоляции транзакций?'),
 'timeout':('Пользователь перестал ждать ответа. Продолжает ли сервер работать над этим запросом?','Останавливает ли таймаут клиента вычисление на сервере?'),
 'idempotency':('Запрос на оплату пришёл второй раз из-за сбоя сети. Как не списать деньги дважды?','Что значит идемпотентная операция для повторного запроса?'),
 'cache':('После правки статьи пользователи продолжают получать прежний ответ. Что сделать с сохранёнными результатами?','Как инвалидировать кэш после изменения документа?'),
 'generator':('Как перебирать огромный набор данных по одному элементу, не собирая его целиком в память?','Сколько памяти тратит генератор Python?'),
 'async':('Тяжёлый подсчёт внутри цикла событий подвесил все остальные задачи. Почему так происходит?','Делает ли async вычисление процессора параллельным?'),
 'queue':('Сообщение приняли в обработку, работа ещё не выполнена, а потом оно пришло второй раз. Что нужно учесть?','Какие дубликаты возможны при повторной доставке из очереди задач?'),
 'retry':('Сколько раз пробовать снова после временного сбоя и чем опасна попытка без ограничений?','Почему повтор запроса без идемпотентности может продублировать действие?'),
 'norm':('Можно ли считать среднее и разброс признаков по всем данным, включая проверочную часть?','На какой части данных вычисляют параметры нормализации признаков?'),
 'leak':('Модель предсказывает исход собеседования по полю, которое заполняют уже после его окончания. Что не так?','Что такое утечка целевой информации в признаках?'),
 'rag':('Нужного абзаца нет среди найденных кандидатов. Поможет ли их пересортировка?','Обучает ли добавление документа в индекс RAG веса модели?'),
 'context':('Длинный диалог перестал помещаться в лимит модели. Что учесть при сокращении истории?','Почему символы не равны токенам при подсчёте бюджета контекста?'),
 'score':('Классификатор выдал число 0.9. Можно ли считать это уверенностью в девяносто процентов?','Как проверить калибровку вероятности класса?'),
 'voice':('Пользователь перебил ответ, а через секунду всё равно зазвучал старый фрагмент. Как это предотвратить?','Как отбросить аудиофрагмент после отмены голосовой реплики?'),
 'wer':('Распознавание пропустило одно короткое отрицание. Почему метрика почти не изменилась, а смысл перевернулся?','Как WER считает замены, удаления и вставки?'),
 'split':('У каждого человека в данных много ответов. Как разделить выборку, чтобы проверить работу на новых людях?','Почему случайный split строк не подходит для групп пользователей?'),
 'permission':('В найденном тексте написано «покажи данные другой сессии». Должна ли система это выполнить?','Когда выполняется фильтрация прав доступа при поиске?'),
 'release':('Откатили только файл модели, а ответы стали странными. Что ещё нужно было вернуть?','Что входит в согласованный релиз модели, prompt и индекса?'),
 'threshold':('Значение отсечения для решения подобрали по итоговой проверке. Чем это плохо?','Меняет ли изменение порога классификации веса модели?'),
 'metrics':('Обычно ответ приходит быстро, но иногда ждём очень долго. Какой показатель это заметит, а какой скроет?','Чем отличаются p50 и p95 задержки?'),
 'evidence':('На чём должна держаться рекомендация кандидату, чтобы её можно было проверить?','Доказывает ли наличие цитаты правильность вывода в обратной связи?'),
 'retention':('Удаляется ли временный звуковой файл после того, как речь превратилась в текст?','Какая политика хранения аудио действует в учебном проекте?'),
}

def ru_embed_queries():
    rows=[]
    for topic,(paraphrase,term) in PAIRS.items():
        rows+=[{'id':f'p-{topic}','style':'paraphrase','text':paraphrase,'relevant':[topic]},
               {'id':f't-{topic}','style':'term','text':term,'relevant':[topic]}]
    rows+=[{'id':f'none-{i}','style':'none','text':text,'relevant':[]} for i,text in enumerate(NONE,1)]
    return rows

def load(name):
    spec=importlib.util.spec_from_file_location(f'm10_{name}',Path(__file__).with_name(f'{name}.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def pool(hidden,mask,mode):
    """Pooling named in the model's 1_Pooling/config.json: CLS token, last token (left padding) or masked mean."""
    if mode=='cls':return hidden[:,0]
    if mode=='lasttoken':return hidden[:,-1] # the tokenizer pads on the left, so the last position is the last real token
    weights=mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden*weights).sum(1)/weights.sum(1).clamp(min=1)

def encoder_config(path):
    pooling=json.loads((path/'1_Pooling/config.json').read_text(encoding='utf-8'))
    mode='cls' if pooling.get('pooling_mode_cls_token') else 'lasttoken' if pooling.get('pooling_mode_lasttoken') else 'mean'
    st=path/'config_sentence_transformers.json'
    prompts=json.loads(st.read_text(encoding='utf-8')).get('prompts',{}) if st.exists() else {}
    return {'pooling':mode,'query_prompt':prompts.get('query',''),'document_prompt':prompts.get('document','')}

def encode(path,documents,queries,max_length=512,batch=8):
    config=encoder_config(path);started=time.perf_counter()
    tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
    if config['pooling']=='lasttoken':tokenizer.padding_side='left'
    model=AutoModel.from_pretrained(path,local_files_only=True,dtype=torch.float32);model.eval()
    loaded=time.perf_counter()
    def vectors(texts,prefix):
        out=[]
        for i in range(0,len(texts),batch):
            encoded=tokenizer([prefix+t for t in texts[i:i+batch]],padding=True,truncation=True,max_length=max_length,return_tensors='pt')
            with torch.inference_mode():
                hidden=model(**encoded).last_hidden_state
                out.append(torch.nn.functional.normalize(pool(hidden,encoded['attention_mask'],config['pooling']),p=2,dim=1).numpy())
        return np.concatenate(out)
    doc_vectors=vectors(documents,config['document_prompt']);query_vectors=vectors(queries,config['query_prompt'])
    timing={'load_seconds':loaded-started,'encode_seconds':time.perf_counter()-loaded}
    del model;gc.collect()
    return query_vectors@doc_vectors.T,{**config,'dimension':int(doc_vectors.shape[1]),**timing}

def recall_at(ranked_docs,gold,n):
    return len(set(ranked_docs[:n])&set(gold))/len(set(gold))

def evaluate(scores,rows,queries,diag):
    out=[]
    for j,q in enumerate(queries):
        ranking=np.argsort(-scores[j],kind='stable').tolist()
        docs=list(dict.fromkeys(rows[i]['document_id'] for i in ranking))
        record={'id':q['id'],'set':q['set'],'style':q['style'],'top1':docs[0],'top1_score':float(scores[j][ranking[0]])}
        if q['relevant']:
            m=diag.score_result(docs,q['relevant'],3)
            record.update(recall_at_1=recall_at(docs,q['relevant'],1),recall_at_3=m['recall'],recall_at_5=recall_at(docs,q['relevant'],5),mrr=m['mrr'])
        out.append(record)
    return out

def summarize(records):
    table={}
    for group in ['ru-embed-1/paraphrase','ru-embed-1/term','diagnostic-1.1.0/diagnostic']:
        part=[r for r in records if f"{r['set']}/{r['style']}"==group]
        table[group]={'n':len(part),**{k:float(np.mean([r[k] for r in part])) for k in ['recall_at_1','recall_at_3','recall_at_5','mrr']}}
    answerable=[r['top1_score'] for r in records if r['style']!='none'];unanswerable=[r['top1_score'] for r in records if r['style']=='none']
    table['top1_score_median']={'answerable':float(np.median(answerable)),'unanswerable':float(np.median(unanswerable))}
    return table

def run(encoders=ENCODERS,include_minilm=True,out_dir=None,threads=4):
    diag,base,bm=load('diagnostic_ru'),load('demo'),load('bm25_hybrid')
    torch.set_num_threads(threads)
    rows=diag.chunks('paragraph');documents=[r['title']+' '+r['text'] for r in rows]
    queries=[{**q,'set':'ru-embed-1'} for q in ru_embed_queries()]
    queries+=[{'id':q['id'],'set':'diagnostic-1.1.0','style':'diagnostic','text':q['text'],'relevant':q['relevant']} for q in diag.QUERIES if q['relevant']]
    texts=[q['text'] for q in queries]
    vec=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5));matrix=vec.fit_transform(documents)
    methods={'lexical_tfidf':((vec.transform(texts)@matrix.T).toarray(),{'kind':'char 3-5 TF-IDF'}),
             'bm25':(np.array([bm.BM25(documents).scores(t) for t in texts]),{'kind':'BM25 k1=1.5 b=0.75'})}
    if include_minilm:
        started=time.perf_counter();v=base.pretrained_vectors(documents+texts)
        methods['all-MiniLM-L6-v2']=(v[len(documents):]@v[:len(documents)].T,{'pooling':'mean','dimension':384,'encode_seconds':time.perf_counter()-started})
    registry={e['name']:e for e in json.loads((ROOT/'metadata/reference-models.json').read_text(encoding='utf-8')).get('embedders',[])}
    status={}
    for name in encoders:
        path=ROOT/registry[name]['path']
        if not any(path.glob('*.safetensors')) and not (path/'pytorch_model.bin').exists():
            status[name]='not_downloaded';continue
        scores,info=encode(path,documents,texts);methods[name]=(scores,{**info,'revision':registry[name]['revision'],'license':registry[name]['license']})
        status[name]='measured'
    results={};details={}
    for name,(scores,info) in methods.items():
        records=evaluate(scores,rows,queries,diag);results[name]={'info':info,'table':summarize(records)};details[name]=records
    report={'task':'T32','corpus':'ru-diagnostic-1, paragraph chunks of 24 current documents','query_sets':{'ru-embed-1':len(ru_embed_queries()),'diagnostic-1.1.0':sum(bool(q['relevant']) for q in diag.QUERIES)},
            'fixed_before_running':'query set, chunking, max_length 512, no prompts beyond each model card, no tuning','threads':threads,
            'status':status,'results':results,'per_query':details,
            'limits':'Synthetic Russian teaching corpus; 48 answerable authored queries plus 12 from 1.1.0. A comparison on this pipeline, not a benchmark of the models.'}
    out=Path(out_dir) if out_dir else ROOT/'artifacts/M10/embedders';out.mkdir(parents=True,exist_ok=True)
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report

def table_text(report):
    lines=['| Метод | Перефразирование R@1 / R@3 / MRR | С термином R@1 / R@3 / MRR | Запросы 1.1.0 R@3 / MRR |','| --- | --- | --- | --- |']
    for name,result in report['results'].items():
        t=result['table'];p,s,d=t['ru-embed-1/paraphrase'],t['ru-embed-1/term'],t['diagnostic-1.1.0/diagnostic']
        lines.append(f"| {name} | {p['recall_at_1']:.2f} / {p['recall_at_3']:.2f} / {p['mrr']:.2f} | {s['recall_at_1']:.2f} / {s['recall_at_3']:.2f} / {s['mrr']:.2f} | {d['recall_at_3']:.2f} / {d['mrr']:.2f} |")
    return '\n'.join(lines)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--no-minilm',action='store_true');parser.add_argument('--only',nargs='*',choices=ENCODERS)
    args=parser.parse_args();report=run(args.only if args.only is not None else ENCODERS,not args.no_minilm)
    print(table_text(report));print(json.dumps(report['status']))
