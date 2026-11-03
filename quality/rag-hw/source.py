"""Complete Russian RAG homework key; actual outputs are never repaired.

Run from the repository root. Models are local. English MiniLM is deliberately
measured on Russian rather than advertised as a multilingual solution.
"""
from pathlib import Path
import argparse
import gc
import hashlib
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from labs.M10 import diagnostic_ru as diagnostic
from labs.M10 import demo as base
from projects.interview_trainer.llm import LocalLLM

NEW_TOPICS = [
    ('pool', 'Пул соединений базы данных',
     'Пул переиспользует соединения базы данных и ограничивает число одновременно выданных соединений. '
     'Если все соединения заняты, новый запрос ждёт освобождения или получает отказ по дедлайну. '
     'Увеличение пула не гарантирует ускорения: база данных может стать перегруженной.',
     'Старая ошибочная учебная заметка обещала, что увеличение пула всегда ускоряет все запросы.'),
    ('backpressure', 'Обратное давление и ограничение приёма',
     'Обратное давление ограничивает приём новых задач, когда обработчик перегружен. '
     'Конечная очередь может отклонить новую задачу при заполнении. '
     'Очередь приёма задач и ожидание соединения из пула являются разными этапами; измеряйте время каждого отдельно.',
     'Старая ошибочная учебная заметка разрешала бесконечную очередь без ограничения ожидания.')]

QUERIES = [
    {'id':'hw-term-pool','kind':'exact','text':'Что делает пул соединений базы данных, когда все соединения заняты?',
     'relevant':['pool'],'support':['ждёт освобождения или получает отказ по дедлайну']},
    {'id':'hw-term-pressure','kind':'exact','text':'Что означает обратное давление при перегрузке обработчика?',
     'relevant':['backpressure'],'support':['ограничивает приём новых задач']},
    {'id':'hw-paraphrase-pool','kind':'paraphrase','text':'Даст ли больше открытых соединений обязательное ускорение работы с базой?',
     'relevant':['pool'],'support':['не гарантирует ускорения']},
    {'id':'hw-paraphrase-pressure','kind':'paraphrase','text':'Обработчик не успевает, новые задания копятся без конца. Как ограничить поступление работы?',
     'relevant':['backpressure'],'support':['ограничивает приём новых задач']},
    {'id':'hw-two-sources','kind':'multi-source','text':'Где запрос может ждать: перед обработчиком и перед получением соединения базы? Почему нужны два измерения?',
     'relevant':['pool','backpressure'],'support':['ждёт освобождения','измеряйте время каждого отдельно']},
    {'id':'hw-no-answer','kind':'unanswerable','text':'Какой точный бюджет в тенге выделила компания Арктур на серверы в ноябре?',
     'relevant':[],'support':[]}]


def documents():
    docs = diagnostic.documents()
    for key, title, current, old in NEW_TOPICS:
        for version, body in [(1, old), (2, current)]:
            paragraphs = [title+'.', body,
                          'Сравнивай одинаковые входы и сохраняй отдельно ожидание и вычисление.']
            docs.append({'id':f'{key}-v{version}','document_id':key,'version':version,
                         'title':title,'paragraphs':paragraphs,'text':'\n\n'.join(paragraphs),
                         'origin':'original synthetic course note, not copied documentation',
                         'status':'current' if version==2 else 'obsolete deliberately incorrect teaching note'})
    return docs


def chunks(docs, strategy):
    rows=[]
    for doc in docs:
        if doc['version'] != 2:
            continue
        if strategy=='paragraph':
            parts=doc['paragraphs']
        else:
            words=doc['text'].split()
            parts=[' '.join(words[i:i+18]) for i in range(0,len(words),18)]
        for i, text in enumerate(parts):
            rows.append({'id':f"{doc['id']}-c{i}",'document_id':doc['document_id'],
                         'version':doc['version'],'title':doc['title'],'text':text})
    return rows


def write(out, name, value):
    (out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def run(retrieval_only=False):
    out=ROOT/'artifacts/M10/hw-ru';out.mkdir(parents=True,exist_ok=True)
    docs=documents()
    assert len(docs)==52 and len(QUERIES)==6
    # Gold is materialized before retrieval and generation, not derived from rank.
    write(out,'corpus.json',docs);write(out,'gold.json',QUERIES)
    policy={'chosen_context':'paragraph/lexical/top3 chunks','abstention_threshold':0.12,
            'llm_cases':['hw-term-pool','hw-paraphrase-pool','hw-two-sources','hw-no-answer'],
            'max_new_tokens':160,'selection':'fixed before viewing outputs'}
    write(out,'policy.json',policy)
    traces=[];timings={}
    base.torch.set_num_threads(2)
    for strategy in ['words','paragraph']:
        rows=chunks(docs,strategy)
        texts=[row['title']+' '+row['text'] for row in rows]
        start=time.perf_counter()
        vectorizer=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),sublinear_tf=True)
        matrix=vectorizer.fit_transform(texts)
        query_matrix=vectorizer.transform([q['text'] for q in QUERIES])
        lexical=(query_matrix@matrix.T).toarray()
        lexical_time=time.perf_counter()-start
        start=time.perf_counter()
        svd=TruncatedSVD(n_components=8,random_state=17)
        lsa_docs=normalize(svd.fit_transform(matrix))
        lsa_queries=normalize(svd.transform(query_matrix))
        lsa=lsa_queries@lsa_docs.T
        lsa_time=time.perf_counter()-start
        start=time.perf_counter()
        vectors=base.pretrained_vectors(texts+[q['text'] for q in QUERIES])
        dense=vectors[len(texts):]@vectors[:len(texts)].T
        dense_time=time.perf_counter()-start
        timings[strategy]={'lexical_build_and_query_seconds':lexical_time,
                           'lsa_fit_and_query_seconds':lsa_time,
                           'minilm_load_encode_query_seconds':dense_time,
                           'caution':'These phases include different setup work, not a serving latency comparison.'}
        for j,query in enumerate(QUERIES):
            rankings={'lexical':base.order(lexical[j]),'lsa':base.order(lsa[j]),
                      'minilm_english_on_russian':base.order(dense[j])}
            rankings['rrf']=base.rrf([rankings['lexical'][:12],rankings['minilm_english_on_russian'][:12]])
            for method, ranking in rankings.items():
                ranked_docs=list(dict.fromkeys(rows[i]['document_id'] for i in ranking))
                top=[rows[i] for i in ranking[:3]]
                text='\n'.join(row['text'] for row in top)
                abstained=float(lexical[j].max())<policy['abstention_threshold']
                traces.append({'case_id':query['id'],'query':query['text'],'strategy':strategy,'method':method,
                               'ranked_chunks':[rows[i]['id'] for i in ranking], 'ranked_documents':ranked_docs,
                               'top_chunks':top,'metrics':diagnostic.score_result(ranked_docs,query['relevant'],3,abstained),
                               'abstained_by_lexical_rule':abstained,'lexical_max':float(lexical[j].max()),
                               'support_phrases_present':[phrase in text for phrase in query['support']]})
    summary={}
    for strategy in ['words','paragraph']:
        for method in ['lexical','lsa','minilm_english_on_russian','rrf']:
            group=[row for row in traces if row['strategy']==strategy and row['method']==method]
            summary[f'{strategy}/{method}']={
                'recall_at_3':base.mean_defined([row['metrics']['recall'] for row in group]),
                'mrr':base.mean_defined([row['metrics']['mrr'] for row in group]),
                'answerable_n':5,'unanswerable_n':1,
                'correct_abstentions':sum(row['metrics']['correct_abstention'] is True for row in group)}
    assert all(row['version']==2 for strategy in ['words','paragraph'] for row in chunks(docs,strategy))
    old=next(doc for doc in docs if doc['id']=='pool-v1')
    current=next(doc for doc in docs if doc['id']=='pool-v2')
    cache={('release-v1','pool-query'):old['text']}
    assert cache.get(('release-v2','pool-query')) is None
    cache[('release-v2','pool-query')]=current['text']
    fault={'type':'deliberate diagnostic fixture, not a real model output',
           'retrieval':{'gold':['pool'],'forced_candidates':['queue','timeout'],'recall':0},
           'generation':{'claim':'Увеличение пула всегда ускоряет запросы.','source_id':'pool-v2',
                         'label':'contradicted','evidence':'Увеличение пула не гарантирует ускорения'},
           'version_check':{'all_current_chunks_version':2,'new_release_cache_value':cache[('release-v2','pool-query')],
                            'old_cache_preserved_for_reproduction':True}}
    report={'summary':summary,'traces':traces,'timings':timings,'controlled_faults':fault,
            'limits':'Synthetic corpus; MiniLM is English; substring support is not semantic entailment.',
            'generation_status':'not_run' if retrieval_only else 'pending',
            'encoder':{'id':base.ENCODER_ID,'revision':base.ENCODER_REVISION},
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    write(out,'report.json',report)
    if not retrieval_only:
        gc.collect()
        model=LocalLLM();outputs=[]
        for case_id in policy['llm_cases']:
            row=next(r for r in traces if r['case_id']==case_id and r['strategy']=='paragraph' and r['method']=='lexical')
            context='\n'.join(f"[{c['id']}] {c['text']}" for c in row['top_chunks'])
            messages=[{'role':'system','content':'Ответь по-русски кратко, максимум два предложения. Используй только предоставленные источники. После фактического утверждения укажи ID источника в квадратных скобках. Если подтверждения нет, скажи «Недостаточно сведений». Не добавляй собственные числа и факты.'},
                      {'role':'user','content':f"Вопрос: {row['query']}\nИсточники:\n{context}"}]
            started=time.perf_counter();raw=model.generate(messages,max_new_tokens=policy['max_new_tokens'])
            outputs.append({'case_id':case_id,'messages':messages,'sources':row['top_chunks'],
                            'raw_output':raw,'generation':dict(model.last_generation),
                            'seconds':time.perf_counter()-started,'semantic_review':'required, separate manual-review.json'})
            write(out,'outputs.json',outputs)
        report['generation_status']='4 actual local Qwen outputs saved; semantic correctness not implied'
        write(out,'report.json',report)
    return report['summary']


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--retrieval-only',action='store_true')
    args=parser.parse_args()
    print(json.dumps(run(args.retrieval_only),ensure_ascii=False,indent=2))
