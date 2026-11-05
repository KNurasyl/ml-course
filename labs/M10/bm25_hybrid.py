"""BM25 as a third retriever next to TF-IDF and MiniLM in RRF (extension 1.2.0, task T33).

The Russian diagnostic corpus, its queries and the metric come unchanged from diagnostic_ru.py; this lab only adds a
word-level BM25 ranking and fuses it. The report goes to artifacts/M10/bm25 and leaves the 1.1.0 reports untouched.
"""
from pathlib import Path
import argparse, importlib.util, json, re, time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT=Path(__file__).resolve().parents[2]
TOKEN=re.compile(r'[0-9a-zа-я]+')

def load(name):
    spec=importlib.util.spec_from_file_location(f'm10_{name}',Path(__file__).with_name(f'{name}.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def tokenize(text):
    """Lower-case words; ё is read as е. No stemming: «запись» and «записи» stay different words."""
    return TOKEN.findall(text.lower().replace('ё','е'))

class BM25:
    """Okapi BM25 over whole words: score(q,d)=Σ IDF(t)·tf·(k1+1)/(tf+k1·(1−b+b·|d|/avgdl))."""
    def __init__(self,documents,k1=1.5,b=.75):
        if k1<0 or not 0<=b<=1:raise ValueError('need k1 >= 0 and 0 <= b <= 1')
        docs=[tokenize(d) for d in documents]
        if not docs or any(not d for d in docs):raise ValueError('every document needs at least one word')
        self.vocab={t:i for i,t in enumerate(sorted({t for d in docs for t in d}))}
        self.tf=np.zeros((len(docs),len(self.vocab)))
        for row,doc in enumerate(docs):
            for t in doc:self.tf[row,self.vocab[t]]+=1
        self.length=self.tf.sum(axis=1);self.avgdl=self.length.mean();self.k1,self.b=k1,b
        df=(self.tf>0).sum(axis=0);n=len(docs)
        self.idf=np.log((n-df+.5)/(df+.5)+1) # +1 keeps IDF positive even for a word found in every document

    def scores(self,query):
        cols=[self.vocab[t] for t in tokenize(query) if t in self.vocab] # unknown words add nothing
        tf=self.tf[:,cols];norm=self.k1*(1-self.b+self.b*self.length/self.avgdl)
        return (self.idf[cols]*tf*(self.k1+1)/(tf+norm[:,None])).sum(axis=1)

def first_relevant(ranked_docs,gold):
    return next((i for i,doc in enumerate(ranked_docs,1) if doc in set(gold)),None)

def run(use_encoder=True,out_dir=None):
    diag,base=load('diagnostic_ru'),load('demo')
    traces=[];timings={}
    for strategy in ['words','paragraph']:
        rows=diag.chunks(strategy);texts=[r['title']+' '+r['text'] for r in rows];queries=[q['text'] for q in diag.QUERIES]
        started=time.perf_counter();vec=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5));matrix=vec.fit_transform(texts)
        tfidf=(vec.transform(queries)@matrix.T).toarray();timings[f'{strategy}/tfidf']=time.perf_counter()-started
        started=time.perf_counter();bm25=BM25(texts);bm=np.array([bm25.scores(q) for q in queries]);timings[f'{strategy}/bm25']=time.perf_counter()-started
        methods={'lexical_tfidf':tfidf,'bm25':bm}
        if use_encoder:
            base.torch.set_num_threads(2)
            vectors=base.pretrained_vectors(texts+queries);methods['minilm']=vectors[len(texts):]@vectors[:len(texts)].T
        def fused(names):
            out=[]
            for j in range(len(queries)):
                # Same fusion as diagnostic_ru: top 12 of each list, c=10, ranks rather than scores.
                ranking=base.rrf([np.argsort(-methods[n][j],kind='stable').tolist()[:12] for n in names])
                scores=np.zeros(len(rows));scores[ranking]=np.arange(len(ranking),0,-1);out.append(scores)
            return np.asarray(out)
        fusions={'rrf_tfidf_bm25':['lexical_tfidf','bm25']}
        if use_encoder:fusions.update(rrf_tfidf_minilm=['lexical_tfidf','minilm'],rrf_three=['lexical_tfidf','minilm','bm25'])
        for name,parts in fusions.items():methods[name]=fused(parts)
        for method,scores in methods.items():
            for j,q in enumerate(diag.QUERIES):
                ranking=np.argsort(-scores[j],kind='stable').tolist();top=ranking[:3]
                ranked_docs=list(dict.fromkeys(rows[i]['document_id'] for i in ranking))
                support=None if not q['support'] else any(q['support'] in rows[i]['text'] for i in top)
                traces.append({'case_id':q['id'],'split':q['split'],'strategy':strategy,'method':method,
                               'top_documents':ranked_docs[:3],'first_relevant_rank':first_relevant(ranked_docs,q['relevant']),
                               'metrics':diag.score_result(ranked_docs,q['relevant'],3),'support_phrase_in_context':support})
    groups={}
    for row in traces:groups.setdefault(f"{row['strategy']}/{row['method']}/{row['split']}",[]).append(row)
    summary={}
    for key,values in groups.items():
        answerable=[r for r in values if r['metrics']['answerable']]
        summary[key]={'answerable_n':len(answerable),'recall_at_3':float(np.mean([r['metrics']['recall'] for r in answerable])),
                      'mrr':float(np.mean([r['metrics']['mrr'] for r in answerable])),
                      'support_present':sum(r['support_phrase_in_context'] is True for r in values)}
    # Per query: did adding BM25 move the first relevant document up or down against the 1.1.0 fusion?
    baseline='rrf_tfidf_minilm' if use_encoder else 'lexical_tfidf';candidate='rrf_three' if use_encoder else 'rrf_tfidf_bm25'
    index={(r['strategy'],r['method'],r['case_id']):r for r in traces};changes=[]
    for (strategy,method,case),row in index.items():
        if method!=candidate:continue
        old=index[(strategy,baseline,case)]['first_relevant_rank'];new=row['first_relevant_rank']
        if old!=new:changes.append({'strategy':strategy,'case_id':case,'split':row['split'],baseline:old,candidate:new})
    report={'task':'T33','corpus_version':'ru-diagnostic-1','baseline':baseline,'candidate':candidate,'bm25':{'k1':1.5,'b':.75,'tokens':'words, ё→е, no stemming'},
            'summary':summary,'rank_changes':changes,'traces':traces,'build_seconds':timings,
            'limits':'Synthetic Russian teaching corpus with 14 queries; a development check, not a benchmark. BM25 matches whole words without stemming. MiniLM is an English encoder.'}
    out=Path(out_dir) if out_dir else ROOT/'artifacts/M10/bm25';out.mkdir(parents=True,exist_ok=True)
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--lexical-only',action='store_true',help='skip the MiniLM encoder');args=parser.parse_args()
    result=run(not args.lexical_only);print(json.dumps({'summary':result['summary'],'rank_changes':result['rank_changes']},ensure_ascii=True,indent=2))
