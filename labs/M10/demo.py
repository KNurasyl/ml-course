"""Local lexical/LSA/pretrained-vector retrieval + actual small-LLM RAG."""
from pathlib import Path
from collections import defaultdict
import argparse, gc, importlib.util, json, re, time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
import torch
from transformers import AutoModel, AutoTokenizer

ROOT=Path(__file__).resolve().parents[2]
ENCODER=ROOT/'models/all-MiniLM-L6-v2'
ENCODER_ID='sentence-transformers/all-MiniLM-L6-v2'
ENCODER_REVISION='1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
CORPUS=[
 {'id':'sql-v1-atomic','document_id':'sql','version':1,'title':'Transaction atomicity','text':'A SQL transaction groups operations. Atomicity means that its changes are committed together or rolled back together after failure.'},
 {'id':'sql-v1-isolation','document_id':'sql','version':1,'title':'Transaction isolation','text':'Transaction isolation controls how concurrent transactions observe changes. Different isolation levels allow different concurrency anomalies.'},
 {'id':'index-v2-cost','document_id':'index','version':2,'title':'Index tradeoffs','text':'A database index can speed up suitable queries but uses storage and adds maintenance work for inserts and updates. The query plan and selectivity matter.'},
 {'id':'http-v1-idempotent','document_id':'http','version':1,'title':'Idempotency','text':'An idempotent operation has the same intended effect when repeated with the same input. A request identifier can prevent duplicate processing in an application.'},
 {'id':'python-v1-generator','document_id':'python','version':1,'title':'Python generators','text':'A Python generator yields values lazily. It produces the next value when requested instead of constructing a full list in advance.'},
 {'id':'api-v1-timeout','document_id':'api','version':1,'title':'Timeout and cancellation','text':'A client timeout stops waiting but does not necessarily stop server computation. Cancellation and stale-result checks are separate application responsibilities.'},
 {'id':'cache-v1-version','document_id':'cache','version':1,'title':'Cache versions','text':'A cached answer may become stale when source data changes. Include relevant release identifiers in the cache key or invalidate affected cached results.'},
 {'id':'ml-v1-leakage','document_id':'ml','version':1,'title':'Data leakage','text':'Fit preprocessing on the training partition only. Using test labels to choose a model leaks evaluation information and invalidates the final estimate.'},
]
for doc in CORPUS:doc['source']='original course-authored educational note; not a copied external document'
QUERIES=[
 {'id':'q1','text':'What happens to transaction changes if one operation fails?','relevant':['sql-v1-atomic']},
 {'id':'q2','text':'Why can a database index make writes more expensive?','relevant':['index-v2-cost']},
 {'id':'q3','text':'How does a Python generator avoid building the entire list?','relevant':['python-v1-generator']},
 {'id':'q4','text':'Does a client timeout cancel server work?','relevant':['api-v1-timeout']},
 {'id':'q5','text':'How can stale cached answers survive a data update?','relevant':['cache-v1-version']},
 {'id':'q6','text':'Why is selecting a model with test labels invalid?','relevant':['ml-v1-leakage']},
 {'id':'q7','text':'What produces items on demand without allocating a collection up front?','relevant':['python-v1-generator']},
]

def download_encoder():
    from huggingface_hub import snapshot_download
    names=['model.safetensors','config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json',
           'vocab.txt','README.md','modules.json','sentence_bert_config.json','1_Pooling/config.json']
    snapshot_download(ENCODER_ID,revision=ENCODER_REVISION,local_dir=str(ENCODER),allow_patterns=names)
    (ENCODER/'course-provenance.json').write_text(json.dumps({'repo_id':ENCODER_ID,'revision':ENCODER_REVISION}),encoding='utf-8')

def pretrained_vectors(texts):
    if not (ENCODER/'model.safetensors').exists():
        raise FileNotFoundError('Run .\\.venv\\Scripts\\python.exe labs\\M10\\demo.py --download')
    tokenizer=AutoTokenizer.from_pretrained(ENCODER,local_files_only=True)
    model=AutoModel.from_pretrained(ENCODER,local_files_only=True);model.eval()
    encoded=tokenizer(texts,padding=True,truncation=True,max_length=256,return_tensors='pt')
    with torch.inference_mode():
        hidden=model(**encoded).last_hidden_state
        mask=encoded['attention_mask'].unsqueeze(-1).to(hidden.dtype)
        pooled=(hidden*mask).sum(1)/mask.sum(1).clamp(min=1)
        vectors=torch.nn.functional.normalize(pooled,p=2,dim=1).numpy().copy()
    assert vectors.shape==(len(texts),384)
    assert np.allclose(np.linalg.norm(vectors,axis=1),1,atol=1e-5)
    del model,hidden,pooled;gc.collect()
    return vectors

def order(scores):return np.argsort(-np.asarray(scores),kind='stable').tolist()

def rrf(rankings,c=10):
    scores=defaultdict(float)
    for ranking in rankings:
        for rank,index in enumerate(ranking,1):scores[index]+=1/(c+rank)
    return sorted(scores,key=lambda i:(-scores[i],i))

def metrics(ranking,relevant,k=3):
    ids=[CORPUS[i]['id'] for i in ranking];gold=set(relevant)
    recall=len(gold&set(ids[:k]))/len(gold)
    first=next((rank for rank,doc in enumerate(ids,1) if doc in gold),None)
    return {'recall_at_3':recall,'mrr':0 if first is None else 1/first}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--download',action='store_true')
    parser.add_argument('--retrieval-only',action='store_true');args=parser.parse_args()
    torch.set_num_threads(2)
    if args.download:download_encoder()
    texts=[d['title']+' '+d['text'] for d in CORPUS]
    vectorizer=TfidfVectorizer(stop_words='english',ngram_range=(1,2));matrix=vectorizer.fit_transform(texts)
    query_matrix=vectorizer.transform([q['text'] for q in QUERIES])
    lexical=(query_matrix@matrix.T).toarray()
    svd=TruncatedSVD(n_components=4,random_state=17)
    doc_lsa=normalize(svd.fit_transform(matrix));q_lsa=normalize(svd.transform(query_matrix))
    lsa=q_lsa@doc_lsa.T
    started=time.perf_counter();vectors=pretrained_vectors(texts+[q['text'] for q in QUERIES])
    dense=vectors[len(texts):]@vectors[:len(texts)].T
    embedding_seconds=time.perf_counter()-started
    traces=[]
    for j,q in enumerate(QUERIES):
        rankings={'lexical':order(lexical[j]),'lsa':order(lsa[j]),'minilm':order(dense[j])}
        combined=rrf([rankings['lexical'][:4],rankings['minilm'][:4]])
        terms=set(re.findall(r'[a-z]+',q['text'].lower()))-{'a','the','to','of','is','does','can','how','why','what'}
        def coverage(i):return len(terms&set(re.findall(r'[a-z]+',texts[i].lower())))
        # Deliberately transparent second-stage heuristic; NOT a neural cross-encoder.
        reranked=sorted(combined,key=lambda i:(-coverage(i),combined.index(i)))
        rankings.update(rrf=combined,heuristic_rerank=reranked)
        traces.append({'query':q,'rankings':{name:[CORPUS[i]['id'] for i in ids] for name,ids in rankings.items()},
                       'metrics':{name:metrics(ids,q['relevant']) for name,ids in rankings.items()}})
    assert rrf([[0,1,2],[1,3,0]])==[1,0,3,2]
    summary={name:{key:float(np.mean([t['metrics'][name][key] for t in traces])) for key in ['recall_at_3','mrr']}
             for name in ['lexical','lsa','minilm','rrf','heuristic_rerank']}
    selected=traces[0]['rankings']['rrf'][:2]
    context='\n'.join(f"[{d['id']}] {d['text']}" for cid in selected for d in CORPUS if d['id']==cid)
    assert 'sql-v1-atomic' in selected,'Diagnostic query lost its relevant source before generation'
    report={'corpus_version':'course-rag-v1','encoder_id':ENCODER_ID,'encoder_revision':ENCODER_REVISION,
            'encoder_language':'English; Russian quality not established','embedding_seconds':embedding_seconds,
            'summary':summary,'traces':traces,'context':context,'extractive_baseline':context,
            'reranker':'term coverage heuristic, not cross-encoder','generation_status':'not_run_retrieval_only'}
    if not args.retrieval_only:
        spec=importlib.util.spec_from_file_location('course_m09',ROOT/'labs/M09/demo.py')
        m09=importlib.util.module_from_spec(spec);spec.loader.exec_module(m09)
        tokenizer,model,load_seconds=m09.load_local()
        messages=[{'role':'system','content':'Answer only from the supplied context. Cite the exact source identifier in brackets. If unsupported, say insufficient information.'},
                  {'role':'user','content':f"Context:\n{context}\nQuestion: {QUERIES[0]['text']}"}]
        output=m09.generate(tokenizer,model,messages,max_new_tokens=64)
        citations=re.findall(r'\[([^\]]+)\]',output['text'])
        report.update(generation_status='actual_local_smollm',generated_answer=output,
                      cited_ids=citations,citation_ids_known=all(c in selected for c in citations),
                      has_citation=bool(citations),support_status='requires claim-by-claim human check; ID existence is insufficient')
    out=ROOT/'artifacts/M10';out.mkdir(parents=True,exist_ok=True)
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'corpus.json').write_text(json.dumps(CORPUS,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'summary':summary,'generation_status':report['generation_status'],
                      'generated_answer':report.get('generated_answer'),'has_citation':report.get('has_citation')},indent=2))

if __name__=='__main__':main()
