"""Five fixed inputs, all greedy/sampling attempts and actual terminal fallback."""
from pathlib import Path
import importlib.util, json

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('m09_demo',Path(__file__).with_name('demo.py'))
demo=importlib.util.module_from_spec(spec);spec.loader.exec_module(demo)

CASES=[
 {'id':'definition','text':'Explain SQL transaction atomicity in one sentence.'},
 {'id':'clarification','text':'An interview candidate said: I improved the API. Ask one question about their own contribution.'},
 {'id':'json','text':'Return only JSON with feedback (string) and score (integer 0,1,2). Evaluate: A transaction groups operations.'},
 {'id':'ambiguous','text':'An interview candidate said: It became better. Ask one question to clarify the evidence.'},
 {'id':'russian','text':'Что значит идемпотентность HTTP? Ответь кратко по-русски.'},
]

def main():
    tokenizer,model,load_seconds=demo.load_local();rows=[]
    for case in CASES:
        messages=[{'role':'system','content':'You are a concise interview tutor.'},{'role':'user','content':case['text']}]
        for mode,temperature,seed in [('greedy',1.,17),('sampling',.7,17),('sampling',.7,23),('sampling',1.1,17),('sampling',1.1,23)]:
            output=demo.generate(tokenizer,model,messages,24,do_sample=mode=='sampling',temperature=temperature,top_p=.9,seed=seed)
            rows.append({'case_id':case['id'],'mode':mode,'messages':messages,
                         'template':tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True),
                         'output':output,'response':demo.feedback_or_abstain(output['text']) if case['id']=='json' else None})
    # Forced faults isolate control-flow correctness from unpredictable LLM output.
    faults={name:demo.feedback_or_abstain(value) for name,value in {
        'empty':'','bool':'{"score":true,"feedback":"ok"}','extra':'{"score":1,"feedback":"ok","x":0}',
        'truncated':'{"feedback":','valid':'{"feedback":"Name your action","score":1}'}.items()}
    assert all(v['status']=='needs_review' and v['score'] is None for k,v in faults.items() if k!='valid')
    assert faults['valid']['status']=='ok'
    exceeded=False
    try:demo.generate(tokenizer,model,[{'role':'user','content':'x'}],model.config.max_position_embeddings)
    except ValueError:exceeded=True
    assert exceeded
    out=ROOT/'artifacts/M09';out.mkdir(parents=True,exist_ok=True)
    report={'repo_id':demo.REPO,'revision':demo.REVISION,'torch_threads':2,'load_seconds':load_seconds,
            'attempt_count':len(rows),'cases':rows,'fault_cases':faults,'context_overflow_rejected':exceeded,
            'interpretation':'Raw model outputs; format validity is not truth. Russian quality requires manual rubric.'}
    (out/'experiments.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'attempts':len(rows),'terminal_fallbacks':sum(r['response']['fallback_used'] for r in rows if r['response']),
                      'context_overflow_rejected':exceeded}))

if __name__=='__main__':main()
