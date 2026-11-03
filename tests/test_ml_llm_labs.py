"""Semantic checks: wrong versions, future leakage and invalid judgments must fail."""
from pathlib import Path
import importlib.util
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]

def load(module):
    spec=importlib.util.spec_from_file_location('course_test_'+module,ROOT/'labs'/module/'demo.py')
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value

def test_release_preprocessing_and_queue():
    m=load('M06')
    release=dict(mean=10,scale=2,weight=3,bias=1)
    assert m.predict(14,release)==7
    assert m.predict(14,dict(release,mean=12))!=7
    assert m.queue_latencies([0,.5,1,4],1)==[1,1.5,2,1]

def test_tiny_lm_does_not_see_future():
    m=load('M08');torch.manual_seed(17);model=m.TinyLM(7,6);model.eval()
    a=torch.tensor([[1,2,3,4]]);b=torch.tensor([[1,2,3,6]])
    with torch.inference_mode():oa=model(a);ob=model(b)
    assert oa.shape==(1,4,7)
    assert torch.allclose(oa[:,:3],ob[:,:3],atol=1e-6)
    assert not torch.allclose(oa[:,3],ob[:,3])

def test_feedback_rejects_boolean_and_extras():
    m=load('M09')
    assert m.validate_feedback('{"score":1,"feedback":"Explain"}')['valid']
    for text in ['{"score":true,"feedback":"x"}','{"score":3,"feedback":"x"}',
                 '{"score":1,"feedback":" "}','{"score":1,"feedback":"x","extra":0}','not json']:
        assert not m.validate_feedback(text)['valid']

def test_rrf_and_retrieval_metrics():
    m=load('M10')
    assert m.rrf([[0,1,2],[1,3,0]])==[1,0,3,2]
    assert m.metrics([1,0,2],[m.CORPUS[0]['id']],k=1)=={'recall_at_3':0.,'mrr':.5}
    assert m.metrics([0,1,2],[m.CORPUS[0]['id']],k=3)=={'recall_at_3':1.,'mrr':1.}

def test_paired_bootstrap_preserves_pair():
    m=load('M11')
    a=np.array([0,1,0,1])
    assert m.paired_bootstrap(a,a,repeats=100)==[0.,0.]
    assert m.paired_bootstrap(a,a+1,repeats=100)==[1.,1.]

def test_judge_contract_and_group_split():
    m=load('M11')
    assert m.validate_judge('{"score":2,"reason":"Correct"}') is not None
    assert m.validate_judge('{"score":true,"reason":"Correct"}') is None
    for name in ['hr','backend']:
        rows=m.read_data(name)
        dev={r['scenario_id'] for r in rows if r['split']=='development'}
        check={r['scenario_id'] for r in rows if r['split']=='check'}
        assert dev and check and not dev&check
        assert all('synthetic' in r['origin'] for r in rows)
