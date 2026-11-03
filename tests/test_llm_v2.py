"""New course checks must accept correct work and reject representative semantic faults."""
from pathlib import Path
import importlib.util, types
import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
def load(relative):
    p=ROOT/relative;s=importlib.util.spec_from_file_location('v2_'+str(relative).replace('/','_').replace('.','_'),p)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

@pytest.mark.parametrize('module',['M07','M08','M09','M10','M12'])
def test_exercise_solution_passes(module):
    checker=load(f'exercises/{module}/check.py');solution=load(f'solutions/exercises/{module}/solution.py')
    assert checker.run_checks(solution)['passed']

@pytest.mark.parametrize('module',['M07','M08','M09','M10','M12'])
def test_starter_is_an_unfinished_student_task(module):
    checker=load(f'exercises/{module}/check.py');starter=load(f'exercises/{module}/starter.py')
    with pytest.raises(NotImplementedError):checker.run_checks(starter)

def test_validation_checker_detects_train_mode():
    checker=load('exercises/M07/check.py')
    wrong=types.SimpleNamespace(validation_loss=lambda model,x,y,c:float(c(model(x),y).detach()))
    with pytest.raises(AssertionError):checker.run_checks(wrong)

def test_attention_checker_detects_missing_mask():
    checker=load('exercises/M08/check.py')
    def wrong(q,k,v,valid):
        a=np.ones((q.shape[0],q.shape[1],q.shape[1]))/q.shape[1];return a@v,a
    with pytest.raises(AssertionError):checker.run_checks(types.SimpleNamespace(causal_attention=wrong))

def test_feedback_checker_detects_made_up_score():
    checker=load('exercises/M09/check.py')
    wrong=types.SimpleNamespace(respond=lambda raw:dict(status='ok',feedback='Fine',score=2,fallback_used=False))
    with pytest.raises(AssertionError):checker.run_checks(wrong)

def test_retrieval_checker_detects_undefined_as_zero():
    checker=load('exercises/M10/check.py');solution=load('solutions/exercises/M10/solution.py')
    def wrong(*args):
        r=solution.retrieval_scores(*args)
        if r['recall'] is None:r['recall']=0.
        return r
    with pytest.raises(AssertionError):checker.run_checks(types.SimpleNamespace(retrieval_scores=wrong))

def test_sft_checker_detects_double_shift():
    checker=load('exercises/M12/check.py');solution=load('solutions/exercises/M12/solution.py')
    def wrong(*args):
        r=solution.training_example(*args);r['labels']=r['labels'][1:]+[-100];return r
    with pytest.raises(AssertionError):checker.run_checks(types.SimpleNamespace(training_example=wrong))

def test_empty_relevance_no_division_by_zero():
    m=load('labs/M10/demo.py')
    assert m.metrics([0,1],[])=={'recall_at_3':None,'mrr':None}
    assert m.mean_defined([1.,0.,None])==.5 and m.mean_defined([None,None]) is None
    for k in [0,-1,True]:
        with pytest.raises(ValueError):m.metrics([0],['x'],k)

def test_terminal_fallback_is_executed():
    m=load('labs/M09/demo.py')
    r=m.feedback_or_abstain('not JSON')
    assert r['status']=='needs_review' and r['score'] is None and r['feedback'] and r['fallback_used']
    assert m.feedback_or_abstain('{"feedback":"Ask for evidence","score":1}')['status']=='ok'

def test_normalization_matches_torch_not_global_axis():
    m=load('labs/M08/forward_trace.py');x=torch.tensor([[1.,3.],[100.,104.]])
    expected=torch.nn.LayerNorm(2)(x)
    assert torch.allclose(m.manual_norm(x),expected)
    assert not torch.allclose((x-x.mean())/x.std(unbiased=False),expected)

def test_lora_frozen_base_and_zero_initialization_trap():
    m=load('labs/M12/sft.py');layer=m.LowRankLinear(torch.nn.Linear(3,2),rank=2,alpha=2)
    x=torch.ones(2,3);base=layer.base.weight.detach().clone()
    layer(x).sum().backward()
    assert layer.base.weight.grad is None and layer.b.grad.abs().sum()>0
    assert torch.equal(base,layer.base.weight)
    with torch.no_grad():layer.a.zero_();layer.b.zero_()
    layer.zero_grad();layer(x).sum().backward()
    assert layer.a.grad.abs().sum()==0 and layer.b.grad.abs().sum()==0

def test_russian_corpus_versions_groups_and_null_metric():
    m=load('labs/M10/diagnostic_ru.py')
    assert len(m.documents())==48 and len({x['document_id'] for x in m.documents()})==24
    assert all(c['version']==2 for c in m.chunks())
    assert any(c['version']==1 for c in m.chunks(include_obsolete=True))
    ids=[q['id'] for q in m.QUERIES];assert len(ids)==len(set(ids))
    assert {q['split'] for q in m.QUERIES}=={'development','check'}
    r=m.score_result(['x'],[],abstained=True);assert r['recall'] is None and r['correct_abstention']
    assert len(m.chunks('words'))>len(m.chunks('paragraph'))
