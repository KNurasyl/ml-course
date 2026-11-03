"""Audit-driven checks: procedure errors, independent variants and real student APIs."""
from pathlib import Path
import importlib.util
import json
import re
import subprocess
import sys
import types
import numpy as np
import pytest
from labs.M04.bridges import binomial_tail,plan,paired_plan
from labs.M05.pipeline import make_data,split_groups,make_pipeline,check_fit_rows,audited_cv

ROOT=Path(__file__).resolve().parents[1]


def load(path):
    spec=importlib.util.spec_from_file_location('test_loaded_module',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('module',[f'M{i:02}' for i in range(6)])
def test_reference_passes_student_contract(module):
    checker=load(ROOT/f'exercises/{module}/check.py')
    reference=load(ROOT/f'solutions/exercises/{module}/solution.py')
    assert len(checker.check(reference))>=4


@pytest.mark.parametrize('module',[f'M{i:02}' for i in range(6)])
def test_unfinished_starter_does_not_get_credit(module):
    checker=load(ROOT/f'exercises/{module}/check.py')
    starter=load(ROOT/f'exercises/{module}/starter.py')
    with pytest.raises(NotImplementedError):checker.check(starter)


def test_wrong_gradient_rejected_against_finite_difference():
    checker=load(ROOT/'exercises/M03/check.py')
    reference=load(ROOT/'solutions/exercises/M03/solution.py')
    def reversed_gradient(*args):return tuple(-v for v in reference.gradient(*args))
    with pytest.raises(AssertionError):checker.check(types.SimpleNamespace(gradient=reversed_gradient))


def test_wrong_denominator_recall_rejected():
    checker=load(ROOT/'exercises/M02/check.py')
    reference=load(ROOT/'solutions/exercises/M02/solution.py')
    bad=types.SimpleNamespace(search=reference.search,recall=lambda found,relevant:len(set(found)&set(relevant))/len(found))
    with pytest.raises(AssertionError):checker.check(bad)


def test_train_test_imputation_leak_rejected():
    checker=load(ROOT/'exercises/M01/check.py')
    reference=load(ROOT/'solutions/exercises/M01/solution.py')
    def leaked(train,test):
        mean=float(np.mean([v for v in train+test if v is not None]))
        return {'mean':mean,'train':train,'test':test}
    with pytest.raises(AssertionError):checker.check(types.SimpleNamespace(prepare=leaked,deduplicate=reference.deduplicate))


def test_threshold_checker_rejects_ignored_error_cost():
    checker=load(ROOT/'exercises/M05/check.py')
    reference=load(ROOT/'solutions/exercises/M05/solution.py')
    def ignores_cost(y,p,thresholds=(.25,.5,.75),fn_cost=2):
        return reference.choose_threshold(y,p,thresholds,fn_cost=1)
    wrong=types.SimpleNamespace(split_records=reference.split_records,build_pipeline=reference.build_pipeline,choose_threshold=ignores_cost)
    with pytest.raises(AssertionError):checker.check(wrong)


def test_boundary_group_is_excluded_not_split():
    df=make_data(2026,difficult=True);s=split_groups(df)
    assert {k:len(v) for k,v in s.items()}=={'train':141,'validation':47,'test':48,'excluded':4}
    assert set(df.loc[s['excluded']].user_id)=={'u070'}
    parts=[df.loc[s[k]] for k in ['train','validation','test']]
    assert max(parts[0].day)<min(parts[1].day)<min(parts[2].day)
    for i in range(3):
        for j in range(i+1,3):assert not set(parts[i].user_id)&set(parts[j].user_id)


def test_actual_global_fit_rejected_even_without_accuracy_threshold():
    df=make_data(120);train=df.iloc[:100]
    correct=make_pipeline().fit(train,train.is_ready);check_fit_rows(correct,train.index)
    leaked=make_pipeline().fit(df,df.is_ready)
    with pytest.raises(AssertionError,match='fit row leakage'):check_fit_rows(leaked,train.index)
    assert all(row['global_fit_detected'] for row in audited_cv(df))


def test_future_target_column_does_not_influence_output():
    df=make_data(2026,difficult=True);parts=split_groups(df)
    train=df.loc[parts['train']];test=df.loc[parts['test']]
    m=make_pipeline().fit(train,train.is_ready);p=m.predict_proba(test)
    changed=test.copy();changed['review_score_after']=10000-changed.review_score_after
    assert np.allclose(p,m.predict_proba(changed))
    assert np.isfinite(p).all()
    with pytest.raises(ValueError):m.predict_proba(test.drop(columns='track'))


def test_exact_probability_and_simulation_agree_with_declared_error():
    assert binomial_tail(4,3)==5/16
    rows=plan()
    for row in rows:
        assert row['actual_alpha']<=.05
        assert abs(row['simulated_power']-row['exact_power'])<.02
    assert next(row['n'] for row in rows if row['exact_power']>=.8)==200
    alt=paired_plan(improve=.1,regress=.05,sizes=(100,200,400,800),seed=75)
    assert next(row['n'] for row in alt if row['exact_power']>=.8)==400
    assert all(abs(row['exact_power']-row['simulated_power'])<.02 for row in alt)
    null=paired_plan(improve=.1,regress=.1)
    assert all(row['exact_power']<=.05+1e-12 for row in null)


@pytest.mark.parametrize('lesson',['M04-L06','M04-L07','M04-L08','M05-L06','M05-L07'])
def test_new_lesson_python_blocks_execute_independently(lesson):
    source=(ROOT/f'course/lessons/{lesson}.md').read_text(encoding='utf-8')
    blocks=re.findall(r'```python\n(.*?)```',source,re.S)
    assert blocks
    for block in blocks:
        process=subprocess.run([sys.executable,'-c',block],cwd=ROOT,capture_output=True,text=True,timeout=90)
        assert process.returncode==0,process.stdout+process.stderr


def test_demo_and_homework_require_different_data_and_decisions():
    hw=make_data(2026,difficult=True);demo=make_data(120)
    assert not hw.equals(demo)
    assert split_groups(demo)['excluded']==[] and len(split_groups(hw)['excluded'])==4
    assert 'rust' in set(hw.track) and 'rust' not in set(demo.track)
    assert '3D_cosine' in load(ROOT/'exercises/M02/check.py').check(load(ROOT/'solutions/exercises/M02/solution.py'))


def test_classification_artifact_recovers_in_a_new_process(tmp_path):
    import joblib
    data=make_data(120);model=make_pipeline().fit(data.iloc[:144],data.is_ready.iloc[:144])
    output=tmp_path/'model.joblib';joblib.dump(model,output)
    data.iloc[144:150].to_csv(tmp_path/'input.csv');np.save(tmp_path/'expected.npy',model.predict_proba(data.iloc[144:150]))
    code="""import joblib,numpy as np,pandas as pd,sys
from pathlib import Path
p=Path(sys.argv[1]);m=joblib.load(p/'model.joblib');x=pd.read_csv(p/'input.csv',index_col='row_id')
assert np.allclose(m.predict_proba(x),np.load(p/'expected.npy'),atol=1e-12)
"""
    process=subprocess.run([sys.executable,'-c',code,str(tmp_path)],cwd=ROOT,capture_output=True,text=True,timeout=60)
    assert process.returncode==0,process.stderr
