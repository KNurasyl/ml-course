"""Validate the student's behavior, not filenames. No model or internet required."""
import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

def check(module):
    s={'topic':0,'turn':0,'clarifications':0,'topics':5};original=dict(s)
    a=module.transition(s,'Пока не знаю.',2)
    assert a==dict(s,turn=1,clarifications=1) and s==original
    b=module.transition(a,'Пока не знаю.',2);c=module.transition(b,'Пока не знаю.',2)
    assert (c['topic'],c['turn'],c['clarifications'])==(1,3,0)
    assert module.transition(s,'Не хочу отвечать.',2)['topic']==1
    assert module.transition(s,'Коротко',0)['topic']==1
    count=0
    while s['topic']<5:s=module.transition(s,'Не знаю.',2);count+=1;assert count<=15
    assert count==15
    try:module.transition(s,'Ещё ответ',2)
    except ValueError:pass
    else:raise AssertionError('finished session accepted an answer')
    return True

def load(path):
    spec=importlib.util.spec_from_file_location('student_exercise',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('submission',type=Path);parser.add_argument('--self-test',action='store_true');args=parser.parse_args()
    check(load(args.submission))
    if args.self_test:
        bad=SimpleNamespace(transition=lambda state, answer, limit: dict(state,turn=state['turn']+1,topic=state['topic']+1,clarifications=0))
        try:check(bad)
        except (AssertionError,ValueError):pass
        else:raise AssertionError('checker failed to detect deliberately wrong implementation')
    print(json.dumps({'behavior_passed':True,'mutant_rejected':args.self_test,'mastery':'not_assessed'}))
