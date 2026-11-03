"""Validate the student's behavior, not filenames. No model or internet required."""
import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

def check(module):
    events=[('begin',5,None),('receive',5,'A'),('cancel',5,None),('receive',5,'B'),('play',0,None),
            ('begin',7,None),('receive',7,'C'),('play',0,None)]
    assert module.playable(events)==['C']
    assert module.playable([('begin',1,None),('receive',1,'A'),('play',0,None),('receive',1,'B'),('play',0,None)])==['A','B']
    assert module.playable([('begin',1,None),('receive',1,'A'),('begin',2,None),('play',0,None)])==[]
    return True

def load(path):
    spec=importlib.util.spec_from_file_location('student_exercise',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('submission',type=Path);parser.add_argument('--self-test',action='store_true');args=parser.parse_args()
    check(load(args.submission))
    if args.self_test:
        bad=SimpleNamespace(playable=lambda events: [e[2] for e in events if e[0]=='receive'])
        try:check(bad)
        except (AssertionError,ValueError):pass
        else:raise AssertionError('checker failed to detect deliberately wrong implementation')
    print(json.dumps({'behavior_passed':True,'mutant_rejected':args.self_test,'mastery':'not_assessed'}))
