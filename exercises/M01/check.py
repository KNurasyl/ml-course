from pathlib import Path
import importlib.util
import sys
import json
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
def load(path):
    spec=importlib.util.spec_from_file_location("learner_submission",path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module
def rejected(call):
    try:call()
    except (ValueError,TypeError):return
    raise AssertionError("invalid input was accepted")
def check(m):
    a=m.prepare([2,None,8],[None,1000])
    b=m.prepare([2,None,8],[None,-1000])
    assert a['mean']==b['mean']==5 and a['train']==[2,5,8] and a['test']==[5,1000]
    rejected(lambda:m.prepare([None,None],[5]))
    rejected(lambda:m.prepare([1,float('nan')],[]))
    rows=[{'session_id':'s1','value':2},{'session_id':'s1','value':2},{'session_id':'s2','value':9}]
    assert len(m.deduplicate(rows))==2
    rejected(lambda:m.deduplicate([{'session_id':'s','value':2},{'session_id':'s','value':3}]))
    return ['train_only_fit','missing_values','conflicting_duplicate','grain']

if __name__=="__main__":
    if len(sys.argv)!=2:raise SystemExit("usage: check.py path/to/your_solution.py")
    try:
        checks=check(load(sys.argv[1]))
    except Exception as error:
        print(json.dumps({"passed":False,"error":type(error).__name__,"message":str(error)},ensure_ascii=True))
        raise SystemExit(1)
    print(json.dumps({"passed":True,"checks":checks,"understanding":"requires your explanation and a new case"},ensure_ascii=True))
