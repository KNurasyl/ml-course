from pathlib import Path
import importlib.util
import json
import sys


def checks(execute):
    a={"id":"A","mean":10,"scale":2,"weight":3,"bias":1}
    b={"id":"B","mean":12,"scale":2,"weight":3,"bias":1}
    cache={};request={"session":"s1","turn":0,"x":14}
    result=execute(request,a,cache)
    assert result=={"prediction":7.0,"release_id":"A"}
    assert execute(request,b,cache)==result, "retry must preserve original release/result"
    result["prediction"]=999
    assert execute(request,a,cache)["prediction"]==7, "do not expose mutable cache internals"
    assert execute(dict(request,session="s2"),b,cache)=={"prediction":4.0,"release_id":"B"}
    for bad in [dict(request,x=15),dict(request,x=True),dict(request,x=float('nan')),dict(request,turn=True),dict(request,extra=1)]:
        snapshot=repr(cache)
        try:execute(bad,a,cache)
        except ValueError:pass
        else:raise AssertionError("invalid input or conflicting retry accepted")
        assert repr(cache)==snapshot, "failed request changed state"
    try:execute(dict(request,turn=2),dict(a,scale=0),cache)
    except ValueError:pass
    else:raise AssertionError("invalid release accepted")
    return {"passed":True,"scope":"behavior only; explain release and failure path separately"}


if __name__=='__main__':
    if len(sys.argv)!=2:raise SystemExit('Usage: python exercises/M06/check.py path/to/your.py')
    spec=importlib.util.spec_from_file_location('learner_m06',Path(sys.argv[1]));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    print(json.dumps(checks(m.execute)))
