"""M00: explicit model artifact, finite input contract, and measured loss."""
from pathlib import Path
import json,math

def mse(actual,prediction):
    if not actual or len(actual)!=len(prediction):
        raise ValueError('nonempty matching observations required')
    return sum((p-y)**2 for y,p in zip(actual,prediction))/len(actual)

def handle(payload,artifact):
    x=payload.get('x')
    if type(x) not in (int,float) or not math.isfinite(x):
        raise ValueError('x must be a finite number, not bool')
    if artifact['architecture']!='linear_v1':
        raise ValueError('unsupported architecture')
    answer=artifact['w']*x+artifact['b']
    if not math.isfinite(answer):
        raise ValueError('prediction overflow')
    return {'prediction':answer,'model_version':artifact['version']}

def run():
    x=[0,1,2,3];y=[1,4,7,10]
    candidates={'A':{'w':3,'b':1},'B':{'w':2,'b':2}}
    losses={name:mse(y,[m['w']*v+m['b'] for v in x]) for name,m in candidates.items()}
    chosen=min(losses,key=losses.get)
    artifact={**candidates[chosen],'architecture':'linear_v1','version':'M00-HW01-v1','input_unit':'synthetic_units'}
    loaded=json.loads(json.dumps(artifact))
    answer=handle({'x':4},loaded)
    assert losses=={'A':0.0,'B':1.5}
    assert answer['prediction']==13
    for invalid in (None,True,'4',float('nan'),float('inf')):
        try:handle({'x':invalid},loaded)
        except ValueError:pass
        else:raise AssertionError(f'invalid input accepted: {invalid!r}')
    result={'mse':losses,'selected':chosen,'new_input':4,'response':answer,'invalid_inputs_rejected':5}
    out=Path(__file__).parent/'output';out.mkdir(exist_ok=True)
    (out/'artifact.json').write_text(json.dumps(artifact,indent=2)+'\n',encoding='utf-8')
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return result

if __name__=='__main__':run()
