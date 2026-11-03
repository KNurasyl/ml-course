"""Reference for a single-process local state snapshot, not a multiwriter database."""
import json
import os
from pathlib import Path

def save(path, state):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    payload={'schema':1,'state':state}
    temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w',encoding='utf-8') as stream:
        json.dump(payload,stream,ensure_ascii=False);stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)

def restore(path):
    payload=json.loads(Path(path).read_text(encoding='utf-8'))
    if set(payload)!={'schema','state'} or payload['schema']!=1:raise ValueError('unsupported snapshot')
    state=payload['state']
    required={'mode','turn','topic','clarifications','history'}
    if not isinstance(state,dict) or set(state)!=required:raise ValueError('state schema')
    if state['mode'] not in ('hr','backend') or any(type(state[k]) is not int or state[k]<0 for k in ('turn','topic','clarifications')):
        raise ValueError('state values')
    if not isinstance(state['history'],list) or len(state['history'])!=state['turn']:raise ValueError('history mismatch')
    return state

def accept_turn(state, turn):
    if type(turn) is not int or turn != state['turn']:raise ValueError('stale turn')
    return True
