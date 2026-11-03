"""Versioned preprocessing, queue and rollback: deterministic CPU exercise."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[2]

def predict(x, release):
    if release['scale'] <= 0:
        raise ValueError('scale must be positive')
    return release['weight'] * (x-release['mean']) / release['scale'] + release['bias']

def queue_latencies(arrivals, service):
    end, values = 0., []
    for arrived in arrivals:
        end = max(end, arrived) + service
        values.append(end-arrived)
    return values

def main():
    stable = dict(id='release-A', mean=10, scale=2, weight=3, bias=1)
    broken = dict(stable, id='release-broken', mean=12)
    good, bad = predict(14, stable), predict(14, broken)
    assert good == 7 and bad == 4
    active = broken
    events = [{'event':'activated','release_id':active['id'],'control':predict(14,active)}]
    if abs(predict(14, active)-7) > 1e-6:
        active = stable
        events.append({'event':'rolled_back','release_id':active['id'],'control':predict(14,active)})
    assert predict(14,active) == 7
    latency = queue_latencies([0,1,2,3],2)
    assert latency == [2,3,4,5]
    turns = {}
    def accept(key, payload):
        if key in turns and turns[key] != payload:
            raise ValueError('conflicting idempotency key')
        turns[key] = payload
    accept('S7:3','answer');accept('S7:3','answer')
    assert len(turns) == 1
    try:
        accept('S7:3','changed')
        raise AssertionError('conflict was accepted')
    except ValueError:
        pass
    report={'stable_prediction':good,'broken_prediction':bad,'restored_prediction':predict(14,active),
            'queue_latencies':latency,'logical_turns':len(turns),'events':events}
    out=ROOT/'artifacts/M06';out.mkdir(parents=True,exist_ok=True)
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
