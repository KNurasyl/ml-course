"""Persistence mechanics; fixture feedback is deliberately not LLM evidence."""
import json
from pathlib import Path
import subprocess
import sys
from contextlib import contextmanager
from urllib.request import Request, urlopen
from urllib.error import HTTPError

ROOT = Path(__file__).resolve().parents[1]


def child(code, path):
    return subprocess.run([sys.executable, '-c', code, str(path)], cwd=ROOT,
                          text=True, capture_output=True, check=True).stdout


def test_restore_in_fresh_process_and_reject_duplicate(tmp_path):
    child("""
import sys
from solutions.exercises.M16.durable_app import configure, app
configure(sys.argv[1]); interview=app.Interview('hr'); app.SESSIONS[interview.id]=interview
interview.prepare('Test answer', 0)
interview.commit('Test answer', 'Fixture feedback, not a model output', [], 0)
""", tmp_path)
    # Interrupted temporary output must not replace the last accepted snapshot.
    (tmp_path / 'incomplete.json.tmp').write_text('{')
    output = child("""
import sys,json
from solutions.exercises.M16.durable_app import configure
sessions=configure(sys.argv[1]); interview=next(iter(sessions.values()))
assert interview.turn==1 and interview.clarification and len(interview.history)==1
try: interview.prepare('Duplicate answer',0)
except ValueError: pass
else: raise AssertionError('accepted stale turn')
interview.prepare('Next answer',1)
interview.commit('Next answer','Fixture feedback',[],0)
print(json.dumps({'turn':interview.turn,'topic':interview.topic_index}))
""", tmp_path)
    assert json.loads(output) == {'turn': 2, 'topic': 1}


def test_incompatible_snapshot_and_changed_policy_fail_closed(tmp_path):
    child("""
import sys
from solutions.exercises.M16.durable_app import configure,app
configure(sys.argv[1]); i=app.Interview('backend')
i.commit('Test answer','Fixture feedback',[],0)
""", tmp_path)
    path = next(tmp_path.glob('*.json'))
    original = json.loads(path.read_text(encoding='utf-8'))
    for changed in ({**original, 'schema': 2},
                    {**original, 'state': {**original['state'], 'topic': 3}}):
        path.write_text(json.dumps(changed), encoding='utf-8')
        result = subprocess.run([sys.executable, '-c',
            'import sys; from solutions.exercises.M16.durable_app import configure; configure(sys.argv[1])', str(tmp_path)],
            cwd=ROOT, text=True, capture_output=True)
        assert result.returncode != 0
        assert 'ValueError' in result.stderr


def test_failed_write_does_not_accept_turn(tmp_path):
    child("""
import sys
from solutions.exercises.M16 import durable_app as d
d.configure(sys.argv[1]); i=d.app.Interview('hr')
def failed(*args): raise OSError('simulated disk full')
d.save=failed
try: i.commit('Test answer','Fixture feedback',[],0)
except OSError: pass
else: raise AssertionError('failure hidden')
assert i.turn==0 and i.history==[] and i.topic_index==0
""", tmp_path)


@contextmanager
def http_process(path):
    # Only the model is a fixture; TCP, HTTP, disk and restart are real.
    code = """
import sys
from solutions.exercises.M16.durable_app import configure,app
class FixtureModel:
    model=None
    last_generation={'source':'fixture; not LLM evaluation'}
    def generate_choice(self,messages,choices): return choices[0]
app.MODEL=FixtureModel()
configure(sys.argv[1])
server=app.ThreadingHTTPServer(('127.0.0.1',0),app.Handler)
print(server.server_port,flush=True)
server.serve_forever()
"""
    process = subprocess.Popen([sys.executable, '-c', code, str(path)], cwd=ROOT,
                               text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        port = int(process.stdout.readline().strip())
        yield f'http://127.0.0.1:{port}'
    finally:
        process.terminate()
        process.wait(timeout=10)
        process.stdout.close()
        process.stderr.close()


def request(base, route, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = Request(base+route, data=data, headers={'Content-Type':'application/json'})
    try:
        with urlopen(req, timeout=5) as response:
            return response.status, json.load(response)
    except HTTPError as error:
        return error.code, json.load(error)


def test_http_restart_recovers_session_and_duplicate_turn(tmp_path):
    with http_process(tmp_path) as base:
        status, session = request(base, '/start', {'mode':'hr'})
        assert status == 200
        session_id = session['id']
        status, first = request(base, '/answer', {'id':session_id, 'turn':0,
                                                'answer':'I implemented a retry policy.'})
        assert status == 200 and first['turn'] == 1
    with http_process(tmp_path) as base:
        status, restored = request(base, '/session/'+session_id)
        assert status == 200 and restored == first
        status, _ = request(base, '/answer', {'id':session_id, 'turn':0,
                                            'answer':'Duplicate response.'})
        assert status == 400
        _, after = request(base, '/session/'+session_id)
        assert after == restored
