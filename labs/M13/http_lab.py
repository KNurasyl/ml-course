"""Real local HTTP concurrency, queue deadlines and process-level recovery."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from urllib.request import Request, urlopen
from urllib.error import HTTPError

ROOT = Path(__file__).resolve().parents[2]


def load_release(path):
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    if value.get('schema') != 1 or value.get('weight') != 2 or value.get('bias') != 1:
        raise ValueError('release failed schema/canary')
    return value


def serve(manifest, capacity_size=2):
    release = load_release(manifest)
    capacity, worker = threading.BoundedSemaphore(capacity_size), threading.Lock()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_): pass
        def reply(self, code, payload):
            data = json.dumps(payload).encode()
            self.send_response(code); self.send_header('Content-Length', str(len(data)))
            self.end_headers(); self.wfile.write(data)
        def do_POST(self):
            start = time.perf_counter()
            try:
                size = int(self.headers.get('Content-Length', 0))
                if not 0 < size <= 1024: raise ValueError('body size')
                data = json.loads(self.rfile.read(size))
                if set(data) != {'x','wait_ms'} or type(data['x']) not in (int,float) or not math.isfinite(data['x']) or type(data['wait_ms']) not in (int,float) or not 0 <= data['wait_ms'] <= 1000:
                    raise ValueError('contract')
            except (ValueError,TypeError,KeyError):
                return self.reply(400, {'error':'contract'})
            if not capacity.acquire(False): return self.reply(503, {'error':'queue_full'})
            locked = False
            try:
                locked = worker.acquire(timeout=data['wait_ms']/1000)
                waited = time.perf_counter()-start
                if not locked: return self.reply(504, {'error':'queue_deadline','queue_ms':waited*1000})
                work_start = time.perf_counter()
                time.sleep(.08)  # explicitly simulated compute, real HTTP and queue
                value = release['weight']*data['x']+release['bias']
                self.reply(200, {'prediction':value,'release':release['version'],
                    'queue_ms':waited*1000,'compute_ms':(time.perf_counter()-work_start)*1000})
            finally:
                if locked: worker.release()
                capacity.release()
    server = ThreadingHTTPServer(('127.0.0.1',0), Handler)
    print(json.dumps({'port':server.server_port,'release':release['version']}),flush=True)
    try: server.serve_forever()
    finally: server.server_close()


@contextmanager
def running(manifest, capacity=2):
    process = subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--serve',str(manifest),'--capacity',str(capacity)],
        stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,encoding='utf-8')
    try:
        line=process.stdout.readline()
        if not line:
            raise RuntimeError(process.stderr.read()[-500:])
        yield json.loads(line)['port']
    finally:
        if process.poll() is None: process.terminate()
        process.wait(timeout=5)
        process.stdout.close(); process.stderr.close()


def request(port, x=2, wait_ms=500):
    started=time.perf_counter()
    req=Request(f'http://127.0.0.1:{port}/predict',data=json.dumps({'x':x,'wait_ms':wait_ms}).encode(),method='POST')
    try:
        with urlopen(req,timeout=3) as result: code,body=result.status,json.loads(result.read())
    except HTTPError as error: code,body=error.code,json.loads(error.read())
    return {'status':code,'body':body,'client_ms':(time.perf_counter()-started)*1000}


def run():
    out=ROOT/'artifacts/M13-http';out.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='course-release-') as folder:
        manifest=Path(folder)/'release.json'
        stable={'schema':1,'weight':2,'bias':1,'version':'A'}
        manifest.write_text(json.dumps(stable),encoding='utf-8')
        with running(manifest) as port:
            batches={}
            for clients in (1,2,4):
                barrier=threading.Barrier(clients)
                def one(_): barrier.wait(timeout=3);return request(port)
                with ThreadPoolExecutor(max_workers=clients) as pool:
                    rows=list(pool.map(one,range(clients)))
                assert all(r['status'] in (200,503) for r in rows)
                assert all(r['body']['prediction']==5 for r in rows if r['status']==200)
                batches[str(clients)]=rows
            barrier=threading.Barrier(2)
            def deadline(_): barrier.wait(timeout=3);return request(port,wait_ms=10)
            with ThreadPoolExecutor(max_workers=2) as pool: timed=list(pool.map(deadline,range(2)))
            assert sorted(r['status'] for r in timed)==[200,504]
        manifest.write_text(json.dumps(dict(stable,schema=2,version='bad')),encoding='utf-8')
        rejected=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--serve',str(manifest)],capture_output=True,timeout=5)
        assert rejected.returncode!=0
        # Restore a complete persisted artifact, then start a NEW server process.
        manifest.write_text(json.dumps(stable),encoding='utf-8')
        with running(manifest) as port: restored=request(port,x=7)
        assert restored['body']==dict(restored['body'],prediction=15,release='A') and restored['status']==200
    report={'clients':batches,'deadline':timed,'bad_release_rejected':True,'restart_restored':restored,
        'scope':'real localhost HTTP/processes; model compute is an explicit 80 ms sleep, not measured LLM throughput'}
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--serve');parser.add_argument('--capacity',type=int,choices=range(1,5),default=2);args=parser.parse_args()
    if args.serve: serve(args.serve,args.capacity)
    else: print(json.dumps(run(),indent=2))
