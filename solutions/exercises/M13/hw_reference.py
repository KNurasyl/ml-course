"""Complete measured capacity comparison, separate from the starter."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json,sys,tempfile,threading
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from labs.M13.http_lab import running,request,run
def main():
    baseline=run();rows=[]
    with tempfile.TemporaryDirectory() as folder:
        manifest=Path(folder)/'release.json';manifest.write_text(json.dumps({'schema':1,'weight':2,'bias':1,'version':'A'}))
        for capacity in (2,3):
            with running(manifest,capacity) as port:
                for clients in (1,2,4):
                    barrier=threading.Barrier(clients)
                    def one(i):barrier.wait(timeout=5);return request(port)
                    with ThreadPoolExecutor(max_workers=clients) as pool:results=list(pool.map(one,range(clients)))
                    assert all(r['status'] in (200,503) for r in results)
                    rows.append({'capacity':capacity,'clients':clients,'results':results})
    out=ROOT/'artifacts/M13-hw';out.mkdir(parents=True,exist_ok=True)
    report={'runs':rows,'recovery_verified':baseline['bad_release_rejected'],'note':'simulated 80 ms model; real HTTP, queue and process restart'}
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report
if __name__=='__main__': print(json.dumps(main(),indent=2))
