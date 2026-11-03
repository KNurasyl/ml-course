"""Isolated CPU float32/int8 runs; parent samples process RSS, no speed promise."""
from pathlib import Path
import argparse, gc, importlib.util, io, json, subprocess, sys, time
import psutil

ROOT=Path(__file__).resolve().parents[2]
CASES=['Explain SQL transaction atomicity in one sentence.',
       'What does an idempotent operation mean?',
       'Что означает индекс базы данных?']

def worker(kind,path):
    import torch
    spec=importlib.util.spec_from_file_location('m09_benchmark_demo',Path(__file__).with_name('demo.py'))
    d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
    tok,model,load_seconds=d.load_local()
    convert_seconds=0.
    if kind=='int8':
        started=time.perf_counter()
        torch.ao.quantization.quantize_dynamic(model,{torch.nn.Linear},dtype=torch.qint8,inplace=True)
        convert_seconds=time.perf_counter()-started
    packed=io.BytesIO();torch.save(model.state_dict(),packed);state_bytes=packed.tell();del packed;gc.collect()
    rss_after_model=psutil.Process().memory_info().rss
    warm=d.generate(tok,model,[{'role':'user','content':'Hello'}],4)
    rows=[]
    for q in CASES:
        rows.append({'prompt':q,'output':d.generate(tok,model,[{'role':'user','content':q}],24)})
    result={'config':kind,'repo_id':d.REPO,'revision':d.REVISION,'threads':torch.get_num_threads(),
            'load_seconds':load_seconds,'conversion_seconds':convert_seconds,
            'serialized_state_bytes':state_bytes,'rss_after_model_bytes':rss_after_model,
            'warmup':warm,'cases':rows,
            'note':'RSS includes interpreter/libraries/model; serialized state is not peak RAM. TTFT not measured.'}
    Path(path).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker',choices=['float32','int8']);parser.add_argument('--out')
    args=parser.parse_args()
    if args.worker:worker(args.worker,args.out);return
    out=ROOT/'artifacts/M09/benchmark';out.mkdir(parents=True,exist_ok=True);reports=[]
    for kind in ['float32','int8']:
        result_path=out/f'{kind}.json';log_path=out/f'{kind}.log'
        with log_path.open('w',encoding='utf-8') as log:
            p=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker',kind,'--out',str(result_path)],stdout=log,stderr=subprocess.STDOUT)
            peak=0;sampled=psutil.Process(p.pid)
            while p.poll() is None:
                try:
                    # Windows venv python.exe may be a small redirector spawning the real interpreter.
                    # Observe the whole worker tree, not only the launcher PID.
                    processes=[sampled]+sampled.children(recursive=True)
                    rss=0
                    for process in processes:
                        try:rss+=process.memory_info().rss
                        except psutil.NoSuchProcess:pass
                    peak=max(peak,rss)
                except psutil.NoSuchProcess:pass
                time.sleep(.03)
            if p.returncode:raise RuntimeError(f'{kind} failed; inspect {log_path}')
        row=json.loads(result_path.read_text(encoding='utf-8'));row['sampled_peak_rss_bytes']=peak;row['rss_sampling_ms']=30
        assert peak>=row['rss_after_model_bytes'],'Sampler missed worker process; do not report launcher RSS as model memory'
        reports.append(row)
    assert reports[1]['serialized_state_bytes']<reports[0]['serialized_state_bytes']
    report={'configs':reports,'quality_review_required':['Factual correctness','Russian language','Completeness'],
            'comparison':'Same model revision, prompts, max_new_tokens, greedy, threads; separate fresh processes.',
            'limits':'Sampled worker-tree RSS includes Windows launcher and may double-count shared pages; it is an operational upper sum, not unique private memory. Sampling may miss short spikes. Cold init includes float32 conversion for int8; no on-disk int8 loader benchmark.'}
    (out/'comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps([{k:r[k] for k in ['config','serialized_state_bytes','sampled_peak_rss_bytes','rss_after_model_bytes']} for r in reports]))

if __name__=='__main__':main()
