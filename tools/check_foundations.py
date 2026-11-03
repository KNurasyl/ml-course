"""Execute all authored foundation examples and CPU labs. No learner grading."""
from pathlib import Path
import contextlib,io,json,re,subprocess,sys,time

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def main():
    results=[]
    for mid in [f'M{i:02}' for i in range(6)]:
        paths=sorted((ROOT/'course'/'lessons').glob(f'{mid}-L*.md'))
        paths.append(ROOT/'solutions'/'modules'/f'{mid}.md')
        for path in paths:
            text=path.read_text(encoding='utf-8')
            for n,code in enumerate(re.findall(r'```python\s*\n(.*?)```',text,re.S),1):
                output=io.StringIO();start=time.perf_counter()
                try:
                    with contextlib.redirect_stdout(output):
                        exec(compile(code,str(path)+f':block{n}','exec'),{'__name__':'__course_example__'})
                    record={'path':str(path.relative_to(ROOT)),'block':n,'status':'passed','stdout':output.getvalue(),'seconds':round(time.perf_counter()-start,3)}
                except Exception as error:
                    record={'path':str(path.relative_to(ROOT)),'block':n,'status':'failed','error':repr(error),'stdout':output.getvalue()}
                results.append(record)
    for mid in [f'M{i:02}' for i in range(6)]:
        lab=ROOT/'labs'/mid/'demo.py'
        if not lab.exists():
            results.append({'path':f'labs/{mid}/demo.py','status':'missing'})
            continue
        start=time.perf_counter()
        run=subprocess.run([sys.executable,str(lab)],cwd=ROOT,capture_output=True,text=True,encoding='utf-8')
        results.append({'path':str(lab.relative_to(ROOT)),'status':'passed' if run.returncode==0 else 'failed','stdout':run.stdout,'stderr':run.stderr,'returncode':run.returncode,'seconds':round(time.perf_counter()-start,3)})
    failures=[r for r in results if r['status']!='passed']
    destination=ROOT/'metadata'/'foundations-verification.json'
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps({'results':results,'passed':len(results)-len(failures),'failures':failures},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'passed':len(results)-len(failures),'failures':failures},ensure_ascii=False))
    return bool(failures)

if __name__=='__main__':
    raise SystemExit(main())
