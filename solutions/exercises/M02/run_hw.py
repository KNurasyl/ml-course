from pathlib import Path
import sys,json,importlib.util
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location("reference",Path(__file__).with_name("solution.py"))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
ids=['V','U','W','X','Y'];D=[[2,0,0],[1,0,0],[0,1,0],[0,0,1],[-1,0,0]]
rows=[]
for q,gold in [([1,1,0],['U','W']),([0,0,1],['X','W'])]:
    ranked=m.search(ids,D,q,3);found=[r[0] for r in ranked]
    rows.append({'top3':ranked,'recall':{str(k):m.recall(found[:k],gold) for k in [1,2,3]}})
    assert m.search(ids[::-1],D[::-1],q,3)==ranked
ranked_b=m.search(ids+['Z'],D+[[0,2,0]],[1,1,0],3)
report={'A':rows,'mean_recall':[sum(row['recall'][str(k)] for row in rows)/2 for k in [1,2,3]],'B':{'top3':ranked_b,'recall_at_3':m.recall([r[0] for r in ranked_b],['U','W','Z'])},'C':{'recall':m.recall(['U','U','Y'],['U','W','X'])}}
assert report['mean_recall']==[.5,.5,.75]
assert report['B']['recall_at_3']==2/3

out=ROOT/"artifacts/foundation-hw"/Path(__file__).parent.name
out.mkdir(exist_ok=True,parents=True)
(out/"report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps(report,indent=2))
