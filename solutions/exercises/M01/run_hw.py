from pathlib import Path
import sys,json,importlib.util
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location("reference",Path(__file__).with_name("solution.py"))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
import sqlite3
records=[{'session_id':'a','person':'A','value':2},{'session_id':'b','person':'A','value':None},{'session_id':'c','person':'B','value':8},{'session_id':'c','person':'B','value':8},{'session_id':'d','person':'C','value':None},{'session_id':'e','person':'C','value':1000}]
unique=m.deduplicate(records)
train=[r['value'] for r in unique if r['person'] in ['A','B']]
test=[r['value'] for r in unique if r['person']=='C']
with sqlite3.connect(':memory:') as connection:
    connection.execute('CREATE TABLE sessions(session_id TEXT PRIMARY KEY,person TEXT,value REAL)')
    connection.executemany('INSERT INTO sessions VALUES(?,?,?)',[(r['session_id'],r['person'],r['value']) for r in unique])
    counts=connection.execute('SELECT person,COUNT(*) FROM sessions GROUP BY person ORDER BY person').fetchall()
report={'A':m.prepare(train,test),'sql_counts':counts,'stage_rows':[len(records),len(unique)],'B':m.prepare([1,None,9],[None,100]),'C':{'session_mean':3,'person_mean':5}}
assert report['A']['mean']==5 and counts==[('A',2),('B',1),('C',2)]
assert m.prepare(train,[None,-1000])['mean']==5
try:m.deduplicate(unique+[{'session_id':'c','person':'B','value':9}])
except ValueError:report['conflict_rejected']=True
else:raise AssertionError('conflict must be rejected')

out=ROOT/"artifacts/foundation-hw"/Path(__file__).parent.name
out.mkdir(exist_ok=True,parents=True)
(out/"report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps(report,indent=2))
