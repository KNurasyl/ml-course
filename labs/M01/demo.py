"""M01: reproducible table, group split, train-only centering, SQL grain."""
from pathlib import Path
import json,sqlite3
import numpy as np
import pandas as pd

def run():
    raw=pd.DataFrame([('a','A',2),('b','A',4),('b','A',4),('c','B',6),('d','C',10),('e','C',14)],columns=['session','person','seconds'])
    clean=raw.drop_duplicates().copy()
    assert clean.session.is_unique and clean.seconds.ge(0).all()
    train=clean[clean.person.isin(['A','B'])].copy()
    test=clean[clean.person.eq('C')].copy()
    assert not set(train.person)&set(test.person)
    artifact={'mean':float(train.seconds.mean()),'feature':'seconds','version':'M01-v1'}
    before=artifact.copy()
    transformed=test.seconds.to_numpy()-artifact['mean']
    augmented=np.r_[test.seconds.to_numpy(),100]-artifact['mean']
    assert artifact==before and np.allclose(augmented[:2],transformed)
    assert np.allclose(transformed,[6,10])
    with sqlite3.connect(':memory:') as connection:
        clean.to_sql('clean_sessions',connection,index=False)
        counts=dict(connection.execute('SELECT person,COUNT(*) FROM clean_sessions GROUP BY person').fetchall())
    assert counts=={'A':2,'B':1,'C':2}
    # Join counterexample: one session repeated three times changes AVG.
    sessions=pd.DataFrame({'session':['a','b'],'seconds':[10,30]})
    answers=pd.DataFrame({'session':['a','a','a','b']})
    joined=sessions.merge(answers,on='session',validate='one_to_many')
    assert sessions.seconds.mean()==20 and joined.seconds.mean()==15
    result={'raw_rows':len(raw),'clean_rows':len(clean),'train_rows':len(train),'test_rows':len(test),'center':artifact['mean'],'test_transformed':transformed.tolist(),'augmented_test':augmented.tolist(),'session_counts':counts,'session_mean':20,'naive_join_mean':15}
    out=Path(__file__).parent/'output';out.mkdir(exist_ok=True)
    clean.to_csv(out/'clean_sessions.csv',index=False)
    (out/'artifact.json').write_text(json.dumps(artifact,indent=2)+'\n',encoding='utf-8')
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
    return result

if __name__=='__main__':run()
