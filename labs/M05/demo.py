"""M05: baseline, validation-only selection and one final synthetic test."""
from pathlib import Path
import hashlib,json
import numpy as np
import sklearn
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error

def run():
    rng=np.random.default_rng(42)
    X=rng.uniform(-3,3,size=(300,1));y=X[:,0]**2+rng.normal(0,.2,size=300)
    train=np.arange(180);validation=np.arange(180,240);test=np.arange(240,300)
    assert not set(train)&set(validation) and not set(train)&set(test) and not set(validation)&set(test)
    models={'mean':DummyRegressor(strategy='mean'),'linear':LinearRegression(),'tree_depth_2':DecisionTreeRegressor(max_depth=2,random_state=42),'tree_depth_4':DecisionTreeRegressor(max_depth=4,random_state=42),'boosting':GradientBoostingRegressor(n_estimators=60,max_depth=2,learning_rate=.1,random_state=42)}
    table=[]
    for name,model in models.items():
        model.fit(X[train],y[train])
        pred=model.predict(X[validation])
        table.append({'model':name,'validation_mae':float(mean_absolute_error(y[validation],pred)),'validation_mse':float(mean_squared_error(y[validation],pred))})
    selected=min(table,key=lambda row:row['validation_mae'])['model']
    # Selection is already final here. Test is not used to adjust hyperparameters.
    pred=models[selected].predict(X[test]);truth=y[test]
    errors=pred-truth;slices={}
    for name,mask in [('x_negative',X[test,0]<0),('x_nonnegative',X[test,0]>=0)]:
        slices[name]={'n':int(mask.sum()),'mae':float(np.abs(errors[mask]).mean()),'bias':float(errors[mask].mean())}
    assert sum(r['n'] for r in slices.values())==60
    assert all(np.isfinite(row['validation_mae']) for row in table)
    assert np.isclose(models['mean'].constant_[0,0],y[train].mean())
    exact_y=np.array([0.,1.,4.,9.]);baseline=np.array([2.,2.,2.,2.]);candidate=np.array([0.,2.,3.,8.])
    assert mean_absolute_error(exact_y,baseline)==3 and mean_squared_error(exact_y,baseline)==14.5
    assert mean_absolute_error(exact_y,candidate)==.75 and mean_squared_error(exact_y,candidate)==.75
    result={'dataset':'synthetic y=x^2+N(0,0.2)','seed':42,'data_sha256':hashlib.sha256(np.c_[X,y].tobytes()).hexdigest(),'numpy':np.__version__,'sklearn':sklearn.__version__,'split_sizes':{'train':180,'validation':60,'test':60},'validation':table,'selected_by_validation_only':selected,'test_mae':float(mean_absolute_error(truth,pred)),'test_mse':float(mean_squared_error(truth,pred)),'slices':slices}
    out=Path(__file__).parent/'output';out.mkdir(exist_ok=True)
    np.savetxt(out/'synthetic.csv',np.c_[X,y],delimiter=',',header='x,y',comments='')
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
    return result

if __name__=='__main__':run()
