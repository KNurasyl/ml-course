"""Local mixed-data classification with audited train-only preprocessing."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import KFold

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
FEATURES=['experience_years','answer_words','track']
NUMERIC=FEATURES[:2]
FORBIDDEN=['is_ready','review_score_after','user_id','day']


class FitAudit(BaseEstimator,TransformerMixin):
    """Records real input row IDs at fit; transforms only allowed features."""
    def fit(self, X, y=None):
        if not isinstance(X,pd.DataFrame) or not X.index.is_unique:
            raise ValueError('DataFrame with unique row IDs required')
        if not set(FEATURES)<=set(X.columns):
            raise ValueError('missing required input columns')
        self.fit_rows_=X.index.tolist()
        self.feature_names_in_=np.asarray(FEATURES,dtype=object)
        return self
    def transform(self, X):
        if not isinstance(X,pd.DataFrame) or not set(FEATURES)<=set(X.columns):
            raise ValueError('missing required input columns')
        return X.loc[:,FEATURES].copy()


def make_data(seed=120, users=120, difficult=False):
    """Authored synthetic records; target is a toy label, not hiring suitability."""
    rng=np.random.default_rng(seed)
    records=[]
    for user in range(users):
        exp=float(rng.uniform(0,8));track=str(rng.choice(['python','go','js']))
        for event in range(2):
            day=user*2+event
            words=int(rng.integers(10,180))
            score=-3 + .48*exp + .017*words + (track=='python')*.4
            y=int(rng.random()<1/(1+np.exp(-score)))
            records.append({'row_id':f'{seed}-{user:03}-{event}','user_id':f'u{user:03}',
                'day':day,'experience_years':exp if rng.random()>.09 else np.nan,
                'answer_words':float(words) if rng.random()>.08 else np.nan,
                'track':track,'review_score_after':float(y*.8+rng.uniform(0,.2)),
                'is_ready':y})
    df=pd.DataFrame(records).set_index('row_id')
    if difficult:
        # A real group spans the first boundary: both rows must be excluded.
        df.loc[df.index[143],'user_id']='u070'
        df.loc[df.index[144],'user_id']='u070'
        # Unknown category arrives only at final evaluation.
        df.loc[df.index[-8:],'track']='rust'
    return df


def split_groups(df, train_before=144, validation_before=192):
    """Group and chronology must both hold; exclude boundary-straddling groups."""
    if train_before>=validation_before or not df.index.is_unique:
        raise ValueError('invalid boundaries or duplicated row IDs')
    if not {'user_id','day'}<=set(df.columns) or df.user_id.isna().any() or not np.isfinite(df.day).all():
        raise ValueError('group and time must be known for every record')
    groups={'train':[],'validation':[],'test':[],'excluded':[]}
    for _,rows in df.groupby('user_id',sort=True):
        days=rows.day
        if (days<train_before).all(): key='train'
        elif ((days>=train_before)&(days<validation_before)).all(): key='validation'
        elif (days>=validation_before).all(): key='test'
        else:key='excluded'
        groups[key].extend(rows.index.tolist())
    return groups


def make_pipeline(C=1., scale=True):
    numeric=[('impute',SimpleImputer(strategy='median'))]
    if scale:numeric.append(('scale',StandardScaler()))
    prep=ColumnTransformer([
        ('num',Pipeline(numeric),NUMERIC),
        ('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),['track'])],
        remainder='drop')
    return Pipeline([('audit',FitAudit()),('prepare',prep),
                     ('model',LogisticRegression(C=C,max_iter=1000,solver='lbfgs'))])


def choose_model(train, validation):
    table=[];models={}
    ytr=train.is_ready.to_numpy();yv=validation.is_ready.to_numpy()
    for C in [.1,1.,10.]:
        model=make_pipeline(C)
        model.fit(train,ytr)
        proba=model.predict_proba(validation)[:,1]
        for threshold in [.25,.5,.75]:
            pred=proba>=threshold
            fn=int(((yv==1)&~pred).sum());fp=int(((yv==0)&pred).sum())
            table.append({'C':C,'threshold':threshold,'validation_cost':(2*fn+fp)/len(yv),
                          'validation_log_loss':float(log_loss(yv,proba,labels=[0,1]))})
        models[C]=model
    selected=min(table,key=lambda r:(r['validation_cost'],r['validation_log_loss'],r['C'],r['threshold']))
    return models[selected['C']],float(selected['threshold']),table


def check_fit_rows(model, allowed_rows):
    actual=set(model.named_steps['audit'].fit_rows_)
    if actual!=set(allowed_rows):
        raise AssertionError(f'fit row leakage/missing training data: {sorted(actual-set(allowed_rows))[:5]}')


def audited_cv(df):
    """The deliberately global fit must fail even if model metrics look excellent."""
    X=df.iloc[:24]
    folds=KFold(3,shuffle=True,random_state=17)
    checks=[]
    for train_idx,val_idx in folds.split(X):
        good=make_pipeline();good.fit(X.iloc[train_idx],X.is_ready.iloc[train_idx])
        check_fit_rows(good,X.index[train_idx])
        leaked=make_pipeline();leaked.fit(X,X.is_ready)
        try:check_fit_rows(leaked,X.index[train_idx])
        except AssertionError:detected=True
        else:detected=False
        checks.append({'correct_fit_rows':len(train_idx),'validation_rows':len(val_idx),'global_fit_detected':detected})
    assert all(r['global_fit_detected'] for r in checks)
    return checks


def run(variant='demo'):
    settings={'demo':(120,False),'hw':(2026,True),'retry':(2027,True),'delayed':(2028,True)}
    seed,difficult=settings[variant];df=make_data(seed,difficult=difficult)
    parts=split_groups(df)
    train,validation,test=[df.loc[parts[k]] for k in ['train','validation','test']]
    assert max(train.day)<min(validation.day) and max(validation.day)<min(test.day)
    assert not set(train.user_id)&set(validation.user_id)
    model,threshold,table=choose_model(train,validation)
    check_fit_rows(model,train.index)
    # Final evaluation occurs only after model/threshold selection is fixed.
    prob=model.predict_proba(test)[:,1];truth=test.is_ready.to_numpy();pred=prob>=threshold
    prior=float(train.is_ready.mean());baseline=np.full(len(test),prior)
    before=model.predict_proba(test)[:,1]
    tampered=test.copy();tampered['review_score_after']=10000-tampered.review_score_after
    assert np.allclose(before,model.predict_proba(tampered)[:,1])
    out=Path(__file__).parent/'output'/variant;out.mkdir(exist_ok=True,parents=True)
    data_path=out/'records.csv';df.to_csv(data_path)
    artifact={'pipeline':model,'threshold':threshold,'features':FEATURES,'version':'1.1.0'}
    joblib.dump(artifact,out/'classifier.joblib')
    restored=joblib.load(out/'classifier.joblib')
    assert np.allclose(prob,restored['pipeline'].predict_proba(test)[:,1])
    exp_scaled=train.copy();exp_scaled.experience_years*=1000
    val_scaled=validation.copy();val_scaled.experience_years*=1000
    scale_exp=[]
    for scale in [True,False]:
        for factor,X,Y in [(1,train,validation),(1000,exp_scaled,val_scaled)]:
            m=make_pipeline(C=.1,scale=scale).fit(X,X.is_ready)
            p=m.predict_proba(Y)[:,1]
            scale_exp.append({'scaler':scale,'experience_unit_multiplier':factor,'validation_log_loss':float(log_loss(Y.is_ready,p))})
    # Controlled composition experiment on validation only: weights stay frozen.
    positive=validation[validation.is_ready==1]
    negative=validation[validation.is_ready==0]
    class_mix=[]
    unit=min(4,len(positive)//2,len(negative)//3)
    if unit<1:raise ValueError('validation needs both classes for composition experiment')
    for label,npos,nneg in [('balanced',2*unit,2*unit),('positive_25_percent',unit,3*unit)]:
        subset=pd.concat([positive.sample(npos,random_state=61),negative.sample(nneg,random_state=62)])
        probabilities=model.predict_proba(subset)[:,1]
        truth_subset=subset.is_ready.to_numpy();decisions=probabilities>=threshold
        class_mix.append({'composition':label,'n':len(subset),'observed_positive_rate':float(truth_subset.mean()),
            'mean_predicted_probability':float(probabilities.mean()),
            'brier_score':float(np.mean((probabilities-truth_subset)**2)),
            'fixed_threshold_cost':float((2*((truth_subset==1)&~decisions).sum()+((truth_subset==0)&decisions).sum())/len(subset)),
            'model_refit':False,'source_split':'validation'})
    result={'variant':variant,'seed':seed,'data_sha256':hashlib.sha256(data_path.read_bytes()).hexdigest(),
       'split_rows':parts,'split_sizes':{k:len(v) for k,v in parts.items()},'feature_contract':FEATURES,
       'forbidden_features':FORBIDDEN,'selection_uses':'validation only','selection_table':table,
       'C':float(model.named_steps['model'].C),'threshold':threshold,'train_class_prior':prior,
       'test_class_prior':float(truth.mean()),'test_log_loss':float(log_loss(truth,prob)),
       'baseline_test_log_loss':float(log_loss(truth,baseline)),
       'test_balanced_accuracy':float(balanced_accuracy_score(truth,pred)),
       'test_confusion_matrix_TN_FP_FN_TP':confusion_matrix(truth,pred,labels=[0,1]).ravel().tolist(),
       'scale_experiment':scale_exp,'class_composition_experiment':class_mix,'fit_audit':audited_cv(df),
       'restored_probabilities_equal':True,'future_feature_ignored':True,
       'limitation':'synthetic task; readiness label does not measure human hiring suitability'}
    (out/'report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='split_rows'},indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--variant',choices=['demo','hw','retry','delayed'],default='demo')
    # Persist the class under an importable module, never __main__.FitAudit.
    from labs.M05.pipeline import run as run_importable
    run_importable(parser.parse_args().variant)
