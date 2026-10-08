import sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0,'explore')
from features import build_features, CATS
from harness import oof_cv
import lightgbm as lgb
tr=pd.read_csv('train.csv'); y=tr['accepted'].values
Xc=build_features(tr,1)
LB=dict(n_estimators=1500,learning_rate=0.03,num_leaves=31,min_child_samples=20,
        subsample=0.9,subsample_freq=1,colsample_bytree=0.9,random_state=42,verbose=-1)
_,_,p=oof_cv(lambda: lgb.LGBMClassifier(**LB),Xc,y,n_repeats=5,fit_kwargs={'categorical_feature':CATS})
pred=(p>=0.5).astype(int)
fn=(pred==0)&(y==1); fp=(pred==1)&(y==0)
print("FN=%d FP=%d  F1=%.4f"%(fn.sum(),fp.sum(),2*((pred*y).sum())/(pred.sum()+y.sum())))
d=tr.assign(p=p,pred=pred)
print("\n=== error rate by smartphone_brand ===")
print(d.groupby('smartphone_brand').agg(n=('y' if 'y' in d else 'accepted','size'),FN=('p',lambda s:0),).shape if False else
      d.groupby('smartphone_brand').apply(lambda g: pd.Series({'n':len(g),'FN':fn[g.index].sum(),'FP':fp[g.index].sum(),'amb':((g.p>=.3)&(g.p<=.7)).sum()}),include_groups=False).to_string())
print("\n=== error rate by region ===")
print(d.groupby('region').apply(lambda g: pd.Series({'n':len(g),'FN':fn[g.index].sum(),'FP':fp[g.index].sum(),'amb':((g.p>=.3)&(g.p<=.7)).sum()}),include_groups=False).to_string())
print("\n=== ambiguous zone (0.3<=p<=0.7): n=%d, ybar=%.3f ==="%(((p>=.3)&(p<=.7)).sum(), y[(p>=.3)&(p<=.7)].mean()))
amb=d[(d.p>=.3)&(d.p<=.7)]
for c in ['smartphone_brand','region','education','employment_type','gender','previous_insurance','has_credit']:
    print(f"  {c}:", amb.groupby(c)['accepted'].agg(['mean','size']).round(3).to_dict('index'))
print("\n=== confidence buckets: n, ybar, mean p ===")
bins=[0,1e-6,1e-3,0.05,0.2,0.35,0.5,0.65,0.8,0.95,0.999,0.999999,1.0]
d['b']=pd.cut(d.p,bins)
print(d.groupby('b',observed=True).agg(n=('accepted','size'),ybar=('accepted','mean'),pmean=('p','mean')).round(4).to_string())
