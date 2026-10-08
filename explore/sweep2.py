import sys, numpy as np, pandas as pd, warnings, time, itertools
warnings.filterwarnings('ignore'); sys.path.insert(0,'explore')
from features import build_features, CATS
from harness import oof_cv
import lightgbm as lgb
tr=pd.read_csv('train.csv'); y=tr['accepted'].values
Xc=build_features(tr,1)
res=[]
t0=time.time()
for ra,rl,nl,md in itertools.product([1.0,2.0,5.0,10.0],[10.0,20.0,40.0,80.0],[7,15],[6]):
    p=dict(n_estimators=1200,learning_rate=0.03,num_leaves=nl,max_depth=md,min_child_samples=5,
           reg_alpha=ra,reg_lambda=rl,subsample=0.9,subsample_freq=1,colsample_bytree=0.9,
           random_state=42,verbose=-1)
    a,f,_=oof_cv(lambda: lgb.LGBMClassifier(**p),Xc,y,n_repeats=3,fit_kwargs={'categorical_feature':CATS})
    res.append(dict(ra=ra,rl=rl,nl=nl,md=md,auc=a,f1=f))
    print(f"ra={ra:5.1f} rl={rl:5.1f} nl={nl:3d} AUC {a:.5f} F1* {f:.4f} [{time.time()-t0:.0f}s]",flush=True)
r=pd.DataFrame(res); r.to_csv('explore/sweep2.csv',index=False)
print("\n=== TOP by AUC ==="); print(r.sort_values('auc',ascending=False).head(10).to_string(index=False))
