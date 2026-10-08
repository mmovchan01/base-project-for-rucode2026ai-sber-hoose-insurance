import sys, numpy as np, pandas as pd, warnings, time
warnings.filterwarnings('ignore'); sys.path.insert(0,'explore')
from features import build_features, onehot, CATS
from harness import oof_cv
from f1fast import f1_curve
import lightgbm as lgb, xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
tr=pd.read_csv('train.csv'); y=tr['accepted'].values
Xc=build_features(tr,1); Xoh=onehot(tr,1)
Xx=Xc.copy();  [Xx.__setitem__(c, Xx[c].astype('int16').astype('category')) for c in CATS]
Xcb=Xc.copy(); [Xcb.__setitem__(c, Xcb[c].astype(int).astype(str)) for c in CATS]
out={}
try: out.update(dict(np.load('explore/zoo.npz')))
except Exception: pass
out.pop('y',None)
def run(name,mk,X,fk=None,nr=5):
    t=time.time()
    try:
        a,f,p=oof_cv(mk,X,y,n_repeats=nr,fit_kwargs=fk); out[name]=p
        print(f"{name:34s} AUC {a:.5f} F1* {f:.4f}  ({time.time()-t:.0f}s)",flush=True)
        return p
    except Exception as e:
        print(f"{name:34s} FAILED: {type(e).__name__}: {str(e)[:90]}",flush=True)
        return None
LB=dict(n_estimators=1500,learning_rate=0.03,num_leaves=31,min_child_samples=20,
        subsample=0.9,subsample_freq=1,colsample_bytree=0.9,verbose=-1)
run('lgb_s42',  lambda: lgb.LGBMClassifier(**{**LB,'random_state':42}),Xc,{'categorical_feature':CATS})
run('lgb_dart', lambda: lgb.LGBMClassifier(**{**LB,'random_state':1,'boosting_type':'dart','n_estimators':600,'learning_rate':0.06}),Xc,{'categorical_feature':CATS})
run('lgb_goss', lambda: lgb.LGBMClassifier(**{**LB,'random_state':2,'boosting_type':'goss','n_estimators':2000,'subsample':1.0,'subsample_freq':0}),Xc,{'categorical_feature':CATS})
run('lgb_l1l2', lambda: lgb.LGBMClassifier(**{**LB,'random_state':3,'reg_alpha':0.5,'reg_lambda':2.0,'num_leaves':15}),Xc,{'categorical_feature':CATS})
run('xgb_d4',   lambda: xgb.XGBClassifier(n_estimators=2000,learning_rate=0.03,max_depth=4,subsample=0.9,colsample_bytree=0.9,reg_lambda=2.0,random_state=42,verbosity=0,tree_method='hist',enable_categorical=True),Xx)
run('cb_d8',    lambda: CatBoostClassifier(iterations=2000,learning_rate=0.04,depth=8,l2_leaf_reg=5,random_seed=42,verbose=0,cat_features=CATS,allow_const_label=True),Xcb)
run('cb_d4',    lambda: CatBoostClassifier(iterations=2000,learning_rate=0.04,depth=4,l2_leaf_reg=3,random_seed=7,verbose=0,cat_features=CATS,allow_const_label=True),Xcb)
run('extratrees',lambda: ExtraTreesClassifier(n_estimators=1500,max_depth=None,min_samples_leaf=2,random_state=42,n_jobs=-1),Xoh)
run('rf',       lambda: RandomForestClassifier(n_estimators=1500,min_samples_leaf=2,random_state=42,n_jobs=-1),Xoh)
run('mlp_128',  lambda: make_pipeline(StandardScaler(),MLPClassifier((128,64,32),alpha=1e-3,max_iter=800,random_state=42,early_stopping=True,n_iter_no_change=25,learning_rate_init=2e-3)),Xoh)
np.savez('explore/zoo.npz', y=y, **out)
print("\n=== best rank-ensembles from zoo ===")
import itertools
R={k:(rankdata(v)-0.5)/len(y) for k,v in out.items()}
res=[]
for r_ in (2,3,4,5):
    for sub in itertools.combinations(R,r_):
        p=np.mean([R[k] for k in sub],0); res.append((sub,roc_auc_score(y,p),f1_curve(y,p)[0]))
for sub,a,f in sorted(res,key=lambda x:-x[2])[:10]: print(f"  {'+'.join(sub):44s} AUC {a:.5f} F1* {f:.4f}")
