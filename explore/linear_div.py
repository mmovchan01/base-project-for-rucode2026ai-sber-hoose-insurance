import sys, numpy as np, pandas as pd, warnings, itertools
warnings.filterwarnings('ignore'); sys.path.insert(0,'explore')
from features import build_features, onehot, CATS
from harness import oof_cv
from f1fast import f1_curve
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.preprocessing import StandardScaler, QuantileTransformer, SplineTransformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
tr=pd.read_csv('train.csv'); y=tr['accepted'].values
Xoh=onehot(tr,1); Xc=build_features(tr,1)
P={}
def run(n,mk,X,fk=None):
    a,f,p=oof_cv(mk,X,y,n_repeats=5,fit_kwargs=fk); P[n]=p
    print(f"{n:30s} AUC {a:.5f} F1* {f:.4f}",flush=True)
run('lr_C1',      lambda: make_pipeline(StandardScaler(),LogisticRegression(max_iter=5000,C=1.0)),Xoh)
run('lr_l1',      lambda: make_pipeline(StandardScaler(),LogisticRegression(max_iter=8000,C=0.3,penalty='l1',solver='liblinear')),Xoh)
run('lr_spline6', lambda: make_pipeline(SplineTransformer(n_knots=6,degree=3,include_bias=False),StandardScaler(),LogisticRegression(max_iter=8000,C=1.0)),Xoh)
run('lr_spline4', lambda: make_pipeline(SplineTransformer(n_knots=4,degree=3,include_bias=False),StandardScaler(),LogisticRegression(max_iter=8000,C=1.0)),Xoh)
run('lr_spline10',lambda: make_pipeline(SplineTransformer(n_knots=10,degree=3,include_bias=False),StandardScaler(),LogisticRegression(max_iter=8000,C=0.5)),Xoh)
run('lgb_tuned',  lambda: lgb.LGBMClassifier(n_estimators=1200,learning_rate=0.03,num_leaves=7,max_depth=6,
    min_child_samples=5,reg_alpha=2.0,reg_lambda=20.0,subsample=0.9,subsample_freq=1,
    colsample_bytree=0.9,random_state=42,verbose=-1),Xc,{'categorical_feature':CATS})
np.savez('explore/linear_div.npz', y=y, **P)
def rk(p): return (rankdata(p)-0.5)/len(y)
R={k:rk(v) for k,v in P.items()}
print("\n=== lgb_tuned + <linear> (rank-mean) ===")
for k in P:
    if k=='lgb_tuned': continue
    p=(R['lgb_tuned']+R[k])/2
    print(f"  lgb+{k:14s} AUC {roc_auc_score(y,p):.5f} F1* {f1_curve(y,p)[0]:.4f}  rankcorr {np.corrcoef(R['lgb_tuned'],R[k])[0,1]:.4f}")
print("\n=== best 2-linear + lgb ===")
res=[]
for a_,b_ in itertools.combinations([k for k in P if k!='lgb_tuned'],2):
    p=(R['lgb_tuned']+R[a_]+R[b_])/3; res.append((a_,b_,roc_auc_score(y,p),f1_curve(y,p)[0]))
for a_,b_,au,f in sorted(res,key=lambda x:-x[2])[:8]: print(f"  lgb+{a_}+{b_:22s} AUC {au:.5f} F1* {f:.4f}")
