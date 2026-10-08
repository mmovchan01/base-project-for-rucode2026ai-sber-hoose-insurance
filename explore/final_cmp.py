import sys, numpy as np, pandas as pd, warnings, time, itertools
warnings.filterwarnings('ignore'); sys.path.insert(0,'explore')
from features import build_features, onehot, CATS
from harness import oof_cv
from f1fast import f1_curve
import lightgbm as lgb, xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
tr=pd.read_csv('train.csv'); y=tr['accepted'].values
Xc=build_features(tr,1); Xoh=onehot(tr,1)
Xx=Xc.copy();  [Xx.__setitem__(c, Xx[c].astype('int16').astype('category')) for c in CATS]
Xcb=Xc.copy(); [Xcb.__setitem__(c, Xcb[c].astype(int).astype(str)) for c in CATS]
t0=time.time(); P={}
def run(n,mk,X,fk=None):
    a,f,p=oof_cv(mk,X,y,n_repeats=5,fit_kwargs=fk); P[n]=p
    print(f"{n:26s} AUC {a:.5f} F1* {f:.4f} [{time.time()-t0:.0f}s]",flush=True)
# --- XGB regularization sweep (short) ---
for md_,rl_ in [(4,5.0),(4,20.0),(3,10.0),(6,10.0)]:
    run(f'xgb_d{md_}_l{int(rl_)}', lambda md_=md_,rl_=rl_: xgb.XGBClassifier(
        n_estimators=1200,learning_rate=0.03,max_depth=md_,reg_lambda=rl_,reg_alpha=1.0,
        subsample=0.9,colsample_bytree=0.9,tree_method='hist',enable_categorical=True,
        random_state=42,verbosity=0),Xx)
# --- CatBoost regularization sweep (short) ---
for d_,l2_ in [(4,10.0),(3,10.0),(4,3.0),(6,20.0)]:
    run(f'cb_d{d_}_l{int(l2_)}', lambda d_=d_,l2_=l2_: CatBoostClassifier(
        iterations=1200,learning_rate=0.05,depth=d_,l2_leaf_reg=l2_,random_seed=42,
        verbose=0,cat_features=CATS,allow_const_label=True),Xcb)
# --- tuned LGBM candidates ---
run('lgb_tuned',  lambda: lgb.LGBMClassifier(n_estimators=1200,learning_rate=0.03,num_leaves=7,max_depth=6,
    min_child_samples=5,reg_alpha=2.0,reg_lambda=20.0,subsample=0.9,subsample_freq=1,
    colsample_bytree=0.9,random_state=42,verbose=-1),Xc,{'categorical_feature':CATS})
run('logreg',     lambda: make_pipeline(StandardScaler(),LogisticRegression(max_iter=5000,C=1.0)),Xoh)
np.savez('explore/final_cmp.npz', y=y, **P)
def rk(p): return (rankdata(p)-0.5)/len(y)
R={k:rk(v) for k,v in P.items()}
print("\n=== ensembles (rank-mean) ===")
for sub in [('lgb_tuned',),('lgb_tuned','logreg'),('lgb_tuned','cb_d4_l10'),
            ('lgb_tuned','xgb_d4_l20','cb_d4_l10'),('lgb_tuned','xgb_d4_l20','cb_d4_l10','logreg')]:
    p=np.mean([R[k] for k in sub],0)
    print(f"  {'+'.join(sub):44s} AUC {roc_auc_score(y,p):.5f} F1* {f1_curve(y,p)[0]:.4f}")
