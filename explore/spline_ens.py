import sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore'); sys.path.insert(0,'explore'); sys.path.insert(0,'.')
from features import onehot, build_features, CATS
from harness import oof_cv
from f1fast import f1_curve
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, SplineTransformer
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
tr=pd.read_csv('train.csv'); y=tr['accepted'].values
Xoh=onehot(tr,1); Xc=build_features(tr,1)
CATCOL=[c for c in Xoh.columns if '=' in c]; NUMCOL=[c for c in Xoh.columns if c not in CATCOL]
def spline_lr(nk=3,deg=3,C=0.3):
    prep=ColumnTransformer([('sp',SplineTransformer(n_knots=nk,degree=deg,include_bias=False),NUMCOL),
                            ('keep','passthrough',CATCOL)])
    return make_pipeline(prep,StandardScaler(),LogisticRegression(max_iter=12000,C=C))
P={}
def run(n,mk,X,fk=None):
    a,f,p=oof_cv(mk,X,y,n_repeats=5,fit_kwargs=fk); P[n]=p
    print(f"{n:22s} AUC {a:.5f} F1* {f:.4f}",flush=True)
run('lr_spline', lambda: spline_lr(), Xoh)
run('lr_plain',  lambda: make_pipeline(StandardScaler(),LogisticRegression(max_iter=5000,C=1.0)), Xoh)
run('lgb_tuned', lambda: lgb.LGBMClassifier(n_estimators=1200,learning_rate=0.03,num_leaves=7,max_depth=6,
    min_child_samples=5,reg_alpha=2.0,reg_lambda=20.0,subsample=0.9,subsample_freq=1,
    colsample_bytree=0.9,random_state=42,verbose=-1), Xc, {'categorical_feature':CATS})
np.savez('explore/spline_ens.npz', y=y, **P)
def rk(p): return (rankdata(p)-0.5)/len(y)
R={k:rk(v) for k,v in P.items()}
print("\n=== ensembles (rank-mean) ===")
for sub in [('lr_spline',),('lgb_tuned',),('lgb_tuned','lr_spline'),('lgb_tuned','lr_plain'),
            ('lgb_tuned','lr_spline','lr_plain')]:
    p=np.mean([R[k] for k in sub],0)
    print(f"  {'+'.join(sub):38s} AUC {roc_auc_score(y,p):.5f} F1* {f1_curve(y,p)[0]:.4f}")
print("\n=== weight scan lgb/lr_spline ===")
for w in [0.3,0.4,0.5,0.6,0.7]:
    p=w*R['lgb_tuned']+(1-w)*R['lr_spline']
    print(f"  w_lgb={w:.1f} AUC {roc_auc_score(y,p):.5f} F1* {f1_curve(y,p)[0]:.4f}")
m=spline_lr().fit(Xoh,y); n=m[-1].coef_.size
print(f"\nПараметров в spline-LR: {n} весов (+{m[-1].intercept_.size} intercept); "
      f"входных колонок после сплайнов: {m[1].mean_.size}")
b=lgb.LGBMClassifier(n_estimators=1200,learning_rate=0.03,num_leaves=7,max_depth=6,min_child_samples=5,
    reg_alpha=2.0,reg_lambda=20.0,subsample=0.9,subsample_freq=1,colsample_bytree=0.9,
    random_state=42,verbose=-1).fit(Xc,y,categorical_feature=CATS)
print("Параметров в LightGBM: %d деревьев x <=7 листьев ~ %d значений"%(b.n_estimators_, b.n_estimators_*7))
