import sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0,'explore')
from features import build_features, onehot, CATS
from harness import oof_cv, THRS
import lightgbm as lgb, xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, f1_score
from scipy.stats import rankdata

tr=pd.read_csv('train.csv'); y=tr['accepted'].values
Xc=build_features(tr,1); Xoh=onehot(tr,1)
Xx=Xc.copy()
for c in CATS: Xx[c]=Xx[c].astype('int16').astype('category')
Xcb=Xc.copy()
for c in CATS: Xcb[c]=Xcb[c].astype(int).astype(str)

LB=dict(n_estimators=1500,learning_rate=0.03,num_leaves=31,min_child_samples=20,
        subsample=0.9,subsample_freq=1,colsample_bytree=0.9,random_state=42,verbose=-1)
XB=dict(n_estimators=1500,learning_rate=0.03,max_depth=6,subsample=0.9,colsample_bytree=0.9,
        reg_lambda=1.0,random_state=42,verbosity=0,tree_method='hist',enable_categorical=True)
CB=dict(iterations=1500,learning_rate=0.05,depth=6,l2_leaf_reg=3,random_seed=42,verbose=0,
        cat_features=CATS,allow_const_label=True)

P={}
P['lgb'],_=None,None
a,f,P['lgb']=oof_cv(lambda: lgb.LGBMClassifier(**LB),Xc,y,n_repeats=5,fit_kwargs={'categorical_feature':CATS})
print("lgb      AUC %.5f F1* %.4f"%(a,f))
a,f,P['xgb']=oof_cv(lambda: xgb.XGBClassifier(**XB),Xx,y,n_repeats=5)
print("xgb      AUC %.5f F1* %.4f"%(a,f))
a,f,P['cb']=oof_cv(lambda: CatBoostClassifier(**CB),Xcb,y,n_repeats=5)
print("catboost AUC %.5f F1* %.4f"%(a,f))
a,f,P['lr']=oof_cv(lambda: make_pipeline(StandardScaler(),LogisticRegression(max_iter=5000,C=1.0)),Xoh,y,n_repeats=5)
print("logreg   AUC %.5f F1* %.4f"%(a,f))
np.savez('explore/oof_all.npz', **P, y=y)

def ev(name,p):
    a=roc_auc_score(y,p); f=max(f1_score(y,(p>=t).astype(int)) for t in THRS); print(f"{name:26s} AUC {a:.5f} F1* {f:.4f}")
print("\n=== ensembles (rank-avg) ===")
def rk(p): return rankdata(p)/len(p)
ev('lgb+cb', rk(P['lgb'])+rk(P['cb']))
ev('lgb+xgb+cb', rk(P['lgb'])+rk(P['xgb'])+rk(P['cb']))
ev('lgb+xgb+cb+lr', rk(P['lgb'])+rk(P['xgb'])+rk(P['cb'])+rk(P['lr']))
ev('mean prob all4', (P['lgb']+P['xgb']+P['cb']+P['lr'])/4)
ev('cb+lr', rk(P['cb'])+rk(P['lr']))
ev('logit-avg all4', np.mean([np.log(np.clip(P[k],1e-6,1-1e-6)/(1-np.clip(P[k],1e-6,1-1e-6))) for k in ['lgb','xgb','cb','lr']],0))
print("\npairwise OOF rank corr:")
ks=list(P); print(pd.DataFrame([[round(np.corrcoef(rk(P[a]),rk(P[b]))[0,1],4) for b in ks] for a in ks],index=ks,columns=ks).to_string())
