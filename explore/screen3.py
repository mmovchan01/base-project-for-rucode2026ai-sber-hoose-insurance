import sys, numpy as np, pandas as pd, warnings, time
warnings.filterwarnings('ignore')
sys.path.insert(0,'explore')
from features import build_features, onehot, CATS
from harness import oof_cv
import lightgbm as lgb, xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

tr=pd.read_csv('train.csv'); y=tr['accepted'].values
Xc=build_features(tr,1)                 # categoricals as codes
Xoh=onehot(tr,1)                        # one-hot, median-filled

LB=dict(n_estimators=1500,learning_rate=0.03,num_leaves=31,min_child_samples=20,
        subsample=0.9,subsample_freq=1,colsample_bytree=0.9,random_state=42,verbose=-1)
XB=dict(n_estimators=1500,learning_rate=0.03,max_depth=6,subsample=0.9,colsample_bytree=0.9,
        reg_lambda=1.0,random_state=42,verbosity=0,tree_method='hist',enable_categorical=True)

res=[]
def run(name,mk,X,fk=None):
    t=time.time(); auc,f,_=oof_cv(mk,X,y,n_repeats=3,fit_kwargs=fk)
    res.append((name,auc,f,time.time()-t)); print(f"{name:32s} AUC {auc:.4f} oofF1* {f:.4f}  ({time.time()-t:.0f}s)",flush=True)

run('LightGBM cat-native', lambda: lgb.LGBMClassifier(**LB), Xc, {'categorical_feature':CATS})
Xx=Xc.copy()
for c in CATS: Xx[c]=Xx[c].astype('int16').astype('category')
run('XGBoost cat-native',  lambda: xgb.XGBClassifier(**XB), Xx)
Xcb=Xc.copy()
for c in CATS: Xcb[c]=Xcb[c].astype(int).astype(str)
run('CatBoost ordered',    lambda: CatBoostClassifier(iterations=1500,learning_rate=0.05,depth=6,
        l2_leaf_reg=3,random_seed=42,verbose=0,cat_features=CATS,allow_const_label=True), Xcb)
run('LogReg onehot',       lambda: make_pipeline(StandardScaler(),LogisticRegression(max_iter=5000,C=1.0)), Xoh)
run('MLP 64-32',           lambda: make_pipeline(StandardScaler(),MLPClassifier((64,32),max_iter=400,random_state=42,early_stopping=True)), Xoh)
print("\n=== sorted by F1 ===")
for n,a,f,t in sorted(res,key=lambda r:-r[2]): print(f"{n:32s} AUC {a:.4f} F1* {f:.4f}")
