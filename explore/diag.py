import sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0,'explore')
from features import build_features, onehot, CATS
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
tr=pd.read_csv('train.csv'); y=tr['accepted'].values
Xc=build_features(tr,1)

# 1) capacity test: can a huge model separate in-sample?
for kw in [dict(num_leaves=255,max_depth=-1,min_child_samples=1,n_estimators=3000,learning_rate=0.05),
           dict(num_leaves=31,min_child_samples=1,n_estimators=5000,learning_rate=0.05)]:
    m=lgb.LGBMClassifier(**kw,random_state=0,verbose=-1).fit(Xc,y,categorical_feature=CATS)
    print("LGBM huge in-sample AUC %.5f"%roc_auc_score(y,m.predict_proba(Xc)[:,1]))

# 2) L1 sparsity: which features matter?
Xoh=onehot(tr,1); Xs=((Xoh-Xoh.mean())/Xoh.std().replace(0,1)).values
for C in [0.01,0.03,0.1,0.3,1.0]:
    lr=LogisticRegression(max_iter=20000,C=C,penalty='l1',solver='liblinear').fit(Xs,y)
    nz=pd.Series(lr.coef_[0],index=Xoh.columns)
    print(f"\nL1 C={C}: nonzero={int((nz!=0).sum())}/{len(nz)}  AUC={roc_auc_score(y,lr.decision_function(Xs)):.5f}")
    print("   ", dict(nz[nz.abs()>0.05].round(3).sort_values(key=abs,ascending=False).head(15)))

# 3) per-feature marginal AUC (which single features carry signal)
print("\n=== single-feature AUC ===")
rows=[]
for c in Xoh.columns:
    v=Xoh[c].values
    if len(np.unique(v))<2: continue
    rows.append((c,roc_auc_score(y,v)))
r=pd.DataFrame(rows,columns=['f','auc']); r['dist']=(r.auc-0.5).abs()
print(r.sort_values('dist',ascending=False).head(18).to_string(index=False))
