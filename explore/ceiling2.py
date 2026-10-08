import sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0,'explore')
from features import build_features, CATS
from harness import oof_cv
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score
import lightgbm as lgb
tr=pd.read_csv('train.csv'); y=tr['accepted'].values
Xc=build_features(tr,1)
LB=dict(n_estimators=1500,learning_rate=0.03,num_leaves=31,min_child_samples=20,
        subsample=0.9,subsample_freq=1,colsample_bytree=0.9,random_state=42,verbose=-1)
auc,f,p=oof_cv(lambda: lgb.LGBMClassifier(**LB),Xc,y,n_repeats=5,fit_kwargs={'categorical_feature':CATS})
print("OOF AUC %.4f  F1* %.4f"%(auc,f))

# isotonic smoothing (out-of-fold to avoid overfit) -> estimate of true p
iso_p=np.zeros(len(y))
for a,b in StratifiedKFold(5,shuffle=True,random_state=7).split(p,y):
    iso=IsotonicRegression(out_of_bounds='clip',y_min=0,y_max=1).fit(p[a],y[a])
    iso_p[b]=iso.predict(p[b])
print("isotonic-smoothed AUC %.4f"%roc_auc_score(y,iso_p))
Ptot=iso_p.sum()
o=np.argsort(-iso_p)
cum=np.cumsum(iso_p[o]); ks=np.arange(1,len(iso_p)+1)
f1c=2*cum/(ks+Ptot)
kbest=ks[np.argmax(f1c)]
print("Expected #pos %.1f (actual %d)"%(Ptot,y.sum()))
print(">>> BAYES-ISH F1 CEILING (isotonic p): %.4f at k=%d"%(f1c.max(),kbest))
print("    ceiling using y.sum as Ptot: %.4f"%(2*cum/(ks+y.sum())).max())
for t in [0.3,0.4,0.475,0.5,0.55,0.6,0.7]:
    s=iso_p>=t; print("   thr %.3f -> n=%4d expF1=%.4f"%(t,s.sum(),2*iso_p[s].sum()/(s.sum()+Ptot)))
