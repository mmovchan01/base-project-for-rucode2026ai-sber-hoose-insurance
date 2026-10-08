import sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0,'explore')
from features import build_features, onehot, CATS
from harness import oof_cv, THRS
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, f1_score
tr=pd.read_csv('train.csv'); y=tr['accepted'].values
base=dict(n_estimators=1500,learning_rate=0.03,num_leaves=31,min_child_samples=20,
          subsample=0.9,subsample_freq=1,colsample_bytree=0.9,random_state=42,verbose=-1)
X1=build_features(tr,1)
auc,f,p=oof_cv(lambda: lgb.LGBMClassifier(**base),X1,y,n_repeats=5,
               fit_kwargs={'categorical_feature':CATS})
print("LGBM v1 cat-native 5x5 OOF: AUC %.4f  oofF1* %.4f"%(auc,f))
np.save('explore/oof_lgb.npy',p)
Ptot=p.sum()
def oracle(p,Ptot):
    o=np.argsort(-p); best=0
    for k in range(1,len(p)+1):
        s=p[o[:k]].sum(); best=max(best,2*s/(s+Ptot))
    return best
print("Expected #pos %.1f  actual %d"%(Ptot,y.sum()))
print("ORACLE F1 ceiling (p_hat as truth): %.4f"%oracle(p,Ptot))
q=pd.qcut(p,10,duplicates='drop')
cal=pd.DataFrame({'p':p,'y':y}).groupby(q,observed=True).agg(phat=('p','mean'),ybar=('y','mean'),n=('y','size'))
cal['diff']=(cal.ybar-cal.phat).round(4); print("\nCalibration deciles:\n",cal.round(4).to_string())
print("\n#p in[0.3,0.7]:",((p>=.3)&(p<=.7)).sum()," #p in[0.45,0.55]:",((p>=.45)&(p<=.55)).sum())
