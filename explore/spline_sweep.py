import sys, numpy as np, pandas as pd, warnings, itertools, time
warnings.filterwarnings('ignore'); sys.path.insert(0,'explore')
from features import onehot, CATS
from harness import oof_cv
import lightgbm as lgb
from features import build_features
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, SplineTransformer
from sklearn.pipeline import make_pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
from f1fast import f1_curve
tr=pd.read_csv('train.csv'); y=tr['accepted'].values
Xoh=onehot(tr,1)
CATCOL=[c for c in Xoh.columns if '=' in c]
res=[]; t0=time.time()
for nk,deg,C,only_num in itertools.product([3,4,5,6],[2,3],[0.1,0.3,1.0,3.0],[True,False]):
    if only_num:
        num=[c for c in Xoh.columns if c not in CATCOL]
        prep=ColumnTransformer([('sp',SplineTransformer(n_knots=nk,degree=deg,include_bias=False),num),
                                ('keep','passthrough',CATCOL)])
    else:
        prep=SplineTransformer(n_knots=nk,degree=deg,include_bias=False)
    mk=lambda prep=prep,C=C: make_pipeline(prep,StandardScaler(),LogisticRegression(max_iter=12000,C=C))
    try:
        a,f,p=oof_cv(mk,Xoh,y,n_repeats=3)
    except Exception as e:
        print(f"nk={nk} d={deg} C={C} only_num={only_num} FAIL {str(e)[:60]}"); continue
    res.append(dict(nk=nk,deg=deg,C=C,only_num=only_num,auc=a,f1=f)); np.save('explore/_p_%d_%d_%s_%s.npy'%(nk,deg,C,only_num),p)
    print(f"nk={nk} d={deg} C={C:<4} only_num={str(only_num):5s} AUC {a:.5f} F1* {f:.4f} [{time.time()-t0:.0f}s]",flush=True)
r=pd.DataFrame(res); r.to_csv('explore/spline_sweep.csv',index=False)
print("\n=== TOP 12 by AUC ==="); print(r.sort_values('auc',ascending=False).head(12).to_string(index=False))
print("\n=== TOP 8 by F1 ==="); print(r.sort_values('f1',ascending=False).head(8).to_string(index=False))
