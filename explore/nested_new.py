import sys, numpy as np, pandas as pd, warnings, time
warnings.filterwarnings('ignore'); sys.path.insert(0,'.')
from common import TARGET, build_features, build_onehot
from evaluate import nested_cv
df=pd.read_csv('train.csv'); y=df[TARGET].to_numpy(int)
Xb,Xo=build_features(df),build_onehot(df)
cands={"lr_spline":["lr_spline"],"lgbm+lr_spline":["lgbm","lr_spline"],
       "catboost+lr_spline":["catboost","lr_spline"],
       "lgbm+lr_spline+logreg":["lgbm","lr_spline","logreg"]}
print(f"{'состав':26s} {'nestedF1':>9s} {'+/-':>7s} {'nestedAUC':>10s} {'thr':>6s}")
for name,names in cands.items():
    t0=time.time()
    f1s,aucs,thrs=nested_cv(names,Xb,Xo,y,42,5,2,verbose=False,df=df)
    print(f"{name:26s} {f1s.mean():9.4f} {f1s.std(ddof=1)/np.sqrt(len(f1s)):7.4f} "
          f"{aucs.mean():10.5f} {thrs.mean():6.3f}  [{time.time()-t0:.0f}s]",flush=True)
    print("   folds:", " ".join(f"{v:.4f}" for v in f1s),flush=True)
