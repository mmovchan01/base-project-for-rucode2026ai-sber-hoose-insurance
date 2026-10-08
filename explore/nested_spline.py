"""Не является ли (nk=3,deg=3,C=0.3) удачной случайностью выбора по OOF?
Сравниваем соседние конфигурации честным nested CV."""
import sys, numpy as np, pandas as pd, warnings, time
warnings.filterwarnings('ignore'); sys.path.insert(0,'.')
import common
from common import TARGET, build_features, build_onehot
from evaluate import nested_cv
df=pd.read_csv('train.csv'); y=df[TARGET].to_numpy(int)
Xb,Xo=build_features(df),build_onehot(df)
print(f"{'конфигурация spline-LR':34s} {'nestedF1':>9s} {'+/-':>7s} {'nestedAUC':>10s}")
for nk,deg,C in [(3,3,0.3),(3,3,0.1),(3,3,1.0),(4,3,0.1),(4,3,0.3),(5,3,0.3),(6,3,1.0),(3,2,0.3)]:
    common.make_lr_spline.__defaults__=(nk,deg,C)
    t0=time.time()
    f1s,aucs,thrs=nested_cv(['lr_spline'],Xb,Xo,y,42,5,2,verbose=False,df=df)
    print(f"nk={nk} deg={deg} C={C:<5} {'':16s} {f1s.mean():9.4f} "
          f"{f1s.std(ddof=1)/np.sqrt(len(f1s)):7.4f} {aucs.mean():10.5f}  [{time.time()-t0:.0f}s]",flush=True)
common.make_lr_spline.__defaults__=(3,3,0.3)
print("\n=== финальная конфигурация, 5x4 repeats ===")
f1s,aucs,thrs=nested_cv(['lr_spline'],Xb,Xo,y,42,5,4,verbose=False,df=df)
print(f"F1  = {f1s.mean():.4f} +/- {f1s.std(ddof=1)/np.sqrt(len(f1s)):.4f}  (n={len(f1s)} folds)")
print(f"AUC = {aucs.mean():.5f} +/- {aucs.std(ddof=1)/np.sqrt(len(aucs)):.5f}")
print(f"thr = {thrs.mean():.4f} +/- {thrs.std(ddof=1):.4f}")
