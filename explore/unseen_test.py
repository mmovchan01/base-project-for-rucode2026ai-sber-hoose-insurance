"""Проверка устойчивости к категориям, которых не было в train.
В public_test есть employment_type='retired', которого в train НЕТ."""
import sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore'); sys.path.insert(0,'.')
from common import (ALL_MODELS, CATS, TARGET, build_features, build_onehot,
                    prepare_for, KNOWN_CATEGORIES)
tr=pd.read_csv('train.csv'); te=pd.read_csv('public_test.csv')
y=tr[TARGET].to_numpy(int)
Xb_tr,Xo_tr=build_features(tr),build_onehot(tr)
Xb_te,Xo_te=build_features(te),build_onehot(te)
print("Коды employment_type, встречающиеся в train :", sorted(set(Xb_tr.employment_type)))
print("Коды employment_type, встречающиеся в test  :", sorted(set(Xb_te.employment_type)))
print("Словарь:", KNOWN_CATEGORIES['employment_type'])
print("retired -> код", KNOWN_CATEGORIES['employment_type'].index('retired'))
print()
for name,(factory,rk,fk) in ALL_MODELS.items():
    m=factory(42); X=prepare_for(rk,Xb_tr,Xo_tr)
    try: m.fit(X,y,**fk)
    except Exception as e: print(f"{name:10s} FIT   FAIL {type(e).__name__}: {str(e)[:70]}"); continue
    Xt=prepare_for(rk,Xb_te,Xo_te)
    try:
        p=(m.predict(Xt.to_numpy(dtype=float)) if name=='lgbm' else m.predict_proba(Xt)[:,1])
        p=np.asarray(p).ravel()
        bad=np.isfinite(p).all()
        print(f"{name:10s} PREDICT OK  n={len(p)} all-finite={bad} mean={p.mean():.4f} "
              f"min={p.min():.4f} max={p.max():.4f}")
    except Exception as e:
        print(f"{name:10s} PREDICT FAIL {type(e).__name__}: {str(e)[:120]}")
