"""Жёсткая проверка: обучение БЕЗ категории 'widowed' (код 3 > max в train),
предсказание НА данных, где 'widowed' есть. Это ровно тот случай, на котором
упал XGBoost в nested CV."""
import sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore'); sys.path.insert(0,'.')
from common import ALL_MODELS, TARGET, build_features, build_onehot, prepare_for
tr=pd.read_csv('train.csv'); y=tr[TARGET].to_numpy(int)
keep=(tr.family_status!='widowed').values
tr_sub=tr[keep].reset_index(drop=True); y_sub=y[keep]
Xb_s,Xo_s=build_features(tr_sub),build_onehot(tr_sub)
Xb_f,Xo_f=build_features(tr),build_onehot(tr)          # содержит widowed
print("family_status коды в обучающей подвыборке:",sorted(set(Xb_s.family_status)))
print("family_status коды в данных для прогноза   :",sorted(set(Xb_f.family_status)))
print("строк с widowed в прогнозной части:",int((Xb_f.family_status==3).sum()))
print()
for name,(factory,rk,fk) in ALL_MODELS.items():
    m=factory(42)
    try: m.fit(prepare_for(rk,Xb_s,Xo_s),y_sub,**fk)
    except Exception as e: print(f"{name:10s} FIT     FAIL {type(e).__name__}"); continue
    Xt=prepare_for(rk,Xb_f,Xo_f)
    try:
        p=(m.predict(Xt.to_numpy(dtype=float)) if name=='lgbm' else m.predict_proba(Xt)[:,1])
        p=np.asarray(p).ravel()
        w=int((Xb_f.family_status==3).values.argmax())
        print(f"{name:10s} PREDICT OK  all-finite={bool(np.isfinite(p).all())} "
              f"min={p.min():.4f} max={p.max():.4f}  p(widowed row)={p[w]:.4f}")
    except Exception as e:
        print(f"{name:10s} PREDICT FAIL {type(e).__name__}: {str(e)[:110]}")
