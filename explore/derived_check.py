"""Нужны ли финальной сплайновой модели 5 производных признаков?
В экспериментах (explore/features.py v1) их не было, а в common.py -- есть.
Проверяем честным nested CV."""
import sys, numpy as np, pandas as pd, warnings, time
warnings.filterwarnings('ignore'); sys.path.insert(0,'.')
import common
from common import TARGET, build_features, build_onehot
from evaluate import nested_cv

df = pd.read_csv('train.csv'); y = df[TARGET].to_numpy(int)
DERIVED = ["income_over_phone_price", "debt_to_income", "balance_to_income",
           "phone_residual_value", "digital_activity"]
LOGS = ["log_" + c for c in common.SKEWED]
ISNA = [c + "_isna" for c in common.MNAR]

BASE_NUMERIC = common.NUM + LOGS + ISNA          # без производных (как в экспериментах)
FULL_NUMERIC = common._ONEHOT_NUMERIC            # с производными (как в финальном коде)

def variant(name, num_cols, n_knots=3, degree=3, C=0.3):
    common._ONEHOT_NUMERIC = list(num_cols)
    common.make_lr_spline.__defaults__ = (n_knots, degree, C)
    Xb, Xo = build_features(df), build_onehot(df)
    t0 = time.time()
    f1s, aucs, _ = nested_cv(['lr_spline'], Xb, Xo, y, 42, 5, 2, verbose=False, df=df)
    print(f"{name:38s} колонок={Xo.shape[1]:3d}  nestedF1 {f1s.mean():.4f} "
          f"+/- {f1s.std(ddof=1)/np.sqrt(len(f1s)):.4f}  AUC {aucs.mean():.5f}  [{time.time()-t0:.0f}s]", flush=True)

variant('только исходные + логи + isna (28)', BASE_NUMERIC)
variant('+ 5 производных (33, финальный код)', FULL_NUMERIC)
