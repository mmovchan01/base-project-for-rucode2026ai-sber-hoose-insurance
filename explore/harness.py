"""Reusable honest CV harness: threshold tuned INSIDE each outer fold (nested)."""
import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score

CATS = ['gender','region','family_status','education','employment_type','smartphone_brand']
TARGET, ID = 'accepted', 'customer_id'
THRS = np.arange(0.10, 0.90, 0.005)

def _asarr(X):
    """Keep pandas (column names/dtypes) intact; else to numpy."""
    return X if hasattr(X, "iloc") else np.asarray(X)

def _rows(X, idx):
    return X.iloc[idx] if hasattr(X, "iloc") else X[idx]

def best_thr(y, p):
    return max(((f1_score(y, (p >= t).astype(int)), t) for t in THRS))[1]

def nested_cv(make_model, X, y, n_splits=5, n_repeats=3, seed=0, fit_kwargs=None):
    """Returns unbiased (mean_f1, std_f1, mean_auc, thr_mean)."""
    X = _asarr(X); y = np.asarray(y); fit_kwargs = fit_kwargs or {}
    f1s, aucs, thrs = [], [], []
    for r in range(n_repeats):
        skf = StratifiedKFold(n_splits, shuffle=True, random_state=seed + r)
        for tr_i, te_i in skf.split(X, y):
            inner = StratifiedKFold(4, shuffle=True, random_state=1000 + r)
            Xi, yi = _rows(X, tr_i), y[tr_i]
            ip = np.zeros(len(yi))
            for a, b in inner.split(Xi, yi):
                m = make_model(); m.fit(_rows(Xi, a), yi[a], **fit_kwargs); ip[b] = m.predict_proba(_rows(Xi, b))[:, 1]
            t = best_thr(yi, ip)
            m = make_model(); m.fit(Xi, yi, **fit_kwargs)
            p = m.predict_proba(_rows(X, te_i))[:, 1]
            f1s.append(f1_score(y[te_i], (p >= t).astype(int)))
            aucs.append(roc_auc_score(y[te_i], p)); thrs.append(t)
    return np.mean(f1s), np.std(f1s), np.mean(aucs), np.mean(thrs)

def oof_cv(make_model, X, y, n_splits=5, n_repeats=1, seed=0, fit_kwargs=None):
    """Plain OOF (threshold tuned on OOF -> optimistic). For fast screening."""
    X = _asarr(X); y = np.asarray(y); fit_kwargs = fit_kwargs or {}
    P = np.zeros((n_repeats, len(y)))
    for r in range(n_repeats):
        skf = StratifiedKFold(n_splits, shuffle=True, random_state=seed + r)
        for a, b in skf.split(X, y):
            m = make_model(); m.fit(_rows(X, a), y[a], **fit_kwargs); P[r, b] = m.predict_proba(_rows(X, b))[:, 1]
    p = P.mean(0)
    return roc_auc_score(y, p), max(f1_score(y, (p >= t).astype(int)) for t in THRS), p
