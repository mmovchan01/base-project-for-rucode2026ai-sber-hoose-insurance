import numpy as np
from sklearn.metrics import f1_score

def f1_curve(y, p):
    """Return (best_f1, best_thr) over all thresholds. O(n log n).
    Threshold candidates are the midpoints between consecutive sorted scores."""
    y = np.asarray(y); p = np.asarray(p, dtype=float)
    o = np.argsort(-p, kind='mergesort')
    ys = y[o]; ps = p[o]
    tp = np.cumsum(ys); k = np.arange(1, len(y) + 1); pos = y.sum()
    f1 = 2 * tp / (k + pos)
    i = int(np.argmax(f1))
    # threshold: any t in (ps[i+1], ps[i]] works; pick midpoint (guard the last index)
    lo = ps[i + 1] if i + 1 < len(ps) else -np.inf
    thr = 0.5 * (ps[i] + lo) if np.isfinite(lo) else ps[i] - 1e-9
    return float(f1[i]), float(thr), float(2 * tp[-1] / (len(y) + pos))

def f1_from_k(y, p, k):
    """F1 when predicting the top-k by score."""
    y = np.asarray(y); o = np.argsort(-np.asarray(p, float), kind='mergesort')[:k]
    return float(2 * y[o].sum() / (k + y.sum()))
