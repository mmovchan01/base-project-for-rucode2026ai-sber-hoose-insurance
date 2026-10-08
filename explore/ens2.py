import sys, numpy as np, pandas as pd, warnings, itertools
warnings.filterwarnings('ignore')
from sklearn.metrics import roc_auc_score, f1_score
from scipy.stats import rankdata
from scipy.optimize import minimize
d=np.load('explore/oof_all.npz'); y=d['y']; ks=[k for k in d.files if k!='y']
P={k:d[k] for k in ks}
def rk(p): return (rankdata(p)-0.5)/len(p)     # -> (0,1), comparable to a probability

def f1_at_thr(y,p):
    ts=np.arange(0.02,0.98,0.002)
    sc=[(f1_score(y,(p>=t).astype(int)),t) for t in ts]
    return max(sc)

print("=== proper rank-ensemble eval (ranks mapped to (0,1)) ===")
R={k:rk(P[k]) for k in ks}
def ev(name,p):
    a=roc_auc_score(y,p); f,t=f1_at_thr(y,p); print(f"{name:28s} AUC {a:.5f} F1* {f:.4f} @thr {t:.3f}")
    return a,f
for n in ks: ev(n, P[n])
print()
ev('rankavg lgb+cb', (R['lgb']+R['cb'])/2)
ev('rankavg lgb+xgb+cb', (R['lgb']+R['xgb']+R['cb'])/3)
ev('rankavg all4', sum(R.values())/4)
ev('probavg all4', sum(P.values())/4)

print("\n=== weight search on rank-space (maximize OOF F1) ===")
M=np.vstack([R[k] for k in ks]).T
def negf1(w):
    w=np.abs(w); w=w/w.sum(); return -f1_at_thr(y, M@w)[0]
best=None
for seed in range(30):
    rng=np.random.default_rng(seed); w0=rng.random(4)
    r=minimize(negf1,w0,method='Nelder-Mead',options=dict(maxiter=2000))
    if best is None or r.fun<best.fun: best=r
w=np.abs(best.x); w=w/w.sum()
print("weights:",dict(zip(ks,np.round(w,3))),"F1=%.4f"%(-best.fun))
ev('weighted rankavg', M@w)

print("\n=== simple grid over subsets (equal weights, rank space) ===")
res=[]
for r_ in range(2,5):
    for sub in itertools.combinations(ks,r_):
        p=np.mean([R[k] for k in sub],0); res.append((sub,roc_auc_score(y,p),f1_at_thr(y,p)[0]))
for sub,a,f in sorted(res,key=lambda x:-x[2])[:8]: print(f"  {'+'.join(sub):24s} AUC {a:.5f} F1* {f:.4f}")

print("\n=== isotonic ceiling for probavg all4 ===")
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold
p=sum(P.values())/4
iso=np.zeros(len(y))
for a_,b_ in StratifiedKFold(5,shuffle=True,random_state=7).split(p,y):
    iso[b_]=IsotonicRegression(out_of_bounds='clip',y_min=0,y_max=1).fit(p[a_],y[a_]).predict(p[b_])
Pt=iso.sum(); o=np.argsort(-iso); cum=np.cumsum(iso[o]); K=np.arange(1,len(iso)+1)
print("CEILING F1 = %.4f at k=%d  (Ptot=%.1f, actual %d)"%((2*cum/(K+Pt)).max(),K[np.argmax(2*cum/(K+Pt))],Pt,y.sum()))
