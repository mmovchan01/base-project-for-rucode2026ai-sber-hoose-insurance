import sys, numpy as np, pandas as pd, warnings, itertools, time
warnings.filterwarnings('ignore'); sys.path.insert(0,'explore')
from f1fast import f1_curve
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
from scipy.optimize import minimize
d=np.load('explore/oof_all.npz'); y=d['y']; ks=[k for k in d.files if k!='y']
P={k:d[k] for k in ks}; R={k:(rankdata(P[k])-0.5)/len(y) for k in ks}
def ev(name,p,show=True):
    a=roc_auc_score(y,p); f,t,_=f1_curve(y,p)
    if show: print(f"{name:28s} AUC {a:.5f} F1* {f:.4f} @thr {t:.4f}")
    return a,f
for n in ks: ev(n,P[n])
print()
ev('rankavg all4', sum(R.values())/4); ev('probavg all4', sum(P.values())/4)
M=np.vstack([R[k] for k in ks]).T
def negf1(w):
    w=np.abs(w); w=w/w.sum(); return -f1_curve(y, M@w)[0]
t0=time.time(); best=None
for s in range(12):
    r=minimize(negf1,np.random.default_rng(s).random(4),method='Nelder-Mead',options=dict(maxiter=600,fatol_tol=1e-6,xatol=1e-6))
    if best is None or r.fun<best.fun: best=r
w=np.abs(best.x); w/=w.sum()
print("\nweights:",dict(zip(ks,np.round(w,3))),"F1=%.4f (%.1fs)"%(-best.fun,time.time()-t0))
ev('weighted rankavg', M@w)
print("\n=== equal-weight subsets (rank space) ===")
res=[]
for r_ in range(1,5):
    for sub in itertools.combinations(ks,r_):
        p=np.mean([R[k] for k in sub],0); res.append((sub,roc_auc_score(y,p),f1_curve(y,p)[0]))
for sub,a,f in sorted(res,key=lambda x:-x[2]): print(f"  {'+'.join(sub):24s} AUC {a:.5f} F1* {f:.4f}")
print("\n=== isotonic ceiling, probavg all4 ===")
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold
p=sum(P.values())/4; iso=np.zeros(len(y))
for a_,b_ in StratifiedKFold(5,shuffle=True,random_state=7).split(p,y):
    iso[b_]=IsotonicRegression(out_of_bounds='clip',y_min=0,y_max=1).fit(p[a_],y[a_]).predict(p[b_])
Pt=iso.sum(); o=np.argsort(-iso); cum=np.cumsum(iso[o]); K=np.arange(1,len(iso)+1); c=2*cum/(K+Pt)
print("CEILING F1 = %.4f at k=%d (Ptot %.1f actual %d)"%(c.max(),K[np.argmax(c)],Pt,y.sum()))
