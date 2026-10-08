import numpy as np, pandas as pd, itertools, sys
sys.path.insert(0,'explore')
from f1fast import f1_curve
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
d=np.load('explore/final_cmp.npz'); y=d['y']; ks=[k for k in d.files if k!='y']
P={k:d[k] for k in ks}; R={k:(rankdata(P[k])-0.5)/len(y) for k in ks}
print("=== single models ===")
for k in ks: print(f"  {k:22s} AUC {roc_auc_score(y,P[k]):.5f} F1* {f1_curve(y,P[k])[0]:.4f}")
print("\n=== all rank-mean subsets, sorted by AUC ===")
res=[]
for r_ in range(1,len(ks)+1):
    for sub in itertools.combinations(ks,r_):
        p=np.mean([R[k] for k in sub],0); res.append((sub,roc_auc_score(y,p),f1_curve(y,p)[0]))
for sub,a,f in sorted(res,key=lambda x:-x[1])[:12]:
    print(f"  {'+'.join(sub):52s} AUC {a:.5f} F1* {f:.4f}")
print("\n=== top by F1 ===")
for sub,a,f in sorted(res,key=lambda x:-x[2])[:6]:
    print(f"  {'+'.join(sub):52s} AUC {a:.5f} F1* {f:.4f}")
# rank correlation with lgb
print("\n=== rank corr with lgb_tuned ===")
for k in ks: print(f"  {k:22s} {np.corrcoef(R['lgb_tuned'],R[k])[0,1]:.4f}")
# weighted lgb+lr
print("\n=== lgb+lr weight scan (rank space) ===")
for w in np.arange(0.3,0.85,0.05):
    p=w*R['lgb_tuned']+(1-w)*R['logreg']
    print(f"  w_lgb={w:.2f} AUC {roc_auc_score(y,p):.5f} F1* {f1_curve(y,p)[0]:.4f}")
