"""Ранговая агрегация против вероятностной: какой способ порога лучше?
Сравниваем честным nested CV на финальной модели."""
import sys, numpy as np, pandas as pd, warnings, time
warnings.filterwarnings('ignore'); sys.path.insert(0,'.')
from common import TARGET, build_features, build_onehot, ALL_MODELS, prepare_for, rank01
from train import pick_threshold, f1_score_at, roc_auc
from sklearn.model_selection import StratifiedKFold

df=pd.read_csv('train.csv'); y=df[TARGET].to_numpy(int)
Xb,Xo=build_features(df),build_onehot(df)
fac,rk,fk=ALL_MODELS['lr_spline']

def nested(agg, splits=5, repeats=4, seed=42):
    f1s=[]
    for rep in range(repeats):
        for tr_i,te_i in StratifiedKFold(splits,shuffle=True,random_state=seed+7919*rep).split(Xb,y):
            ytr=y[tr_i]
            ip=np.zeros(len(ytr))
            for a_i,b_i in StratifiedKFold(4,shuffle=True,random_state=1000+rep).split(Xb.iloc[tr_i],ytr):
                m=fac(seed).fit(Xo.iloc[tr_i].iloc[a_i],ytr[a_i],**fk)
                ip[b_i]=m.predict_proba(Xo.iloc[tr_i].iloc[b_i])[:,1]
            ip = rank01(ip) if agg=='rank' else ip
            thr,_=pick_threshold(ytr,ip)
            m=fac(seed).fit(Xo.iloc[tr_i],ytr,**fk)
            p=m.predict_proba(Xo.iloc[te_i])[:,1]
            p = rank01(p) if agg=='rank' else p
            f1s.append(f1_score_at(y[te_i],p,thr))
    return np.array(f1s)

for agg in ['rank','prob']:
    t0=time.time(); f=nested(agg)
    print(f"агрегация {agg:5s}: F1 = {f.mean():.4f} +/- {f.std(ddof=1)/np.sqrt(len(f)):.4f}  [{time.time()-t0:.0f}s]")
