import sys, numpy as np, pandas as pd, warnings, time, json, os
warnings.filterwarnings('ignore'); sys.path.insert(0,'explore')
from features import build_features, CATS
from harness import oof_cv
import lightgbm as lgb
tr=pd.read_csv('train.csv'); y=tr['accepted'].values
Xc=build_features(tr,1)
BASE=dict(n_estimators=1500,learning_rate=0.03,subsample=0.9,subsample_freq=1,
          colsample_bytree=0.9,random_state=42,verbose=-1)
res=[]
if os.path.exists('explore/sweep_lgb.csv'):
    res=pd.read_csv('explore/sweep_lgb.csv').to_dict('records')
seen={(r['nl'],r['md'],r['mcs'],r['ra'],r['rl']) for r in res}
# coordinate descent from the promising (15,3/5,20,0.5,2.0) region
start=dict(nl=15,md=-1,mcs=20,ra=0.5,rl=2.0)
grids=dict(nl=[7,15,24,31],md=[3,4,5,6,-1],mcs=[5,20,50,100],ra=[0.0,0.25,0.5,1.0,2.0],rl=[0.0,1.0,2.0,5.0,10.0])
cur=dict(start); t0=time.time()
def ev(cfg,nr=3):
    p={**BASE,'num_leaves':cfg['nl'],'max_depth':cfg['md'],'min_child_samples':cfg['mcs'],
       'reg_alpha':cfg['ra'],'reg_lambda':cfg['rl']}
    a,f,_=oof_cv(lambda: lgb.LGBMClassifier(**p),Xc,y,n_repeats=nr,fit_kwargs={'categorical_feature':CATS})
    return a,f
for it in range(3):
    improved=False
    for k,vals in grids.items():
        best=(None,-1,None)
        for v in vals:
            cfg=dict(cur); cfg[k]=v
            key=(cfg['nl'],cfg['md'],cfg['mcs'],cfg['ra'],cfg['rl'])
            hit=[r for r in res if (r['nl'],r['md'],r['mcs'],r['ra'],r['rl'])==key]
            if hit: a,f=hit[0]['auc'],hit[0]['f1']
            else:
                a,f=ev(cfg); res.append(dict(nl=cfg['nl'],md=cfg['md'],mcs=cfg['mcs'],ra=cfg['ra'],rl=cfg['rl'],auc=a,f1=f))
                pd.DataFrame(res).to_csv('explore/sweep_lgb.csv',index=False)
            print(f"  it{it} {k}={v} AUC {a:.5f} F1* {f:.4f} [{time.time()-t0:.0f}s]",flush=True)
            if a>best[1]: best=(v,a,f)
        if best[0]!=cur[k]: cur[k]=best[0]; improved=True
        print(f"-> best {k}={cur[k]}  cfg={cur} [{time.time()-t0:.0f}s]",flush=True)
    if not improved: break
print("\nFINAL cfg:",cur)
a,f=ev(cur,nr=6); print("FINAL 6x5 OOF: AUC %.5f F1* %.4f"%(a,f))
json.dump({k:(int(v) if isinstance(v,(int,np.integer)) else float(v)) for k,v in cur.items()},open('explore/best_lgb.json','w'))
