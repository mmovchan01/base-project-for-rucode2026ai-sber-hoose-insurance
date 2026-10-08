import pandas as pd, numpy as np
from scipy.stats import spearmanr
tr=pd.read_csv('train.csv'); y=tr['accepted'].values
print("=== CORRECTED spearman (non-NaN subset) ===")
rows=[]
for c in tr.columns:
    if c in ('customer_id','accepted') or tr[c].dtype==object: continue
    m=tr[c].notna().values
    if m.sum()<50: continue
    sp=spearmanr(tr[c].values[m], y[m]).statistic
    pe=np.corrcoef(tr[c].values[m], y[m])[0,1]
    rows.append((c, round(pe,4), round(sp,4), round(1-m.mean(),3)))
print(pd.DataFrame(rows,columns=['feat','pearson','spearman','na']).sort_values('spearman',key=abs,ascending=False).to_string(index=False))

# credit_score shape
cs=tr[tr.credit_score.notna()]
print("\ncredit_score: n=%d  pearson=%.4f"%(len(cs), np.corrcoef(cs.credit_score,y[cs.index])[0,1]))
print(pd.qcut(cs.credit_score,10,duplicates='drop').value_counts().sort_index().to_dict())
b=pd.qcut(cs.credit_score,10,duplicates='drop')
print(pd.DataFrame({'y':y[cs.index]}).groupby(b,observed=True)['y'].agg(['mean','count']).round(3).to_string())
