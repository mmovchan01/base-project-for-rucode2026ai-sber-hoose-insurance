import sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0,'explore')
from harness import oof_cv, CATS
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression

tr=pd.read_csv('train.csv'); y=tr['accepted'].values
SKEW=['average_monthly_balance','loan_amount','smartphone_price','monthly_income','city_population']

def feats(df, v):
    d=df.copy()
    X=pd.DataFrame(index=d.index)
    for c in d.columns:
        if c in ('customer_id','accepted'): continue
        if c in CATS: continue
        X[c]=d[c]
    if v>=1:
        for c in SKEW: X['log_'+c]=np.log1p(d[c].fillna(-1).clip(lower=0)+1)
        for c in ['credit_score','mobile_app_usage']: X[c+'_isna']=d[c].isna().astype(float)
    if v>=2:
        X['income_over_price']=d.monthly_income/(d.smartphone_price+1)
        X['debt_burden']=d.loan_amount.fillna(0)/(d.monthly_income+1)
        X['bal_over_income']=d.average_monthly_balance/(d.monthly_income+1)
        X['loan_over_bal']=d.loan_amount.fillna(0)/(d.average_monthly_balance+1)
        X['phone_residual']=d.smartphone_price/(d.smartphone_age_months+1)
        X['phone_new']=(d.smartphone_age_months<=6).astype(float)
        X['age_x_income']=d.age*d.monthly_income
    if v>=3:
        X['tx_x_online']=d.number_of_card_transactions_month*d.online_payments_share
        X['city_tier']=pd.cut(d.city_population,[-1,1.1e6,2e6,6e6,2e7],labels=[0,1,2,3]).astype(float)
        X['app_x_tx']=d.mobile_app_usage*d.number_of_card_transactions_month
        X['prod_x_previns']=d.number_of_bank_products*d.previous_insurance
    X=pd.concat([X, pd.get_dummies(d[CATS], columns=CATS, drop_first=True).astype(float)],axis=1)
    return X

base=dict(n_estimators=1500,learning_rate=0.03,num_leaves=31,min_child_samples=20,
          subsample=0.9,subsample_freq=1,colsample_bytree=0.9,random_state=42,verbose=-1)
print(f"{'variant':28s} {'AUC':>7s} {'oofF1*':>7s}")
for name,mk in [
  ('lgb v0 raw',      lambda: (0,lgb.LGBMClassifier(**base))),
  ('lgb v1 log+na',   lambda: (1,lgb.LGBMClassifier(**base))),
  ('lgb v2 +ratios',  lambda: (2,lgb.LGBMClassifier(**base))),
  ('lgb v3 +inter',   lambda: (3,lgb.LGBMClassifier(**base))),
  ('lr  v1 log+na',   lambda: (1,'lr')),
  ('lr  v2 +ratios',  lambda: (2,'lr')),
  ('lr  v3 +inter',   lambda: (3,'lr')),
]:
    v,model=mk(); X=feats(tr,v)
    if model=='lr':
        Xf=X.fillna(X.median()); Xs=((Xf-Xf.mean())/Xf.std().replace(0,1)).values
        mkf=lambda: LogisticRegression(max_iter=5000,C=1.0); M=Xs
    else:
        mkf=lambda m=model: m; M=X.values
    auc,f,_=oof_cv(mkf,M,y,n_repeats=3)
    print(f"{name:28s} {auc:7.4f} {f:7.4f}")
