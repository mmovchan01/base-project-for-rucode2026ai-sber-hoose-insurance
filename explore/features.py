import numpy as np, pandas as pd

CATS = ['gender','region','family_status','education','employment_type','smartphone_brand']
KNOWN = {
 'gender':['F','M'],
 'region':['Екатеринбург','Казань','Краснодар','Москва','Нижний Новгород','Новосибирск',
           'Омск','Ростов-на-Дону','Самара','Санкт-Петербург','Уфа','Челябинск'],
 'family_status':['divorced','married','single','widowed'],
 'education':['secondary','bachelor','master','phd'],
 'employment_type':['employed','retired','self_employed','student','unemployed'],
 'smartphone_brand':['Apple','Google','Huawei','OnePlus','Other','Samsung','Xiaomi'],
}
NUM = ['age','city_population','children','monthly_income','owns_car','owns_house','years_with_bank',
 'number_of_bank_products','average_monthly_balance','has_credit','loan_amount','credit_score',
 'number_of_card_transactions_month','online_payments_share','mobile_app_usage','smartphone_price',
 'smartphone_age_months','marketing_contacts_last_year','previous_campaign_response',
 'previous_insurance','insurance_claims']
SKEW = ['average_monthly_balance','loan_amount','smartphone_price','monthly_income','city_population']

def build_features(df, version=1):
    d = df.reset_index(drop=True)
    X = pd.DataFrame(index=d.index)
    for c in NUM:
        X[c] = pd.to_numeric(d[c], errors='coerce')
    if version >= 1:
        for c in SKEW:
            X['log_'+c] = np.log1p(X[c].fillna(-1).clip(lower=0) + 1)
        for c in ['credit_score','mobile_app_usage']:
            X[c+'_isna'] = X[c].isna().astype(float)
    if version >= 2:
        X['income_over_price'] = X.monthly_income/(X.smartphone_price+1)
        X['debt_burden'] = X.loan_amount.fillna(0)/(X.monthly_income+1)
        X['bal_over_income'] = X.average_monthly_balance/(X.monthly_income+1)
        X['loan_over_bal'] = X.loan_amount.fillna(0)/(X.average_monthly_balance+1)
        X['phone_residual'] = X.smartphone_price/(X.smartphone_age_months+1)
    for c in CATS:
        s = d[c].astype(str).where(d[c].astype(str).isin(KNOWN[c]), 'UNSEEN')
        X[c] = pd.Categorical(s, categories=KNOWN[c]+['UNSEEN']).codes.astype('float32')
    return X

def onehot(df, version=1):
    """One-hot variant for linear models."""
    X = build_features(df, version)
    X = X.drop(columns=CATS)
    for c in CATS:
        raw = df.reset_index(drop=True)[c].astype(str)
        for lv in KNOWN[c]:
            X[f'{c}={lv}'] = (raw == lv).astype(float)
    return X.fillna(X.median())
