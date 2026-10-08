import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict

CAT_COLS = [
    'gender',
    'region',
    'family_status',
    'education',
    'employment_type',
    'smartphone_brand'
]

def compute_group_stats(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Compute reference group statistics from training data."""
    brand_stats = df.groupby('smartphone_brand')['smartphone_price'].agg(['mean', 'std']).rename(
        columns={'mean': 'brand_mean_price', 'std': 'brand_std_price'}
    ).reset_index()
    
    region_stats = df.groupby('region')['monthly_income'].agg(['mean', 'std']).rename(
        columns={'mean': 'region_mean_income', 'std': 'region_std_income'}
    ).reset_index()
    
    return {
        'brand_stats': brand_stats,
        'region_stats': region_stats
    }

def engineer_features(
    df_raw: pd.DataFrame, 
    group_stats: Optional[Dict[str, pd.DataFrame]] = None
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Apply comprehensive domain feature engineering for insurance propensity prediction.
    """
    df = df_raw.copy()
    
    # 1. Missing Value Indicators & Domain Logical Imputations
    df['has_loan_amount'] = (~df['loan_amount'].isna()).astype(int)
    df['has_credit_score'] = (~df['credit_score'].isna()).astype(int)
    df['has_mobile_app'] = (~df['mobile_app_usage'].isna()).astype(int)
    df['has_insurance_claims'] = (~df['insurance_claims'].isna()).astype(int)
    
    # Domain imputation: loan_amount is 0 if no credit, insurance_claims is 0 if no prev insurance
    df['loan_amount_filled'] = df['loan_amount'].fillna(0.0)
    df['insurance_claims_filled'] = df['insurance_claims'].fillna(0.0)
    df['mobile_app_usage_filled'] = df['mobile_app_usage'].fillna(0.0)
    median_credit = df['credit_score'].median() if not df['credit_score'].dropna().empty else 650.0
    df['credit_score_filled'] = df['credit_score'].fillna(median_credit)
    
    # 2. Financial Ratios & Liquidity Proxies
    df['price_to_income'] = df['smartphone_price'] / (df['monthly_income'] + 1.0)
    df['balance_to_income'] = df['average_monthly_balance'] / (df['monthly_income'] + 1.0)
    df['loan_to_income'] = df['loan_amount_filled'] / (df['monthly_income'] + 1.0)
    df['income_per_child'] = df['monthly_income'] / (df['children'] + 1.0)
    df['net_liquid_wealth'] = df['average_monthly_balance'] - df['loan_amount_filled']
    df['total_annual_liquidity'] = df['monthly_income'] * 12.0 + df['average_monthly_balance']
    df['phone_affordability_index'] = (df['average_monthly_balance'] + df['monthly_income'] * 3.0) / (df['smartphone_price'] + 1.0)
    
    # 3. Smartphone Valuation & Life-cycle Metrics
    df['price_per_month_age'] = df['smartphone_price'] / (df['smartphone_age_months'] + 1.0)
    df['estimated_current_value'] = df['smartphone_price'] * np.maximum(0.1, (1.0 - 0.02 * df['smartphone_age_months']))
    df['is_apple'] = (df['smartphone_brand'] == 'Apple').astype(int)
    df['is_samsung'] = (df['smartphone_brand'] == 'Samsung').astype(int)
    df['is_xiaomi'] = (df['smartphone_brand'] == 'Xiaomi').astype(int)
    df['is_flagship'] = (df['smartphone_price'] >= 60000).astype(int)
    df['is_budget'] = (df['smartphone_price'] <= 20000).astype(int)
    df['is_new_phone'] = (df['smartphone_age_months'] <= 6).astype(int)
    df['is_old_phone'] = (df['smartphone_age_months'] >= 24).astype(int)
    
    # 4. Digital Engagement & Daily Banking Activity
    df['digital_intensity'] = df['mobile_app_usage_filled'] * df['online_payments_share']
    df['online_tx_count'] = df['number_of_card_transactions_month'] * df['online_payments_share']
    df['offline_tx_count'] = df['number_of_card_transactions_month'] * (1.0 - df['online_payments_share'])
    df['digital_native_score'] = df['online_payments_share'] * (1.0 - df['age'] / 100.0) * (df['mobile_app_usage_filled'] + 1.0)
    df['card_tx_daily'] = df['number_of_card_transactions_month'] / 30.0
    
    # 5. Bank Relationship & Creditworthiness
    df['products_per_year'] = df['number_of_bank_products'] / (df['years_with_bank'] + 1.0)
    df['balance_per_product'] = df['average_monthly_balance'] / (df['number_of_bank_products'] + 1.0)
    df['credit_to_balance'] = df['loan_amount_filled'] / (df['average_monthly_balance'] + 1.0)
    df['bank_relationship_score'] = df['years_with_bank'] * df['number_of_bank_products']
    
    # 6. Insurance Propensity & Marketing Responsiveness
    df['claims_per_insurance'] = df['insurance_claims_filled'] / (df['previous_insurance'] + 1.0)
    df['had_claims'] = (df['insurance_claims_filled'] > 0).astype(int)
    df['campaign_and_insurance'] = df['previous_campaign_response'] * df['previous_insurance']
    df['marketing_responsiveness'] = df['previous_campaign_response'] / (df['marketing_contacts_last_year'] + 1.0)
    df['insurance_propensity'] = (
        df['previous_insurance'] * 2.0 +
        df['previous_campaign_response'] * 2.0 +
        (df['insurance_claims_filled'] > 0).astype(int)
    )
    
    # 7. Demographics & Geographic Tiers
    df['is_capital'] = df['region'].isin(['Москва', 'Санкт-Петербург']).astype(int)
    df['is_young'] = (df['age'] < 30).astype(int)
    df['is_senior'] = (df['age'] >= 60).astype(int)
    
    # 8. Group Aggregations
    if group_stats is None:
        group_stats = compute_group_stats(df)
        
    df = df.merge(group_stats['brand_stats'], on='smartphone_brand', how='left')
    df['brand_mean_price'] = df['brand_mean_price'].fillna(df['smartphone_price'].mean())
    df['brand_std_price'] = df['brand_std_price'].fillna(0.0)
    df['price_to_brand_mean'] = df['smartphone_price'] / (df['brand_mean_price'] + 1.0)
    df['price_diff_brand_mean'] = df['smartphone_price'] - df['brand_mean_price']
    
    df = df.merge(group_stats['region_stats'], on='region', how='left')
    df['region_mean_income'] = df['region_mean_income'].fillna(df['monthly_income'].mean())
    df['region_std_income'] = df['region_std_income'].fillna(0.0)
    df['income_to_region_mean'] = df['monthly_income'] / (df['region_mean_income'] + 1.0)
    df['phone_to_region_income'] = df['smartphone_price'] / (df['region_mean_income'] + 1.0)
    
    # 9. Non-linear / Log Transforms for skewed distributions
    df['log_income'] = np.log1p(np.maximum(0, df['monthly_income']))
    df['log_balance'] = np.log1p(np.maximum(0, df['average_monthly_balance']))
    df['log_price'] = np.log1p(np.maximum(0, df['smartphone_price']))
    df['log_population'] = np.log1p(df['city_population'])
    df['log_loan'] = np.log1p(np.maximum(0, df['loan_amount_filled']))
    
    # Ensure categorical types are consistent strings
    for col in CAT_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str)
            
    return df, group_stats
