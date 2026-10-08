import os
import json
import pickle
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier
import lightgbm as lgb
import xgboost as xgb

from src.features import CAT_COLS, engineer_features, compute_group_stats
from src.utils import find_optimal_threshold, evaluate_predictions, seed_everything

class InsurancePropensityEnsemble:
    """
    Ensemble Model for Insurance Acceptance Classification.
    Combines diverse CatBoost, LightGBM, and XGBoost models trained on Stratified K-Fold CV.
    """
    def __init__(self, n_splits: int = 5, seed: int = 42):
        self.n_splits = n_splits
        self.seed = seed
        self.catboost_models: List[CatBoostClassifier] = []
        self.catboost_d5_models: List[CatBoostClassifier] = []
        self.lightgbm_models: List[lgb.LGBMClassifier] = []
        self.xgboost_models: List[xgb.XGBClassifier] = []
        
        self.weights = {
            'catboost_d6': 0.40,
            'catboost_d5': 0.30,
            'lightgbm': 0.15,
            'xgboost': 0.15
        }
        self.feature_names: List[str] = []
        self.group_stats: Dict[str, pd.DataFrame] = {}
        self.optimal_threshold: float = 0.5
        self.cv_metrics: Dict[str, Any] = {}
        
    def fit(self, train_df: pd.DataFrame) -> Dict[str, Any]:
        """Train the ensemble on training data with Stratified K-Fold CV."""
        seed_everything(self.seed)
        
        # 1. Feature Engineering
        train_feat, self.group_stats = engineer_features(train_df)
        self.feature_names = [c for c in train_feat.columns if c not in ['customer_id', 'accepted']]
        
        X = train_feat[self.feature_names].copy()
        y = train_df['accepted'].values.astype(int)
        
        # Categorical handling for GBDT models
        X_cb = X.copy()
        for c in CAT_COLS:
            X_cb[c] = X_cb[c].astype(str)
            
        X_tree = X.copy()
        for c in CAT_COLS:
            X_tree[c] = X_tree[c].astype('category')
            
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        
        oof_cb_d6 = np.zeros(len(y))
        oof_cb_d5 = np.zeros(len(y))
        oof_lgb = np.zeros(len(y))
        oof_xgb = np.zeros(len(y))
        
        self.catboost_models = []
        self.catboost_d5_models = []
        self.lightgbm_models = []
        self.xgboost_models = []
        
        print(f"=== Starting Training with {self.n_splits}-Fold Stratified CV (Seed={self.seed}) ===")
        print(f"Total features: {len(self.feature_names)}")
        
        for fold, (trn_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"--- Training Fold {fold + 1}/{self.n_splits} ---")
            
            X_tr_cb, y_tr = X_cb.iloc[trn_idx], y[trn_idx]
            X_va_cb, y_va = X_cb.iloc[val_idx], y[val_idx]
            X_tr_tree, X_va_tree = X_tree.iloc[trn_idx], X_tree.iloc[val_idx]
            
            # 1. CatBoost Depth 6
            cb_d6 = CatBoostClassifier(
                iterations=850,
                learning_rate=0.045,
                depth=6,
                l2_leaf_reg=4.0,
                cat_features=CAT_COLS,
                eval_metric='Logloss',
                random_seed=self.seed + fold,
                verbose=0,
                thread_count=2
            )
            cb_d6.fit(X_tr_cb, y_tr, eval_set=(X_va_cb, y_va), early_stopping_rounds=50)
            oof_cb_d6[val_idx] = cb_d6.predict_proba(X_va_cb)[:, 1]
            self.catboost_models.append(cb_d6)
            
            # 2. CatBoost Depth 5
            cb_d5 = CatBoostClassifier(
                iterations=850,
                learning_rate=0.045,
                depth=5,
                l2_leaf_reg=3.0,
                cat_features=CAT_COLS,
                eval_metric='Logloss',
                random_seed=self.seed + 100 + fold,
                verbose=0,
                thread_count=2
            )
            cb_d5.fit(X_tr_cb, y_tr, eval_set=(X_va_cb, y_va), early_stopping_rounds=50)
            oof_cb_d5[val_idx] = cb_d5.predict_proba(X_va_cb)[:, 1]
            self.catboost_d5_models.append(cb_d5)
            
            # 3. LightGBM
            lgbm = lgb.LGBMClassifier(
                n_estimators=850,
                learning_rate=0.035,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.seed + fold,
                verbose=-1,
                n_jobs=2
            )
            lgbm.fit(X_tr_tree, y_tr, eval_set=[(X_va_tree, y_va)], callbacks=[lgb.early_stopping(50, verbose=False)], eval_metric='logloss')
            oof_lgb[val_idx] = lgbm.predict_proba(X_va_tree)[:, 1]
            self.lightgbm_models.append(lgbm)
            
            # 4. XGBoost
            xgb_m = xgb.XGBClassifier(
                n_estimators=850,
                learning_rate=0.035,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                enable_categorical=True,
                random_state=self.seed + fold,
                eval_metric='logloss',
                early_stopping_rounds=50,
                n_jobs=2
            )
            xgb_m.fit(X_tr_tree, y_tr, eval_set=[(X_va_tree, y_va)], verbose=False)
            oof_xgb[val_idx] = xgb_m.predict_proba(X_va_tree)[:, 1]
            self.xgboost_models.append(xgb_m)
        
        # Ensemble Blending
        oof_ensemble = (
            self.weights['catboost_d6'] * oof_cb_d6 +
            self.weights['catboost_d5'] * oof_cb_d5 +
            self.weights['lightgbm'] * oof_lgb +
            self.weights['xgboost'] * oof_xgb
        )
        
        # Optimize Decision Threshold on Out-of-Fold predictions
        best_thresh, best_f1 = find_optimal_threshold(y, oof_ensemble)
        self.optimal_threshold = best_thresh
        
        # Metrics report
        cb_d6_metrics = evaluate_predictions(y, oof_cb_d6, threshold=find_optimal_threshold(y, oof_cb_d6)[0])
        cb_d5_metrics = evaluate_predictions(y, oof_cb_d5, threshold=find_optimal_threshold(y, oof_cb_d5)[0])
        lgb_metrics = evaluate_predictions(y, oof_lgb, threshold=find_optimal_threshold(y, oof_lgb)[0])
        xgb_metrics = evaluate_predictions(y, oof_xgb, threshold=find_optimal_threshold(y, oof_xgb)[0])
        ensemble_metrics = evaluate_predictions(y, oof_ensemble, threshold=self.optimal_threshold)
        
        self.cv_metrics = {
            'catboost_d6': cb_d6_metrics,
            'catboost_d5': cb_d5_metrics,
            'lightgbm': lgb_metrics,
            'xgboost': xgb_metrics,
            'ensemble': ensemble_metrics,
            'optimal_threshold': self.optimal_threshold
        }
        
        print("\n=== Cross-Validation Results ===")
        print(f"CatBoost (Depth 6) : F1={cb_d6_metrics['f1_score']:.5f}, AUC={cb_d6_metrics['roc_auc']:.5f}")
        print(f"CatBoost (Depth 5) : F1={cb_d5_metrics['f1_score']:.5f}, AUC={cb_d5_metrics['roc_auc']:.5f}")
        print(f"LightGBM           : F1={lgb_metrics['f1_score']:.5f}, AUC={lgb_metrics['roc_auc']:.5f}")
        print(f"XGBoost            : F1={xgb_metrics['f1_score']:.5f}, AUC={xgb_metrics['roc_auc']:.5f}")
        print(f"Final Ensemble     : F1={ensemble_metrics['f1_score']:.5f}, AUC={ensemble_metrics['roc_auc']:.5f}, Accuracy={ensemble_metrics['accuracy']:.5f}, Precision={ensemble_metrics['precision']:.5f}, Recall={ensemble_metrics['recall']:.5f}")
        print(f"Calibrated Threshold: {self.optimal_threshold}")
        
        return self.cv_metrics

    def predict_proba(self, test_df: pd.DataFrame) -> np.ndarray:
        """Generate ensemble probability predictions on unseen test data."""
        test_feat, _ = engineer_features(test_df, group_stats=self.group_stats)
        
        X = test_feat[self.feature_names].copy()
        
        X_cb = X.copy()
        for c in CAT_COLS:
            X_cb[c] = X_cb[c].astype(str)
            
        X_tree = X.copy()
        for c in CAT_COLS:
            X_tree[c] = X_tree[c].astype('category')
            
        cb_d6_preds = np.mean([model.predict_proba(X_cb)[:, 1] for model in self.catboost_models], axis=0)
        cb_d5_preds = np.mean([model.predict_proba(X_cb)[:, 1] for model in self.catboost_d5_models], axis=0)
        lgb_preds = np.mean([model.predict_proba(X_tree)[:, 1] for model in self.lightgbm_models], axis=0)
        xgb_preds = np.mean([model.predict_proba(X_tree)[:, 1] for model in self.xgboost_models], axis=0)
        
        ensemble_preds = (
            self.weights['catboost_d6'] * cb_d6_preds +
            self.weights['catboost_d5'] * cb_d5_preds +
            self.weights['lightgbm'] * lgb_preds +
            self.weights['xgboost'] * xgb_preds
        )
        return ensemble_preds

    def predict(self, test_df: pd.DataFrame, threshold: float = None) -> np.ndarray:
        """Generate binary 0/1 predictions using the optimal threshold."""
        if threshold is None:
            threshold = self.optimal_threshold
        probs = self.predict_proba(test_df)
        return (probs >= threshold).astype(int)

    def save(self, model_dir: str = 'models') -> None:
        """Save all ensemble weights and metadata to disk."""
        os.makedirs(model_dir, exist_ok=True)
        
        # Save models
        with open(os.path.join(model_dir, 'catboost_d6_models.pkl'), 'wb') as f:
            pickle.dump(self.catboost_models, f)
        with open(os.path.join(model_dir, 'catboost_d5_models.pkl'), 'wb') as f:
            pickle.dump(self.catboost_d5_models, f)
        with open(os.path.join(model_dir, 'lightgbm_models.pkl'), 'wb') as f:
            pickle.dump(self.lightgbm_models, f)
        with open(os.path.join(model_dir, 'xgboost_models.pkl'), 'wb') as f:
            pickle.dump(self.xgboost_models, f)
            
        # Save group stats
        with open(os.path.join(model_dir, 'group_stats.pkl'), 'wb') as f:
            pickle.dump(self.group_stats, f)
            
        # Save metadata JSON
        meta = {
            'n_splits': self.n_splits,
            'seed': self.seed,
            'weights': self.weights,
            'feature_names': self.feature_names,
            'optimal_threshold': self.optimal_threshold,
            'cv_metrics': self.cv_metrics
        }
        with open(os.path.join(model_dir, 'metadata.json'), 'w') as f:
            json.dump(meta, f, indent=2)
            
        print(f"Successfully saved all model artifacts to '{model_dir}/'")

    @classmethod
    def load(cls, model_dir: str = 'models') -> 'InsurancePropensityEnsemble':
        """Load trained ensemble weights and metadata from disk."""
        with open(os.path.join(model_dir, 'metadata.json'), 'r') as f:
            meta = json.load(f)
            
        instance = cls(n_splits=meta['n_splits'], seed=meta['seed'])
        instance.weights = meta['weights']
        instance.feature_names = meta['feature_names']
        instance.optimal_threshold = meta['optimal_threshold']
        instance.cv_metrics = meta['cv_metrics']
        
        with open(os.path.join(model_dir, 'catboost_d6_models.pkl'), 'rb') as f:
            instance.catboost_models = pickle.load(f)
        with open(os.path.join(model_dir, 'catboost_d5_models.pkl'), 'rb') as f:
            instance.catboost_d5_models = pickle.load(f)
        with open(os.path.join(model_dir, 'lightgbm_models.pkl'), 'rb') as f:
            instance.lightgbm_models = pickle.load(f)
        with open(os.path.join(model_dir, 'xgboost_models.pkl'), 'rb') as f:
            instance.xgboost_models = pickle.load(f)
        with open(os.path.join(model_dir, 'group_stats.pkl'), 'rb') as f:
            instance.group_stats = pickle.load(f)
            
        return instance
