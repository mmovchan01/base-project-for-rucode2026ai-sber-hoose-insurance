import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
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
        # Category levels (one list per CAT_COL) as seen by the models during training.
        # They are persisted in metadata.json and are what keeps the integer codes of
        # the tree models stable between training and inference.
        self.cat_levels: Dict[str, List[str]] = {}
        # Per-fold levels recovered from the LightGBM twins (used to validate the
        # category containers that pickled XGBoost boosters carry around).
        self._fold_cat_levels: List[Dict[str, List[str]]] = []
        # How XGBoost probabilities are produced: "native" | "codes" | "repaired".
        self._xgb_strategy: Optional[str] = None
        
    def fit(self, train_df: pd.DataFrame) -> Dict[str, Any]:
        """Train the ensemble on training data with Stratified K-Fold CV."""
        seed_everything(self.seed)
        # A fresh fit redefines the category levels (and rebuilds every model)
        self.cat_levels = {}
        self._fold_cat_levels = []
        self._xgb_strategy = None
        
        # 1. Feature Engineering
        train_feat, self.group_stats = engineer_features(train_df)
        self.feature_names = [c for c in train_feat.columns if c not in ['customer_id', 'accepted']]
        
        X = train_feat[self.feature_names].copy()
        y = train_df['accepted'].values.astype(int)
        
        # Categorical handling for GBDT models
        X_cb = X.copy()
        for c in CAT_COLS:
            X_cb[c] = X_cb[c].astype(str)
            
        # Fixed categorical levels -> identical integer codes at train and inference time
        X_tree = self._aligned_categories(X)
        self._capture_cat_levels(X_tree)
        
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
        
        # XGBoost mangles non-ASCII levels in its own category container, so the
        # freshly fitted boosters may already need the code based scoring path.
        self._fold_cat_levels = self._recover_fold_cat_levels()
        self._check_xgb_categories()
        
        return self.cv_metrics

    def predict_proba(self, test_df: pd.DataFrame) -> np.ndarray:
        """Generate ensemble probability predictions on unseen test data."""
        test_feat, _ = engineer_features(test_df, group_stats=self.group_stats)
        
        X = test_feat[self.feature_names].copy()
        
        X_cb = X.copy()
        for c in CAT_COLS:
            X_cb[c] = X_cb[c].astype(str)
            
        # Categorical levels are pinned to the training ones, so the integer codes
        # fed to LightGBM/XGBoost are exactly the codes they were fitted on.
        X_tree = self._aligned_categories(X)
            
        cb_d6_preds = np.mean([model.predict_proba(X_cb)[:, 1] for model in self.catboost_models], axis=0)
        cb_d5_preds = np.mean([model.predict_proba(X_cb)[:, 1] for model in self.catboost_d5_models], axis=0)
        lgb_preds = np.mean([model.predict_proba(X_tree)[:, 1] for model in self.lightgbm_models], axis=0)
        xgb_preds = self._predict_xgboost(X_tree)
        
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

    # ------------------------------------------------------------------
    # Categorical encoding
    # ------------------------------------------------------------------
    def _aligned_categories(self, X: pd.DataFrame) -> pd.DataFrame:
        """Cast CAT_COLS to the categorical dtype the models were fitted on.

        Deriving the levels from the incoming data (``astype('category')``) is
        unsafe: a test batch that misses a level - or, as in ``public_test.csv``,
        brings an extra one (``employment_type == 'retired'``) - shifts every
        integer code and silently corrupts the XGBoost/LightGBM predictions.
        Pinning the levels also turns genuinely unknown categories into NaN
        ("missing") instead of letting XGBoost abort with
        ``Found a category not in the training set ...``.
        """
        X_tree = X.copy()
        for col in CAT_COLS:
            if col not in X_tree.columns:
                continue
            values = X_tree[col].astype(str)
            levels = self.cat_levels.get(col)
            if not levels:
                X_tree[col] = values.astype('category')
                continue
            # Explicit code mapping: unknown levels become -1 (= missing) instead of
            # raising, and the codes of the known ones stay aligned with training.
            mapping = {level: code for code, level in enumerate(levels)}
            codes = values.map(mapping).fillna(-1).to_numpy(dtype=np.int64)
            X_tree[col] = pd.Categorical.from_codes(codes, categories=list(levels))
        return X_tree

    def _capture_cat_levels(self, X_tree: pd.DataFrame) -> None:
        """Remember the training-time category levels (shared by every fold)."""
        self.cat_levels = {
            col: list(X_tree[col].cat.categories)
            for col in CAT_COLS
            if col in X_tree.columns and hasattr(X_tree[col], 'cat')
        }

    def _recover_fold_cat_levels(self) -> List[Dict[str, List[str]]]:
        """Rebuild the training-time levels from the LightGBM twins.

        LightGBM stores ``pandas_categorical`` inside every booster, i.e. the
        categories of the very same frames XGBoost was fitted on.  This is what
        lets models saved before ``cat_levels`` was written to metadata.json be
        served correctly.
        """
        cat_cols = self._cat_columns()
        recovered: List[Dict[str, List[str]]] = []
        for model in self.lightgbm_models:
            booster = getattr(model, 'booster_', None)
            pandas_categorical = getattr(booster, 'pandas_categorical', None)
            if not pandas_categorical or len(pandas_categorical) != len(cat_cols):
                recovered.append({})
            else:
                recovered.append({
                    col: [str(v) for v in levels]
                    for col, levels in zip(cat_cols, pandas_categorical)
                })
        return recovered

    def _cat_columns(self) -> List[str]:
        """Categorical features in the order they appear in the model matrix."""
        return [c for c in self.feature_names if c in CAT_COLS]

    @property
    def xgb_scoring_strategy(self) -> str:
        """XGBoost scoring path in use: 'native', 'codes' or 'repaired'."""
        return self._xgb_strategy or 'native'

    # ------------------------------------------------------------------
    # XGBoost prediction (robust against stale category containers)
    # ------------------------------------------------------------------
    def _predict_xgboost(self, X_tree: pd.DataFrame) -> np.ndarray:
        """Blend the XGBoost folds, falling back to safer prediction paths.

        XGBoost stores a corrupted category container whenever the levels are
        not ASCII (here the Cyrillic ``region`` values come back as
        ``'Екатер' + 'инб' + ...`` - reproduced with XGBoost 3.2.0, and pickles
        written by older releases show the same damage).  Recent XGBoost
        validates the incoming categories against that container and raises
        ``Found a category not in the training set ...`` even for perfectly
        valid rows, so those boosters have to be scored through a path that
        does not consult the container:

        * ``native``   - the regular ``predict_proba`` call;
        * ``codes``    - the matrix of (aligned) category codes is passed as a
                         plain numpy array, which XGBoost consumes without any
                         categorical validation.  The trees split on those very
                         codes, so the result is identical;
        * ``repaired`` - the booster is reloaded from its own model dump, which
                         drops the stale container.
        """
        strategies = ('native', 'codes', 'repaired')
        if self._xgb_strategy in strategies:
            strategies = (self._xgb_strategy,) + tuple(s for s in strategies if s != self._xgb_strategy)

        X_num = None
        fold_preds = []
        for model in self.xgboost_models:
            last_error: Optional[Exception] = None
            for strategy in strategies:
                try:
                    if strategy == 'native':
                        preds = model.predict_proba(X_tree)[:, 1]
                    else:
                        if strategy == 'codes':
                            if X_num is None:
                                X_num = self._category_codes_matrix(X_tree)
                            preds = self._xgb_predict_codes(model, X_num)
                        else:
                            preds = self._xgb_predict_repaired(model, X_tree)
                    if strategy != self._xgb_strategy:
                        if strategy != 'native':
                            warnings.warn(
                                "XGBoost models could not be scored through the regular "
                                f"path ({last_error.__class__.__name__}: {last_error}); "
                                f"using the '{strategy}' fallback instead.",
                                RuntimeWarning,
                            )
                        self._xgb_strategy = strategy
                    break
                except Exception as exc:  # noqa: BLE001 - try the next strategy
                    last_error = exc
            else:
                raise RuntimeError(
                    "XGBoost scoring failed for every available prediction path "
                    f"({', '.join(strategies)}). Last error: {last_error}"
                ) from last_error
            fold_preds.append(preds)

        return np.mean(fold_preds, axis=0)

    def _category_codes_matrix(self, X_tree: pd.DataFrame) -> np.ndarray:
        """Numeric matrix whose categorical columns hold the training-code values."""
        X_num = X_tree.copy()
        for col in CAT_COLS:
            if col in X_num.columns and isinstance(X_num[col].dtype, pd.CategoricalDtype):
                codes = X_num[col].cat.codes.to_numpy(dtype=np.float64)
                codes[codes < 0] = np.nan  # unknown category -> missing
                X_num[col] = codes
        return X_num.to_numpy(dtype=np.float64)

    @staticmethod
    def _xgb_iteration_range(model) -> Tuple[int, int]:
        """Mirror the iteration range scikit-learn's ``predict`` would use."""
        try:
            return (0, int(model.get_booster().best_iteration) + 1)
        except Exception:  # noqa: BLE001 - no early stopping -> use all trees
            return (0, 0)

    def _xgb_predict_codes(self, model, X_num: np.ndarray) -> np.ndarray:
        booster = model.get_booster()
        return booster.inplace_predict(
            X_num,
            iteration_range=self._xgb_iteration_range(model),
            predict_type='value',
            missing=getattr(model, 'missing', np.nan),
        )

    def _xgb_predict_repaired(self, model, X_tree: pd.DataFrame) -> np.ndarray:
        booster = model.get_booster()
        fresh = xgb.Booster()
        fresh.load_model(bytearray(booster.save_raw('json')))
        preds = fresh.inplace_predict(
            X_tree,
            iteration_range=self._xgb_iteration_range(model),
            predict_type='value',
            missing=getattr(model, 'missing', np.nan),
        )
        model._Booster = fresh
        return preds

    def _xgb_categories_are_stale(self, model, expected: Dict[str, List[str]]) -> bool:
        """True when a booster's stored category container cannot be trusted.

        XGBoost mangles non-ASCII levels when it builds that container
        (``region`` comes back as ``'Екатер' + 'инб' + ...``), so any container
        that disagrees with the LightGBM twins - which were fitted on the very
        same frames - is unusable.
        """
        booster = model.get_booster()
        getter = getattr(booster, 'get_categories', None)
        if getter is None:
            return False  # XGBoost < 3.1 stores no container at all
        try:
            cats = getter(export_to_arrow=True)
            empty = cats.empty
            if empty() if callable(empty) else empty:
                return False
            table = list(cats.to_arrow())
        except Exception:  # noqa: BLE001 - container not inspectable here
            return False

        for col, levels in expected.items():
            if col not in self.feature_names:
                continue
            stored = table[self.feature_names.index(col)][1]
            if stored is None:
                continue
            try:
                stored_levels = [value.as_py() for value in stored]
            except Exception:  # noqa: BLE001 - broken UTF-8 payload
                return True
            if stored_levels != [str(level) for level in levels]:
                return True
        return False

    def _check_xgb_categories(self) -> None:
        """Warn (and switch strategy) when pickled containers are unusable."""
        for fold, model in enumerate(self.xgboost_models):
            expected = (
                self._fold_cat_levels[fold]
                if fold < len(self._fold_cat_levels) and self._fold_cat_levels[fold]
                else self.cat_levels
            )
            if not expected:
                continue
            if self._xgb_categories_are_stale(model, expected):
                self._xgb_strategy = 'codes'
                warnings.warn(
                    "The XGBoost boosters carry a corrupted categorical container "
                    "(XGBoost mangles non-ASCII levels such as the Cyrillic `region` "
                    "values). Scoring them through the category-code path, which "
                    "reproduces the training-time encoding exactly and avoids the "
                    "'Found a category not in the training set' error.",
                    RuntimeWarning,
                )
                return

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
            'cv_metrics': self.cv_metrics,
            'cat_levels': self.cat_levels
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
        # Category levels as seen during training. Models saved before they were
        # persisted get them back from the LightGBM twins below.
        instance.cat_levels = meta.get('cat_levels') or {}
        
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
            
        # Restore the training-time category levels for models saved without them
        instance._fold_cat_levels = instance._recover_fold_cat_levels()
        if not instance.cat_levels:
            for fold_levels in instance._fold_cat_levels:
                if fold_levels:
                    instance.cat_levels = fold_levels
                    break
        instance._check_xgb_categories()
        
        return instance
