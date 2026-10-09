#!/usr/bin/env python3
"""Regression tests for categorical handling at inference time.

The tree models are fed the *integer codes* of the categorical columns, so the
mapping "level -> code" has to be the one they were fitted on.  Deriving it from
the incoming batch breaks as soon as the batch misses a level (codes shift) or
brings a new one (XGBoost aborts with "Found a category not in the training
set").  On top of that, XGBoost boosters pickled by an older release can carry a
corrupted category container, which has to be detected and bypassed.

Run with:  python tests/test_categorical_levels.py
"""

import os
import sys
import shutil
import tempfile
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features import CAT_COLS, engineer_features  # noqa: E402
from src.models import InsurancePropensityEnsemble  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(REPO_ROOT, 'models')
TRAIN_CSV = os.path.join(REPO_ROOT, 'train.csv')
TEST_CSV = os.path.join(REPO_ROOT, 'public_test.csv')


def _check(name, condition, detail=''):
    status = 'PASS' if condition else 'FAIL'
    print(f'[{status}] {name}{(" - " + detail) if detail else ""}')
    if not condition:
        raise AssertionError(name)


def _small_frame(n=420):
    """A slice of the real training data (keeps every engineered column valid)."""
    return pd.read_csv(TRAIN_CSV).head(n).reset_index(drop=True)


def test_aligned_categories():
    """Levels are pinned: known levels keep their code, unknown ones go missing."""
    ens = InsurancePropensityEnsemble.load(MODELS_DIR)
    test_df = pd.read_csv(TEST_CSV)
    feat, _ = engineer_features(test_df, group_stats=ens.group_stats)
    X = feat[ens.feature_names].copy()
    aligned = ens._aligned_categories(X)

    for col in CAT_COLS:
        _check(
            f'{col}: levels match the training ones',
            list(aligned[col].cat.categories) == list(ens.cat_levels[col]),
        )

    # 'retired' is absent from train.csv -> missing, and it must not shift anyone
    unknown = test_df['employment_type'] == 'retired'
    _check(
        'unknown level -> -1 (missing)',
        (aligned.loc[unknown, 'employment_type'].cat.codes == -1).all(),
    )
    known = ~unknown
    expected = test_df.loc[known, 'employment_type'].map(
        {lvl: i for i, lvl in enumerate(ens.cat_levels['employment_type'])}
    )
    _check(
        'known levels keep their training codes',
        (aligned.loc[known, 'employment_type'].cat.codes.to_numpy() == expected.to_numpy()).all(),
    )


def test_unseen_and_missing_levels_do_not_crash():
    """A batch with an extra level and a missing level still scores fine."""
    ens = InsurancePropensityEnsemble.load(MODELS_DIR)
    test_df = pd.read_csv(TEST_CSV)
    # drop one region entirely and inject a brand new one
    test_df.loc[test_df['region'] == 'Омск', 'region'] = 'Новый Город'
    probs = ens.predict_proba(test_df)
    _check(
        'probabilities are finite and in [0, 1]',
        np.isfinite(probs).all() and probs.min() >= 0.0 and probs.max() <= 1.0,
        f'n={len(probs)}',
    )


def test_stale_container_fallback():
    """When the regular XGBoost path refuses the data, the code path takes over."""
    from xgboost.core import XGBoostError

    ens = InsurancePropensityEnsemble.load(MODELS_DIR)
    test_df = pd.read_csv(TEST_CSV)
    reference = ens.predict_proba(test_df)

    ens = InsurancePropensityEnsemble.load(MODELS_DIR)
    ens._xgb_strategy = 'native'  # pretend the container looks healthy
    for model in ens.xgboost_models:
        model.predict_proba = lambda X: (_ for _ in ()).throw(
            XGBoostError('Found a category not in the training set for the 2th '
                         '(0-based) column: `Екатеринбург`'))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        probs = ens.predict_proba(test_df)

    _check('fallback strategy selected', ens.xgb_scoring_strategy == 'codes',
           ens.xgb_scoring_strategy)
    _check('fallback probabilities match the reference run', np.allclose(probs, reference))
    _check('a warning was emitted', len(caught) > 0)


def test_fit_save_load_roundtrip():
    """Freshly trained models persist their levels and score perturbed batches."""
    train_df = _small_frame()
    tmp_dir = tempfile.mkdtemp(prefix='insurance_models_')
    try:
        ens = InsurancePropensityEnsemble(n_splits=2, seed=42)
        ens.fit(train_df)
        ens.save(tmp_dir)

        with open(os.path.join(tmp_dir, 'metadata.json'), 'r') as f:
            import json
            meta = json.load(f)
        _check('cat_levels persisted', bool(meta.get('cat_levels')))

        reloaded = InsurancePropensityEnsemble.load(tmp_dir)
        _check('levels restored from metadata',
               reloaded.cat_levels == ens.cat_levels)

        # same distribution -> stable predictions
        plain = ens.predict_proba(train_df)
        _check('training data scores fine', np.isfinite(plain).all())

        # perturbed batch: level removed + unknown level added
        odd = train_df.copy()
        odd['region'] = odd['region'].where(odd['region'] != 'Москва', 'Владивосток')
        odd['employment_type'] = odd['employment_type'].where(
            odd['employment_type'] != 'student', 'retired')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            expected = ens.predict_proba(odd)      # in-memory models, native path
            probs = reloaded.predict_proba(odd)    # reloaded models, container bypassed
        _check('perturbed batch scores fine',
               np.isfinite(probs).all() and probs.min() >= 0.0 and probs.max() <= 1.0,
               f'mean={probs.mean():.4f}')
        _check('reloaded models reproduce the in-memory predictions',
               np.allclose(expected, probs),
               f"strategy={reloaded.xgb_scoring_strategy}, max|delta|="
               f"{np.abs(expected - probs).max():.2e}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def main():
    tests = [
        test_aligned_categories,
        test_unseen_and_missing_levels_do_not_crash,
        test_stale_container_fallback,
    ]
    if os.path.exists(TRAIN_CSV):
        tests.append(test_fit_save_load_roundtrip)
    for test in tests:
        print(f'\n--- {test.__name__} ---')
        test()
    print('\nAll checks passed.')


if __name__ == '__main__':
    main()
