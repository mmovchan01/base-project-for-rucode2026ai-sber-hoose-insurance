"""
evaluate.py -- НЕСМЕЩЁННАЯ оценка пайплайна и честный выбор состава ансамбля.

Зачем: в train.py порог подбирается по тем же OOF-предсказаниям, на которых
считается F1, -- это даёт оптимистичное смещение. Здесь и подбор порога, и
обучение происходят внутри тренировочной части каждой внешней фолды, а F1
меряется на отложенной части, которую не видели ни модель, ни порог.

Запуск:
    python evaluate.py                                  # nested CV финального ансамбля
    python evaluate.py --compare-candidates             # честный выбор состава
    python evaluate.py --splits 5 --repeats 3 --seed 42
"""
from __future__ import annotations

import argparse
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from common import (ALL_MODELS, FINAL_ENSEMBLE, TARGET, build_features,
                    build_onehot, prepare_for, rank01)
from train import f1_score_at, pick_threshold, roc_auc


def _rows(X, idx):
    return X.iloc[idx]


def _fit_ensemble(names, seed, Xb_tr, Xo_tr, y_tr, tr_idx=None):
    """Обучает все модели ансамбля (на tr_idx внутри Xb_tr/Xo_tr, если задан)
    и возвращает список обученных моделей."""
    models = []
    for n in names:
        factory, repr_kind, fit_kw = ALL_MODELS[n]
        model = factory(seed)          # все фабрики в ALL_MODELS принимают seed
        X = prepare_for(repr_kind,
                        Xb_tr if tr_idx is None else _rows(Xb_tr, tr_idx),
                        Xo_tr if tr_idx is None else _rows(Xo_tr, tr_idx))
        yy = y_tr if tr_idx is None else y_tr[tr_idx]
        model.fit(X, yy, **fit_kw)
        models.append((n, repr_kind, model))
    return models


def _predict_ensemble(models, Xb, Xo):
    """Сырые вероятности каждой модели ансамбля, строки = модели."""
    return np.vstack([np.asarray(m.predict_proba(
        prepare_for(rk, Xb, Xo))[:, 1]).ravel() for _, rk, m in models])


def _aggregate(prob_matrix: np.ndarray) -> np.ndarray:
    """Среднее нормированных рангов по строкам (строка = одна модель).

    ВАЖНО: ранги нормализуются по всему переданному вектору сразу. В проде
    (train.py) порог подбирается по полному OOF, поэтому и здесь внутренний
    OOF нужно сначала собрать целиком, и только потом ранжировать -- иначе
    ранги сравнивались бы внутри отдельных фолдов, чего на инференсе нет.
    """
    return np.mean([rank01(p) for p in prob_matrix], axis=0)


def nested_cv(names, Xb, Xo, y, seed, splits=5, repeats=2, verbose=True, df=None):
    """df: исходная таблица. Если задана, one-hot представление пересобирается
    внутри каждой внешней фолды по медианам её тренировочной части -- ровно так,
    как это делает predict.py (медианы из meta.json). Без этого заполнение
    пропусков подглядывало бы в тестовую фолду."""
    from sklearn.model_selection import StratifiedKFold
    f1s, aucs, thrs = [], [], []
    t0 = time.time()
    for rep in range(repeats):
        outer = StratifiedKFold(splits, shuffle=True, random_state=seed + 7919 * rep)
        for fold, (tr_i, te_i) in enumerate(outer.split(Xb, y)):
            Xb_tr, Xb_te = _rows(Xb, tr_i), _rows(Xb, te_i)
            if df is None:
                Xo_tr, Xo_te = _rows(Xo, tr_i), _rows(Xo, te_i)
            else:
                med = _rows(Xb, tr_i).median()
                Xo_tr = build_onehot(_rows(df, tr_i), medians=med)
                Xo_te = build_onehot(_rows(df, te_i), medians=med)
            y_tr = y[tr_i]

            # подбор порога на внутреннем CV тренировочной части.
            # Ранги нормализуются ПО ПУЛУ всего внутреннего OOF -- ровно так
            # же, как в train.py порог подбирается по всему OOF сразу.
            inner = StratifiedKFold(4, shuffle=True, random_state=1000 + rep)
            P_inner = np.zeros((len(names), len(y_tr)))
            for a_i, b_i in inner.split(Xb_tr, y_tr):
                P_inner[:, b_i] = _predict_ensemble(
                    _fit_ensemble(names, seed, Xb_tr, Xo_tr, y_tr, a_i),
                    _rows(Xb_tr, b_i), _rows(Xo_tr, b_i))
            thr, _ = pick_threshold(y_tr, _aggregate(P_inner))

            p_te = _aggregate(_predict_ensemble(
                _fit_ensemble(names, seed, Xb_tr, Xo_tr, y_tr), Xb_te, Xo_te))
            f1s.append(f1_score_at(y[te_i], p_te, thr))
            aucs.append(roc_auc(y[te_i], p_te))
            thrs.append(thr)
            if verbose:
                print(f"    rep{rep} fold{fold}: F1={f1s[-1]:.4f} AUC={aucs[-1]:.5f} "
                      f"thr={thr:.3f}  [{time.time()-t0:.0f}s]", flush=True)
    return np.array(f1s), np.array(aucs), np.array(thrs)


def oof_ensemble(names, Xb, Xo, y, seed, splits=5, repeats=3):
    """Обычный OOF ансамбля (быстро, но порог потом подбирается по нему же).
    Возвращает агрегированный скоринг, пул рангов -- по всему набору."""
    from sklearn.model_selection import StratifiedKFold
    P = np.zeros((len(names), len(y)))
    for rep in range(repeats):
        skf = StratifiedKFold(splits, shuffle=True, random_state=seed + 1000 * rep)
        acc = np.zeros_like(P)
        for tr_i, va_i in skf.split(Xb, y):
            acc[:, va_i] = _predict_ensemble(
                _fit_ensemble(names, seed, Xb, Xo, y, tr_i),
                _rows(Xb, va_i), _rows(Xo, va_i))
        P += acc
    P /= repeats
    return np.mean([rank01(p) for p in P], axis=0)


CANDIDATES = {
    "lgbm":                    ["lgbm"],
    "logreg":                  ["logreg"],
    "lgbm+logreg":             ["lgbm", "logreg"],
    "lgbm+catboost":           ["lgbm", "catboost"],
    "lgbm+catboost+logreg":    ["lgbm", "catboost", "logreg"],
    "lr_spline":               ["lr_spline"],
    "lgbm+lr_spline":          ["lgbm", "lr_spline"],
    "lgbm+lr_spline+logreg":   ["lgbm", "lr_spline", "logreg"],
    "catboost+lr_spline":      ["catboost", "lr_spline"],
    # XGBoost намеренно не входит в набор кандидатов: он дублирует LightGBM
    # (ранговая корреляция 0.9975) и падает на невидимых категориях.
    # Проверить его можно так:
    # "lgbm+xgb+catboost+logreg": ["lgbm", "xgb", "catboost", "logreg"],
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="train.csv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--compare-candidates", action="store_true")
    ap.add_argument("--ensemble", default=",".join(FINAL_ENSEMBLE),
                    help="состав ансамбля через запятую; default = FINAL_ENSEMBLE")
    args = ap.parse_args()

    df = pd.read_csv(args.train)
    y = df[TARGET].to_numpy(int)
    Xb, Xo = build_features(df), build_onehot(df)

    print("=" * 78)
    print("НЕСМЕЩЁННАЯ ОЦЕНКА (nested CV: модель и порог не видят тестовую фолду)")
    print(f"seed={args.seed}  splits={args.splits}  repeats={args.repeats}")
    print("=" * 78)

    if args.compare_candidates:
        print("\n-- сравнение составов ансамбля --")
        print(f"{'состав':28s} {'F1':>8s} {'+/-':>7s} {'AUC':>9s} {'thr':>7s}")
        rows = []
        for name, names in CANDIDATES.items():
            t0 = time.time()
            f1s, aucs, thrs = nested_cv(names, Xb, Xo, y, args.seed,
                                        args.splits, args.repeats,
                                        verbose=False, df=df)
            se = f1s.std(ddof=1) / np.sqrt(len(f1s))
            rows.append((name, f1s.mean(), se, aucs.mean(), thrs.mean()))
            print(f"{name:28s} {f1s.mean():8.4f} {se:7.4f} {aucs.mean():9.5f} "
                  f"{thrs.mean():7.3f}   [{time.time()-t0:.0f}s]", flush=True)
        best = max(rows, key=lambda r: r[3])
        print(f"\nЛучший по AUC: {best[0]}")

    names = [n.strip() for n in args.ensemble.split(",") if n.strip()]
    print(f"\n-- финальный ансамбль {names}, nested CV --")
    f1s, aucs, thrs = nested_cv(names, Xb, Xo, y, args.seed, args.splits,
                                args.repeats, df=df)
    print("-" * 78)
    print(f"F1  = {f1s.mean():.4f} +/- {f1s.std(ddof=1)/np.sqrt(len(f1s)):.4f} "
          f"(std {f1s.std(ddof=1):.4f}, min {f1s.min():.4f}, max {f1s.max():.4f})")
    print(f"AUC = {aucs.mean():.5f} +/- {aucs.std(ddof=1)/np.sqrt(len(aucs)):.5f}")
    print(f"порог = {thrs.mean():.4f} +/- {thrs.std(ddof=1):.4f}")
    print("Это честная оценка того, что модель покажет на невидимом тесте.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
