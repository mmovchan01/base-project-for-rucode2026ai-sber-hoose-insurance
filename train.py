"""
train.py -- обучение финальной модели для задачи «Защищённый смартфон».

Запуск:
    python train.py                      # seed выбирается случайно
    python train.py --seed 42            # зафиксировать seed
    python train.py --train data/train.csv --model-dir models

Что делает:
  1. Читает train.csv.
  2. Строит признаки (common.build_features / build_onehot).
  3. Для каждого участника ансамбля считает out-of-fold предсказания
     (Repeated Stratified K-Fold) и обучает итоговую модель на всех данных.
  4. Усредняет предсказания ансамбля по нормированным рангам и подбирает
     порог, максимизирующий F1 на OOF.
  5. Сохраняет веса всех моделей и метаинформацию (порог, seed, метрики,
     список колонок, версии библиотек) в каталог моделей.

Результат полностью детерминирован при фиксированном --seed.
"""
from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import os
import random
import sys
import time
import warnings

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from common import (CATS, ID, TARGET, build_features, build_onehot,
                    ensemble_specs, prepare_for, rank01)

N_SPLITS = 5
N_REPEATS = 5
LIBS = ["numpy", "pandas", "scikit-learn", "scipy", "lightgbm",
        "catboost", "joblib", "matplotlib"]


# --------------------------------------------------------------------------- #
def set_global_seed(seed: int) -> None:
    """Фиксируем все генераторы случайных чисел (требование воспроизводимости)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import lightgbm as lgb
        lgb.reset_parameter()          # noqa: F401  (сброс глобальных дефолтов)
    except Exception:
        pass


def pick_threshold(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Порог, максимизирующий F1. Перебор всех порогов за O(n log n):
    сортируем по убыванию score, F1(top-k) = 2*TP_k / (k + P)."""
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    order = np.argsort(-p, kind="mergesort")
    ys, ps = y[order], p[order]
    tp = np.cumsum(ys)
    k = np.arange(1, len(y) + 1)
    f1 = 2.0 * tp / (k + y.sum())
    i = int(np.argmax(f1))
    lo = ps[i + 1] if i + 1 < len(ps) else -np.inf
    thr = 0.5 * (ps[i] + lo) if np.isfinite(lo) else ps[i] - 1e-9
    return float(thr), float(f1[i])


def f1_score_at(y, p, thr) -> float:
    pred = (np.asarray(p) >= thr).astype(int)
    tp = float((pred * np.asarray(y)).sum())
    return float(2 * tp / (pred.sum() + np.asarray(y).sum())) if (pred.sum() + y.sum()) else 0.0


def roc_auc(y, p) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p))


def save_model(name: str, model, model_dir: str) -> dict:
    """Сохраняем веса нативным форматом каждой библиотеки.

    Нативный формат переносимее pickle: бустеры читаются без установленной
    sklearn, а joblib используется только для sklearn-пайплайнов.
    """
    if name == "lgbm":
        path = os.path.join(model_dir, "lgbm.txt")
        model.booster_.save_model(path)
        return {"name": name, "kind": "lgbm_booster", "path": os.path.basename(path)}
    if name == "catboost":
        path = os.path.join(model_dir, "catboost.cbm")
        model.save_model(path)
        return {"name": name, "kind": "catboost", "path": os.path.basename(path)}
    if name in ("logreg", "lr_spline"):
        path = os.path.join(model_dir, f"{name}.joblib")
        joblib.dump(model, path)
        return {"name": name, "kind": "sklearn_pipeline", "path": os.path.basename(path)}
    raise ValueError(f"Неизвестная модель: {name}")


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", default="train.csv")
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--seed", type=int, default=None,
                    help="seed генератора; по умолчанию выбирается случайно и "
                         "записывается в метаинформацию и в имя файла ответа")
    ap.add_argument("--splits", type=int, default=N_SPLITS)
    ap.add_argument("--repeats", type=int, default=N_REPEATS)
    args = ap.parse_args()

    seed = args.seed if args.seed is not None else int(np.random.default_rng().integers(0, 2**31 - 1))
    set_global_seed(seed)
    os.makedirs(args.model_dir, exist_ok=True)
    t_start = time.time()

    print("=" * 78)
    print(f"ОБУЧЕНИЕ  |  seed = {seed}  |  splits = {args.splits}  |  repeats = {args.repeats}")
    print("=" * 78)

    # ---------------------------------------------------------------- данные
    train = pd.read_csv(args.train)
    if TARGET not in train.columns:
        sys.exit(f"В {args.train} нет колонки '{TARGET}'")
    y = train[TARGET].to_numpy(dtype=int)
    print(f"train: {train.shape[0]} строк, {train.shape[1]} колонок, "
          f"доля класса 1 = {y.mean():.4f}")

    X_boost = build_features(train)
    # Медианы для заполнения пропусков считаются на ОБУЧАЮЩЕЙ выборке и
    # сохраняются в meta.json: predict.py применяет именно их, чтобы замена
    # пропусков не зависела от того, какой батч данных пришёл на инференс.
    medians = X_boost.median()
    X_onehot = build_onehot(train, medians=medians)
    print(f"признаков: бустинг {X_boost.shape[1]}, линейная модель {X_onehot.shape[1]}")

    # ------------------------------------------------------- oof + полный фит
    from sklearn.model_selection import StratifiedKFold

    specs = ensemble_specs(seed)
    oof: dict[str, np.ndarray] = {}
    saved: list[dict] = []
    per_model: dict[str, dict] = {}

    for name, factory, repr_kind, fit_kw in specs:
        t0 = time.time()
        X = prepare_for(repr_kind, X_boost, X_onehot)
        p_oof = np.zeros(len(y), dtype=np.float64)

        for rep in range(args.repeats):
            skf = StratifiedKFold(args.splits, shuffle=True,
                                  random_state=seed + 1000 * rep)
            for fold, (tr_i, va_i) in enumerate(skf.split(X, y)):
                m = factory()
                m.fit(X.iloc[tr_i], y[tr_i], **fit_kw)
                p_oof[va_i] += m.predict_proba(X.iloc[va_i])[:, 1]
        p_oof /= args.repeats
        oof[name] = p_oof

        full = factory()
        full.fit(X, y, **fit_kw)
        saved.append(save_model(name, full, args.model_dir))

        thr_i, f1_i = pick_threshold(y, p_oof)
        per_model[name] = {"oof_auc": round(roc_auc(y, p_oof), 5),
                           "oof_f1": round(f1_i, 5), "oof_thr": round(thr_i, 4)}
        print(f"  {name:9s} OOF AUC {per_model[name]['oof_auc']:.5f}  "
              f"F1 {per_model[name]['oof_f1']:.4f} (thr {thr_i:.3f})  "
              f"[{time.time() - t0:.0f}s]")

    # ------------------------------------------------------------- ансамбль
    R = np.mean([rank01(oof[n]) for n in oof], axis=0)
    thr, f1_rank = pick_threshold(y, R)
    P = np.mean([oof[n] for n in oof], axis=0)
    thr_p, f1_prob = pick_threshold(y, P)

    print("-" * 78)
    print(f"АНСАМБЛЬ (среднее рангов)      : AUC {roc_auc(y, R):.5f}  F1 {f1_rank:.4f}  thr {thr:.4f}")
    print(f"АНСАМБЛЬ (среднее вероятностей): AUC {roc_auc(y, P):.5f}  F1 {f1_prob:.4f}  thr {thr_p:.4f}")

    # Среднее рангов даёт более устойчивый порядок (каждая модель вносит вклад
    # одинакового масштаба, насыщенные вероятности деревьев не доминируют).
    final_f1 = f1_score_at(y, R, thr)
    print(f"ИТОГ (выбранное усреднение = ранги): CV F1 = {final_f1:.4f}, порог = {thr:.4f}")
    print(f"NB: порог подобран на тех же OOF-предсказаниях, поэтому оценка слегка "
          f"оптимистична; несмещённая оценка -- в evaluate.py")

    # ------------------------------------------- какие коды категорий видели
    # Нужно predict.py, чтобы честно сообщить о категории, которой не было в
    # обучении (например, employment_type='retired' в train отсутствует, но
    # встречается в тесте). LightGBM и логистическая регрессия такой случай
    # обрабатывают корректно, но молчать об этом нельзя.
    observed_codes = {c: sorted(int(v) for v in X_boost[c].unique()) for c in CATS}

    # --------------------------------------------------------------- сохранение
    versions = {}
    for lib in LIBS:
        try:
            versions[lib] = md.version(lib)
        except md.PackageNotFoundError:
            versions[lib] = "unknown"
    meta = {
        "seed": seed,
        "threshold": thr,
        "aggregation": "rank_mean",
        "cv": {"splits": args.splits, "repeats": args.repeats,
               "oof_auc": round(roc_auc(y, R), 5),
               "oof_f1_rank_mean": round(f1_rank, 5),
               "oof_f1_prob_mean": round(f1_prob, 5),
               "per_model": per_model},
        "features_boost": list(X_boost.columns),
        "features_onehot": list(X_onehot.columns),
        "cats": CATS,
        "observed_codes": observed_codes,
        "medians": {k: float(v) for k, v in medians.items()},
        "models": saved,
        "python": sys.version.split()[0],
        "libraries": versions,
        "train_file": args.train,
        "train_rows": int(len(train)),
        "positive_rate": float(y.mean()),
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "train_seconds": round(time.time() - t_start, 1),
    }
    with open(os.path.join(args.model_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("-" * 78)
    print(f"Веса и метаинформация сохранены в '{args.model_dir}/': "
          f"{sorted(os.listdir(args.model_dir))}")
    print(f"Готово за {time.time() - t_start:.0f}s.  Для предсказаний: python predict.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
