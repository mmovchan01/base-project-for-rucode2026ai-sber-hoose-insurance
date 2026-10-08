"""
predict.py -- предсказание на тестовом наборе БЕЗ обучения.

Запуск:
    python predict.py                                  # public_test.csv -> submission_seed_{SEED}.csv
    python predict.py --test private_test.csv          # другой тестовый файл
    python predict.py --model-dir models --out-dir submissions

Скрипт:
  1. Читает обученные веса из каталога моделей (models/*.txt|json|cbm|joblib)
     и метаинформацию (порог, список колонок, seed).
  2. Строит признаки точно так же, как train.py (общий модуль common.py).
  3. Прогоняет каждую модель, усредняет предсказания по нормированным рангам,
     применяет сохранённый порог.
  4. Пишет submission_seed_{SEED}.csv со столбцами customer_id, accepted.

Обучение здесь НЕ выполняется -- только загрузка весов и инференс.
"""
from __future__ import annotations

import argparse
import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from common import (CATS, ID, TARGET, build_features, build_onehot,
                    prepare_for, rank01)


def load_model(spec: dict, model_dir: str):
    """Загрузка весов нативным загрузчиком соответствующей библиотеки."""
    kind, path = spec["kind"], os.path.join(model_dir, spec["path"])
    if not os.path.exists(path):
        raise FileNotFoundError(f"Не найден файл весов: {path}")

    if kind == "lgbm_booster":
        import lightgbm as lgb
        return kind, lgb.Booster(model_file=path)
    if kind == "catboost":
        from catboost import CatBoostClassifier
        m = CatBoostClassifier()
        m.load_model(path)
        return kind, m
    if kind == "sklearn_pipeline":
        return kind, joblib.load(path)
    raise ValueError(f"Неизвестный тип модели: {kind}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test", default="public_test.csv")
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--seed", type=int, default=None,
                    help="переопределить seed в имени файла (по умолчанию берётся из meta.json)")
    args = ap.parse_args()

    meta_path = os.path.join(args.model_dir, "meta.json")
    if not os.path.exists(meta_path):
        raise SystemExit(f"Не найдена метаинформация '{meta_path}'. Сначала выполните: python train.py")
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    seed = args.seed if args.seed is not None else int(meta["seed"])
    thr = float(meta["threshold"])
    print("=" * 78)
    print(f"ПРЕДСКАЗАНИЕ  |  seed = {seed}  |  порог = {thr:.4f}  "
          f"|  агрегация = {meta['aggregation']}")
    print(f"Модель обучена {meta.get('trained_at','?')} на {meta.get('train_file','?')} "
          f"({meta.get('train_rows','?')} строк), CV F1 = {meta['cv']['oof_f1_rank_mean']}")
    print("=" * 78)

    test = pd.read_csv(args.test)
    if ID not in test.columns:
        raise SystemExit(f"В {args.test} нет колонки '{ID}'")
    print(f"test: {test.shape[0]} строк, {test.shape[1]} колонок")

    # --- признаки (та же функция, что и при обучении) -----------------------
    X_boost = build_features(test)
    # медианы берём из meta.json (обучающая выборка), а не из теста
    medians = meta.get("medians")
    X_onehot = build_onehot(test, medians=medians)

    missing_b = [c for c in meta["features_boost"] if c not in X_boost.columns]
    missing_o = [c for c in meta["features_onehot"] if c not in X_onehot.columns]
    if missing_b or missing_o:
        raise SystemExit(f"Рассогласование признаков с обучением: {missing_b + missing_o}")
    X_boost = X_boost[meta["features_boost"]]
    X_onehot = X_onehot[meta["features_onehot"]]

    # --- контроль невидимых категорий --------------------------------------
    # В данных такое реально встречается: employment_type='retired' в train
    # отсутствует, а в public_test есть. LightGBM и логистическая регрессия
    # обрабатывают это корректно, но факт надо зафиксировать в логе.
    observed = meta.get("observed_codes", {})
    for c in meta.get("cats", []):
        if c not in X_boost.columns or c not in observed:
            continue
        unseen = sorted(set(int(v) for v in X_boost[c].unique()) - set(observed[c]))
        if unseen:
            n = int(X_boost[c].isin(unseen).sum())
            print(f"  ВНИМАНИЕ: '{c}' содержит не встречавшиеся при обучении "
                  f"коды {unseen} в {n} строках -- обработаны штатно")

    # --- инференс по каждой модели -----------------------------------------
    ranks, probs = [], []
    for spec in meta["models"]:
        kind, model = load_model(spec, args.model_dir)
        name = spec["name"]

        if kind == "lgbm_booster":
            # нативный Booster принимает только числовую матрицу
            p = model.predict(X_boost.to_numpy(dtype=float))
        elif kind == "catboost":
            p = model.predict_proba(prepare_for("boost_str", X_boost, X_onehot))[:, 1]
        else:                                   # sklearn-пайплайн (logreg / lr_spline)
            p = model.predict_proba(X_onehot)[:, 1]

        p = np.asarray(p, dtype=float).ravel()
        ranks.append(rank01(p))
        probs.append(p)
        # NB: сохранённый `thr` относится к агрегированному ранговому скорингу,
        # поэтому по каждой модели показываем только её собственное распределение.
        print(f"  {name:9s} mean p = {p.mean():.4f}  медиана p = {np.median(p):.4f}  "
              f"доля p>=0.5 = {(p >= 0.5).mean():.4f}")

    if meta["aggregation"] == "rank_mean":
        agg = np.mean(ranks, axis=0)
    elif meta["aggregation"] == "prob_mean":
        agg = np.mean(probs, axis=0)
    else:
        raise SystemExit(f"Неизвестная агрегация: {meta['aggregation']}")

    y_pred = (agg >= thr).astype(int)

    # --- сохранение ответа --------------------------------------------------
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"submission_seed_{seed}.csv")
    sub = pd.DataFrame({ID: test[ID].to_numpy(), TARGET: y_pred})
    sub.to_csv(out_path, index=False)

    print("-" * 78)
    print(f"Сохранено: {out_path}  ({len(sub)} строк)")
    print(f"Прогноз: 1 -> {int(y_pred.sum())} ({y_pred.mean():.4f}), "
          f"0 -> {int((1 - y_pred).sum())}   [агрегированный скоринг >= {thr:.4f}]")
    base = meta.get("positive_rate")
    if base is not None:
        print(f"Для справки: доля класса 1 в train = {base:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
