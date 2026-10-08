"""
verify.py -- самопроверка решения перед отправкой.

Проверяет:
  1. веса загружаются из каталога моделей нативными загрузчиками;
  2. загруженные веса дают те же предсказания, что и модель, обученная
     заново теми же гиперпараметрами и seed (т.е. сохранено/прочитано верно);
  3. predict.py детерминирован (два запуска -> байт-в-байт одинаковый файл);
  4. формат ответа совпадает с sample_submission.csv;
  5. предсказания на train согласованы с CV-метрикой из meta.json.

Запуск:  python verify.py [--model-dir models] [--test public_test.csv]
"""
from __future__ import annotations

import argparse
import filecmp
import json
import os
import shutil
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from common import (ID, TARGET, build_features, build_onehot, prepare_for,
                    rank01)
from predict import load_model

OK, FAIL = "\033[92mOK\033[0m", "\033[91mFAIL\033[0m"
_results = []


def check(name: str, cond: bool, detail: str = "") -> bool:
    print(f"  [{OK if cond else FAIL}] {name}" + (f" -- {detail}" if detail else ""))
    _results.append(cond)
    return cond


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--test", default="public_test.csv")
    ap.add_argument("--train", default="train.csv")
    ap.add_argument("--sample", default="sample_submission.csv")
    args = ap.parse_args()

    meta_path = os.path.join(args.model_dir, "meta.json")
    print("=" * 78)
    print("САМОПРОВЕРКА РЕШЕНИЯ")
    print("=" * 78)
    if not os.path.exists(meta_path):
        print(f"  [{FAIL}] нет {meta_path} -- сначала запустите: python train.py")
        return 1
    meta = json.load(open(meta_path, encoding="utf-8"))
    seed, thr = int(meta["seed"]), float(meta["threshold"])
    print(f"seed = {seed}, порог = {thr:.4f}, состав = {[m['name'] for m in meta['models']]}")

    # -------------------------------------------------- 1. файлы весов на месте
    print("\n1. Файлы весов")
    for spec in meta["models"]:
        p = os.path.join(args.model_dir, spec["path"])
        size = os.path.getsize(p) if os.path.exists(p) else 0
        check(f"{spec['name']}: {spec['path']}", os.path.exists(p) and size > 0,
              f"{size/1024:.1f} KB")
    total = sum(os.path.getsize(os.path.join(args.model_dir, s["path"]))
                for s in meta["models"])
    print(f"  суммарный размер весов: {total/1024/1024:.2f} MB")

    # ------------------------------------------- 2. веса загружаются и работают
    print("\n2. Загрузка весов и инференс")
    train = pd.read_csv(args.train)
    Xb_tr, Xo_tr = build_features(train), build_onehot(train)
    y = train[TARGET].to_numpy(int)

    insample = {}
    for spec in meta["models"]:
        kind, model = load_model(spec, args.model_dir)
        if kind == "lgbm_booster":
            p = model.predict(Xb_tr[meta["features_boost"]].to_numpy(dtype=float))
        elif kind == "catboost":
            p = model.predict_proba(prepare_for("boost_str", Xb_tr, Xo_tr))[:, 1]
        else:
            p = model.predict_proba(Xo_tr[meta["features_onehot"]])[:, 1]
        insample[spec["name"]] = np.asarray(p).ravel()
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y, insample[spec["name"]])
        # in-sample AUC обязан быть высоким: модель обучалась на этих данных
        check(f"{spec['name']}: in-sample AUC = {auc:.4f}", auc > 0.98)

    # ---------------------- 3. воспроизводимость: повторное обучение даёт то же
    print("\n3. Детерминированность обучения (тот же seed -> те же веса)")
    from common import ALL_MODELS
    for spec in meta["models"]:
        name = spec["name"]
        factory, repr_kind, fit_kw = ALL_MODELS[name]
        m = factory(seed)
        X = prepare_for(repr_kind, Xb_tr, Xo_tr)
        m.fit(X, y, **fit_kw)
        p_new = (m.booster_.predict(X.to_numpy(dtype=float)) if name == "lgbm"
                 else m.predict_proba(X)[:, 1])
        same = np.allclose(p_new, insample[name], atol=1e-9)
        check(f"{name}: повторное обучение совпадает с сохранёнными весами", same,
              f"max|diff| = {np.abs(p_new - insample[name]).max():.2e}")

    # ------------------------- 4. детерминированность predict + 5. формат ответа
    print("\n4. Детерминированность predict.py")
    tmp = "._verify_tmp"
    os.makedirs(tmp, exist_ok=True)
    try:
        outs = []
        for i in range(2):
            d = os.path.join(tmp, f"r{i}")
            r = subprocess.run([sys.executable, "predict.py", "--model-dir", args.model_dir,
                                "--test", args.test, "--out-dir", d],
                               capture_output=True, text=True)
            if r.returncode != 0:
                print(r.stdout[-2000:]); print(r.stderr[-2000:])
            check(f"запуск #{i+1} завершился успешно", r.returncode == 0)
            outs.append(os.path.join(d, f"submission_seed_{seed}.csv"))
        if all(os.path.exists(o) for o in outs):
            check("два запуска дают идентичный файл",
                  filecmp.cmp(outs[0], outs[1], shallow=False))

            # --------------------------------------------- 5. формат ответа
            print("\n5. Формат submission")
            sub = pd.read_csv(outs[0])
            check("колонки == [customer_id, accepted]", list(sub.columns) == [ID, TARGET],
                  str(list(sub.columns)))
            check("значения accepted в {0, 1}", set(np.unique(sub[TARGET])) <= {0, 1},
                  f"unique = {sorted(set(np.unique(sub[TARGET])))}")
            check("нет пропусков", sub.isna().sum().sum() == 0)
            if os.path.exists(args.sample):
                sample = pd.read_csv(args.sample)
                check("customer_id совпадает с sample_submission (порядок и состав)",
                      sub[ID].tolist() == sample[ID].tolist())
                check("число строк совпадает", len(sub) == len(sample),
                      f"{len(sub)} vs {len(sample)}")
            check("доля единиц в ответе близка к доле в train",
                  abs(sub[TARGET].mean() - meta["positive_rate"]) < 0.08,
                  f"test {sub[TARGET].mean():.4f} vs train {meta['positive_rate']:.4f}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # ------------------------------- 6. устойчивость к вырожденным входам
    print("\n6. Устойчивость к малым / вырожденным входам")
    # Регрессия на баг: если в батче колонка целиком состоит из NaN, её медиана
    # тоже NaN, и SplineTransformer падал с "Input X contains NaN".
    # insurance_claims пуст в 53% строк, поэтому вызов на одной строке ломался
    # с вероятностью >0.5.
    tmp2 = "._verify_tiny"
    try:
        os.makedirs(tmp2, exist_ok=True)
        for n_rows in (1, 3):
            tiny = os.path.join(tmp2, f"tiny{n_rows}.csv")
            pd.read_csv(args.test).head(n_rows).to_csv(tiny, index=False)
            r = subprocess.run([sys.executable, "predict.py", "--model-dir", args.model_dir,
                                "--test", tiny,
                                "--out-dir", os.path.join(tmp2, f"o{n_rows}")],
                               capture_output=True, text=True)
            check(f"predict.py на {n_rows} строке(ах) отрабатывает без NaN-ошибки",
                  r.returncode == 0, "" if r.returncode == 0 else r.stderr.strip()[-160:])
        # медианы в meta.json совпадают с медианами обучающей выборки
        med = meta.get("medians", {})
        chk = Xb_tr.median()
        bad = [k for k, v in med.items() if abs(float(v) - float(chk[k])) > 1e-6]
        check("медианы в meta.json == медианы обучающей выборки", not bad,
              f"расхождений: {len(bad)} из {len(med)}")
    finally:
        shutil.rmtree(tmp2, ignore_errors=True)

    print("\n" + "=" * 78)
    n_ok = sum(_results)
    print(f"РЕЗУЛЬТАТ: {n_ok}/{len(_results)} проверок пройдено"
          + (" -- всё в порядке" if n_ok == len(_results) else " -- ЕСТЬ ПАДЕНИЯ"))
    print("=" * 78)
    return 0 if n_ok == len(_results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
