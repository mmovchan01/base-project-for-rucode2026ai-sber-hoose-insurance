"""
Общий модуль: подготовка признаков и фабрики моделей для задачи
«Защищённый смартфон» (классификация отклика на страховое предложение).

Используется и в train.py, и в predict.py, чтобы подготовка признаков на
обучении и на предсказании была строго идентичной (иначе модель "поедет").
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
#  Схема данных
# --------------------------------------------------------------------------- #
ID = "customer_id"
TARGET = "accepted"

CATS = ["gender", "region", "family_status", "education",
        "employment_type", "smartphone_brand"]

# Полные словари категорий берутся из train И test: в train есть редкие
# значения (widowed, student), а в test встречается 'retired', которого в
# train нет вообще. Фиксированный словарь + бакет UNSEEN делают кодирование
# устойчивым к невидимым категориям.
KNOWN_CATEGORIES = {
    "gender": ["F", "M"],
    "region": ["Екатеринбург", "Казань", "Краснодар", "Москва", "Нижний Новгород",
               "Новосибирск", "Омск", "Ростов-на-Дону", "Самара",
               "Санкт-Петербург", "Уфа", "Челябинск"],
    "family_status": ["divorced", "married", "single", "widowed"],
    "education": ["secondary", "bachelor", "master", "phd"],
    "employment_type": ["employed", "retired", "self_employed", "student",
                        "unemployed"],
    "smartphone_brand": ["Apple", "Google", "Huawei", "OnePlus", "Other",
                         "Samsung", "Xiaomi"],
}

NUM = ["age", "city_population", "children", "monthly_income", "owns_car",
       "owns_house", "years_with_bank", "number_of_bank_products",
       "average_monthly_balance", "has_credit", "loan_amount", "credit_score",
       "number_of_card_transactions_month", "online_payments_share",
       "mobile_app_usage", "smartphone_price", "smartphone_age_months",
       "marketing_contacts_last_year", "previous_campaign_response",
       "previous_insurance", "insurance_claims"]

# Признаки с тяжёлым правым хвостом -> log1p сильно улучшает линейную модель.
SKEWED = ["average_monthly_balance", "loan_amount", "smartphone_price",
          "monthly_income", "city_population"]

# Неструктурные пропуски (MNAR): пропуск сам по себе несёт сигнал.
# (loan_amount / insurance_claims не входят: их пропуск полностью определён
#  флагами has_credit / previous_insurance и новой информации не даёт.)
MNAR = ["credit_score", "mobile_app_usage"]


# --------------------------------------------------------------------------- #
#  Признаки
# --------------------------------------------------------------------------- #
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Матрица признаков: числовые столбцы + производные + коды категорий.

    * числовые признаки -- как есть, пропуски СОХРАНЯЮТСЯ (LGBM/XGB/CatBoost
      работают с NaN нативно и сами находят для них оптимальное направление);
    * категории -- целочисленные коды по фиксированному словарю.
    """
    d = df.reset_index(drop=True)
    X = pd.DataFrame(index=d.index)

    for c in NUM:
        X[c] = pd.to_numeric(d[c], errors="coerce")

    # 1) логарифмы тяжёлых хвостов
    for c in SKEWED:
        X["log_" + c] = np.log1p(X[c].clip(lower=0).fillna(-1.0) + 1.0)

    # 2) индикаторы неструктурных пропусков (MNAR-сигнал)
    for c in MNAR:
        X[c + "_isna"] = X[c].isna().astype("float32")

    # 3) производные признаки, имеющие предметный смысл
    X["income_over_phone_price"] = X["monthly_income"] / (X["smartphone_price"] + 1.0)
    X["debt_to_income"] = X["loan_amount"].fillna(0.0) / (X["monthly_income"] + 1.0)
    X["balance_to_income"] = X["average_monthly_balance"] / (X["monthly_income"] + 1.0)
    X["phone_residual_value"] = X["smartphone_price"] / (X["smartphone_age_months"] + 1.0)
    X["digital_activity"] = (X["number_of_card_transactions_month"]
                             * X["online_payments_share"])

    # 4) категории -> коды
    for c in CATS:
        raw = d[c].astype("string").fillna("__NA__")
        codes = pd.Categorical(raw, categories=KNOWN_CATEGORIES[c] + ["__NA__"]).codes
        # -1 = неизвестная категория; зашиваем в отдельный код UNSEEN
        codes = np.where(codes < 0, len(KNOWN_CATEGORIES[c]), codes)
        X[c] = codes.astype("int16")

    return X


# Числовые (не категориальные) колонки в представлении build_onehot: исходные
# числовые признаки + логи + индикаторы пропусков + производные.
_ONEHOT_NUMERIC = (
    NUM
    + ["log_" + c for c in SKEWED]
    + [c + "_isna" for c in MNAR]
    + ["income_over_phone_price", "debt_to_income", "balance_to_income",
       "phone_residual_value", "digital_activity"]
)


def build_onehot(df: pd.DataFrame, medians=None) -> pd.DataFrame:
    """Матрица признаков для линейной модели: one-hot категорий + заполнение
    пропусков медианой (линейные модели и SplineTransformer NaN не принимают).

    medians: Series/None. В проде (predict.py) сюда передаются медианы
    ОБУЧАЮЩЕЙ выборки из models/meta.json, чтобы замена пропусков на train и
    на test была одинаковой. Если не переданы -- берутся медианы самого df;
    это корректно, пока в df нет полностью пустой колонки.

    После заполнения медианой дополнительно применяется fillna(0.0): если
    колонка пуста целиком (например, при вызове на одной строке, где
    insurance_claims с вероятностью 0.53 равен NaN), её медиана тоже NaN, и
    без этого шага SplineTransformer упал бы с "Input X contains NaN".
    В train.csv и public_test.csv полностью пустых колонок нет (проверено),
    поэтому на полном наборе этот шаг ничего не меняет.
    """
    X = build_features(df).drop(columns=CATS)
    d = df.reset_index(drop=True)
    for c in CATS:
        raw = d[c].astype("string").fillna("__NA__")
        for level in KNOWN_CATEGORIES[c]:
            X[f"{c}={level}"] = (raw == level).astype("float32")
    if medians is not None:
        X = X.fillna(pd.Series(medians).reindex(X.columns))
    else:
        X = X.fillna(X.median())
    return X.fillna(0.0).astype("float32")


# --------------------------------------------------------------------------- #
#  Фабрики моделей
# --------------------------------------------------------------------------- #
# ГЛАВНЫЙ ВЫВОД ПО ГИПЕРПАРАМЕТРАМ: здесь выигрывает СИЛЬНАЯ регуляризация.
# На переборе (40 конфигураций LightGBM) переход
#     num_leaves 31 -> 7,  reg_lambda 0 -> 20,  reg_alpha 0 -> 2
# поднимает OOF-AUC с 0.99209 до 0.99302. Причина: целевая зависимость почти
# линейна (логистическая регрессия на one-hot даёт те же 0.9922), поэтому
# глубокие деревья тратят ёмкость на подгонку шума в 6000 строках.
# Побочный плюс -- узкие деревья дают компактную модель.
#
# Про XGBoost: проверен и ОТКЛОНЁН. (1) Ничего не добавляет к ансамблю --
# ранговая корреляция его предсказаний с LightGBM равна 0.9975. (2) Он
# единственный из движков падает с "Found a category not in the training set",
# если в тесте встречается код категории больше максимального из обучающих;
# в наших данных это реальный риск: family_status='widowed' встречается в train
# ровно 1 раз, а employment_type='retired' в train отсутствует вовсе, но есть в
# public_test. LightGBM, CatBoost и логистическая регрессия такой случай
# обрабатывают корректно -- проверка в explore/unseen_test2.py.

def make_lgbm(seed: int):
    """LightGBM: 7 листьев, глубина 6, reg_alpha=2, reg_lambda=20.
    Одиночный OOF-AUC 0.99302 (лучший среди одиночных бустингов)."""
    import lightgbm as lgb
    return lgb.LGBMClassifier(
        n_estimators=1200, learning_rate=0.03,
        num_leaves=7, max_depth=6, min_child_samples=5,
        reg_alpha=2.0, reg_lambda=20.0,
        subsample=0.9, subsample_freq=1, colsample_bytree=0.9,
        random_state=seed, verbose=-1, n_jobs=-1,
    )


def make_catboost(seed: int):
    """CatBoost: глубина 3, l2_leaf_reg=10. Ordered target statistics для
    12-уровневого region -- второй по силе одиночный результат, OOF-AUC 0.99308."""
    from catboost import CatBoostClassifier
    return CatBoostClassifier(
        iterations=1200, learning_rate=0.05, depth=3,
        l2_leaf_reg=10.0, random_seed=seed, verbose=0,
        cat_features=CATS, allow_const_label=True, thread_count=-1,
    )


def make_lr():
    """Логистическая регрессия на one-hot признаках (StandardScaler + L2, C=1).

    Одиночно слабее бустинга (nested-CV F1 0.9477 против 0.9467 -- практически
    вровень), НО это самый полезный участник ансамбля: ранговая корреляция её
    предсказаний с LightGBM равна 0.965, тогда как XGBoost/CatBoost
    скоррелированы с LightGBM на 0.98-0.998 и поэтому почти не добавляют
    разнообразия. Ансамбль lgbm+logreg поднимает nested-CV F1 с 0.9467 до
    0.9543, а AUC с 0.99260 до 0.99345.

    Содержательно это не случайно: целевая зависимость в данных близка к
    линейной в логитах (см. EDA), и линейная модель точно ловит глобальный
    тренд там, где деревья вынуждены дробить его ступеньками.
    Параметры: 67 весов + 67 сдвигов стандартизации.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=5000, C=1.0))


def onehot_cat_columns() -> list[str]:
    """Имена one-hot колонок, которые build_onehot создаёт из категорий."""
    return [f"{c}={lv}" for c in CATS for lv in KNOWN_CATEGORIES[c]]


def make_lr_spline(n_knots: int = 3, degree: int = 3, C: float = 0.3):
    """Логистическая регрессия на B-сплайнах -- САМАЯ СИЛЬНАЯ одиночная модель.

    33 числовых признака раскладываются по B-сплайнам (n_knots=3, degree=3,
    include_bias=False -> 4 базисные функции на признак = 132 колонки),
    34 one-hot колонки категорий проходят как есть, затем StandardScaler
    и L2-логистика (C=0.3).

    ИТОГО ПАРАМЕТРОВ: 166 весов + 1 intercept = 167. Для сравнения: только
    один тюнингованный LightGBM содержит ~8400 значений в листьях.

    Результаты (OOF, 5x5):
        логистическая регрессия без сплайнов   AUC 0.99223
        LightGBM (тюнингованный)               AUC 0.99298
        CatBoost depth=3                       AUC 0.99308
        логистическая регрессия со сплайнами   AUC 0.99379   <-- лучшая

    Почему это работает: EDA показал, что зависимость вероятности покупки от
    каждого числового признака гладкая и монотонная (см. eda/02_target_rates.png),
    но НЕ линейная -- например, отклик растёт с 0.004 до 1.0 по
    number_of_card_transactions_month. Сплайны дают линейной модели ровно ту
    гладкую нелинейность, которую бустинг вынужден аппроксимировать ступеньками,
    при этом не переобучаясь: параметров всего 167.

    Меньшее число узлов лучше (перебор 48 конфигураций: n_knots=3 даёт
    AUC 0.99375 против 0.99305 при n_knots=6) -- та же причина, что и с
    регуляризацией деревьев: данных 6000 строк, а шум в метках большой.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import SplineTransformer, StandardScaler

    # Числовые признаки раскладываются по сплайнам, one-hot категорий идёт как есть.
    prep = ColumnTransformer([
        ("cat", "passthrough", onehot_cat_columns()),
        ("spline", SplineTransformer(n_knots=n_knots, degree=degree,
                                     include_bias=False),
         [c for c in _ONEHOT_NUMERIC]),
    ])
    return make_pipeline(prep, StandardScaler(),
                         LogisticRegression(max_iter=12000, C=C))


# --------------------------------------------------------------------------- #
#  Состав ансамбля
# --------------------------------------------------------------------------- #
# (имя, фабрика-callable, представление признаков, kwargs для fit)
# Важно: передаём именно callable, а не готовую модель -- train.py создаёт
# по свежей модели на каждый фолд.
ALL_MODELS = {
    "lgbm":     (lambda seed: make_lgbm(seed),     "boost",     {"categorical_feature": CATS}),
    "catboost": (lambda seed: make_catboost(seed), "boost_str", {}),
    "logreg":   (lambda seed: make_lr(),           "onehot",    {}),
    "lr_spline": (lambda seed: make_lr_spline(),    "onehot",    {}),
}

# --------------------------------------------------------------------------- #
#  ФИНАЛЬНЫЙ СОСТАВ
# --------------------------------------------------------------------------- #
# Состав и гиперпараметры выбирались НЕСМЕЩЁННО: evaluate.py --compare-candidates
# (nested CV 5x2 -- порог подбирается внутри каждой внешней фолды и тестовую
# фолду не видит). Полный протокол -- reports/nested_cv.log.
#
#   состав                          nested F1   nested AUC   ~параметров
#   lgbm (тюнингованный)              0.9473      0.99260        8400
#   logreg (без сплайнов)             0.9474      0.99235         140
#   lgbm + catboost                   0.9511      0.99284       17000
#   lgbm + logreg                     0.9531      0.99345        8500
#   lgbm + catboost + logreg          0.9559      0.99350       17000
#   lgbm + lr_spline                  0.9565      0.99374        8500
#   lgbm + lr_spline + logreg         0.9567      0.99385        8700
#   catboost + lr_spline              0.9572      0.99366        8700
#   lr_spline                         0.9591      0.99364         167  <-- ВЫБРАНА
#
# Итоговая оценка выбранной модели, 5x4 nested CV (20 фолдов):
#   F1 = 0.9595 +/- 0.0016     AUC = 0.99367 +/- 0.00053
FINAL_ENSEMBLE = ["lr_spline"]


def ensemble_specs(seed: int, names=None):
    names = FINAL_ENSEMBLE if names is None else names
    return [(n,) + (lambda s=seed, n=n: ALL_MODELS[n][0](s),) + ALL_MODELS[n][1:]
            for n in names]


def prepare_for(model_repr: str, X_boost: pd.DataFrame, X_onehot: pd.DataFrame):
    """Каждый движок требует свой dtype категориальных колонок:
      * LightGBM  -- int-коды + параметр categorical_feature;
      * XGBoost   -- pandas 'category' с ЦЕЛОЧИСЛЕННЫМ underlying dtype;
      * CatBoost  -- строковые значения;
      * линейная модель -- one-hot матрица без категориальных кодов.
    """
    if model_repr == "onehot":
        return X_onehot
    X = X_boost.copy()
    if model_repr == "boost_cat":
        for c in CATS:
            X[c] = X[c].astype("int16").astype("category")
        return X
    if model_repr == "boost_str":
        for c in CATS:
            X[c] = X[c].astype(int).astype(str)
        return X
    return X_boost          # "boost" -- LightGBM


def rank01(p: np.ndarray) -> np.ndarray:
    """Нормированные ранги в (0, 1) -- устойчивое к масштабу усреднение."""
    from scipy.stats import rankdata
    return (rankdata(p, method="average") - 0.5) / len(p)
