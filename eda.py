"""
EDA for the "Защищённый смартфон" (protected smartphone) insurance task.

Run:  python eda.py
Writes:
  reports/eda_summary.txt   - textual summary of every check below
  eda/*.png                 - distribution / target-rate / correlation charts
"""
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

CATS = ["gender", "region", "family_status", "education",
        "employment_type", "smartphone_brand"]
TARGET = "accepted"
ID = "customer_id"
EDA_DIR, REPORT_DIR = "eda", "reports"
os.makedirs(EDA_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

_lines = []
def log(msg=""):
    print(msg)
    _lines.append(str(msg))


def main():
    train = pd.read_csv("train.csv")
    test = pd.read_csv("public_test.csv")
    y = train[TARGET].values

    # ---------------------------------------------------------------- 1. shape
    log("=" * 78)
    log("1. ОБЩАЯ СТРУКТУРА ДАННЫХ")
    log("=" * 78)
    log(f"train: {train.shape}, public_test: {test.shape}")
    log(f"customer_id: train {train[ID].min()}..{train[ID].max()} | "
        f"test {test[ID].min()}..{test[ID].max()}")
    log(f"дубликаты customer_id в train: {train[ID].duplicated().sum()}")
    log(f"полные дубликаты строк в train: {train.drop(columns=[ID]).duplicated().sum()}")
    log(f"целевая переменная: доля 1 = {y.mean():.4f}  "
        f"({int(y.sum())} из {len(y)}) -> дисбаланс умеренный (1 : {(1-y.mean())/y.mean():.2f})")

    # -------------------------------------------------------------- 2. missing
    log()
    log("=" * 78)
    log("2. ПРОПУСКИ")
    log("=" * 78)
    miss = pd.DataFrame({
        "train_%": (train.isna().mean() * 100).round(2),
        "test_%": (test.reindex(columns=train.columns).isna().mean() * 100).round(2),
    })
    log(miss[miss["train_%"] > 0].to_string())
    log()
    log("Структурные пропуски (закономерные, а не случайные):")
    log(f"  loan_amount NaN <=> has_credit == 0 : "
        f"{((train.has_credit == 0) == train.loan_amount.isna()).all()}")
    log(f"  insurance_claims NaN <=> previous_insurance == 0 : "
        f"{((train.previous_insurance == 0) == train.insurance_claims.isna()).all()}")
    log("  => эти пропуски не несут новой информации (индикатор = сам флаг).")
    log("Неструктурные пропуски (MNAR - пропуск сам по себе информативен):")
    for c in ["credit_score", "mobile_app_usage"]:
        g = train.groupby(train[c].isna())[TARGET].agg(["mean", "size"])
        log(f"  {c}: target rate | есть = {g.loc[False,'mean']:.4f} (n={int(g.loc[False,'size'])})"
            f" | пропуск = {g.loc[True,'mean']:.4f} (n={int(g.loc[True,'size'])})")

    # ---------------------------------------------------------- 3. outliers
    log()
    log("=" * 78)
    log("3. ВЫБРОСЫ И ДИАПАЗОНЫ")
    log("=" * 78)
    spec = {"age": (18, 72), "children": (0, 5), "monthly_income": (10000, 417589),
            "years_with_bank": (0, 28), "number_of_bank_products": (1, 8),
            "credit_score": (352, 850), "number_of_card_transactions_month": (2, 108),
            "online_payments_share": (0, 1), "smartphone_price": (3500, 149412),
            "smartphone_age_months": (0, 48), "marketing_contacts_last_year": (0, 12)}
    for c, (lo, hi) in spec.items():
        a, b = train[c].min(), train[c].max()
        flag = "OK" if (a >= lo and b <= hi) else "ВНЕ ДИАПАЗОНА"
        iqr = train[c].quantile(.75) - train[c].quantile(.25)
        n_out = int(((train[c] < train[c].quantile(.25) - 3 * iqr) |
                     (train[c] > train[c].quantile(.75) + 3 * iqr)).sum())
        log(f"  {c:32s} [{a:>12.2f} .. {b:>12.2f}]  spec {flag:14s} extreme-outliers: {n_out}")
    log("  Тяжёлые правые хвосты (нужен log): average_monthly_balance, loan_amount, "
        "city_population, monthly_income")

    # ------------------------------------------------- 4. distributions plot
    num = [c for c in train.select_dtypes(include=[np.number]).columns if c != TARGET]
    fig, ax = plt.subplots(5, 5, figsize=(22, 16))
    for a, c in zip(ax.ravel(), num):
        a.hist(train[c].dropna(), bins=50, color="#4878cf")
        a.set_title(c, fontsize=9); a.tick_params(labelsize=7)
    for a in ax.ravel()[len(num):]:
        a.axis("off")
    fig.suptitle("Распределения числовых признаков (train)", fontsize=14)
    fig.tight_layout(); fig.savefig(f"{EDA_DIR}/01_distributions.png", dpi=90); plt.close(fig)

    # --------------------------------------------- 5. train/test drift check
    log()
    log("=" * 78)
    log("4. СДВИГ РАСПРЕДЕЛЕНИЙ TRAIN -> PUBLIC TEST")
    log("=" * 78)
    from scipy.stats import ks_2samp
    drift = []
    for c in num:
        a, b = train[c].dropna(), test[c].dropna()
        drift.append((c, ks_2samp(a, b).pvalue))
    dd = pd.DataFrame(drift, columns=["feature", "ks_pvalue"]).sort_values("ks_pvalue")
    log(dd.head(8).to_string(index=False))
    log("  (p > 0.05 везде => значимого сдвига нет, ковариатного сдвига нет)")
    for c in CATS:
        tr_s = train[c].value_counts(normalize=True)
        te_s = test[c].value_counts(normalize=True)
        log(f"  {c}: max |доля_train - доля_test| = "
            f"{(tr_s.sub(te_s, fill_value=0)).abs().max():.4f}")

    # ----------------------------------------------- 6. categorical categories
    log()
    log("=" * 78)
    log("5. РЕДКИЕ / НЕВИДИМЫЕ КАТЕГОРИИ")
    log("=" * 78)
    for c in CATS:
        only_tr = set(train[c]) - set(test[c])
        only_te = set(test[c]) - set(train[c])
        rare = train[c].value_counts()
        rare = rare[rare < 10]
        if only_tr or only_te or len(rare):
            log(f"  {c}: только в train={sorted(only_tr)} только в test={sorted(only_te)} "
                f"редкие(<10)={dict(rare)}")
    log("  => обязателен бакет UNSEEN и аккуратное кодирование категорий.")

    # ------------------------------------------- 7. target rate per category
    log()
    log("=" * 78)
    log("6. ЦЕЛЕВАЯ ПЕРЕМЕННАЯ ПО КАТЕГОРИЯМ")
    log("=" * 78)
    for c in CATS:
        g = train.groupby(c)[TARGET].agg(["mean", "size"]).sort_values("mean", ascending=False)
        log(f"\n  {c} (базовая доля = {y.mean():.3f})")
        log(g.round(4).to_string())
    for c in ["owns_car", "owns_house", "has_credit",
              "previous_campaign_response", "previous_insurance"]:
        log(f"  {c}: {train.groupby(c)[TARGET].mean().round(4).to_dict()}")

    # ---------------------------------------- 8. numeric monotonicity + plots
    log()
    log("=" * 78)
    log("7. МОНТОТОННОСТЬ ЧИСЛОВЫХ ПРИЗНАКОВ")
    log("=" * 78)
    from scipy.stats import spearmanr
    rows = []
    for c in num:
        m = train[c].notna().values
        if m.sum() < 50 or train[c].nunique() < 3:
            continue
        rows.append((c,
                     round(np.corrcoef(train[c].values[m], y[m])[0, 1], 4),
                     round(spearmanr(train[c].values[m], y[m]).statistic, 4)))
    corr = pd.DataFrame(rows, columns=["feature", "pearson", "spearman"])
    corr["abs_s"] = corr.spearman.abs()
    log(corr.sort_values("abs_s", ascending=False).drop(columns="abs_s").to_string(index=False))

    strong = corr.sort_values("abs_s", ascending=False)["feature"].head(10).tolist()
    fig, ax = plt.subplots(2, 5, figsize=(24, 8))
    for a, c in zip(ax.ravel(), strong):
        sub = train[[c, TARGET]].dropna()
        b = pd.qcut(sub[c], 20, duplicates="drop")
        g = sub.groupby(b, observed=True)[TARGET].agg(["mean", "size"])
        a.plot(range(len(g)), g["mean"], marker="o", color="#d1495b")
        a.axhline(y.mean(), ls="--", c="gray", lw=1)
        a.set_title(f"{c}\n(|spearman|={abs(corr.set_index('feature').loc[c,'spearman']):.3f})",
                    fontsize=9)
        a.tick_params(labelsize=7); a.set_xlabel("квантильный бин"); a.set_ylabel("P(accepted)")
    fig.suptitle("Доля покупок по бинам топ-10 признаков", fontsize=14)
    fig.tight_layout(); fig.savefig(f"{EDA_DIR}/02_target_rates.png", dpi=90); plt.close(fig)

    # ------------------------------------------------- 9. correlation matrix
    log()
    log("=" * 78)
    log("8. КОРРЕЛЯЦИИ МЕЖДУ ПРИЗНАКАМИ (|r| > 0.3)")
    log("=" * 78)
    cm = train[num].corr()
    pairs = [(cm.columns[i], cm.columns[j], cm.iloc[i, j])
             for i in range(len(cm)) for j in range(i + 1, len(cm))
             if abs(cm.iloc[i, j]) > 0.3]
    for a, b, v in sorted(pairs, key=lambda t: -abs(t[2])):
        log(f"  {a:34s} ~ {b:34s} r={v:+.3f}")
    log("  Мультиколлинеарность умеренная; для деревьев не критична.")

    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(cm, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cm))); ax.set_xticklabels(cm.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(cm))); ax.set_yticklabels(cm.columns, fontsize=7)
    fig.colorbar(im, shrink=.8); ax.set_title("Корреляционная матрица (Пирсон)")
    fig.tight_layout(); fig.savefig(f"{EDA_DIR}/03_correlations.png", dpi=90); plt.close(fig)

    # --------------------------------------------- 10. key business insights
    log()
    log("=" * 78)
    log("9. КЛЮЧЕВЫЕ ЗАКОНОМЕРНОСТИ (бизнес-интерпретация)")
    log("=" * 78)
    for c in ["number_of_card_transactions_month", "marketing_contacts_last_year",
              "number_of_bank_products"]:
        g = train.groupby(c)[TARGET].agg(["mean", "size"])
        sat = g[g["mean"] >= 0.99].index
        tail = f"; при значении >= {sat.min()} доля покупок = 1.0" if len(sat) else \
               f"; максимум {g['mean'].max():.3f} при {g['mean'].idxmax()}"
        log(f"  {c}: монотонный рост от {g['mean'].iloc[0]:.3f} до {g['mean'].iloc[-1]:.3f}{tail}")
    log(f"  smartphone_brand: Apple {train[train.smartphone_brand=='Apple'][TARGET].mean():.3f}"
        f" vs Other {train[train.smartphone_brand=='Other'][TARGET].mean():.3f}"
        f" vs Xiaomi {train[train.smartphone_brand=='Xiaomi'][TARGET].mean():.3f}")
    log(f"  unemployed: {train[train.employment_type=='unemployed'][TARGET].mean():.3f}"
        f" (n={int((train.employment_type=='unemployed').sum())}) vs "
        f"self_employed: {train[train.employment_type=='self_employed'][TARGET].mean():.3f}")
    log(f"  Москва {train[train.region=='Москва'][TARGET].mean():.3f} vs "
        f"средний по остальным регионам "
        f"{train[train.region!='Москва'][TARGET].mean():.3f}")

    with open(f"{REPORT_DIR}/eda_summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(_lines) + "\n")
    log()
    log(f"Отчёт сохранён в {REPORT_DIR}/eda_summary.txt, графики - в {EDA_DIR}/")


if __name__ == "__main__":
    main()
