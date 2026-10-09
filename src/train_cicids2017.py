"""Balanced, regularised intrusion detector trained on CIC-IDS2017.

Dataset: CIC-IDS2017 (Canadian Institute for Cybersecurity), official
"MachineLearningCSV" flow features, 8 capture files (~2.83M flows).
Source mirror: https://huggingface.co/datasets/bvsam/cic-ids-2017 (machine_learning/*.parquet)

Protocol (same as src/train_unsw_nb15.py):
1. Clean: strip column names, replace +/-inf with NaN, drop NaN rows.
2. Remove exact duplicate rows so no flow appears in both a training and a testing fold.
3. Memory: each file is deduplicated and sampled (SAMPLE_FRAC) on load, stored as float32.
4. Binary task: BENIGN = 0, any attack = 1.
5. Stratified 5-fold CV. Inside each fold 15% of the training part is an inner
   validation set used ONLY to choose the precision-recall break-even threshold.
6. Regularised RF + XGBoost + LightGBM soft-voting ensemble; train accuracy is
   reported per fold to measure the overfitting gap.
"""

from pathlib import Path
import gc
import numpy as np
import pandas as pd
import joblib

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             fbeta_score, roc_auc_score, confusion_matrix)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "external" / "cicids2017"
REPORTS_DIR = BASE_DIR / "results" / "reports"
MODELS_DIR = BASE_DIR / "results" / "models"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
N_JOBS = 4          # capped to limit peak memory on an 8 GB machine
SAMPLE_FRAC = 0.20  # fraction of each file's unique flows kept (natural class ratio preserved)

# ------------------------------------------------------------
# 1. LOAD, CLEAN, DEDUPLICATE, SAMPLE
# ------------------------------------------------------------
parts, n_raw, n_unique = [], 0, 0
for f in sorted(DATA_DIR.glob("*.parquet")):
    d = pd.read_parquet(f)
    d.columns = d.columns.str.strip()
    n_raw += len(d)
    label = d.pop("Label").astype(str).str.strip()
    d = d.apply(pd.to_numeric, errors="coerce").astype(np.float32)
    d = d.replace([np.inf, -np.inf], np.nan)
    d["Label"] = label
    d = d.dropna().drop_duplicates()
    n_unique += len(d)
    parts.append(d.sample(frac=SAMPLE_FRAC, random_state=SEED))
    print(f"  {f.stem[:45]:45s} unique={len(d):>8,}", flush=True)
    del d
    gc.collect()

data = pd.concat(parts, ignore_index=True).drop_duplicates().reset_index(drop=True)
del parts
gc.collect()
# The Web Attack labels contain a non-ASCII separator; normalise for printing.
data["Label"] = data["Label"].str.replace("�", "-", regex=False).str.replace("–", "-", regex=False)
categories = data.pop("Label")
y = (categories != "BENIGN").astype(int).to_numpy()
# Drop constant columns (carry no information)
const = [c for c in data.columns if data[c].nunique() <= 1]
X = data.drop(columns=const)
del data
print(f"\nRaw flows: {n_raw:,} | unique after cleaning: {n_unique:,} | "
      f"sampled ({SAMPLE_FRAC:.0%}): {len(X):,} | features: {X.shape[1]} "
      f"(dropped {len(const)} constant) | attack share: {y.mean():.1%}", flush=True)
print(categories.value_counts().to_string(), flush=True)

# ------------------------------------------------------------
# 2. MODELS (regularised: depth caps, min leaf sizes, L2 penalties)
# ------------------------------------------------------------
models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=18, min_samples_leaf=10, max_features=0.5,
        n_jobs=N_JOBS, random_state=SEED),
    "XGBoost": XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.7, min_child_weight=10, reg_lambda=5, gamma=1,
        n_jobs=N_JOBS, random_state=SEED, eval_metric="logloss"),
    "LightGBM": LGBMClassifier(
        n_estimators=400, num_leaves=63, learning_rate=0.05,
        min_child_samples=50, reg_lambda=5, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.7, n_jobs=N_JOBS, random_state=SEED, verbose=-1),
}
GRID = np.linspace(0.05, 0.95, 181)


def balanced_threshold(y_val, p):
    """Precision-recall break-even threshold (precision ~= recall), validation data only."""
    gaps = [abs(precision_score(y_val, p >= t, zero_division=0) - recall_score(y_val, p >= t))
            for t in GRID]
    return float(GRID[int(np.argmin(gaps))])


def metrics(y_true, p, thr):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "Threshold": round(thr, 3),
        "Accuracy (%)": round(accuracy_score(y_true, pred) * 100, 2),
        "Precision (%)": round(precision_score(y_true, pred) * 100, 2),
        "Recall (%)": round(recall_score(y_true, pred) * 100, 2),
        "F1 (%)": round(f1_score(y_true, pred) * 100, 2),
        "F2 (%)": round(fbeta_score(y_true, pred, beta=2) * 100, 2),
        "ROC_AUC": round(roc_auc_score(y_true, p), 4),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
    }


# ------------------------------------------------------------
# 3. STRATIFIED 5-FOLD CROSS-VALIDATION
# ------------------------------------------------------------
rows, per_cat = [], []
for fold, (tr_idx, te_idx) in enumerate(
        StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(X, y), 1):
    fit_idx, val_idx = train_test_split(
        tr_idx, test_size=0.15, stratify=y[tr_idx], random_state=SEED)
    p_te, p_fit, p_val, single = [], [], [], {}
    for name, base in models.items():
        m = clone(base).fit(X.iloc[fit_idx], y[fit_idx])
        p_te.append(m.predict_proba(X.iloc[te_idx])[:, 1])
        p_fit.append(m.predict_proba(X.iloc[fit_idx])[:, 1])
        p_val.append(m.predict_proba(X.iloc[val_idx])[:, 1])
        single[name] = round(roc_auc_score(y[te_idx], p_te[-1]), 4)
        del m
    p_te, p_fit, p_val = (np.mean(a, axis=0) for a in (p_te, p_fit, p_val))
    thr = balanced_threshold(y[val_idx], p_val)
    r = {"Fold": fold, **metrics(y[te_idx], p_te, thr)}
    r["Train Accuracy (%)"] = round(accuracy_score(y[fit_idx], p_fit >= thr) * 100, 2)
    r["Overfit Gap (%)"] = round(r["Train Accuracy (%)"] - r["Accuracy (%)"], 2)
    rows.append(r)
    per_cat.append(pd.Series((p_te >= thr) == y[te_idx]).groupby(categories.iloc[te_idx].values).mean())
    print(f"Fold {fold}: thr={thr:.3f}  train_acc={r['Train Accuracy (%)']:.2f}%  "
          f"acc={r['Accuracy (%)']:.2f}%  gap={r['Overfit Gap (%)']:.2f}  P={r['Precision (%)']:.2f}%  "
          f"R={r['Recall (%)']:.2f}%  F1={r['F1 (%)']:.2f}%  F2={r['F2 (%)']:.2f}%  "
          f"AUC={r['ROC_AUC']:.4f}  single-model AUCs={single}", flush=True)
    gc.collect()

cv = pd.DataFrame(rows)
cv.to_csv(REPORTS_DIR / "cicids2017_cv5_results.csv", index=False)
summary = cv.drop(columns=["Fold", "Threshold"]).agg(["mean", "std"]).round(4)
summary.to_csv(REPORTS_DIR / "cicids2017_cv5_summary.csv")
cat_acc = (pd.concat(per_cat, axis=1).mean(axis=1) * 100).round(2).rename("Correctly classified (%)")
counts = categories.value_counts().rename("Rows")
pd.concat([counts, cat_acc], axis=1).to_csv(REPORTS_DIR / "cicids2017_per_category.csv")

print("\n5-fold CV (deduplicated) mean +/- std:")
for c in ["Train Accuracy (%)", "Accuracy (%)", "Overfit Gap (%)", "Precision (%)",
          "Recall (%)", "F1 (%)", "F2 (%)", "ROC_AUC"]:
    print(f"  {c:18s} {summary.loc['mean', c]:.4f} +/- {summary.loc['std', c]:.4f}")
print("\nPer-category correct rate (%):")
print(pd.concat([counts, cat_acc], axis=1).to_string())

# ------------------------------------------------------------
# 4. FINAL MODEL: refit on all sampled data, threshold = mean fold threshold
# ------------------------------------------------------------
final = {n: clone(m).fit(X, y) for n, m in models.items()}
joblib.dump({"selected_model": "Ensemble (RF+XGB+LGBM), CIC-IDS2017 deduplicated sample",
             "models": final, "threshold": round(float(cv["Threshold"].mean()), 3),
             "feature_columns": list(X.columns)},
            MODELS_DIR / "cicids2017_final_model.joblib", compress=3)
print(f"Saved final model -> {MODELS_DIR / 'cicids2017_final_model.joblib'}")
