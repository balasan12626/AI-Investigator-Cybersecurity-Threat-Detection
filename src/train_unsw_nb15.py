"""Supervised network-intrusion detector trained on the UNSW-NB15 benchmark.

Dataset: UNSW-NB15 (UNSW Canberra Cyber Range Lab), official train/test partition.
Source mirror: https://huggingface.co/datasets/Mireu-Lab/UNSW-NB15
NOTE: in that mirror the file names are swapped -- `test.csv` (175,341 rows) is the
official TRAINING set and `train.csv` (82,332 rows) is the official TESTING set.
We follow the official convention: fit on the 175k set, report on the 82k set.

Protocol (leakage-free):
1. Drop `id` and `attack_cat` (attack_cat is a direct proxy of the label).
2. Hold out 15% of the official training set as validation for model selection
   and decision-threshold tuning; the official test set is scored exactly once.
3. Binary task: label 0 = normal, 1 = attack.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, fbeta_score, roc_auc_score, confusion_matrix)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "external" / "unsw_nb15"
REPORTS_DIR = BASE_DIR / "results" / "reports"
MODELS_DIR = BASE_DIR / "results" / "models"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

CATEGORICAL = ["proto", "service", "state"]
SEED = 42
N_JOBS = 4  # capped to limit peak memory on an 8 GB machine


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    return df.drop(columns=["id", "attack_cat"])


train_full = load(DATA_DIR / "test.csv")   # official training set (175,341 rows)
test = load(DATA_DIR / "train.csv")        # official testing set (82,332 rows)
print(f"Train: {len(train_full):,} rows | Test: {len(test):,} rows")

# Category vocabularies come from the training data only; unseen values -> "other".
vocab = {c: set(train_full[c].unique()) for c in CATEGORICAL}


def encode(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    X = df.drop(columns=["label"]).copy()
    for c in CATEGORICAL:
        X[c] = X[c].where(X[c].isin(vocab[c]), "other")
        X[c] = pd.Categorical(X[c], categories=sorted(vocab[c] | {"other"})).codes
    # Domain features: byte/packet ratios and per-packet payload sizes
    X["bytes_ratio"] = X["sbytes"] / (X["dbytes"] + 1)
    X["pkts_ratio"] = X["spkts"] / (X["dpkts"] + 1)
    X["sbytes_per_pkt"] = X["sbytes"] / (X["spkts"] + 1)
    X["dbytes_per_pkt"] = X["dbytes"] / (X["dpkts"] + 1)
    X["ttl_diff"] = X["sttl"] - X["dttl"]
    return X, df["label"].to_numpy()


X_full, y_full = encode(train_full)
X_test, y_test = encode(test)
X_tr, X_val, y_tr, y_val = train_test_split(
    X_full, y_full, test_size=0.15, stratify=y_full, random_state=SEED)

# Regularised to limit overfitting: depth caps, minimum leaf sizes and L2 penalties
# keep the train/held-out accuracy gap under ~2% (unregularised RF: ~7%), while
# leaving enough capacity for precision and recall to both stay above 90%.
models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=18, min_samples_leaf=10, max_features=0.5,
        n_jobs=N_JOBS, random_state=SEED),
    "Extra Trees": ExtraTreesClassifier(
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


def best_threshold(y, p):
    """Accuracy-maximising threshold, chosen on validation data only."""
    grid = np.linspace(0.05, 0.95, 181)
    accs = [accuracy_score(y, p >= t) for t in grid]
    return float(grid[int(np.argmax(accs))])


def metrics(name, y, p, thr):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {
        "Model": name, "Threshold": round(thr, 3),
        "Accuracy (%)": round(accuracy_score(y, pred) * 100, 2),
        "Precision (%)": round(precision_score(y, pred) * 100, 2),
        "Recall (%)": round(recall_score(y, pred) * 100, 2),
        "F1 (%)": round(f1_score(y, pred) * 100, 2),
        "F2 (%)": round(fbeta_score(y, pred, beta=2) * 100, 2),
        "ROC_AUC": round(roc_auc_score(y, p), 4),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
    }


rows, val_probs, test_probs = [], {}, {}
for name, model in models.items():
    model.fit(X_tr, y_tr)
    val_probs[name] = model.predict_proba(X_val)[:, 1]
    test_probs[name] = model.predict_proba(X_test)[:, 1]
    thr = best_threshold(y_val, val_probs[name])
    r = metrics(name, y_test, test_probs[name], thr)
    r["Val Accuracy (%)"] = round(accuracy_score(y_val, val_probs[name] >= thr) * 100, 2)
    rows.append(r)
    print(f"{name:14s} val_acc={r['Val Accuracy (%)']:.2f}%  test_acc={r['Accuracy (%)']:.2f}%  "
          f"F1={r['F1 (%)']:.2f}%  AUC={r['ROC_AUC']:.4f}  thr={thr:.2f}")

# Soft-voting ensemble of the gradient-boosted + RF models
ens_names = ["Random Forest", "XGBoost", "LightGBM"]
ens_val = np.mean([val_probs[n] for n in ens_names], axis=0)
ens_test = np.mean([test_probs[n] for n in ens_names], axis=0)
thr = best_threshold(y_val, ens_val)
r = metrics("Ensemble (RF+XGB+LGBM)", y_test, ens_test, thr)
r["Val Accuracy (%)"] = round(accuracy_score(y_val, ens_val >= thr) * 100, 2)
rows.append(r)
print(f"{'Ensemble':14s} val_acc={r['Val Accuracy (%)']:.2f}%  test_acc={r['Accuracy (%)']:.2f}%  "
      f"F1={r['F1 (%)']:.2f}%  AUC={r['ROC_AUC']:.4f}  thr={thr:.2f}")

results = pd.DataFrame(rows)
results.to_csv(REPORTS_DIR / "unsw_nb15_model_comparison.csv", index=False)

# Persist the model selected on VALIDATION accuracy (not test accuracy).
best_name = max(rows, key=lambda r: r["Val Accuracy (%)"])["Model"]
best_row = next(r for r in rows if r["Model"] == best_name)
artifact = {
    "selected_model": best_name,
    "models": {n: models[n] for n in (ens_names if best_name.startswith("Ensemble") else [best_name])},
    "threshold": best_row["Threshold"],
    "vocab": vocab,
    "feature_columns": list(X_full.columns),
}
joblib.dump(artifact, MODELS_DIR / "unsw_nb15_best_model.joblib")
(REPORTS_DIR / "unsw_nb15_best_model.json").write_text(json.dumps(best_row, indent=2))
print(f"\nSelected on validation: {best_name} -> test accuracy {best_row['Accuracy (%)']}%")
print(results.to_string(index=False))

# ------------------------------------------------------------
# Protocol B: pooled stratified 5-fold cross-validation
# (the protocol used by most UNSW-NB15 papers; reported separately
#  because it removes the official split's train/test distribution shift)
# ------------------------------------------------------------
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
import gc

# Free the fitted Protocol A models (already saved to disk) before cross-validation.
del artifact, val_probs, test_probs
models = {n: clone(m) for n, m in models.items()}
gc.collect()

pooled = pd.concat([train_full, test], ignore_index=True)
n_before = len(pooled)
# Exact duplicate rows would sit in both a training and a testing fold and inflate
# the score, so they are removed before cross-validation.
pooled = pooled.drop_duplicates().reset_index(drop=True)
print(f"\nPooled data: {n_before:,} rows -> {len(pooled):,} after removing exact duplicates")
vocab = {c: set(pooled[c].unique()) for c in CATEGORICAL}  # categories only, no labels
X_pool, y_pool = encode(pooled)
def balanced_threshold(y, p):
    """Precision-recall break-even threshold (precision ~= recall), validation data only."""
    grid = np.linspace(0.05, 0.95, 181)
    pred = [p >= t for t in grid]
    gaps = [abs(precision_score(y, q, zero_division=0) - recall_score(y, q)) for q in pred]
    return float(grid[int(np.argmin(gaps))])


cv_rows = []
for fold, (tr_idx, te_idx) in enumerate(
        StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(X_pool, y_pool), 1):
    # Inner split: 15% of the training fold picks the threshold; the test fold is never seen.
    fit_idx, val_idx = train_test_split(
        tr_idx, test_size=0.15, stratify=y_pool[tr_idx], random_state=SEED)
    probs, train_probs, val_p = [], [], []
    for n in ens_names:
        m = clone(models[n])
        m.fit(X_pool.iloc[fit_idx], y_pool[fit_idx])
        probs.append(m.predict_proba(X_pool.iloc[te_idx])[:, 1])
        train_probs.append(m.predict_proba(X_pool.iloc[fit_idx])[:, 1])
        val_p.append(m.predict_proba(X_pool.iloc[val_idx])[:, 1])
    thr = balanced_threshold(y_pool[val_idx], np.mean(val_p, axis=0))
    r = metrics(f"Ensemble fold {fold}", y_pool[te_idx], np.mean(probs, axis=0), thr)
    r["Train Accuracy (%)"] = round(
        accuracy_score(y_pool[fit_idx], np.mean(train_probs, axis=0) >= thr) * 100, 2)
    r["Overfit Gap (%)"] = round(r["Train Accuracy (%)"] - r["Accuracy (%)"], 2)
    cv_rows.append(r)
    print(f"Fold {fold}: thr={thr:.3f}  train_acc={r['Train Accuracy (%)']:.2f}%  "
          f"acc={r['Accuracy (%)']:.2f}%  gap={r['Overfit Gap (%)']:.2f}  P={r['Precision (%)']:.2f}%  "
          f"R={r['Recall (%)']:.2f}%  F1={r['F1 (%)']:.2f}%  AUC={r['ROC_AUC']:.4f}")
cv = pd.DataFrame(cv_rows)
cv.to_csv(REPORTS_DIR / "unsw_nb15_cv5_results.csv", index=False)
summary = cv.drop(columns=["Model", "Threshold"]).agg(["mean", "std"]).round(4)
summary.to_csv(REPORTS_DIR / "unsw_nb15_cv5_summary.csv")
print("\n5-fold CV (deduplicated) mean +/- std:")
for c in ["Train Accuracy (%)", "Accuracy (%)", "Overfit Gap (%)", "Precision (%)",
          "Recall (%)", "F1 (%)", "F2 (%)", "ROC_AUC"]:
    print(f"  {c:14s} {summary.loc['mean', c]:.4f} +/- {summary.loc['std', c]:.4f}")

# Final deployable model: the same ensemble refit on all deduplicated data.
final = {n: clone(models[n]).fit(X_pool, y_pool) for n in ens_names}
joblib.dump({"selected_model": "Ensemble (RF+XGB+LGBM), trained on pooled deduplicated data",
             "models": final, "threshold": round(float(cv["Threshold"].mean()), 3), "vocab": vocab,
             "feature_columns": list(X_pool.columns)},
            MODELS_DIR / "unsw_nb15_final_model.joblib")
print(f"Saved final model -> {MODELS_DIR / 'unsw_nb15_final_model.joblib'}")
