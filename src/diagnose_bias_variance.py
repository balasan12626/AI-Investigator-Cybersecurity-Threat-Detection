"""Bias/variance diagnostics for the UNSW-NB15 models: train vs validation vs test
accuracy, a learning curve, and fold-to-fold spread."""
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

BASE = Path(__file__).resolve().parent.parent
D = BASE / "data" / "external" / "unsw_nb15"
CAT = ["proto", "service", "state"]; SEED = 42
load = lambda p: pd.read_csv(p, encoding="utf-8-sig").drop(columns=["id", "attack_cat"])
train_full, test = load(D / "test.csv"), load(D / "train.csv")   # mirror names are swapped
vocab = {c: set(train_full[c].unique()) for c in CAT}

def encode(df):
    X = df.drop(columns=["label"]).copy()
    for c in CAT:
        X[c] = pd.Categorical(X[c].where(X[c].isin(vocab[c]), "other"),
                              categories=sorted(vocab[c] | {"other"})).codes
    X["bytes_ratio"] = X["sbytes"] / (X["dbytes"] + 1); X["pkts_ratio"] = X["spkts"] / (X["dpkts"] + 1)
    X["sbytes_per_pkt"] = X["sbytes"] / (X["spkts"] + 1); X["dbytes_per_pkt"] = X["dbytes"] / (X["dpkts"] + 1)
    X["ttl_diff"] = X["sttl"] - X["dttl"]
    return X, df["label"].to_numpy()

Xf, yf = encode(train_full); Xt, yt = encode(test)
Xtr, Xv, ytr, yv = train_test_split(Xf, yf, test_size=0.15, stratify=yf, random_state=SEED)
print(f"Split sizes: train={len(ytr):,}  validation={len(yv):,}  test={len(yt):,}")
print(f"Attack share: train={ytr.mean():.1%}  validation={yv.mean():.1%}  test={yt.mean():.1%}\n")

models = {
  "Logistic-like shallow tree (depth 2)": RandomForestClassifier(n_estimators=50, max_depth=2, n_jobs=-1, random_state=SEED),
  "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=SEED),
  "Extra Trees": ExtraTreesClassifier(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=SEED),
  "XGBoost": XGBClassifier(n_estimators=600, max_depth=8, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=SEED),
  "LightGBM": LGBMClassifier(n_estimators=600, num_leaves=63, learning_rate=0.05, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, n_jobs=-1, random_state=SEED, verbose=-1),
}
rows = []
for n, m in models.items():
    m.fit(Xtr, ytr)
    a = [accuracy_score(y, m.predict(X)) * 100 for X, y in [(Xtr, ytr), (Xv, yv), (Xt, yt)]]
    rows.append({"Model": n, "Train acc": round(a[0], 2), "Val acc": round(a[1], 2), "Test acc": round(a[2], 2),
                 "Train-Val gap": round(a[0] - a[1], 2), "Val-Test gap": round(a[1] - a[2], 2)})
print(pd.DataFrame(rows).to_string(index=False))

print("\nLearning curve (LightGBM): accuracy vs amount of training data")
lc = []
for frac in [0.05, 0.1, 0.25, 0.5, 1.0]:
    n = int(len(ytr) * frac)
    m = LGBMClassifier(**models["LightGBM"].get_params()).fit(Xtr.iloc[:n], ytr[:n])
    lc.append({"Train rows": n, "Train acc": round(accuracy_score(ytr[:n], m.predict(Xtr.iloc[:n])) * 100, 2),
               "Val acc": round(accuracy_score(yv, m.predict(Xv)) * 100, 2),
               "Test acc": round(accuracy_score(yt, m.predict(Xt)) * 100, 2)})
print(pd.DataFrame(lc).to_string(index=False))

print("\nVariance across 5 folds (LightGBM, official training set only)")
accs = []
for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xf, yf):
    m = LGBMClassifier(**models["LightGBM"].get_params()).fit(Xf.iloc[tr], yf[tr])
    accs.append(accuracy_score(yf[te], m.predict(Xf.iloc[te])) * 100)
print("Fold accuracies:", [round(a, 2) for a in accs], f"-> mean {np.mean(accs):.2f}%, std {np.std(accs):.2f}")
pd.DataFrame(rows).to_csv(BASE / "results/reports/unsw_nb15_bias_variance.csv", index=False)
pd.DataFrame(lc).to_csv(BASE / "results/reports/unsw_nb15_learning_curve.csv", index=False)
