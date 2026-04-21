# ============================================================
# 03_baseline_model.py  —  Baseline Models on Raw Features
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json, os

from sklearn.linear_model    import LogisticRegression
from sklearn.metrics         import (accuracy_score, roc_auc_score,
                                     f1_score, classification_report,
                                     ConfusionMatrixDisplay)
from lightgbm                import LGBMClassifier

RES_PATH = "outputs/results"
FIG_PATH = "outputs/figures"
os.makedirs(RES_PATH, exist_ok=True)
os.makedirs(FIG_PATH, exist_ok=True)

# ── load splits ───────────────────────────────────────────────
X_train = pd.read_csv(f"{RES_PATH}/X_train.csv")
X_val   = pd.read_csv(f"{RES_PATH}/X_val.csv")
X_test  = pd.read_csv(f"{RES_PATH}/X_test.csv")
y_train = pd.read_csv(f"{RES_PATH}/y_train.csv").squeeze()
y_val   = pd.read_csv(f"{RES_PATH}/y_val.csv").squeeze()
y_test  = pd.read_csv(f"{RES_PATH}/y_test.csv").squeeze()

print("=" * 55)
print("BASELINE MODELS — RAW FEATURES")
print("=" * 55)

# ── helper ────────────────────────────────────────────────────
def evaluate(name, model, X_tr, y_tr, X_val, y_val, X_te, y_te):
    model.fit(X_tr, y_tr)

    results = {}
    for split_name, Xs, ys in [("train", X_tr, y_tr),
                                ("val",   X_val, y_val),
                                ("test",  X_te,  y_te)]:
        preds = model.predict(Xs)
        proba = model.predict_proba(Xs)[:, 1]
        results[split_name] = {
            "accuracy" : round(accuracy_score(ys, preds),  4),
            "roc_auc"  : round(roc_auc_score(ys, proba),   4),
            "f1"       : round(f1_score(ys, preds),        4),
        }

    print(f"\n── {name} ──")
    for s, m in results.items():
        print(f"  {s:5s} → Acc={m['accuracy']:.4f}  "
              f"AUC={m['roc_auc']:.4f}  F1={m['f1']:.4f}")
    print(f"\n  Test Classification Report:")
    print(classification_report(y_te, model.predict(X_te),
                                 target_names=["No","Yes"]))

    # confusion matrix
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_te, model.predict(X_te),
        display_labels=["No", "Yes"],
        cmap="Blues", ax=ax)
    ax.set_title(f"{name} — Confusion Matrix (Test)")
    plt.tight_layout()
    safe = name.lower().replace(" ", "_")
    plt.savefig(f"{FIG_PATH}/baseline_{safe}_cm.png", dpi=150)
    plt.close()
    print(f"  ✅ Confusion matrix saved")

    return results

all_results = {}

# ── 1. Logistic Regression ────────────────────────────────────
lr = LogisticRegression(max_iter=1000, random_state=42)
all_results["Logistic Regression"] = evaluate(
    "Logistic Regression", lr,
    X_train, y_train, X_val, y_val, X_test, y_test)

# ── 2. LightGBM ───────────────────────────────────────────────
lgbm = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
    verbose=-1)
all_results["LightGBM"] = evaluate(
    "LightGBM", lgbm,
    X_train, y_train, X_val, y_val, X_test, y_test)

# ── feature importance (LightGBM) ────────────────────────────
feat_imp = pd.Series(
    lgbm.feature_importances_,
    index=X_train.columns).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(8, 6))
feat_imp.plot(kind="bar", color="steelblue", ax=ax)
ax.set_title("LightGBM Feature Importance (Baseline)")
ax.set_ylabel("Importance")
plt.tight_layout()
plt.savefig(f"{FIG_PATH}/baseline_lgbm_feature_importance.png", dpi=150)
plt.close()
print(f"\n✅ Feature importance plot saved")
print(f"\nTop 5 features:\n{feat_imp.head()}")

# ── save baseline metrics ─────────────────────────────────────
with open(f"{RES_PATH}/baseline_metrics.json", "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\n✅ Metrics saved → {RES_PATH}/baseline_metrics.json")

# ── summary table ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("BASELINE SUMMARY (TEST SET)")
print("=" * 55)
print(f"{'Model':<22} {'Accuracy':>10} {'ROC-AUC':>10} {'F1':>10}")
print("-" * 55)
for model_name, res in all_results.items():
    m = res["test"]
    print(f"{model_name:<22} {m['accuracy']:>10.4f} "
          f"{m['roc_auc']:>10.4f} {m['f1']:>10.4f}")
print("=" * 55)
print("✅ BASELINE COMPLETE — these are our numbers to beat!")
print("=" * 55)