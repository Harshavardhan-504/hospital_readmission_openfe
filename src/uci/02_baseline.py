# ============================================================
# src/uci/02_baseline.py
# Baseline Models on UCI Diabetes Dataset
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json, os, warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, roc_auc_score,
                             f1_score, classification_report,
                             ConfusionMatrixDisplay)
from lightgbm import LGBMClassifier

RES_PATH = "outputs/uci/results"
FIG_PATH = "outputs/uci/figures"
os.makedirs(RES_PATH, exist_ok=True)
os.makedirs(FIG_PATH, exist_ok=True)

def evaluate(name, model, X_tr, y_tr,
             X_val, y_val, X_te, y_te):
    model.fit(X_tr, y_tr)
    results = {}
    for sname, Xs, ys in [("train", X_tr,  y_tr),
                           ("val",   X_val, y_val),
                           ("test",  X_te,  y_te)]:
        preds = model.predict(Xs)
        proba = model.predict_proba(Xs)[:, 1]
        results[sname] = {
            "accuracy": round(accuracy_score(ys, preds),  4),
            "roc_auc" : round(roc_auc_score(ys, proba),   4),
            "f1"      : round(f1_score(ys, preds),        4),
        }
    print(f"\n── {name} ──")
    for s, m in results.items():
        print(f"  {s:5s} → Acc={m['accuracy']:.4f}  "
              f"AUC={m['roc_auc']:.4f}  "
              f"F1={m['f1']:.4f}")
    print(f"\n  Classification Report (Test):")
    print(classification_report(
        y_te, model.predict(X_te),
        target_names=["No", "Yes"]))

    # confusion matrix
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_te, model.predict(X_te),
        display_labels=["No", "Yes"],
        cmap="Blues", ax=ax)
    ax.set_title(f"{name} — Confusion Matrix")
    plt.tight_layout()
    safe = name.lower().replace(" ", "_")
    plt.savefig(f"{FIG_PATH}/baseline_{safe}_cm.png", dpi=150)
    plt.close()
    return results

def main():
    # ── load ──────────────────────────────────────────────────
    X_train = pd.read_csv(f"{RES_PATH}/X_train.csv")
    X_val   = pd.read_csv(f"{RES_PATH}/X_val.csv")
    X_test  = pd.read_csv(f"{RES_PATH}/X_test.csv")
    y_train = pd.read_csv(f"{RES_PATH}/y_train.csv").squeeze()
    y_val   = pd.read_csv(f"{RES_PATH}/y_val.csv").squeeze()
    y_test  = pd.read_csv(f"{RES_PATH}/y_test.csv").squeeze()

    print("=" * 60)
    print("BASELINE MODELS — UCI DIABETES (42 features)")
    print("=" * 60)
    print(f"Train: {X_train.shape} | "
          f"Val: {X_val.shape} | "
          f"Test: {X_test.shape}")

    all_results = {}

    # ── logistic regression ───────────────────────────────────
    lr = LogisticRegression(max_iter=1000, random_state=42)
    all_results["Logistic Regression"] = evaluate(
        "Logistic Regression", lr,
        X_train, y_train, X_val, y_val, X_test, y_test)

    # ── lightgbm ──────────────────────────────────────────────
    lgbm = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        verbose=-1)
    all_results["LightGBM"] = evaluate(
        "LightGBM", lgbm,
        X_train, y_train, X_val, y_val, X_test, y_test)

    # ── feature importance ────────────────────────────────────
    feat_imp = pd.Series(
        lgbm.feature_importances_,
        index=X_train.columns
    ).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 8))
    feat_imp.head(20).plot(kind="bar", color="steelblue", ax=ax)
    ax.set_title("LightGBM Top 20 Feature Importances (Baseline)")
    ax.set_ylabel("Importance")
    plt.tight_layout()
    plt.savefig(f"{FIG_PATH}/baseline_feature_importance.png",
                dpi=150)
    plt.close()

    print(f"\nTop 10 features (baseline):")
    print(feat_imp.head(10))

    # ── save ──────────────────────────────────────────────────
    with open(f"{RES_PATH}/baseline_metrics.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # ── summary ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("BASELINE SUMMARY — TEST SET")
    print("=" * 60)
    print(f"{'Model':<22} {'Acc':>8} {'AUC':>8} {'F1':>8}")
    print("-" * 60)
    for name, res in all_results.items():
        m = res["test"]
        print(f"{name:<22} "
              f"{m['accuracy']:>8.4f} "
              f"{m['roc_auc']:>8.4f} "
              f"{m['f1']:>8.4f}")
    print("=" * 60)
    print("✅ BASELINE COMPLETE — numbers to beat with OpenFE!")
    print("=" * 60)

if __name__ == "__main__":
    main()