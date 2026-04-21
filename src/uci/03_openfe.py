# ============================================================
# src/uci/03_openfe.py
# OpenFE Feature Generation on UCI Diabetes Dataset
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json, os, pickle, warnings
warnings.filterwarnings("ignore")

from openfe import OpenFE, transform
from lightgbm import LGBMClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score,
                             f1_score)

RES_PATH = "outputs/uci/results"
FIG_PATH = "outputs/uci/figures"
os.makedirs(RES_PATH, exist_ok=True)
os.makedirs(FIG_PATH, exist_ok=True)

def main():
    # ── load ──────────────────────────────────────────────────
    X_train = pd.read_csv(f"{RES_PATH}/X_train.csv")
    X_val   = pd.read_csv(f"{RES_PATH}/X_val.csv")
    X_test  = pd.read_csv(f"{RES_PATH}/X_test.csv")
    y_train = pd.read_csv(f"{RES_PATH}/y_train.csv").squeeze()
    y_val   = pd.read_csv(f"{RES_PATH}/y_val.csv").squeeze()
    y_test  = pd.read_csv(f"{RES_PATH}/y_test.csv").squeeze()

    print("=" * 60)
    print("OPENFE — UCI DIABETES FEATURE GENERATION")
    print("=" * 60)
    print(f"Train shape before OpenFE: {X_train.shape}")

    # ── run OpenFE ────────────────────────────────────────────
    print("\n⏳ Running OpenFE (5-10 mins on this dataset)...\n")

    ofe = OpenFE()
    features = ofe.fit(
        data=X_train,
        label=y_train,
        n_jobs=1,
        n_data_blocks=8,
        min_candidate_features=2000,
        task="classification",
        verbose=True
    )

    print(f"\n✅ OpenFE complete!")
    print(f"Total features found: {len(features)}")

    # keep top 30 — good balance for this dataset size
    features = features[:20]
    print(f"Using top 20 features")

    # ── show top 30 ───────────────────────────────────────────
    print("\n── Top 30 Generated Features ──")
    for i, f in enumerate(features):
        print(f"  {i+1:2d}. {f.name}")

    # ── transform ─────────────────────────────────────────────
    print("\n⏳ Transforming datasets...")
    X_train_new, X_test_new = transform(
        X_train, X_test, features, n_jobs=1)
    _, X_val_new = transform(
        X_train, X_val, features, n_jobs=1)

    print(f"\nShape after OpenFE:")
    print(f"  X_train : {X_train_new.shape}")
    print(f"  X_val   : {X_val_new.shape}")
    print(f"  X_test  : {X_test_new.shape}")

    # ── eval ──────────────────────────────────────────────────
    print("\n── LightGBM with OpenFE features ──")
    lgbm = LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.01,
        num_leaves=15,
        min_child_samples=100,
        reg_alpha=0.5,
        reg_lambda=0.5,
        subsample=0.7,
        colsample_bytree=0.6,
        random_state=42,
        verbose=-1)

    lgbm.fit(X_train_new, y_train,
             eval_set=[(X_val_new, y_val)])

    results = {}
    for sname, Xs, ys in [("train", X_train_new, y_train),
                           ("val",   X_val_new,   y_val),
                           ("test",  X_test_new,  y_test)]:
        preds = lgbm.predict(Xs)
        proba = lgbm.predict_proba(Xs)[:, 1]
        results[sname] = {
            "accuracy": round(accuracy_score(ys, preds),  4),
            "roc_auc" : round(roc_auc_score(ys, proba),   4),
            "f1"      : round(f1_score(ys, preds),        4),
        }
        print(f"  {sname:5s} → "
              f"Acc={results[sname]['accuracy']:.4f}  "
              f"AUC={results[sname]['roc_auc']:.4f}  "
              f"F1={results[sname]['f1']:.4f}")

    # ── feature importance ────────────────────────────────────
    feat_imp = pd.Series(
        lgbm.feature_importances_,
        index=X_train_new.columns
    ).sort_values(ascending=False)

    original_cols = list(X_train.columns)
    top10 = feat_imp.head(10).index.tolist()
    new_in_top10 = [f for f in top10
                    if f not in original_cols]

    print(f"\nTop 10 features after OpenFE:")
    print(feat_imp.head(10))
    print(f"\nNew generated features in top 10: {new_in_top10}")

    fig, ax = plt.subplots(figsize=(12, 7))
    feat_imp.head(25).plot(kind="bar", color="steelblue", ax=ax)
    ax.set_title("Top 25 Feature Importances after OpenFE\n"
                 "UCI Diabetes Dataset")
    ax.set_ylabel("Importance")
    plt.tight_layout()
    plt.savefig(f"{FIG_PATH}/openfe_feature_importance.png",
                dpi=150)
    plt.close()
    print(f"\n✅ Feature importance plot saved")

    # ── save ──────────────────────────────────────────────────
    X_train_new.to_csv(f"{RES_PATH}/X_train_openfe.csv",
                       index=False)
    X_val_new.to_csv(f"{RES_PATH}/X_val_openfe.csv",
                     index=False)
    X_test_new.to_csv(f"{RES_PATH}/X_test_openfe.csv",
                      index=False)

    with open(f"{RES_PATH}/openfe_features.pkl", "wb") as f:
        pickle.dump(features, f)

    with open(f"{RES_PATH}/openfe_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    # ── compare vs baseline ───────────────────────────────────
    with open(f"{RES_PATH}/baseline_metrics.json") as f:
        baseline = json.load(f)

    print("\n" + "=" * 60)
    print("COMPARISON: BASELINE vs OPENFE (TEST — LightGBM)")
    print("=" * 60)
    print(f"{'Metric':<12} {'Baseline':>10} "
          f"{'OpenFE':>10} {'Delta':>10}")
    print("-" * 60)
    for metric in ["accuracy", "roc_auc", "f1"]:
        base = baseline["LightGBM"]["test"][metric]
        ofe  = results["test"][metric]
        delta = ofe - base
        arrow = "↑" if delta > 0 else "↓"
        print(f"{metric:<12} {base:>10.4f} {ofe:>10.4f} "
              f"{arrow}{abs(delta):>8.4f}")
    print("=" * 60)
    print("✅ OPENFE COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()