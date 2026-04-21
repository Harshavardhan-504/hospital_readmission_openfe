# ============================================================
# 04_openfe_generation.py  —  OpenFE Feature Generation
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json, os, pickle, warnings
warnings.filterwarnings("ignore")

from openfe import OpenFE, transform
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

RES_PATH = "outputs/results"
FIG_PATH = "outputs/figures"
os.makedirs(RES_PATH, exist_ok=True)

def main():
    # ── load splits ───────────────────────────────────────────
    X_train = pd.read_csv(f"{RES_PATH}/X_train.csv")
    X_val   = pd.read_csv(f"{RES_PATH}/X_val.csv")
    X_test  = pd.read_csv(f"{RES_PATH}/X_test.csv")
    y_train = pd.read_csv(f"{RES_PATH}/y_train.csv").squeeze()
    y_val   = pd.read_csv(f"{RES_PATH}/y_val.csv").squeeze()
    y_test  = pd.read_csv(f"{RES_PATH}/y_test.csv").squeeze()

    print("=" * 55)
    print("OPENFE — AUTOMATED FEATURE GENERATION")
    print("=" * 55)
    print(f"Train shape before OpenFE : {X_train.shape}")

    # ── run OpenFE ────────────────────────────────────────────
    print("\n⏳ Running OpenFE (this takes 2-5 mins)...\n")

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

    # Keep only top 10 features like the OpenFE paper does
    # for moderate-sized datasets
    features = features[:20]
    print(f"Using top 20 features only to avoid overfitting")

    print(f"\n✅ OpenFE complete!")
    print(f"Top features selected : {len(features)}")

    # ── show top 20 generated features ───────────────────────
    print("\n── Top 20 Generated Features ──")
    for i, f in enumerate(features[:20]):
        print(f"  {i+1:2d}. {f.name}")

    # ── transform train / val / test ──────────────────────────
    print("\n⏳ Transforming datasets with new features...")
    X_train_new, X_test_new = transform(
        X_train, X_test, features, n_jobs=1)
    X_train_new2, X_val_new = transform(
        X_train, X_val,  features, n_jobs=1)

    print(f"\nShape after OpenFE:")
    print(f"  X_train : {X_train_new.shape}")
    print(f"  X_val   : {X_val_new.shape}")
    print(f"  X_test  : {X_test_new.shape}")

    # ── eval with ALL generated features ──────────────────────
    print("\n── LightGBM with ALL OpenFE features ──")
    lgbm = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=20,
        min_child_samples=50,
        reg_alpha=0.2,
        reg_lambda=0.2,
        subsample=0.8,
        colsample_bytree=0.7,
        random_state=42,
        verbose=-1)

    lgbm.fit(X_train_new, y_train,
         eval_set=[(X_val_new, y_val)])

    results_openfe = {}
    for split_name, Xs, ys in [("train", X_train_new, y_train),
                                ("val",   X_val_new,   y_val),
                                ("test",  X_test_new,  y_test)]:
        preds = lgbm.predict(Xs)
        proba = lgbm.predict_proba(Xs)[:, 1]
        results_openfe[split_name] = {
            "accuracy" : round(accuracy_score(ys, preds), 4),
            "roc_auc"  : round(roc_auc_score(ys, proba),  4),
            "f1"       : round(f1_score(ys, preds),       4),
        }
        print(f"  {split_name:5s} → "
              f"Acc={results_openfe[split_name]['accuracy']:.4f}  "
              f"AUC={results_openfe[split_name]['roc_auc']:.4f}  "
              f"F1={results_openfe[split_name]['f1']:.4f}")

    # ── feature importance ────────────────────────────────────
    feat_imp = pd.Series(
        lgbm.feature_importances_,
        index=X_train_new.columns
    ).sort_values(ascending=False)

    print(f"\nTop 10 features after OpenFE:")
    print(feat_imp.head(10))

    original_cols = list(X_train.columns)
    top10 = feat_imp.head(10).index.tolist()
    new_in_top10 = [f for f in top10 if f not in original_cols]
    print(f"\nNew generated features in top 10: {new_in_top10}")

    fig, ax = plt.subplots(figsize=(10, 7))
    feat_imp.head(20).plot(kind="bar", color="steelblue", ax=ax)
    ax.set_title("Top 20 Feature Importances after OpenFE")
    ax.set_ylabel("Importance")
    plt.tight_layout()
    plt.savefig(f"{FIG_PATH}/openfe_feature_importance.png", dpi=150)
    plt.close()
    print(f"\n✅ Feature importance plot saved")

    # ── save everything ───────────────────────────────────────
    X_train_new.to_csv(f"{RES_PATH}/X_train_openfe.csv", index=False)
    X_val_new.to_csv(f"{RES_PATH}/X_val_openfe.csv",     index=False)
    X_test_new.to_csv(f"{RES_PATH}/X_test_openfe.csv",   index=False)

    with open(f"{RES_PATH}/openfe_features.pkl", "wb") as f:
        pickle.dump(features, f)

    with open(f"{RES_PATH}/openfe_metrics.json", "w") as f:
        json.dump(results_openfe, f, indent=2)

    print(f"✅ All files saved to {RES_PATH}/")

    # ── compare baseline vs openfe ────────────────────────────
    with open(f"{RES_PATH}/baseline_metrics.json") as f:
        baseline = json.load(f)

    print("\n" + "=" * 60)
    print("COMPARISON: BASELINE vs OPENFE (TEST SET — LightGBM)")
    print("=" * 60)
    print(f"{'Metric':<12} {'Baseline':>12} {'OpenFE':>12} {'Delta':>10}")
    print("-" * 60)
    for metric in ["accuracy", "roc_auc", "f1"]:
        base_val = baseline["LightGBM"]["test"][metric]
        ofe_val  = results_openfe["test"][metric]
        delta    = ofe_val - base_val
        arrow    = "↑" if delta > 0 else "↓"
        print(f"{metric:<12} {base_val:>12.4f} {ofe_val:>12.4f} "
              f"{arrow}{abs(delta):>8.4f}")
    print("=" * 60)
    print("✅ OPENFE COMPLETE")
    print("=" * 60)

if __name__ == '__main__':
    main()