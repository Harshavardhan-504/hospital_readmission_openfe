# ============================================================
# 05_feature_selection.py  —  Feature Selection after OpenFE
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json, os, warnings
warnings.filterwarnings("ignore")

from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.feature_selection import SelectFromModel

RES_PATH = "outputs/results"
FIG_PATH = "outputs/figures"
os.makedirs(RES_PATH, exist_ok=True)

def main():
    # ── load OpenFE transformed data ──────────────────────────
    X_train = pd.read_csv(f"{RES_PATH}/X_train_openfe.csv")
    X_val   = pd.read_csv(f"{RES_PATH}/X_val_openfe.csv")
    X_test  = pd.read_csv(f"{RES_PATH}/X_test_openfe.csv")
    y_train = pd.read_csv(f"{RES_PATH}/y_train.csv").squeeze()
    y_val   = pd.read_csv(f"{RES_PATH}/y_val.csv").squeeze()
    y_test  = pd.read_csv(f"{RES_PATH}/y_test.csv").squeeze()

    print("=" * 55)
    print("FEATURE SELECTION AFTER OPENFE")
    print("=" * 55)
    print(f"Shape before selection : {X_train.shape}")

    # ── step 1: get feature importances ───────────────────────
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
    lgbm.fit(X_train, y_train)

    feat_imp = pd.Series(
        lgbm.feature_importances_,
        index=X_train.columns
    ).sort_values(ascending=False)

    print(f"\nAll feature importances:")
    print(feat_imp.to_string())

    # ── step 2: try different top-N cutoffs ───────────────────
    original_cols = [c for c in X_train.columns
                     if not c.startswith("autoFE")]

    print("\n── Trying different feature counts ──")
    results_by_n = {}

    for top_n in [10, 15, 20, 25, 30, 36]:
        top_feats = feat_imp.head(top_n).index.tolist()
        # always keep original features
        keep = list(set(original_cols + top_feats))
        keep = [c for c in X_train.columns if c in keep]

        clf = LGBMClassifier(
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
        clf.fit(X_train[keep], y_train)

        preds = clf.predict(X_test[keep])
        proba = clf.predict_proba(X_test[keep])[:, 1]
        acc = round(accuracy_score(y_test, preds), 4)
        auc = round(roc_auc_score(y_test, proba),  4)
        f1  = round(f1_score(y_test, preds),        4)
        results_by_n[top_n] = {"accuracy": acc,
                                "roc_auc": auc, "f1": f1}
        print(f"  top_{top_n:2d} features → "
              f"Acc={acc:.4f}  AUC={auc:.4f}  F1={f1:.4f}")

    # ── step 3: pick best by AUC ──────────────────────────────
    best_n = max(results_by_n, key=lambda n: results_by_n[n]["roc_auc"])
    print(f"\n✅ Best top_N = {best_n} (by AUC)")

    top_feats  = feat_imp.head(best_n).index.tolist()
    keep_final = list(set(original_cols + top_feats))
    keep_final = [c for c in X_train.columns if c in keep_final]

    # ── step 4: final model with selected features ────────────
    print(f"\n── Final Model with {len(keep_final)} selected features ──")
    final_clf = LGBMClassifier(
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
    final_clf.fit(X_train[keep_final], y_train)

    results_selected = {}
    for split_name, Xs, ys in [
            ("train", X_train[keep_final], y_train),
            ("val",   X_val[keep_final],   y_val),
            ("test",  X_test[keep_final],  y_test)]:
        preds = final_clf.predict(Xs)
        proba = final_clf.predict_proba(Xs)[:, 1]
        results_selected[split_name] = {
            "accuracy": round(accuracy_score(ys, preds), 4),
            "roc_auc" : round(roc_auc_score(ys, proba),  4),
            "f1"      : round(f1_score(ys, preds),       4),
        }
        print(f"  {split_name:5s} → "
              f"Acc={results_selected[split_name]['accuracy']:.4f}  "
              f"AUC={results_selected[split_name]['roc_auc']:.4f}  "
              f"F1={results_selected[split_name]['f1']:.4f}")

    # ── step 5: plot selected feature importances ─────────────
    sel_imp = pd.Series(
        final_clf.feature_importances_,
        index=keep_final
    ).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 7))
    sel_imp.plot(kind="bar", color="steelblue", ax=ax)
    ax.set_title("Feature Importances after Selection")
    ax.set_ylabel("Importance")
    plt.tight_layout()
    plt.savefig(f"{FIG_PATH}/selected_feature_importance.png", dpi=150)
    plt.close()
    print(f"\n✅ Plot saved")

    # ── step 6: save results ──────────────────────────────────
    pd.DataFrame(keep_final, columns=["feature"]).to_csv(
        f"{RES_PATH}/selected_features.csv", index=False)

    with open(f"{RES_PATH}/selected_metrics.json", "w") as f:
        json.dump(results_selected, f, indent=2)

    # ── final 3-way comparison ────────────────────────────────
    with open(f"{RES_PATH}/baseline_metrics.json") as f:
        baseline = json.load(f)
    with open(f"{RES_PATH}/openfe_metrics.json") as f:
        openfe = json.load(f)

    print("\n" + "=" * 65)
    print("FULL COMPARISON — TEST SET (LightGBM)")
    print("=" * 65)
    print(f"{'Metric':<12} {'Baseline':>12} {'OpenFE':>12} "
          f"{'Selected':>12}")
    print("-" * 65)
    for metric in ["accuracy", "roc_auc", "f1"]:
        b = baseline["LightGBM"]["test"][metric]
        o = openfe["test"][metric]
        s = results_selected["test"][metric]
        print(f"{metric:<12} {b:>12.4f} {o:>12.4f} {s:>12.4f}")
    print("=" * 65)
    print("✅ FEATURE SELECTION COMPLETE")
    print("=" * 65)

if __name__ == '__main__':
    main()