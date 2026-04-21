# ============================================================
# src/uci/04_feature_selection.py
# Multiple Models on OpenFE Features — Find the Best!
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json, os, warnings
warnings.filterwarnings("ignore")

from lightgbm  import LGBMClassifier
from xgboost   import XGBClassifier
from sklearn.ensemble       import RandomForestClassifier
from sklearn.linear_model   import LogisticRegression
from sklearn.metrics        import (accuracy_score, roc_auc_score,
                                    f1_score, roc_curve,
                                    ConfusionMatrixDisplay)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing  import StandardScaler
from sklearn.pipeline       import Pipeline
from sklearn.impute import SimpleImputer

RES_PATH = "outputs/uci/results"
FIG_PATH = "outputs/uci/figures"
os.makedirs(RES_PATH, exist_ok=True)
os.makedirs(FIG_PATH, exist_ok=True)

def main():
    # ── load OpenFE transformed data ──────────────────────────
    X_train = pd.read_csv(f"{RES_PATH}/X_train_openfe.csv")
    X_val   = pd.read_csv(f"{RES_PATH}/X_val_openfe.csv")
    X_test  = pd.read_csv(f"{RES_PATH}/X_test_openfe.csv")
    y_train = pd.read_csv(f"{RES_PATH}/y_train.csv").squeeze()
    y_val   = pd.read_csv(f"{RES_PATH}/y_val.csv").squeeze()
    y_test  = pd.read_csv(f"{RES_PATH}/y_test.csv").squeeze()

    # combine train+val for CV
    X_tv = pd.concat([X_train, X_val], ignore_index=True)
    y_tv = pd.concat([y_train, y_val], ignore_index=True)

    print("=" * 65)
    print("MULTIPLE MODELS ON OPENFE FEATURES")
    print("=" * 65)
    print(f"Features: {X_train.shape[1]}")

    # ── feature selection: keep top N by importance ───────────
    print("\n⏳ Selecting best features via LightGBM importance...")
    selector = LGBMClassifier(
        n_estimators=1000, learning_rate=0.01,
        num_leaves=15, min_child_samples=100,
        reg_alpha=0.5, reg_lambda=0.5,
        subsample=0.7, colsample_bytree=0.6,
        random_state=42, verbose=-1)
    selector.fit(X_train, y_train)

    feat_imp = pd.Series(
        selector.feature_importances_,
        index=X_train.columns
    ).sort_values(ascending=False)

    # try different top-N and pick best by val AUC
    print("\n── Finding best feature count ──")
    best_n, best_val_auc = 62, 0
    for n in [20, 25, 30, 40, 50, 62]:
        top_feats = feat_imp.head(n).index.tolist()
        clf = LGBMClassifier(
            n_estimators=1000, learning_rate=0.01,
            num_leaves=15, min_child_samples=100,
            reg_alpha=0.5, reg_lambda=0.5,
            subsample=0.7, colsample_bytree=0.6,
            random_state=42, verbose=-1)
        clf.fit(X_train[top_feats], y_train)
        val_auc = roc_auc_score(
            y_val, clf.predict_proba(
                X_val[top_feats])[:, 1])
        print(f"  top_{n:2d} → val AUC={val_auc:.4f}")
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_n = n

    best_feats = feat_imp.head(best_n).index.tolist()
    print(f"\n✅ Best feature count = {best_n}  "
          f"(val AUC={best_val_auc:.4f})")

    X_tr  = X_train[best_feats]
    X_va  = X_val[best_feats]
    X_te  = X_test[best_feats]
    X_tv2 = X_tv[best_feats]

    # ── define all models ─────────────────────────────────────
    models = {
        "LightGBM" : LGBMClassifier(
            n_estimators=1000, learning_rate=0.01,
            num_leaves=15, min_child_samples=100,
            reg_alpha=0.5, reg_lambda=0.5,
            subsample=0.7, colsample_bytree=0.6,
            random_state=42, verbose=-1),

        "XGBoost"  : XGBClassifier(
            n_estimators=1000, learning_rate=0.01,
            max_depth=4, min_child_weight=10,
            reg_alpha=0.5, reg_lambda=1.0,
            subsample=0.7, colsample_bytree=0.6,
            random_state=42, verbosity=0,
            eval_metric="logloss"),

        "RandomForest" : RandomForestClassifier(
            n_estimators=500,
            max_depth=10,
            min_samples_leaf=50,
            max_features="sqrt",
            random_state=42,
            n_jobs=1),

        "LogisticRegression" : Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
            ("clf",     LogisticRegression(
                            max_iter=1000, C=0.1,
                            random_state=42))]),
    }

    # ── train and evaluate each model ─────────────────────────
    print("\n" + "=" * 65)
    print("MODEL EVALUATION ON OPENFE FEATURES")
    print("=" * 65)

    all_results = {}
    all_probas  = {}
    skf = StratifiedKFold(n_splits=5,
                          shuffle=True, random_state=42)

    for name, model in models.items():
        print(f"\n── {name} ──")
        model.fit(X_tr, y_train)

        preds = model.predict(X_te)
        proba = model.predict_proba(X_te)[:, 1]

        acc = round(accuracy_score(y_test, preds),  4)
        auc = round(roc_auc_score(y_test, proba),   4)
        f1  = round(f1_score(y_test, preds),        4)

        # 5-fold CV AUC
        cv_auc = cross_val_score(
            model, X_tv2, y_tv,
            cv=skf, scoring="roc_auc", n_jobs=1)

        all_results[name] = {
            "accuracy"   : acc,
            "roc_auc"    : auc,
            "f1"         : f1,
            "cv_auc_mean": round(float(cv_auc.mean()), 4),
            "cv_auc_std" : round(float(cv_auc.std()),  4),
        }
        all_probas[name] = proba

        print(f"  Test → Acc={acc:.4f}  "
              f"AUC={auc:.4f}  F1={f1:.4f}")
        print(f"  5-fold CV AUC = "
              f"{cv_auc.mean():.4f} ± {cv_auc.std():.4f}")

    # ── ensemble: average all probas ──────────────────────────
    print(f"\n── Ensemble (avg all models) ──")
    ens_proba = np.mean(list(all_probas.values()), axis=0)
    ens_preds = (ens_proba >= 0.5).astype(int)
    ens_acc   = round(accuracy_score(y_test, ens_preds), 4)
    ens_auc   = round(roc_auc_score(y_test, ens_proba),  4)
    ens_f1    = round(f1_score(y_test, ens_preds),       4)
    all_results["Ensemble"] = {
        "accuracy": ens_acc, "roc_auc": ens_auc,
        "f1": ens_f1, "cv_auc_mean": 0, "cv_auc_std": 0}
    all_probas["Ensemble"] = ens_proba
    print(f"  Test → Acc={ens_acc:.4f}  "
          f"AUC={ens_auc:.4f}  F1={ens_f1:.4f}")

    # ── load baseline for comparison ──────────────────────────
    with open(f"{RES_PATH}/baseline_metrics.json") as f:
        baseline = json.load(f)
    bl_acc = baseline["LightGBM"]["test"]["accuracy"]
    bl_auc = baseline["LightGBM"]["test"]["roc_auc"]
    bl_f1  = baseline["LightGBM"]["test"]["f1"]

    # ── ROC curves ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = ["tomato","seagreen","darkorange","purple","steelblue"]
    for (name, proba), color in zip(
            all_probas.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc = all_results[name]["roc_auc"]
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{name} (AUC={auc:.4f})")

    # baseline ROC
    X_train_raw = pd.read_csv(f"{RES_PATH}/X_train.csv")
    X_test_raw  = pd.read_csv(f"{RES_PATH}/X_test.csv")
    bl_model = LGBMClassifier(n_estimators=500,
                               learning_rate=0.05,
                               num_leaves=31,
                               random_state=42, verbose=-1)
    bl_model.fit(X_train_raw, y_train)
    fpr_b, tpr_b, _ = roc_curve(
        y_test, bl_model.predict_proba(X_test_raw)[:, 1])
    ax.plot(fpr_b, tpr_b, "k--", lw=2,
            label=f"Baseline LGBM (AUC={bl_auc:.4f})")
    ax.plot([0,1],[0,1], "gray", lw=1,
            linestyle=":", label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — All Models on OpenFE Features\n"
                 "UCI Diabetes Dataset",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{FIG_PATH}/all_models_roc.png", dpi=150)
    plt.close()
    print(f"\n✅ ROC curves saved")

    # ── bar chart ─────────────────────────────────────────────
    metrics = ["accuracy", "roc_auc", "f1"]
    x       = np.arange(len(metrics))
    width   = 1 / (len(all_results) + 2)
    colors2 = ["steelblue","tomato","seagreen",
               "darkorange","purple","crimson"]

    fig, ax = plt.subplots(figsize=(13, 6))
    for i, (name, res) in enumerate(all_results.items()):
        vals = [res[m] for m in metrics]
        bars = ax.bar(x + i*width, vals, width,
                      label=name, color=colors2[i], alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.002,
                    f"{val:.3f}", ha="center",
                    va="bottom", fontsize=6.5)

    # baseline line
    for j, (metric, bl_val) in enumerate(
            zip(metrics, [bl_acc, bl_auc, bl_f1])):
        ax.hlines(bl_val, j - 0.1,
                  j + len(all_results)*width + 0.1,
                  colors="black", linestyles="--",
                  linewidth=1.2)

    ax.set_xticks(x + width * len(all_results)/2)
    ax.set_xticklabels(["Accuracy","ROC-AUC","F1"], fontsize=12)
    ax.set_ylim(0.45, 0.80)
    ax.set_ylabel("Score")
    ax.set_title("All Models on OpenFE Features vs Baseline\n"
                 "(dashed line = baseline LightGBM)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{FIG_PATH}/all_models_comparison.png", dpi=150)
    plt.close()
    print(f"✅ Bar chart saved")

    # ── save results ──────────────────────────────────────────
    with open(f"{RES_PATH}/all_models_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    pd.DataFrame(all_results).T.to_csv(
        f"{RES_PATH}/all_models_results.csv")

    # ── final summary ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FINAL SUMMARY — ALL MODELS ON OPENFE FEATURES (TEST SET)")
    print("=" * 70)
    print(f"{'Model':<22} {'Acc':>8} {'AUC':>8} "
          f"{'F1':>8} {'CV-AUC':>12}")
    print("-" * 70)
    print(f"{'Baseline LGBM':<22} "
          f"{bl_acc:>8.4f} {bl_auc:>8.4f} "
          f"{bl_f1:>8.4f} {'—':>12}")
    print("-" * 70)
    for name, res in all_results.items():
        auc_mark = "✅" if res["roc_auc"] > bl_auc else "❌"
        f1_mark  = "✅" if res["f1"]      > bl_f1  else "❌"
        cv_str   = (f"{res['cv_auc_mean']:.4f}"
                    f"±{res['cv_auc_std']:.4f}"
                    if res["cv_auc_mean"] > 0 else "—")
        print(f"{name:<22} "
              f"{res['accuracy']:>8.4f} "
              f"{res['roc_auc']:>8.4f}{auc_mark} "
              f"{res['f1']:>8.4f}{f1_mark} "
              f"{cv_str:>12}")
    print("=" * 70)
    print("✅ ALL MODELS COMPLETE!")
    print("=" * 70)

if __name__ == "__main__":
    main()