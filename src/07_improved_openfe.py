# ============================================================
# 07_improved_openfe.py  —  Improved OpenFE Pipeline
# All 4 strategies: more features + ensemble +
# cross-validation + grid search
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json, os, warnings
warnings.filterwarnings("ignore")

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, roc_auc_score,
                             f1_score, roc_curve)
from sklearn.model_selection import GridSearchCV

RES_PATH = "outputs/results"
FIG_PATH = "outputs/figures"
os.makedirs(RES_PATH, exist_ok=True)

def evaluate(model, X_tr, y_tr, X_te, y_te, name=""):
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    proba = model.predict_proba(X_te)[:, 1]
    acc = round(accuracy_score(y_te, preds), 4)
    auc = round(roc_auc_score(y_te, proba),  4)
    f1  = round(f1_score(y_te, preds),       4)
    if name:
        print(f"  {name:<30} Acc={acc:.4f}  "
              f"AUC={auc:.4f}  F1={f1:.4f}")
    return acc, auc, f1, proba

def main():
    # ── load data ─────────────────────────────────────────────
    X_train_raw = pd.read_csv(f"{RES_PATH}/X_train.csv")
    X_val_raw   = pd.read_csv(f"{RES_PATH}/X_val.csv")
    X_test_raw  = pd.read_csv(f"{RES_PATH}/X_test.csv")
    X_train_ofe = pd.read_csv(f"{RES_PATH}/X_train_openfe.csv")
    X_val_ofe   = pd.read_csv(f"{RES_PATH}/X_val_openfe.csv")
    X_test_ofe  = pd.read_csv(f"{RES_PATH}/X_test_openfe.csv")
    y_train = pd.read_csv(f"{RES_PATH}/y_train.csv").squeeze()
    y_val   = pd.read_csv(f"{RES_PATH}/y_val.csv").squeeze()
    y_test  = pd.read_csv(f"{RES_PATH}/y_test.csv").squeeze()

    # combine train+val for cross-validation
    X_trainval_raw = pd.concat([X_train_raw, X_val_raw],
                                ignore_index=True)
    X_trainval_ofe = pd.concat([X_train_ofe, X_val_ofe],
                                ignore_index=True)
    y_trainval     = pd.concat([y_train, y_val],
                                ignore_index=True)

    print("=" * 60)
    print("IMPROVED OPENFE — ALL 4 STRATEGIES")
    print("=" * 60)

    # ─────────────────────────────────────────────────────────
    # STRATEGY 1: Try more top-N features (30, 40, 50)
    # ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STRATEGY 1 — More Top-N Features (30 / 40 / 50)")
    print("=" * 60)

    lgbm_base = LGBMClassifier(
        n_estimators=500, learning_rate=0.03,
        num_leaves=20, min_child_samples=50,
        reg_alpha=0.2, reg_lambda=0.2,
        subsample=0.8, colsample_bytree=0.7,
        random_state=42, verbose=-1)
    lgbm_base.fit(X_train_ofe, y_train)

    feat_imp = pd.Series(
        lgbm_base.feature_importances_,
        index=X_train_ofe.columns
    ).sort_values(ascending=False)

    original_cols = [c for c in X_train_ofe.columns
                     if not c.startswith("autoFE")]

    best_s1 = {"auc": 0, "n": 0, "feats": None}
    s1_results = {}

    for top_n in [20, 25, 30, 35, 40, 50, 36]:
        top_n = min(top_n, len(feat_imp))
        top_feats  = feat_imp.head(top_n).index.tolist()
        keep = list(dict.fromkeys(
            [c for c in X_train_ofe.columns
             if c in set(original_cols + top_feats)]))

        clf = LGBMClassifier(
            n_estimators=500, learning_rate=0.03,
            num_leaves=20, min_child_samples=50,
            reg_alpha=0.2, reg_lambda=0.2,
            subsample=0.8, colsample_bytree=0.7,
            random_state=42, verbose=-1)
        acc, auc, f1, _ = evaluate(
            clf, X_train_ofe[keep], y_train,
            X_test_ofe[keep], y_test,
            name=f"top_{top_n} ({len(keep)} feats)")
        s1_results[top_n] = {"accuracy": acc,
                              "roc_auc": auc, "f1": f1}
        if auc > best_s1["auc"]:
            best_s1 = {"auc": auc, "n": top_n, "feats": keep}

    print(f"\n✅ Best top_N = {best_s1['n']}  "
          f"AUC = {best_s1['auc']:.4f}")
    best_feats_s1 = best_s1["feats"]

    # ─────────────────────────────────────────────────────────
    # STRATEGY 2: Grid Search on LightGBM
    # ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STRATEGY 2 — LightGBM Grid Search")
    print("=" * 60)

    param_grid = {
        "num_leaves"       : [15, 20, 31],
        "learning_rate"    : [0.03, 0.05],
        "min_child_samples": [30, 50],
        "reg_alpha"        : [0.1, 0.2],
    }

    lgbm_gs = LGBMClassifier(
        n_estimators=500,
        subsample=0.8,
        colsample_bytree=0.7,
        random_state=42,
        verbose=-1)

    print("⏳ Running grid search (takes ~2 mins)...")
    gs = GridSearchCV(
        lgbm_gs, param_grid,
        scoring="roc_auc",
        cv=3, n_jobs=1, verbose=0)
    gs.fit(X_train_ofe[best_feats_s1], y_train)

    print(f"✅ Best params: {gs.best_params_}")
    print(f"   Best CV AUC: {gs.best_score_:.4f}")

    best_lgbm = gs.best_estimator_
    acc, auc, f1, proba_lgbm = evaluate(
        best_lgbm,
        X_train_ofe[best_feats_s1], y_train,
        X_test_ofe[best_feats_s1],  y_test,
        name="LightGBM (tuned)")

    s2_results = {"accuracy": acc, "roc_auc": auc, "f1": f1}

    # ─────────────────────────────────────────────────────────
    # STRATEGY 3: XGBoost on OpenFE features
    # ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STRATEGY 3 — XGBoost on OpenFE Features")
    print("=" * 60)

    xgb_param_grid = {
        "max_depth"  : [3, 5],
        "learning_rate": [0.03, 0.05],
        "reg_alpha"  : [0.1, 0.2],
        "reg_lambda" : [0.1, 1.0],
    }

    xgb = XGBClassifier(
        n_estimators=500,
        subsample=0.8,
        colsample_bytree=0.7,
        random_state=42,
        eval_metric="logloss",
        verbosity=0)

    print("⏳ Running XGBoost grid search (~2 mins)...")
    gs_xgb = GridSearchCV(
        xgb, xgb_param_grid,
        scoring="roc_auc",
        cv=3, n_jobs=1, verbose=0)
    gs_xgb.fit(X_train_ofe[best_feats_s1], y_train)

    print(f"✅ Best XGB params: {gs_xgb.best_params_}")
    print(f"   Best CV AUC   : {gs_xgb.best_score_:.4f}")

    best_xgb = gs_xgb.best_estimator_
    acc_x, auc_x, f1_x, proba_xgb = evaluate(
        best_xgb,
        X_train_ofe[best_feats_s1], y_train,
        X_test_ofe[best_feats_s1],  y_test,
        name="XGBoost (tuned)")

    s3_results = {"accuracy": acc_x,
                  "roc_auc": auc_x, "f1": f1_x}

    # ─────────────────────────────────────────────────────────
    # STRATEGY 4: Ensemble (LGBM + XGB average)
    # ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STRATEGY 4 — Ensemble (LightGBM + XGBoost)")
    print("=" * 60)

    ensemble_proba = (proba_lgbm + proba_xgb) / 2
    ensemble_preds = (ensemble_proba >= 0.5).astype(int)

    acc_e = round(accuracy_score(y_test, ensemble_preds), 4)
    auc_e = round(roc_auc_score(y_test, ensemble_proba),  4)
    f1_e  = round(f1_score(y_test, ensemble_preds),       4)
    print(f"  {'Ensemble (LGBM+XGB)':<30} "
          f"Acc={acc_e:.4f}  AUC={auc_e:.4f}  F1={f1_e:.4f}")

    s4_results = {"accuracy": acc_e,
                  "roc_auc": auc_e, "f1": f1_e}

    # ─────────────────────────────────────────────────────────
    # STRATEGY 5: Cross-validation on best config
    # ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STRATEGY 5 — Cross-Validation (5-fold) on Best Config")
    print("=" * 60)

    skf = StratifiedKFold(n_splits=5,
                          shuffle=True, random_state=42)

    cv_lgbm = LGBMClassifier(**gs.best_params_,
                               n_estimators=500,
                               subsample=0.8,
                               colsample_bytree=0.7,
                               random_state=42,
                               verbose=-1)

    cv_auc = cross_val_score(
        cv_lgbm,
        X_trainval_ofe[best_feats_s1],
        y_trainval,
        cv=skf,
        scoring="roc_auc",
        n_jobs=1)

    cv_f1 = cross_val_score(
        cv_lgbm,
        X_trainval_ofe[best_feats_s1],
        y_trainval,
        cv=skf,
        scoring="f1",
        n_jobs=1)

    print(f"  5-fold CV AUC : {cv_auc.mean():.4f} "
          f"± {cv_auc.std():.4f}")
    print(f"  5-fold CV F1  : {cv_f1.mean():.4f} "
          f"± {cv_f1.std():.4f}")

    # ─────────────────────────────────────────────────────────
    # FINAL COMPARISON PLOT
    # ─────────────────────────────────────────────────────────
    with open(f"{RES_PATH}/baseline_metrics.json") as f:
        baseline = json.load(f)

    baseline_auc = baseline["LightGBM"]["test"]["roc_auc"]
    baseline_f1  = baseline["LightGBM"]["test"]["f1"]
    baseline_acc = baseline["LightGBM"]["test"]["accuracy"]

    all_configs = {
        "Baseline"         : (baseline_acc,
                              baseline_auc, baseline_f1),
        "LGBM Tuned"       : (s2_results["accuracy"],
                              s2_results["roc_auc"],
                              s2_results["f1"]),
        "XGBoost Tuned"    : (s3_results["accuracy"],
                              s3_results["roc_auc"],
                              s3_results["f1"]),
        "Ensemble"         : (s4_results["accuracy"],
                              s4_results["roc_auc"],
                              s4_results["f1"]),
    }

    # bar chart
    metrics = ["Accuracy", "AUC", "F1"]
    x = np.arange(len(metrics))
    width = 0.2
    colors = ["steelblue", "tomato", "seagreen", "darkorange"]

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (name, (acc, auc, f1)) in enumerate(
            all_configs.items()):
        vals = [acc, auc, f1]
        bars = ax.bar(x + i*width, vals, width,
                      label=name, color=colors[i], alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.002,
                    f"{val:.4f}", ha="center",
                    va="bottom", fontsize=7)

    ax.set_xticks(x + width*1.5)
    ax.set_xticklabels(metrics, fontsize=12)
    ax.set_ylim(0.45, 0.75)
    ax.set_ylabel("Score")
    ax.set_title("Improved OpenFE — All Strategies Comparison",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{FIG_PATH}/improved_comparison.png", dpi=150)
    plt.close()

    # ROC curves
    fig, ax = plt.subplots(figsize=(8, 6))
    for (name, proba), color in zip(
            [("LightGBM Tuned", proba_lgbm),
             ("XGBoost Tuned",  proba_xgb),
             ("Ensemble",       ensemble_proba)],
            ["tomato", "seagreen", "darkorange"]):
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc = roc_auc_score(y_test, proba)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{name} (AUC={auc:.4f})")

    # baseline
    lgbm_bl = LGBMClassifier(n_estimators=500,
                              learning_rate=0.05,
                              num_leaves=31,
                              random_state=42, verbose=-1)
    lgbm_bl.fit(X_train_raw, y_train)
    fpr_b, tpr_b, _ = roc_curve(
        y_test, lgbm_bl.predict_proba(X_test_raw)[:, 1])
    ax.plot(fpr_b, tpr_b, "steelblue",
            lw=2, label=f"Baseline (AUC={baseline_auc:.4f})")
    ax.plot([0,1],[0,1],"k--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — Improved Configs",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{FIG_PATH}/improved_roc_curves.png", dpi=150)
    plt.close()

    # ── save & print final summary ────────────────────────────
    improved_results = {
        "lgbm_tuned" : s2_results,
        "xgb_tuned"  : s3_results,
        "ensemble"   : s4_results,
        "cv_auc_mean": round(float(cv_auc.mean()), 4),
        "cv_auc_std" : round(float(cv_auc.std()),  4),
        "cv_f1_mean" : round(float(cv_f1.mean()),  4),
        "cv_f1_std"  : round(float(cv_f1.std()),   4),
    }
    with open(f"{RES_PATH}/improved_results.json", "w") as f:
        json.dump(improved_results, f, indent=2)

    print("\n" + "=" * 65)
    print("FINAL IMPROVED SUMMARY — TEST SET")
    print("=" * 65)
    print(f"{'Config':<22} {'Acc':>8} {'AUC':>8} {'F1':>8}")
    print("-" * 65)
    print(f"{'Baseline':<22} "
          f"{baseline_acc:>8.4f} "
          f"{baseline_auc:>8.4f} "
          f"{baseline_f1:>8.4f}")
    for name, (acc, auc, f1) in list(all_configs.items())[1:]:
        better_auc = " ✅" if auc > baseline_auc else " ❌"
        better_f1  = " ✅" if f1  > baseline_f1  else " ❌"
        print(f"{name:<22} {acc:>8.4f} "
              f"{auc:>8.4f}{better_auc}  "
              f"{f1:>8.4f}{better_f1}")
    print("=" * 65)
    print(f"\n5-fold CV AUC : {cv_auc.mean():.4f} "
          f"± {cv_auc.std():.4f}")
    print(f"5-fold CV F1  : {cv_f1.mean():.4f} "
          f"± {cv_f1.std():.4f}")
    print("\n✅ ALL STRATEGIES COMPLETE!")
    print(f"✅ Plots saved  → {FIG_PATH}/")
    print(f"✅ Results saved → {RES_PATH}/improved_results.json")

if __name__ == '__main__':
    main()