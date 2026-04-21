# ============================================================
# 06_final_evaluation.py  —  Final Comparison + Report
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json, os, warnings
warnings.filterwarnings("ignore")

from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
                             classification_report, roc_curve,
                             ConfusionMatrixDisplay)

RES_PATH = "outputs/results"
FIG_PATH = "outputs/figures"
os.makedirs(RES_PATH, exist_ok=True)
os.makedirs(FIG_PATH, exist_ok=True)

def main():
    # ── load all data ─────────────────────────────────────────
    X_train_raw = pd.read_csv(f"{RES_PATH}/X_train.csv")
    X_val_raw   = pd.read_csv(f"{RES_PATH}/X_val.csv")
    X_test_raw  = pd.read_csv(f"{RES_PATH}/X_test.csv")

    X_train_ofe = pd.read_csv(f"{RES_PATH}/X_train_openfe.csv")
    X_val_ofe   = pd.read_csv(f"{RES_PATH}/X_val_openfe.csv")
    X_test_ofe  = pd.read_csv(f"{RES_PATH}/X_test_openfe.csv")

    y_train = pd.read_csv(f"{RES_PATH}/y_train.csv").squeeze()
    y_val   = pd.read_csv(f"{RES_PATH}/y_val.csv").squeeze()
    y_test  = pd.read_csv(f"{RES_PATH}/y_test.csv").squeeze()

    # load selected features
    sel_feats = pd.read_csv(
        f"{RES_PATH}/selected_features.csv")["feature"].tolist()

    X_train_sel = X_train_ofe[sel_feats]
    X_val_sel   = X_val_ofe[sel_feats]
    X_test_sel  = X_test_ofe[sel_feats]

    print("=" * 60)
    print("FINAL EVALUATION — ALL CONFIGURATIONS")
    print("=" * 60)

    # ── model factory ─────────────────────────────────────────
    def make_lgbm():
        return LGBMClassifier(
            n_estimators=500, learning_rate=0.03,
            num_leaves=20, min_child_samples=50,
            reg_alpha=0.2, reg_lambda=0.2,
            subsample=0.8, colsample_bytree=0.7,
            random_state=42, verbose=-1)

    # ── train all configs ─────────────────────────────────────
    configs = {
        "Baseline (Raw)"        : (X_train_raw, X_val_raw, X_test_raw),
        "OpenFE (All)"          : (X_train_ofe, X_val_ofe, X_test_ofe),
        "OpenFE (Selected)"     : (X_train_sel, X_val_sel, X_test_sel),
    }

    all_results  = {}
    all_models   = {}
    all_probas   = {}

    for name, (Xtr, Xva, Xte) in configs.items():
        print(f"\n── Training: {name} ──")
        model = make_lgbm()
        model.fit(Xtr, y_train, eval_set=[(Xva, y_val)])

        preds = model.predict(Xte)
        proba = model.predict_proba(Xte)[:, 1]

        all_results[name] = {
            "accuracy" : round(accuracy_score(y_test, preds), 4),
            "roc_auc"  : round(roc_auc_score(y_test, proba),  4),
            "f1"       : round(f1_score(y_test, preds),       4),
            "n_features": Xtr.shape[1],
        }
        all_models[name]  = model
        all_probas[name]  = proba

        m = all_results[name]
        print(f"  Features={m['n_features']}  "
              f"Acc={m['accuracy']:.4f}  "
              f"AUC={m['roc_auc']:.4f}  "
              f"F1={m['f1']:.4f}")

    # ── plot 1: bar chart comparison ──────────────────────────
    metrics  = ["accuracy", "roc_auc", "f1"]
    labels   = list(all_results.keys())
    colors   = ["steelblue", "tomato", "seagreen"]
    x        = np.arange(len(metrics))
    width    = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (label, color) in enumerate(zip(labels, colors)):
        vals = [all_results[label][m] for m in metrics]
        bars = ax.bar(x + i*width, vals, width,
                      label=label, color=color, alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.002,
                    f"{val:.4f}", ha="center",
                    va="bottom", fontsize=7.5)

    ax.set_xticks(x + width)
    ax.set_xticklabels(["Accuracy", "ROC-AUC", "F1-Score"],
                        fontsize=12)
    ax.set_ylim(0.45, 0.75)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison\n"
                 "Baseline vs OpenFE vs OpenFE+Selection",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{FIG_PATH}/final_comparison_bar.png", dpi=150)
    plt.close()
    print(f"\n✅ Bar chart saved")

    # ── plot 2: ROC curves ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    for (name, proba), color in zip(all_probas.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc = all_results[name]["roc_auc"]
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{name} (AUC={auc:.4f})")
    ax.plot([0,1],[0,1], "k--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — All Configurations",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{FIG_PATH}/final_roc_curves.png", dpi=150)
    plt.close()
    print(f"✅ ROC curves saved")

    # ── plot 3: confusion matrices ────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (name, (_, _, Xte)) in zip(axes, configs.items()):
        model = all_models[name]
        ConfusionMatrixDisplay.from_predictions(
            y_test, model.predict(Xte),
            display_labels=["No", "Yes"],
            cmap="Blues", ax=ax)
        ax.set_title(name, fontsize=10, fontweight="bold")
    plt.suptitle("Confusion Matrices — Test Set",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{FIG_PATH}/final_confusion_matrices.png", dpi=150)
    plt.close()
    print(f"✅ Confusion matrices saved")

    # ── final summary table ───────────────────────────────────
    print("\n" + "=" * 65)
    print("FINAL RESULTS SUMMARY — TEST SET")
    print("=" * 65)
    print(f"{'Config':<22} {'Feats':>6} {'Acc':>8} "
          f"{'AUC':>8} {'F1':>8}")
    print("-" * 65)
    for name, res in all_results.items():
        print(f"{name:<22} {res['n_features']:>6} "
              f"{res['accuracy']:>8.4f} "
              f"{res['roc_auc']:>8.4f} "
              f"{res['f1']:>8.4f}")
    print("=" * 65)

    # ── save final results ────────────────────────────────────
    with open(f"{RES_PATH}/final_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # results as CSV too
    pd.DataFrame(all_results).T.to_csv(
        f"{RES_PATH}/final_results.csv")

    print(f"\n✅ Results saved → {RES_PATH}/final_results.json")
    print(f"✅ Results saved → {RES_PATH}/final_results.csv")
    print("\n" + "=" * 65)
    print("✅ PROJECT COMPLETE!")
    print("=" * 65)
    print("\nAll outputs saved:")
    print(f"  Figures  → {FIG_PATH}/")
    print(f"  Results  → {RES_PATH}/")

if __name__ == '__main__':
    main()