# ============================================================
# src/uci/00_eda.py
# EDA for UCI Diabetes Dataset
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os, warnings
warnings.filterwarnings("ignore")

DATA_PATH = "data/diabetes_uci.csv"
FIG_PATH  = "outputs/uci/figures"
os.makedirs(FIG_PATH, exist_ok=True)

def main():
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df.replace("?", np.nan, inplace=True)

    # binary target for plotting
    df["readmitted_binary"] = (df["readmitted"] != "NO").astype(int)

    print("=" * 60)
    print("UCI DIABETES — EDA")
    print("=" * 60)
    print(f"Shape: {df.shape}")

    # ── 01: target distribution ───────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    vc = df["readmitted"].value_counts()
    colors = ["steelblue", "tomato", "seagreen"]
    bars = ax.bar(vc.index, vc.values, color=colors)
    ax.set_title("Target Distribution — Readmitted\n"
                 "UCI Diabetes Dataset", fontweight="bold")
    ax.set_xlabel("Readmitted")
    ax.set_ylabel("Count")
    for bar, val in zip(bars, vc.values):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 200,
                f"{val:,}\n({val/len(df)*100:.1f}%)",
                ha="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{FIG_PATH}/01_target_distribution.png", dpi=150)
    plt.close()
    print("✅ 01_target_distribution.png")

    # ── 02: numerical distributions ──────────────────────────
    num_cols = ["time_in_hospital", "num_lab_procedures",
                "num_procedures", "num_medications",
                "number_outpatient", "number_emergency",
                "number_inpatient", "number_diagnoses"]
    num_cols = [c for c in num_cols if c in df.columns]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    for i, col in enumerate(num_cols):
        axes[i].hist(df[col].dropna(), bins=30,
                     color="steelblue", edgecolor="white")
        axes[i].set_title(col)
        axes[i].set_xlabel("Value")
        axes[i].set_ylabel("Count")
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle("Numerical Feature Distributions — UCI Diabetes",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{FIG_PATH}/02_numerical_distributions.png",
                dpi=150)
    plt.close()
    print("✅ 02_numerical_distributions.png")

    # ── 03: categorical distributions ────────────────────────
    cat_cols = ["race", "gender", "age",
                "medical_specialty", "diag_1",
                "max_glu_serum", "A1Cresult",
                "change", "diabetesMed"]
    cat_cols = [c for c in cat_cols if c in df.columns]

    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    axes = axes.flatten()
    for i, col in enumerate(cat_cols):
        vc2 = df[col].value_counts().head(10)
        axes[i].bar(vc2.index.astype(str),
                    vc2.values, color="steelblue",
                    edgecolor="white")
        axes[i].set_title(col)
        axes[i].set_ylabel("Count")
        axes[i].tick_params(axis="x", rotation=30)
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle("Categorical Feature Distributions — UCI Diabetes",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{FIG_PATH}/03_categorical_distributions.png",
                dpi=150)
    plt.close()
    print("✅ 03_categorical_distributions.png")

    # ── 04: readmission rate by category ─────────────────────
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    axes = axes.flatten()
    for i, col in enumerate(cat_cols):
        rate = df.groupby(col)["readmitted_binary"].mean() * 100
        rate = rate.sort_values(ascending=False)
        axes[i].bar(rate.index.astype(str),
                    rate.values, color="tomato",
                    edgecolor="white")
        axes[i].set_title(f"Readmission Rate by {col}")
        axes[i].set_ylabel("Readmission %")
        axes[i].tick_params(axis="x", rotation=30)
        axes[i].axhline(df["readmitted_binary"].mean()*100,
                        color="black", linestyle="--",
                        linewidth=1, label="Overall avg")
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle("Readmission Rate by Categorical Features",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{FIG_PATH}/04_readmission_rates_by_category.png",
                dpi=150)
    plt.close()
    print("✅ 04_readmission_rates_by_category.png")

    # ── 05: correlation heatmap ───────────────────────────────
    num_df = df[num_cols + ["readmitted_binary"]].copy()
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(num_df.corr().round(2),
                annot=True, fmt=".2f",
                cmap="coolwarm", ax=ax,
                square=True, linewidths=0.5)
    ax.set_title("Correlation Heatmap\n"
                 "Numerical Features + Target",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{FIG_PATH}/05_correlation_heatmap.png", dpi=150)
    plt.close()
    print("✅ 05_correlation_heatmap.png")

    # ── 06: boxplots numerical vs readmitted ──────────────────
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    for i, col in enumerate(num_cols):
        df.boxplot(column=col,
                   by="readmitted_binary",
                   ax=axes[i],
                   boxprops=dict(color="steelblue"),
                   medianprops=dict(color="tomato",
                                    linewidth=2))
        axes[i].set_title(col)
        axes[i].set_xlabel("Readmitted (0=No, 1=Yes)")
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle("Numerical Features vs Readmitted — UCI Diabetes",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{FIG_PATH}/06_boxplots_vs_target.png", dpi=150)
    plt.close()
    print("✅ 06_boxplots_vs_target.png")

    # ── 07: drug usage heatmap ────────────────────────────────
    drug_cols = ["metformin","repaglinide","nateglinide",
                 "glimepiride","glipizide","glyburide",
                 "pioglitazone","rosiglitazone","insulin"]
    drug_cols = [c for c in drug_cols if c in df.columns]

    drug_map = {"No": 0, "Steady": 1, "Up": 2, "Down": 3}
    drug_df  = df[drug_cols].copy()
    for col in drug_cols:
        drug_df[col] = drug_df[col].map(drug_map).fillna(0)
    drug_df["readmitted"] = df["readmitted_binary"]

    rate_df = drug_df.groupby("readmitted")[drug_cols].mean()

    fig, ax = plt.subplots(figsize=(12, 4))
    sns.heatmap(rate_df, annot=True, fmt=".2f",
                cmap="YlOrRd", ax=ax, linewidths=0.5)
    ax.set_title("Average Drug Usage by Readmission Status\n"
                 "(0=No, 1=Steady, 2=Up, 3=Down)",
                 fontsize=12, fontweight="bold")
    ax.set_yticklabels(["Not Readmitted", "Readmitted"],
                       rotation=0)
    plt.tight_layout()
    plt.savefig(f"{FIG_PATH}/07_drug_usage_heatmap.png", dpi=150)
    plt.close()
    print("✅ 07_drug_usage_heatmap.png")

    # ── 08: age vs readmission ────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    age_rate = df.groupby("age")["readmitted_binary"].mean()*100
    age_order = ["[0-10)","[10-20)","[20-30)","[30-40)",
                 "[40-50)","[50-60)","[60-70)",
                 "[70-80)","[80-90)","[90-100)"]
    age_order = [a for a in age_order if a in age_rate.index]
    ax.bar(age_order, age_rate[age_order],
           color="steelblue", edgecolor="white")
    ax.axhline(df["readmitted_binary"].mean()*100,
               color="tomato", linestyle="--",
               linewidth=2, label="Overall avg")
    ax.set_title("Readmission Rate by Age Group",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Age Group")
    ax.set_ylabel("Readmission %")
    ax.legend()
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(f"{FIG_PATH}/08_age_vs_readmission.png", dpi=150)
    plt.close()
    print("✅ 08_age_vs_readmission.png")

    print("\n" + "=" * 60)
    print("✅ EDA COMPLETE — 8 plots saved to outputs/uci/figures/")
    print("=" * 60)

if __name__ == "__main__":
    main()