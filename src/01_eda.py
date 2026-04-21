# ============================================================
# 01_eda.py  —  Exploratory Data Analysis
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ── paths ────────────────────────────────────────────────────
DATA_PATH = "data/hospital_readmissions.csv"
FIG_PATH  = "outputs/figures"
os.makedirs(FIG_PATH, exist_ok=True)

# ── load ─────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
print("=" * 55)
print("DATASET OVERVIEW")
print("=" * 55)
print(f"Shape        : {df.shape}")
print(f"Features     : {df.shape[1] - 1}")
print(f"Target       : readmitted")
print(f"\nColumn names :\n{list(df.columns)}")
print(f"\nData types   :\n{df.dtypes}")
print(f"\nMissing values:\n{df.isnull().sum()}")

# ── target distribution ───────────────────────────────────────
print("\n" + "=" * 55)
print("TARGET DISTRIBUTION")
print("=" * 55)
vc = df["readmitted"].value_counts()
print(vc)
print(f"Class balance: {round(vc['no']/len(df)*100,1)}% No  |  "
      f"{round(vc['yes']/len(df)*100,1)}% Yes")

fig, ax = plt.subplots(figsize=(6, 4))
vc.plot(kind="bar", color=["steelblue", "tomato"], ax=ax)
ax.set_title("Target Distribution — Readmitted")
ax.set_xlabel("Readmitted")
ax.set_ylabel("Count")
ax.set_xticklabels(["No", "Yes"], rotation=0)
for p in ax.patches:
    ax.annotate(f"{int(p.get_height()):,}",
                (p.get_x() + p.get_width()/2, p.get_height()),
                ha="center", va="bottom", fontsize=11)
plt.tight_layout()
plt.savefig(f"{FIG_PATH}/01_target_distribution.png", dpi=150)
plt.close()
print(f"\n✅ Saved → {FIG_PATH}/01_target_distribution.png")

# ── numerical feature distributions ──────────────────────────
num_cols = df.select_dtypes(include=np.number).columns.tolist()
print(f"\nNumerical columns: {num_cols}")
print(df[num_cols].describe().round(2))

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()
for i, col in enumerate(num_cols):
    axes[i].hist(df[col], bins=30, color="steelblue", edgecolor="white")
    axes[i].set_title(col)
    axes[i].set_xlabel("Value")
    axes[i].set_ylabel("Count")
for j in range(i+1, len(axes)):
    axes[j].set_visible(False)
plt.suptitle("Numerical Feature Distributions", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIG_PATH}/02_numerical_distributions.png", dpi=150)
plt.close()
print(f"✅ Saved → {FIG_PATH}/02_numerical_distributions.png")

# ── categorical feature distributions ────────────────────────
cat_cols = ["age", "medical_specialty", "diag_1", "diag_2",
            "diag_3", "glucose_test", "A1Ctest", "change", "diabetes_med"]

fig, axes = plt.subplots(3, 3, figsize=(18, 12))
axes = axes.flatten()
for i, col in enumerate(cat_cols):
    vc2 = df[col].value_counts()
    axes[i].bar(vc2.index, vc2.values, color="steelblue", edgecolor="white")
    axes[i].set_title(col)
    axes[i].set_ylabel("Count")
    axes[i].tick_params(axis="x", rotation=30)
plt.suptitle("Categorical Feature Distributions", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIG_PATH}/03_categorical_distributions.png", dpi=150)
plt.close()
print(f"✅ Saved → {FIG_PATH}/03_categorical_distributions.png")

# ── readmission rate per category ────────────────────────────
fig, axes = plt.subplots(3, 3, figsize=(18, 12))
axes = axes.flatten()
for i, col in enumerate(cat_cols):
    rate = df.groupby(col)["readmitted"].apply(
        lambda x: (x == "yes").mean() * 100
    ).sort_values(ascending=False)
    axes[i].bar(rate.index, rate.values, color="tomato", edgecolor="white")
    axes[i].set_title(f"Readmission Rate by {col}")
    axes[i].set_ylabel("Readmission %")
    axes[i].tick_params(axis="x", rotation=30)
plt.suptitle("Readmission Rate by Categorical Features",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIG_PATH}/04_readmission_rates_by_category.png", dpi=150)
plt.close()
print(f"✅ Saved → {FIG_PATH}/04_readmission_rates_by_category.png")

# ── correlation heatmap (numerical only) ─────────────────────
df_corr = df[num_cols].copy()
df_corr["readmitted"] = (df["readmitted"] == "yes").astype(int)

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(df_corr.corr().round(2), annot=True, fmt=".2f",
            cmap="coolwarm", ax=ax, square=True, linewidths=0.5)
ax.set_title("Correlation Heatmap (Numerical Features + Target)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIG_PATH}/05_correlation_heatmap.png", dpi=150)
plt.close()
print(f"✅ Saved → {FIG_PATH}/05_correlation_heatmap.png")

# ── boxplots: numerical vs readmitted ────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()
for i, col in enumerate(num_cols):
    df.boxplot(column=col, by="readmitted", ax=axes[i],
               boxprops=dict(color="steelblue"),
               medianprops=dict(color="tomato", linewidth=2))
    axes[i].set_title(col)
    axes[i].set_xlabel("Readmitted")
for j in range(i+1, len(axes)):
    axes[j].set_visible(False)
plt.suptitle("Numerical Features vs Readmitted", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIG_PATH}/06_boxplots_vs_target.png", dpi=150)
plt.close()
print(f"✅ Saved → {FIG_PATH}/06_boxplots_vs_target.png")

print("\n" + "=" * 55)
print("✅ EDA COMPLETE — all plots saved to outputs/figures/")
print("=" * 55)