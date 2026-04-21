# ============================================================
# 02_preprocessing.py  —  Preprocessing + Train/Val/Test Split
# ============================================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os, pickle

DATA_PATH    = "data/hospital_readmissions.csv"
OUT_PATH     = "outputs/results"
os.makedirs(OUT_PATH, exist_ok=True)

# ── load ─────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
print("=" * 55)
print("PREPROCESSING")
print("=" * 55)
print(f"Raw shape: {df.shape}")

# ── encode target ─────────────────────────────────────────────
df["readmitted"] = (df["readmitted"] == "yes").astype(int)
print(f"Target encoded → yes=1, no=0")
print(f"Target distribution:\n{df['readmitted'].value_counts()}")

# ── encode categoricals ───────────────────────────────────────
cat_cols = ["age", "medical_specialty", "diag_1", "diag_2",
            "diag_3", "glucose_test", "A1Ctest", "change", "diabetes_med"]

encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le
    print(f"  Encoded {col}: {list(le.classes_)}")

# ── features and target ───────────────────────────────────────
X = df.drop(columns=["readmitted"])
y = df["readmitted"]

print(f"\nFeature matrix shape : {X.shape}")
print(f"Feature columns      : {list(X.columns)}")

# ── train / val / test split  (60 / 20 / 20) ─────────────────
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.40, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

print(f"\nSplit sizes:")
print(f"  Train : {X_train.shape}  |  readmitted={y_train.mean():.3f}")
print(f"  Val   : {X_val.shape}   |  readmitted={y_val.mean():.3f}")
print(f"  Test  : {X_test.shape}   |  readmitted={y_test.mean():.3f}")

# ── save splits ───────────────────────────────────────────────
X_train.to_csv(f"{OUT_PATH}/X_train.csv", index=False)
X_val.to_csv(f"{OUT_PATH}/X_val.csv",   index=False)
X_test.to_csv(f"{OUT_PATH}/X_test.csv",  index=False)
y_train.to_csv(f"{OUT_PATH}/y_train.csv", index=False)
y_val.to_csv(f"{OUT_PATH}/y_val.csv",   index=False)
y_test.to_csv(f"{OUT_PATH}/y_test.csv",  index=False)

# save encoders for later use
with open(f"{OUT_PATH}/encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)

print(f"\n✅ All splits saved to {OUT_PATH}/")
print(f"✅ Encoders saved to {OUT_PATH}/encoders.pkl")
print("=" * 55)
print("✅ PREPROCESSING COMPLETE")
print("=" * 55)