# ============================================================
# src/uci/01_preprocessing.py
# UCI Diabetes 130-US Hospitals Dataset Preprocessing
# ============================================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os, pickle, warnings
warnings.filterwarnings("ignore")

DATA_PATH = "data/diabetes_uci.csv"
OUT_PATH  = "outputs/uci/results"
os.makedirs(OUT_PATH, exist_ok=True)

def categorize_diag(code):
    try:
        code = str(code)
        if code.startswith("V") or code.startswith("E"):
            return "Other"
        c = float(code)
        if 390 <= c <= 459 or c == 785: return "Circulatory"
        if 460 <= c <= 519 or c == 786: return "Respiratory"
        if 520 <= c <= 579 or c == 787: return "Digestive"
        if c == 250:                     return "Diabetes"
        if 800 <= c <= 999:              return "Injury"
        if 710 <= c <= 739:              return "Musculoskeletal"
        if 580 <= c <= 629 or c == 788: return "Genitourinary"
        if 140 <= c <= 239:              return "Neoplasms"
        return "Other"
    except:
        return "Other"

def main():
    # ── load ──────────────────────────────────────────────────
    df = pd.read_csv(DATA_PATH, low_memory=False)
    print("=" * 60)
    print("UCI DIABETES — PREPROCESSING")
    print("=" * 60)
    print(f"Raw shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # ── drop useless columns ──────────────────────────────────
    drop_cols = ["weight", "examide", "citoglipton",
                 "payer_code", "medical_specialty"]
    drop_cols = [c for c in drop_cols if c in df.columns]
    df.drop(columns=drop_cols, inplace=True)
    print(f"\nAfter dropping low-info cols : {df.shape}")

    # ── replace ? with NaN ────────────────────────────────────
    df.replace("?", np.nan, inplace=True)

    # ── drop missing race ─────────────────────────────────────
    df.dropna(subset=["race"], inplace=True)
    print(f"After dropping missing race  : {df.shape}")

    # ── remove deceased / hospice ─────────────────────────────
    remove_ids = [11, 13, 14, 19, 20, 21]
    df = df[~df["discharge_disposition_id"].isin(remove_ids)]
    print(f"After removing deceased      : {df.shape}")

    # ── one encounter per patient ─────────────────────────────
    df = df.sort_values("time_in_hospital", ascending=False)
    if "patient_nbr" in df.columns:
        df = df.drop_duplicates(
            subset=["patient_nbr"], keep="first")
        print(f"After dedup patients         : {df.shape}")
    else:
        print("No patient_nbr — skipping dedup")
        print(f"Shape                        : {df.shape}")

    # ── drop ID columns if present ────────────────────────────
    id_cols = ["encounter_id", "patient_nbr"]
    id_cols = [c for c in id_cols if c in df.columns]
    if id_cols:
        df.drop(columns=id_cols, inplace=True)
        print(f"Dropped ID cols: {id_cols}")

    # ── binary target ─────────────────────────────────────────
    df["readmitted"] = (df["readmitted"] != "NO").astype(int)
    vc = df["readmitted"].value_counts()
    print(f"\nTarget distribution:\n{vc}")
    print(f"Readmission rate: {df['readmitted'].mean()*100:.1f}%")

    # ── categorize diag codes ─────────────────────────────────
    for col in ["diag_1", "diag_2", "diag_3"]:
        if col in df.columns:
            df[col] = df[col].apply(categorize_diag)
            print(f"  {col}: {sorted(df[col].unique())}")

    # ── encode drug columns ───────────────────────────────────
    drug_map  = {"No": 0, "Steady": 1, "Up": 2, "Down": 3}
    drug_cols = [
        "metformin","repaglinide","nateglinide","chlorpropamide",
        "glimepiride","acetohexamide","glipizide","glyburide",
        "tolbutamide","pioglitazone","rosiglitazone","acarbose",
        "miglitol","troglitazone","tolazamide","insulin",
        "glyburide-metformin","glipizide-metformin",
        "glimepiride-pioglitazone","metformin-rosiglitazone",
        "metformin-pioglitazone"
    ]
    for col in drug_cols:
        if col in df.columns:
            df[col] = df[col].map(drug_map).fillna(0).astype(int)
    print(f"\nDrug columns encoded")

    # ── encode remaining categoricals ─────────────────────────
    cat_cols = df.select_dtypes(
        include=["object", "str"]).columns.tolist()
    cat_cols = [c for c in cat_cols if c != "readmitted"]
    print(f"Encoding categoricals: {cat_cols}")

    encoders = {}
    for col in cat_cols:
        df[col] = df[col].fillna("Missing")
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    print(f"\nFinal shape : {df.shape}")
    print(f"Features    : {df.shape[1] - 1}")

    # ── train / val / test  60/20/20 ─────────────────────────
    X = df.drop(columns=["readmitted"])
    y = df["readmitted"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.40, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50,
        random_state=42, stratify=y_temp)

    print(f"\nSplit sizes:")
    print(f"  Train : {X_train.shape} | "
          f"readmitted={y_train.mean():.3f}")
    print(f"  Val   : {X_val.shape}  | "
          f"readmitted={y_val.mean():.3f}")
    print(f"  Test  : {X_test.shape}  | "
          f"readmitted={y_test.mean():.3f}")

    # ── save ──────────────────────────────────────────────────
    X_train.to_csv(f"{OUT_PATH}/X_train.csv", index=False)
    X_val.to_csv(f"{OUT_PATH}/X_val.csv",     index=False)
    X_test.to_csv(f"{OUT_PATH}/X_test.csv",   index=False)
    y_train.to_csv(f"{OUT_PATH}/y_train.csv", index=False)
    y_val.to_csv(f"{OUT_PATH}/y_val.csv",     index=False)
    y_test.to_csv(f"{OUT_PATH}/y_test.csv",   index=False)

    with open(f"{OUT_PATH}/encoders.pkl", "wb") as f:
        pickle.dump(encoders, f)

    print(f"\n✅ All splits saved → {OUT_PATH}/")
    print("=" * 60)
    print("✅ PREPROCESSING COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()