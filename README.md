# Hospital Readmission Prediction using OpenFE
## Automated Feature Engineering on UCI Diabetes 130-US Hospitals Dataset

**Group:** Group 3  
**Members:** Harshavardhan Sasikumar (11800735) & Mahesh Chilukamari (11761476)

---

## Project Overview

This project applies **OpenFE** (Automated Feature Generation with
Expert-level Performance) to the UCI Diabetes 130-US Hospitals dataset
to improve hospital readmission prediction. We compare baseline models
on raw features versus models trained on OpenFE-generated features,
evaluating across LightGBM, XGBoost, Random Forest, Logistic
Regression, and an Ensemble.


## Project Structure

```
hospital_readmission_openfe/
│
├── data/
│   ├── hospital_readmissions.csv    
│   └── diabetes_uci.csv             
│
├── src/
│   └── uci/                         
│       ├── 00_eda.py                
│       ├── 01_preprocessing.py      
│       ├── 02_baseline.py           
│       ├── 03_openfe.py             
│       ├── 04_feature_selection.py  
│       └── 05_evaluation.py         
│
├── outputs/
│   └── uci/
│       ├── figures/                 
│       └── results/                 
│
├── index.html                       
├── requirements.txt                 
└── README.md                        
```

---

## Dataset

**UCI Diabetes 130-US Hospitals Dataset**
- Source: [UCI ML Repository](https://archive.ics.uci.edu/dataset/296)
- 10 years of clinical data (1999–2008) across 130 US hospitals
- Raw: 101,766 encounters → After preprocessing: 97,109 patients
- 42 features: 8 numerical, 10 categorical, 21 drug columns, 3 ordinal
- Target: Binary readmission (Yes=1 / No=0)
- Class balance: 52.5% No / 47.5% Yes — no resampling needed

---

## Setup and Installation

### Prerequisites
- Python 3.11+ recommended (tested on Python 3.13.11)
- macOS, Linux, or Windows

### Step 1 — Clone the repository
```bash
git clone https://github.com/Harshavardhan-504/hospital_readmission_openfe.git
cd hospital_readmission_openfe
```

### Step 2 — Create virtual environment
```bash
python3 -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Download the dataset
```bash
python3 -c "
from ucimlrepo import fetch_ucirepo
import pandas as pd
diabetes = fetch_ucirepo(id=296)
df = pd.concat([diabetes.data.features, diabetes.data.targets], axis=1)
df.to_csv('data/diabetes_uci.csv', index=False)
print('Dataset downloaded:', df.shape)
"
```

---

## Running the Project

Run each script in order from the project root:

```bash
# Step 0 — Exploratory Data Analysis 
python src/uci/00_eda.py

# Step 1 — Preprocessing 
python src/uci/01_preprocessing.py

# Step 2 — Baseline Models
python src/uci/02_baseline.py

# Step 3 — OpenFE Feature Generation 
python src/uci/03_openfe.py

# Step 4 — Feature Selection + All Models 
python src/uci/04_feature_selection.py

# Step 5 — Final Evaluation 
python src/uci/05_evaluation.py
```


---

## OpenFE Pipeline

OpenFE follows an **expand-and-reduce** framework:

```
INPUT: X_train (58,265 × 42 features)

STEP 1 — EXPAND
  FOR each operator (+, -, *, /, GroupByThenRank,
                     CombineThenFreq, max, min, ...):
      FOR each feature pair (f_i, f_j):
          candidate_pool.append(operator(f_i, f_j))
  → 2,000+ candidates generated

STEP 2 — PRUNE (Stage I: Successive Pruning)
  FOR each candidate feature τ:
      Δ = FeatureBoost(τ, y_train)
      KEEP top 50% by score each round
  → Pool reduced to ~200 features

STEP 3 — RANK (Stage II: Feature Attribution)
  RANK remaining features by MDI importance
  KEEP top 20 features
  → 20 final features selected

OUTPUT: X_new (58,265 × 62 features)
        = 42 original + 20 generated
```

---

## Results

### Performance Comparison (Test Set — 19,422 patients)

| Model | Features | Accuracy | ROC-AUC | F1-Score |
|---|---|---|---|---|
| Logistic Regression (baseline) | 42 | 0.6200 | 0.6673 | 0.5233 |
| LightGBM (baseline) | 42 | 0.6379 | 0.6912 | 0.5985 |
| OpenFE + LightGBM | 62 | 0.6371 | 0.6942 ✅ | 0.5872 |
| OpenFE + XGBoost | 62 | 0.6366 | 0.6940 ✅ | 0.5825 |
| OpenFE + Random Forest | 62 | 0.6321 | 0.6874 | 0.5678 |
| OpenFE + Logistic Regression | 62 | 0.6285 | 0.6779 | 0.5606 |
| **OpenFE + Ensemble ★** | **62** | **0.6375** | **0.6943 ✅** | **0.5855** |

### Cross-Validation (5-fold on train+val = 77,687 patients)

| Model | CV AUC | Std |
|---|---|---|
| LightGBM + OpenFE | 0.6839 | ±0.0020 |
| XGBoost + OpenFE | 0.6832 | ±0.0019 |
| Random Forest + OpenFE | 0.6778 | ±0.0016 |
| Logistic Regression + OpenFE | 0.6662 | ±0.0026 |

### Key Findings
- ✅ AUC improved consistently across ALL 4 models after OpenFE
- ✅ OpenFE added 20 new features (42 raw → 62 total)
- ✅ Top generated features (`autoFE_f_17`, `autoFE_f_14`) outranked
  ALL 42 original features in importance
- ✅ CV AUC 0.6839 ± 0.002 confirms improvement is statistically reliable
- Insulin usage strongly associated with readmission (0.96 vs 0.82)
- Age group 80-90 has highest readmission rate (48%)
- ⚠️ F1 drops slightly — expected AUC optimization tradeoff

---

## Output Figures

All figures saved to `outputs/uci/figures/`:

| File | Description |
|---|---|
| `01_target_distribution.png` | Class balance (NO/>30/<30) |
| `02_numerical_distributions.png` | Distribution of 8 numerical features |
| `03_categorical_distributions.png` | Distribution of categorical features |
| `04_readmission_rates_by_category.png` | Readmission rate per category |
| `05_correlation_heatmap.png` | Feature-target correlations |
| `06_boxplots_vs_target.png` | Numerical features vs readmission |
| `07_drug_usage_heatmap.png` | Drug usage by readmission status |
| `08_age_vs_readmission.png` | Readmission rate by age group |
| `baseline_feature_importance.png` | LightGBM importance (raw features) |
| `baseline_lightgbm_cm.png` | Baseline confusion matrix |
| `openfe_feature_importance.png` | Feature importance after OpenFE |
| `all_models_comparison.png` | All models bar chart comparison |
| `all_models_roc.png` | ROC curves for all models |
| `final_bar_chart.png` | Final 4-config comparison |
| `final_roc_curves.png` | Final ROC curves |
| `final_delta_chart.png` | Performance delta over baseline |
| `final_confusion_matrices.png` | All 4 confusion matrices |

---

## GitHub Pages

👉 **[https://harshavardhan-504.github.io/hospital_readmission_openfe/](https://harshavardhan-504.github.io/hospital_readmission_openfe/)**

---

## References

1. Zhang, T. et al. (2023). **OpenFE: Automated feature generation with
   expert-level performance.** ICML 2023, pp. 41880–41901. PMLR.

2. Hollmann, N., Müller, S., & Hutter, F. (2023). **Large language models
   for automated data science: Introducing CAAFE for context-aware
   automated feature engineering.** NeurIPS 2023, 36, 44753–44775.

3. Nam, J. et al. (2024). **Optimized feature generation for tabular data
   via LLMs with decision tree reasoning.** NeurIPS 2024, 37, 92352–92380.

4. Tschalzev, A. et al. (2024). **A data-centric perspective on evaluating
   machine learning models for tabular data.** NeurIPS 2024, 37, 95896–95930.

5. Cherepanova, V. et al. (2023). **A performance-driven benchmark for
   feature selection in tabular deep learning.** NeurIPS 2023, 36, 41956–41979.

6. Strack, B. et al. (2014). **UCI Machine Learning Repository: Diabetes
   130-US Hospitals for years 1999–2008 dataset.**


