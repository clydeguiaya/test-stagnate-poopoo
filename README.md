# Philippine Labour Force Survey — Employee Stagnation Prediction

Predicts employee stagnation from the PSA Labour Force Survey (LFS) 2021–2024 using a rule-based target construction and three classification models: Logistic Regression, K-Nearest Neighbors, and Linear SVM.

---

## Problem

Employee stagnation — being stuck in a low-quality job with no path to upward mobility — is not directly measured in any Philippine survey. This project operationalises it from three PSA-defined labour market indicators, then builds a predictive model using only demographic and structural features available before employment quality is observed.

---

## Dataset

**Source:** Philippine Statistics Authority (PSA) — Labour Force Survey Public Use Files (PUF), Jan 2021 – Dec 2024  
**Files:** 48 monthly CSVs, ~30,000–60,000 respondents per file — 6,554,055 total rows before filtering  
**Schemas:** Three questionnaire versions across the 4-year span (Aug–Dec 2021 used a redesigned form)  
**Employed working-age subset:** 2,740,496 rows (emp_status == 1, age ≥ 15); 8.1% stagnation rate

**GDP covariate:** Our World in Data — "GDP per person employed (constant 2021 PPP $)" — Philippines rows for 2021–2024

```
data/
├── PHL-PSA-LFS-2021-2024-PUF/   # 48 monthly LFS CSVs + data dictionaries
│   ├── PHL-PSA-LFS-2021-PUF/
│   ├── PHL-PSA-LFS-2022-PUF/
│   ├── PHL-PSA-LFS-2023-PUF/
│   └── PHL-PSA-LFS-2024-PUF/
└── gdp-per-person-employed-constant-ppp/
    └── gdp-per-person-employed-constant-ppp.csv
```

---

## Target Variable: `is_stagnant`

An **employed person (age ≥ 15)** is labelled `is_stagnant = 1` if **2 or more** of the following criteria are met simultaneously:

| Criterion | PSA Variable | Condition |
|-----------|-------------|-----------|
| C1 — Visible underemployment | `PUFC20_PWMORE` | = 1 (wants additional work/hours) |
| C2 — Precarious employment | `PUFC17_NATEM`, `PUFC23_PCLASS` | temporary/casual job OR unpaid family worker |
| C3 — Education-occupation mismatch | `PUFC07_GRADE`, `PUFC14_PROCC` | College+ education AND elementary occupation (PSOC 9) |

The 2-of-3 composite rule reduces noise from each individual indicator. See [arguments.md](arguments.md) for full justification.

**Leakage prevention:** All criterion variables are dropped from the feature matrix — C1 (`want_more_work`), C2 (`nature_employment`, `worker_class`), and C3 (`education_level`, `occupation_major`). Keeping C3 variables would allow the model to reconstruct the mismatch rule directly.

---

## Pipeline

### `notebooks/data-pipeline.ipynb`

1. **Load & harmonise** — detects schema version per file, renames to a unified canonical column set, concatenates all 48 months
2. **Filter** — employed persons only (`emp_status == 1`, age ≥ 15)
3. **Construct target** — applies 2-of-3 stagnation rule; derives `education_level` and `occupation_major` for C3
4. **Merge GDP** — year-level join on `survey_year`
5. **Feature engineering** — industry sector (1–10 binned), cyclical month encoding (sin/cos)
6. **Drop criterion columns** — removes all C1/C2/C3 inputs (`want_more_work`, `nature_employment`, `worker_class`, `education_level`, `occupation_major`, `education_grade`, `occupation_code`) so no downstream model can access them
7. **Save** — `data/employed_processed.csv`

Each model notebook (log_reg and svm) follows the same six-step structure independently: select features → train/test split (2021–2023 / 2024) → impute from training set → scale → train model → evaluate with classification report, confusion matrix, and AUC-PR curve.

**Train set:** 2,060,789 rows (8.45% stagnation rate)  
**Test set:** 679,707 rows (6.95% stagnation rate)

### `notebooks/log_reg.ipynb`

1. **Pearson correlation** — linear association between each feature and `is_stagnant`
2. **Impute + scale** — mode for categorical, median for continuous; `StandardScaler` fit on train only
3. **Train** — `LogisticRegression(C=1.0, class_weight='balanced', solver='saga', max_iter=500)`
4. **Classification report** — precision, recall, F1 at default threshold (0.5)
5. **PR curve + AUC-PR** — primary metric; uses `predict_proba`
6. **Coefficients** — standardised log-odds coefficients sorted by magnitude

### `notebooks/svm.ipynb`

1. **Pearson correlation** — linear association between each feature and `is_stagnant`
2. **Impute + scale** — same strategy as log_reg
3. **Train** — `LinearSVC(C=0.1, class_weight='balanced', max_iter=2000)`
4. **Classification report + confusion matrix**
5. **PR curve + AUC-PR** — uses `decision_function` (signed distance from hyperplane) as ranking score since `LinearSVC` does not produce probabilities
6. **Coefficients** — margin coefficients sorted by magnitude

---

## Feature Matrix (12 features)

| Feature | Type | Description |
|---------|------|-------------|
| `age` | Continuous | Respondent age |
| `sex` | Categorical | 1=male, 2=female |
| `marital_status` | Categorical | 1=single … 5=common-law |
| `region` | Categorical | PSA region code (1–17) |
| `urban_rural` | Binary | 1=urban, 2=rural |
| `hh_size` | Continuous | Household size |
| `industry_sector` | Categorical | Binned PSIC sector (1–10) |
| `normal_hours` | Continuous | Normal hours per week contracted |
| `actual_hours` | Continuous | Actual hours worked last week |
| `month_sin` | Continuous | sin(2π·month/12) — seasonal encoding |
| `month_cos` | Continuous | cos(2π·month/12) — seasonal encoding |
| `gdp_per_employed` | Continuous | Philippines GDP per person employed (constant 2021 PPP $) |

---

## Key Design Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Target construction | 2-of-3 composite | Reduces false positives from noisy individual indicators |
| Train/test split | Temporal (2021–2023 / 2024) | Prevents temporal leakage; reflects real deployment scenario |
| Class imbalance | `class_weight='balanced'` (LR, SVM) | LR/SVM: loss-function weighting at 2M+ scale
| Primary metric | AUC-PR | ROC-AUC is misleading under class imbalance (inflated by true negatives) |
| Logistic regression | `LogisticRegression(solver='saga', C=1.0)` | Interpretable log-odds coefficients; calibrated probabilities; SAGA solver scales to 2M+ rows |
| Linear SVM | `LinearSVC(C=0.1)` + `decision_function` for AUC-PR | Maximum-margin boundary; primal solver avoids quadratic memory cost of kernel SVM; no native probabilities — decision scores used for ranking |
| Imputation | Mode (categorical) / median (continuous) from train only | Prevents test leakage; mode for `urban_rural` because 2024 survey dropped that column entirely |
| Month encoding | sin/cos cyclical | Respects circular nature of calendar months |

Full technical justifications in [arguments.md](arguments.md).

---

## Outputs

After running all notebooks in order:

```
data/
├── employed_processed.csv    # 2,740,496 employed working-age rows with engineered features
│                             # Each model notebook performs its own train/test split and imputation
├── stagnation_class_distribution.png
├── feature_distributions.png
├── confusion_matrix.png      # Generated per model notebook
└── pr_roc_curves.png         # Generated per model notebook
```

Each model notebook holds its train/test split, imputation fill values, scaler, and model in memory rather than persisting them to disk — rerun the notebook to reproduce results.

---

## How to Run

```bash
# 1. Activate virtual environment
source venv/Scripts/activate  # Windows
# or: source venv/bin/activate  # Unix/Mac

# 2. Install dependencies (if not already installed)
pip install pandas numpy scikit-learn matplotlib seaborn

# 3. Run pipeline first — produces employed_processed.csv
jupyter nbconvert --to notebook --execute notebooks/data-pipeline.ipynb

# 4. Run model notebooks in any order (each is self-contained)
jupyter nbconvert --to notebook --execute notebooks/log_reg.ipynb
jupyter nbconvert --to notebook --execute notebooks/svm.ipynb
```

Or open in Jupyter Lab and run cells sequentially. The data pipeline must run before any model notebook.

---

## Limitations

1. **Cross-sectional, not longitudinal:** The LFS does not track individuals across months. Stagnation is inferred from a single point-in-time snapshot per respondent — it cannot capture dynamic transitions in and out of stagnation.

2. **Imputed urban/rural for 2024:** The 2024 questionnaire dropped the urban/rural classification. Mode imputation introduces measurement error for all 2024 test samples.

3. **Target definition subjectivity:** The 2-of-3 rule is theoretically grounded but operationally chosen. Different threshold choices (any-1-of-3, all-3-of-3) produce different prevalence rates and label sets.

4. **No wage data:** The LFS records basic pay (`PUFC25_PBASIC`) but with high missingness and inconsistent reporting. Wage-based stagnation measures (earning below living wage) were excluded due to data quality.

5. **Linear decision boundaries (LR and SVM):** Cannot capture interaction effects (e.g., age × industry traps). 

6. **No probability calibration for SVM:** `LinearSVC.decision_function` scores are used for ranking (AUC-PR) but are not calibrated probabilities — they cannot be directly interpreted as "probability of stagnation." Platt scaling or isotonic regression would be needed for probability-based deployment.
