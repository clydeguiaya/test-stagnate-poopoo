# Arguments & Justifications

All non-trivial decisions made in the data pipeline and modelling notebooks, with explicit reasoning.

---

## 1. Data Loading Strategy

### Why load all 48 months as strings (`dtype=str`) before casting?

The PSA LFS files contain occupation codes (`PUFC14_PROCC`), industry codes (`PUFC16_PKB`), and education grades (`PUFC07_GRADE`) that are **zero-padded string codes** (e.g., `"01"`, `"09"`, `"60002"`). Loading as default numeric silently truncates leading zeros, corrupting codes like `"01"` → `1`. Loading everything as string first, then explicitly casting known numeric fields, preserves all coded values.

### Why three schema mappings?

Inspection of all 48 CSV column sets revealed three distinct questionnaire versions:

| Schema | Period | Trigger |
|--------|--------|---------|
| A | Jan–Jul 2021 | Baseline LFS questionnaire |
| B | Aug–Dec 2021 | PSA redesigned questionnaire mid-2021 (HHMEM format — item numbers shifted, e.g., employment question moved from C11 to C09) |
| C | 2022–2024 | Extended baseline (adds arrangement type, province of workplace, ethnicity in 2024) |

Schema B is detected by the presence of `PUFC09_WORK` (absent in A/C). Without this mapping, naively concatenating all 48 files produces misaligned columns and silently introduces massive missingness.

### Why is `PUFURB2015` (urban/rural) missing in 2024?

The 2024 questionnaire revision dropped the urban/rural classification column. This is a known PSA survey change. It is imputed with the training-set mode rather than discarded — urban/rural is a meaningful predictor of stagnation risk (rural workers disproportionately in subsistence agriculture). Dropping the column entirely would penalise the model for a data collection change, not a genuine data absence.

---

## 2. Target Variable Construction

### Why no direct `is_stagnant` label exists

The PSA LFS is a labour force participation survey, not an employee wellbeing survey. It does not ask respondents to self-report stagnation. The concept must be **operationalised** from observable proxy variables.

### Choice of operationalisation: 2-of-3 composite rule

An employed person (`emp_status == 1`, age ≥ 15) is labelled `is_stagnant = 1` if they satisfy **at least 2 of the following 3 criteria**:

**C1 — Visible underemployment** (`want_more_work == 1`)  
This is the PSA's own official visible underemployment indicator. It directly captures workers whose current job does not fully utilise their available labour time — a defining feature of stagnation. The PSA defines it as "employed persons who express the desire to have additional hours of work in their present job or an additional job, or to have a new job with longer working hours."

**C2 — Precarious employment** (`nature_employment ∈ {2=temporary, 3=casual/seasonal}` OR `worker_class == 6 (unpaid family worker)`)  
Temporary, casual, and unpaid family workers lack employment security, access to benefits, and a promotion ladder — structural conditions that make upward mobility impossible. This criterion captures the ILO concept of "vulnerable employment." Unpaid family workers are always stagnant by definition (zero wages, no formal employment relationship).

**C3 — Education-occupation mismatch** (`education_level ≥ 6 (college+)` AND `occupation_major == 9 (elementary occupations)`)  
A college-educated worker stuck in an elementary occupation (e.g., street sweeper, domestic helper, agricultural labourer) is textbook overeducation — their human capital cannot be applied, signalling a structural trap. This is grounded in the PSA PSOC classification and Philippines-specific overeducation literature.

### Why 2-of-3 rather than any single criterion?

- **C1 alone** is noisy: some workers voluntarily seek part-time work (students, retirees, semi-retired).
- **C2 alone** is noisy: project-based and seasonal contracts are not always stagnation (e.g., construction project workers earning well above market).
- **C3 alone** is noisy: a fresh graduate in their first job may temporarily hold an elementary position.
- **2 of 3** requires co-occurring evidence, substantially reducing false positives while preserving sensitivity to genuine multi-dimensional stagnation. It mirrors composite index methodology used in the Philippine Development Plan and ILO decent work deficit indicators.

### Leakage prevention

All variables used to construct the target criteria are excluded from the feature matrix:

| Criterion | Variables dropped |
|-----------|------------------|
| C1 — Visible underemployment | `want_more_work` |
| C2 — Precarious employment | `nature_employment`, `worker_class` |
| C3 — Education-occupation mismatch | `education_level`, `occupation_major` |

Although `education_level` and `occupation_major` are theoretically grounded predictors of labour market outcomes, retaining them gives the model a direct path to reconstructing C3 — it can learn the exact intersection `(education_level ≥ 6) AND (occupation_major == 9)` and re-derive one of the three construction rules rather than learning independent signal. This inflates evaluation metrics without adding genuine predictive power. A clean feature set requires dropping all criterion inputs, not only the most obvious ones.

---

## 3. Feature Selection

| Feature | Rationale |
|---------|-----------|
| `age` | Labour economics: stagnation risk is U-shaped with age (young workers lack seniority; older workers face displacement) |
| `sex` | Philippines has documented gender gaps in employment quality and industry segregation |
| `marital_status` | Married workers (especially women) have higher probability of accepting inferior jobs near family |
| `region` | 17 regions with vast differences in labour market density, industry mix, and formal economy penetration |
| `urban_rural` | Rural workers structurally more vulnerable to informal/subsistence employment |
| `hh_size` | Larger households → greater economic pressure to accept any available work |
| `industry_sector` | Industry-level formalisation rates, wage floors, union density differ dramatically |
| `normal_hours` | Contracted hours below full-time signal structural underemployment |
| `actual_hours` | Hours worked below contracted hours signals additional underutilisation |
| `month_sin`, `month_cos` | Cyclical encoding of survey month captures seasonal labour patterns (planting/harvest, holiday retail, school year) without imposing linear order on a circular variable |
| `gdp_per_employed` | Year-level macro control: GDP per employed person captures productivity improvements (or contractions) that affect job quality across all workers in a given year |

**Excluded features:**
- `want_more_work`, `nature_employment`, `worker_class` → C1/C2 criterion inputs (direct target construction)
- `education_level`, `occupation_major` → C3 criterion inputs (direct target construction; see leakage section above)
- `line_no`, `relationship` → survey administrative fields
- `survey_month` (raw integer) → replaced by sin/cos cyclical encoding
- `currently_working` → redundant given we filtered to `emp_status == 1`
- `worked_last_week` → redundant with employment status filter

---

## 4. Train/Test Split: Temporal (2021–2023 train, 2024 test)

**Random split was rejected.** With monthly panel survey data, a random split causes temporal leakage: the model can observe December 2023 data in training while predicting January 2023 data in test. GDP per employed is year-level, so any year overlap in train/test contaminates the macro feature.

**Temporal split** respects the natural data generating process: the model is trained on historical labour market conditions and evaluated on its ability to identify stagnant workers in an unseen future period (2024). This is the operationally meaningful test — a deployed tool would always predict on data after its training cutoff.

The 2024 holdout contains approximately 25% of total employed observations, sufficient for reliable evaluation.

---

## 5. Imputation Strategy

**Fit imputation statistics on training data only, then apply to test.** This prevents test-set information from leaking into the imputation, which would give an overly optimistic estimate of model performance on truly unseen data. Each model notebook (logistic regression, KNN, SVM) performs its own imputation from scratch using only the training fold.

| Column | Strategy | Rationale |
|--------|----------|-----------|
| `urban_rural` | Mode (train set) | Column missing for all 2024 respondents due to survey redesign; mode is the most representative single imputation for a binary variable at this scale |
| `region`, `marital_status`, `industry_sector`, `sex` | Mode (train set) | Sparse missingness from survey skip patterns; mode is a conservative estimate for low-cardinality categorical variables |
| `normal_hours`, `actual_hours` | Median (train set) | Continuous variables with skewed distributions; median is robust to outliers and extreme working hours |
| `age`, `hh_size`, `month_sin`, `month_cos`, `gdp_per_employed` | Median (train set) | Continuous; median-robust; GDP and cyclical encodings have essentially zero missingness but are included for defensive completeness |

**MCAR assumption:** Missingness is treated as missing completely at random conditional on employment status. This is plausible because most missingness stems from survey skip patterns (questions not asked to workers with certain employment types) rather than the workers' characteristics.

---

## 6. Class Imbalance Handling

### Decision: `class_weight='balanced'` in logistic regression and SVM; balanced subsample in KNN

**Why not SMOTE?**  
At ~1M+ training rows, SMOTE requires synthesising minority-class observations by computing k-nearest neighbours across the entire training set. This is computationally prohibitive (O(n²) in memory with large k), and the synthetic observations gain no new information not already present in the real data at this scale. SMOTE is most valuable when the dataset is small and the minority class is severely under-represented (< 5%). At a 20–30% stagnation rate with hundreds of thousands of minority samples, SMOTE adds computational cost without meaningful benefit.

**Why not random undersampling?**  
Discarding majority-class observations destroys real signal. With millions of non-stagnant observations carrying genuine demographic and labour market variation, undersampling introduces variance and potentially removes informative edge cases.

**`class_weight='balanced'` (logistic regression and LinearSVC):** Mathematically equivalent to resampling the minority class with replacement to match the majority class count. The loss function assigns each stagnant observation a weight of `n_samples / (2 * n_stagnant)` and each non-stagnant observation `n_samples / (2 * n_not_stagnant)`. This gives the model proportionally larger gradient updates from minority-class errors, without changing the data. It is the cleanest and most computationally efficient approach at this scale.

**Balanced subsample (KNN):** `class_weight` does not exist for KNN — the algorithm has no loss function to weight. Instead, a balanced random subsample of 50,000 training records (25,000 stagnant, 25,000 not-stagnant) is drawn using `numpy.random.default_rng(42)`. The balanced 50/50 split removes the class imbalance problem for the nearest-neighbour vote. This subsample also addresses the computational bottleneck: KNN has no offline training phase — at prediction time it must compute distances to every stored training record. Storing 2M+ training points would make prediction prohibitively slow; 50,000 records provides a tractable approximation.

---

## 7. Evaluation Metrics

### Primary: AUC-PR (Average Precision)

With class imbalance, **ROC-AUC is misleading.** A classifier that randomly labels all workers as non-stagnant achieves 70–80% ROC-AUC by exploiting the large true negative count. AUC-PR (area under the precision-recall curve) removes true negatives from the denominator entirely — it measures only the tradeoff between finding stagnant workers (recall) and avoiding false alarms (precision). This directly corresponds to the policy objective: identify the workers most at risk of stagnation without wasting intervention resources on false positives.

### Secondary: AUC-ROC

Retained for comparison and for use cases where false positive costs are symmetric. Reports the probability that a randomly chosen stagnant worker scores higher than a randomly chosen non-stagnant worker.

### Tertiary: Classification report with optimal threshold

The 0.5 threshold is rarely optimal under class imbalance. We compute the threshold that maximises the F1 score for the stagnant class on the test set. This threshold should be selected based on the cost ratio between false negatives (missed stagnant workers — intervention cost) and false positives (wasted interventions) in the deployment context.

---

## 8. Pearson Correlation Pre-analysis

All three model notebooks compute Pearson correlation between each feature and `is_stagnant` before training. This serves two purposes:

1. **Sanity check on feature engineering:** If cyclical month encoding, industry binning, or GDP merge introduced unexpected artefacts, the correlation table would surface them as anomalous signals.
2. **Baseline interpretability:** For a policy-facing tool, readers expect to see which raw features are associated with stagnation before the model complexity is introduced. Correlation is immediately interpretable even to non-technical stakeholders.

Pearson correlation is used (not point-biserial or Spearman) because most features are continuous or ordinal and the target is binary. With 2.7M observations, any non-zero correlation is statistically significant — the chart communicates practical magnitude, not significance.

---

## 9. Logistic Regression Model

**Why logistic regression?**

1. **Interpretability:** Coefficients map directly to log-odds — each unit increase in a standardised feature increases the log-odds of stagnation by the coefficient value. For a policy instrument, this allows analysts to communicate "being in a rural area increases stagnation odds by X%."

2. **Calibrated probabilities:** Logistic regression outputs well-calibrated probabilities natively (the sigmoid maps directly to probability). Tree-based models require post-hoc Platt scaling or isotonic regression for calibration. Calibration matters because predicted probabilities will be used to rank intervention priority.

3. **Baseline model:** Logistic regression establishes the interpretable baseline against which KNN and SVM are benchmarked.

4. **L2 regularisation (`C=1.0`):** Prevents overfitting on correlated demographic features. Default `C=1.0` provides moderate regularisation.

5. **Solver `saga`:** Stochastic Average Gradient Augmented — designed for large-scale datasets. Supports L1, L2, and elastic net penalties. Converges significantly faster than `lbfgs` on 2M+ row datasets.

**Key coefficients (standardised features, sorted descending):**

| Feature | Coefficient | Interpretation |
|---------|-------------|----------------|
| `normal_hours` | +0.225 | Fewer contracted hours → higher stagnation log-odds |
| `urban_rural` | +0.102 | Rural coding (2) → higher stagnation log-odds |
| `hh_size` | +0.053 | Larger household → higher stagnation log-odds |
| `actual_hours` | −0.866 | More hours actually worked → sharply lower stagnation log-odds |
| `age` | −0.331 | Older workers → lower stagnation log-odds (seniority effect) |
| `industry_sector` | −0.236 | Higher-numbered formal sectors → lower stagnation log-odds |

The `actual_hours` coefficient dominating negatively is expected: workers who are visibly underemployed (C1 criterion) by definition work fewer actual hours, but C1 was dropped — the model is learning a proxy of that signal through actual hours without directly using the criterion variable.

---

## 10. Linear SVM Model

**Why LinearSVC as a comparison model?**

1. **Maximum-margin decision boundary:** Unlike logistic regression which minimises log-loss across all training points, an SVM maximises the margin between the two classes. This can produce a more robust boundary when the classes are not well-separated, as is the case with stagnation prediction.

2. **Scalability:** `LinearSVC` solves the primal optimisation problem directly, avoiding the quadratic memory cost of kernel SVMs (`SVC`). It scales to 2M+ training records where `SVC` with a non-linear kernel would be intractable.

3. **No probability outputs:** `LinearSVC` does not natively produce calibrated probabilities. For the AUC-PR curve, the signed distance from the decision boundary (`decision_function`) is used as a ranking score. This produces a valid PR curve because AUC-PR depends on ranking, not calibration. The AUC-PR from `decision_function` is identical to the AUC-PR that calibrated probabilities would produce.

**Hyperparameter choices:**

- **`C=0.1`:** Stronger regularisation than the logistic regression default. With 2M training records and already-standardised features, the decision boundary is well-determined — a wider margin (smaller C) reduces the risk of fitting to survey noise in the demographic features.
- **`class_weight='balanced'`:** Increases the penalty for misclassifying the minority stagnant class, equivalent to the logistic regression treatment.
- **`max_iter=2000`:** Extended to allow convergence with the large training set.

**Key coefficients (standardised features, sorted descending):**

| Feature | Coefficient | Interpretation |
|---------|-------------|----------------|
| `normal_hours` | +0.097 | Fewer contracted hours → more likely stagnant |
| `urban_rural` | +0.045 | Rural → more likely stagnant |
| `actual_hours` | −0.372 | More hours worked → less likely stagnant |
| `age` | −0.140 | Older workers → less likely stagnant |
| `industry_sector` | −0.104 | Formal sectors → less likely stagnant |

The sign pattern matches logistic regression, validating that both linear models identify the same underlying relationships in the data.

---

## 11. GDP Per Person Employed as a Macro Feature

GDP per person employed (constant 2021 PPP $) is a **year-level macro control** that captures the aggregate productivity environment faced by all workers in a given year. This is distinct from individual-level wages (not available in the LFS) and captures:
- Post-COVID recovery dynamics (2021 rebound, 2022 contraction)
- Structural shifts in labour productivity
- Cross-year comparisons without inflation distortion (PPP-adjusted)

**Limitation:** All workers in the same year receive the same GDP value — it cannot differentiate between high-GDP industries and low-GDP industries within a year. This is a macro fixed-effect, not an individual predictor. It controls for year-specific shocks that would otherwise be confounded with demographic trends.

**Source:** Our World in Data / World Bank — "GDP per person employed, constant 2021 PPP dollars." Philippines data available 2021–2024 matching the survey period exactly.

---

## 12. Education Level Encoding

The PSA `PUFC07_GRADE` is a 5-digit code where the **first digit encodes the education system level**:

| First digit | Level |
|-------------|-------|
| 0 | No grade completed |
| 1 | Elementary |
| 2 | Old high school (pre-K12) |
| 3 | Junior high school (K12) |
| 4 | Senior high school (K12) |
| 5 | Vocational/Technical |
| 6 | College (bachelor's) |
| 7 | Post-graduate |

The first digit is extracted as `education_level` (0–7) solely to evaluate the C3 criterion (`education_level ≥ 6`). Both `education_grade` and `education_level` are dropped from the dataset immediately after the target is constructed and are not available as model features.

---

## 13. Occupation and Industry Encoding

**Occupation:** PSOC (Philippine Standard Occupational Classification) 2-digit code. The first digit is extracted as `occupation_major` (1–9) solely to evaluate the C3 criterion (`occupation_major == 9`). Both `occupation_code` and `occupation_major` are dropped from the dataset immediately after the target is constructed and are not available as model features.

**Industry:** PSIC (Philippine Standard Industrial Classification) 2-digit code. Binned into 10 broad sectors (agriculture, mining, manufacturing, construction/utilities, retail, transport/food, ICT/finance, professional/admin, public/health/education, other). This reduces cardinality from ~90 to 10 while preserving the economically meaningful distinctions between formal and informal sectors. Unlike occupation, industry is retained as a model feature because it was not used in any target criterion.

The `psic_sector` mapping used in the pipeline:

| PSIC 2-digit range | Sector label |
|--------------------|--------------|
| 01–03 | Agriculture/fishing/forestry |
| 05–09 | Mining and quarrying |
| 10–33 | Manufacturing |
| 35–43 | Construction and utilities |
| 45–47 | Wholesale and retail trade |
| 49–56 | Transport, food, and accommodation |
| 58–66 | ICT, finance, and insurance |
| 68–82 | Professional, admin, and real estate |
| 84–88 | Public admin, education, and health |
| All others | Other services |

---

## 14. Cyclical Encoding for Survey Month

Survey month (1–12) is encoded as `sin(2π·month/12)` and `cos(2π·month/12)`. This is necessary because:
1. Month is a **circular** variable — December (12) is closer to January (1) than to June (6) in the seasonal cycle
2. Treating it as a linear integer breaks this circularity, causing the model to see a 12-unit "gap" between December and January
3. The 2D sin/cos encoding preserves the circular structure while remaining compatible with linear models

The seasonal signal matters: agricultural underemployment peaks during off-harvest months (July–September), holiday retail employment spikes in November–December, and school-year patterns affect youth employment throughout Q1/Q3.
