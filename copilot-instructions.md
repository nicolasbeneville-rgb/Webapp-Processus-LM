---
name: ml-studio-coding-guide
description: "Use when: writing ML code in ML Studio project. Ensures complete ML workflows: model training, feature engineering, data validation, hyperparameter tuning. Applies to src/ and modules/ ML implementation files. Automatically invoked when editing model training, feature engineering, or validation code."
applyTo: "{src,modules}/**/*.py"
---

# ML Studio — Specialized Coding Guide for Copilot

This instruction file specializes Copilot for ML Studio — an interactive 12-step ML pipeline platform. Use this when implementing or reviewing ML code in `src/` and `modules/`.

## Project Context

**ML Studio** is a Streamlit-based no-code platform guiding users through a complete ML workflow:
- **12 steps**: Data import → Typing → Consolidation → Audit → Target selection → Cleaning → Transformation → Training → Evaluation → Optimization → Prediction → Reporting
- **4 problem types**: Regression, Classification, Time Series (ARIMA/SARIMA), Anomaly Detection
- **16+ models**: Ridge, Lasso, Random Forest, Gradient Boosting, SVM, KNN, Logistic Regression, Naive Bayes, Decision Trees, and polynomial regression
- **Architecture**: Slim orchestrator (app.py) + 6 modular stages (m1-m6) + core engine (src/) + utilities (utils/)
- **Quality gates**: Strict validation at each step (quality score ≥60%, missing <20%, correlation threshold 0.80, min 50 rows, etc.)
- **Rules Engine**: `src/rules_engine.py` centralises all business rules: problem type inference, automatic split policy (stratified/chronological/group/cross-val), context-aware metrics, leakage/compliance heuristics, and stage gate evaluation (`evaluate_stage_gates()`)
- **Anomaly reporting**: Step 9 supports dedicated anomaly exports: full predictions CSV, anomaly cases CSV, and HTML summary report
- **Data Template Export** (NEW): After Step 1 typing, ML Studio auto-generates downloadable data templates (CSV + Excel) with final columns & inferred types. Templates are embedded in model exports (`.mlmodel`) and displayed in the standalone prediction page for easy data input on new datasets.

---

## Code Patterns — Extract & Follow

### 1. Model Instantiation Pattern

**From `src/models.py:get_model()`:**

```python
def get_model(name: str, problem_type: str, params: dict = None):
    """Instantiate a scikit-learn model by name, with safe parameter handling."""
    params = params or {}
    
    # Special case: polynomial regression uses Pipeline
    if name.startswith("Régression Polynomiale"):
        degree = 2 if "degré 2" in name else 3
        return Pipeline([
            ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
            ("reg", LinearRegression()),
        ])
    
    # Registry-based model lookup by (name, problem_type) tuple
    models_registry = {
        ("Régression Linéaire", "Régression"): LinearRegression,
        ("Ridge", "Régression"): Ridge,
        # ... more combinations
    }
    
    model_class = models_registry[(name, problem_type)]
    
    # Validate parameters before passing to constructor
    safe_params = {}
    for p, v in params.items():
        try:
            model_class(**{p: v})
            safe_params[p] = v
        except TypeError:
            continue  # Skip invalid params gracefully
    
    # Always set random_state if model supports it
    try:
        return model_class(random_state=DEFAULT_RANDOM_STATE, **safe_params)
    except TypeError:
        return model_class(**safe_params)
```

**When writing new models:**
- Use registry-based lookup, not if-else chains
- Validate parameters before instantiation to avoid crashes
- Always attempt to set `random_state=42` for reproducibility
- Handle special cases (Pipelines, custom preprocessing) explicitly

---

### 2. Model Training Pattern

**From `src/models.py:train_model()`:**

```python
def train_model(model, X_train, y_train, X_test, y_test,
                problem_type: str, cv_folds: int = 0) -> dict:
    """Train a model and return comprehensive metrics."""
    start = time.time()
    result = {"model": model, "error": None}
    
    try:
        model.fit(X_train, y_train)
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        elapsed = round(time.time() - start, 2)
        result["time"] = elapsed
        
        # Problem-specific scoring
        if problem_type == "Régression":
            result["train_score"] = round(r2_score(y_train, train_pred), 4)
            result["test_score"] = round(r2_score(y_test, test_pred), 4)
            result["rmse"] = round(np.sqrt(mean_squared_error(y_test, test_pred)), 4)
            result["mae"] = round(mean_absolute_error(y_test, test_pred), 4)
        else:  # Classification
            result["train_score"] = round(accuracy_score(y_train, train_pred), 4)
            result["test_score"] = round(accuracy_score(y_test, test_pred), 4)
            result["f1"] = round(f1_score(y_test, test_pred, average="weighted", zero_division=0), 4)
            # AUC-ROC if model has predict_proba
            try:
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X_test)
                    result["auc"] = round(roc_auc_score(y_test, proba, multi_class="ovr"), 4)
            except:
                result["auc"] = None
        
        # Detect overfitting
        result["overfit_pct"] = round(abs(result["train_score"] - result["test_score"]) * 100, 2)
        
        # Cross-validation if requested
        if cv_folds > 0:
            scoring = "r2" if problem_type == "Régression" else "accuracy"
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring=scoring)
            result["cv_mean"] = round(cv_scores.mean(), 4)
            result["cv_std"] = round(cv_scores.std(), 4)
        
        return result
        
    except Exception as e:
        result["error"] = str(e)
        result["time"] = round(time.time() - start, 2)
        return result
```

**When implementing training functions:**
- Always capture train & test predictions separately
- Return comprehensive result dict (scores, metrics, timing, error handling)
- Calculate both R² (regression) and Accuracy + F1 + AUC (classification)
- **Detect overfitting**: compare train_score vs test_score
- **Enable cross-validation**: offer cv_folds parameter
- **Error handling**: catch exceptions and return error field, don't raise
- **Timing**: always measure execution time for UX feedback

---

### 3. Hyperparameter Tuning Defaults

**From `src/models.py:DEFAULT_PARAM_GRIDS`:**

```python
DEFAULT_PARAM_GRIDS = {
    "Ridge": {"alpha": [0.01, 0.1, 1, 10, 100]},
    "Lasso": {"alpha": [0.001, 0.01, 0.1, 1, 10]},
    "ElasticNet": {
        "alpha": [0.01, 0.1, 1],
        "l1_ratio": [0.2, 0.5, 0.8],
    },
    "Arbre de décision": {"max_depth": [3, 5, 7, 10, 15, None]},
    "Random Forest": {
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, 15, None],
    },
    "Gradient Boosting": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 5, 7],
    },
    "SVR": {"C": [0.1, 1, 10], "kernel": ["rbf", "linear"]},
    "Régression Logistique": {"C": [0.01, 0.1, 1, 10, 100]},
    "KNN": {"n_neighbors": [3, 5, 7, 11, 15]},
    "SVM": {"C": [0.1, 1, 10], "kernel": ["rbf", "linear"]},
}
```

**When tuning hyperparameters:**
- Start with the default grids above — they are validated for this dataset type
- Ridge/Lasso/ElasticNet: tune alpha (regularization strength)
- Tree-based models: tune max_depth (prevent overfitting) and n_estimators (more trees = better but slower)
- SVM/KNN: tune C (margin tolerance) and neighbors count
- Use GridSearchCV for small grids (< 20 combinations), RandomizedSearchCV for large grids
- Always use cross-validation (cv=5) to avoid overfitting during tuning

---

### 4. Feature Engineering Patterns

**From `src/feature_engineering.py`:**

```python
# Column combination
def combine_columns(df: pd.DataFrame, col_a: str, col_b: str,
                    operation: str, new_name: str = None) -> pd.DataFrame:
    """Create new column by combining two numeric columns."""
    result = df.copy()
    if new_name is None:
        new_name = f"{col_a}_{operation}_{col_b}"
    
    if operation == "sum":
        result[new_name] = result[col_a] + result[col_b]
    elif operation == "diff":
        result[new_name] = result[col_a] - result[col_b]
    elif operation == "ratio":
        result[new_name] = result[col_a] / result[col_b].replace(0, np.nan)
    elif operation == "product":
        result[new_name] = result[col_a] * result[col_b]
    else:
        raise ValueError(f"Unknown operation: {operation}")
    
    return result

# Column transformation
def transform_column(df: pd.DataFrame, col: str,
                     transformation: str) -> pd.DataFrame:
    """Apply mathematical transformation (log, sqrt, square)."""
    result = df.copy()
    
    if transformation == "log":
        min_val = result[col].min()
        result[col] = np.log1p(result[col] - min_val if min_val <= 0 else result[col])
    elif transformation == "sqrt":
        min_val = result[col].min()
        result[col] = np.sqrt(result[col] - min_val if min_val < 0 else result[col])
    elif transformation == "square":
        result[col] = result[col] ** 2
    else:
        raise ValueError(f"Unknown transformation: {transformation}")
    
    return result

# Column discretization
def discretize_column(df: pd.DataFrame, col: str, n_bins: int = 5,
                      strategy: str = "quantile", labels: list = None) -> pd.DataFrame:
    """Discretize numeric column into bins."""
    result = df.copy()
    new_col = f"{col}_binned"
    
    if strategy == "quantile":
        result[new_col] = pd.qcut(result[col], q=n_bins, labels=labels, duplicates="drop")
    elif strategy == "uniform":
        result[new_col] = pd.cut(result[col], bins=n_bins, labels=labels)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    return result
```

**When implementing feature engineering:**
- Always return a copy (`.copy()`) to avoid SettingWithCopyWarning
- Handle edge cases: division by zero (→ NaN), negative values for log/sqrt
- Create descriptive feature names: `f"{original}_{operation}_{context}"`
- Support multiple strategies (quantile uses distribution percentiles, uniform uses fixed widths)
- Validate parameter input and raise clear error messages

---

## ML Best Practices Checklist

Check these when coding ML features or reviewing ML code:

### Model Selection
- [ ] **Regression decision:** Use R² (coefficient of determination) as primary metric. R² ≥ 0.80 = good, 0.60–0.80 = acceptable, < 0.60 = poor
- [ ] **Classification decision:** Use Accuracy (overall correctness) + F1-Score (balance precision/recall) + AUC-ROC (discrimination ability)
- [ ] **Start simple:** Try Linear models first, then tree-based (Random Forest, Gradient Boosting), then complex (SVM, Neural Networks)
- [ ] **Avoid data leakage:** Never use target or future information when creating features
- [ ] **Problem type detection:** If target has >10 unique values → Regression; ≤10 and categorical → Classification

### Data Preparation
- [ ] **Train-test split:** Call `recommend_split_strategy()` from `src/rules_engine.py` to auto-select the right strategy:
  - Chronological (time series, **mandatory**)
  - Stratified via `split_data_stratified()` (imbalanced classification, ratio >10:1)
  - Group split warning (ID column detected in features)
  - Cross-validation preferred (<100 rows)
  - Random (default)
- [ ] **Anomaly mode:** Use `problem_type="Détection d'anomalies"` with numeric features only (no business target required)
- [ ] **Standardization:** Use StandardScaler (mean=0, std=1) for distance-based models (SVM, KNN); MinMaxScaler (0-1) for tree-based models acceptable
- [ ] **Encoding:** One-Hot Encoding for low-cardinality (<20 categories), Label Encoding or embeddings for high-cardinality
- [ ] **Missing values:** Check quality score ≥60% and missing <20% of data BEFORE training
- [ ] **Outliers:** Use IQR method (factor=1.5) for detection; capping at 1st percentile (lower) and 99th percentile (upper)
- [ ] **Leakage & compliance checks:** run `detect_leakage_suspects()` and `detect_compliance_risks()` before validating features

### Feature Engineering
- [ ] **Derived features:** Create ratio, interaction, aggregation features if domain knowledge suggests relationship
- [ ] **Scaling features:** Log transform for skewed distributions (e.g., price data)
- [ ] **Feature selection:** Keep top N features by correlation with target or via tree-based feature importance (avoid overfitting)
- [ ] **Multicollinearity:** Drop features correlated >0.80 with other features (reduces model stability)

### Hyperparameter Tuning
- [ ] **Use defaults from DEFAULT_PARAM_GRIDS above** — they're validated for this project
- [ ] **Regularization (Ridge/Lasso):** Higher alpha = stronger regularization. Balance bias-variance tradeoff
- [ ] **Tree depth (Trees/RF/GB):** Limit max_depth to prevent overfitting; start with 5-7
- [ ] **Learning rate (Gradient Boosting):** Lower = slower but often better models; typical range 0.01–0.2
- [ ] **CV strategy:** Use cross_val_score with cv=5 folds; validates generalization without test set leakage

### Model Validation
- [ ] **Overfitting detection:** If (train_score - test_score) > 10%, model is overfitting. Reduce complexity or regularization
- [ ] **Underfitting detection:** If test_score < 0.60, model is underfitting. Try more features, less regularization, or complex model
- [ ] **Residuals (Regression):** Check residuals are normally distributed with constant variance (homoscedasticity)
- [ ] **Confusion matrix (Classification):** Analyze false positives vs false negatives; may indicate class imbalance
- [ ] **Cross-validation stability:** std(cv_scores) should be small (<0.05); high std = unstable model
- [ ] **Production gate:** Run `validate_production_readiness()` before export; checks score ≥ threshold, overfit <15%, model saved, model card present

---

## Quality Validation Rules — From config.py

**These are hard constraints applied at each step:**

```python
# Data Quality (Step 3 — Audit)
DEFAULT_QUALITY_THRESHOLD = 60              # Score qualité minimum sur 100
CORRELATION_THRESHOLD = 0.80                # Multicollinéarité : drop si corr > 0.80

# Data Integrity (Step 2 — Typing)
MAX_NAN_AFTER_CONVERSION_PCT = 20           # Max % de NaN après conversion de type

# Data Completeness (Step 5 — Cleaning)
MIN_ROWS = 50                               # Minimum rows per file
MIN_ROWS_AFTER_CLEANING = 50                # Rows remaining after cleaning
MIN_FEATURES = 2                            # Minimum feature columns

# Data Distribution (Anomalies)
OUTLIER_IQR_FACTOR = 1.5                    # IQR × 1.5 for outlier detection
CAPPING_LOWER_PERCENTILE = 1                # Percentile 1 for lower cap
CAPPING_UPPER_PERCENTILE = 99               # Percentile 99 for upper cap
QUASI_CONSTANT_PCT = 95                     # Quasi-constant if 95%+ same value
CLASS_IMBALANCE_RATIO = 10                  # Max ratio between class counts

# Model Training (Step 7)
DEFAULT_TEST_SIZE = 0.20                    # 80/20 train-test split
DEFAULT_RANDOM_STATE = 42                   # Reproducibility seed
DEFAULT_CV_FOLDS = 5                        # Cross-validation folds
DEFAULT_MIN_SCORE = 0.60                    # Minimum acceptable R² or Accuracy
DEFAULT_MAX_OVERFIT_PCT = 10                # Max gap train_score - test_score

# Optimization (Step 8)
OPTIMIZATION_DEFAULT_ITERATIONS = 50        # Random Search iterations
MIN_IMPROVEMENT_PCT = 1.0                   # Min improvement to accept new hyperparams
```

**When validating input or generating warnings:**
- Quality score < 60 → block workflow progression, suggest data cleaning
- Missing data > 20% → warn user, suggest imputation strategy
- Correlation > 0.80 → auto-drop one of correlated pair or prompt user
- Test score < 0.60 → suggest more features, data cleaning, or model complexity increase
- Overfitting > 10% → suggest regularization increase, depth reduction, or dropout

---

## Common Pitfalls & Gotchas

### 1. **Silent File Save Failures**
**Problem:** Model or CSV saved to disk, but exception caught silently → user thinks save succeeded but file doesn't exist.
```python
# ❌ BAD
try:
    joblib.dump(model, filepath)
except Exception:
    pass  # Silent failure!

# ✅ GOOD
try:
    joblib.dump(model, filepath)
    st.success(f"✅ Model saved to {filepath}")
except Exception as e:
    st.error(f"❌ Save failed: {e}")
    return None
```
**Action:** Always report errors to user via `st.error()` or raise exception, never silently ignore.

---

### 2. **Session State Fragility Across Reruns**
**Problem:** Streamlit reruns entire script on each interaction. Session state can become inconsistent.
```python
# ❌ BAD — model_trained might not exist after rerun
if st.button("Train"):
    model_trained = train(X, y)  # Local variable, lost on rerun!

# ✅ GOOD — store in session_state
if st.button("Train"):
    st.session_state.model_trained = train(X, y)
    st.session_state.training_done = True

if st.session_state.get("training_done"):
    st.write(st.session_state.model_trained.score(...))
```
**Action:** Always use `st.session_state` for multi-step workflows, never rely on local variables.

---

### 3. **Overfitting on Small Datasets**
**Problem:** With <100 rows, random chance can inflate train_score artificially.
```python
# ✅ BETTER — Use stratified CV to estimate real generalization
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5, scoring="r2")
mean_score = scores.mean()  # More reliable than train_score on small data
```
**Action:** For small datasets (< 100 rows), prioritize cross-validation scores over train-test split scores.

---

### 4. **Data Leakage in Feature Engineering**
**Problem:** Using target statistics (mean, std) fitted on full data, then applied at prediction time.
```python
# ❌ BAD — Leakage! Scaler sees entire dataset including test target statistics
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X) # X contains both train AND test data!
X_scaled = scaler.transform(X)

# ✅ GOOD — Fit on train, transform train & test separately
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```
**Action:** Always fit preprocessing on train data only, then apply to both train & test.

---

### 5. **Confusing Problem Type Detection**
**Problem:** Is this regression or classification? User picks wrong target type.
```python
# ✅ BETTER — Auto-detect, but let user confirm
if target.nunique() > 10:
    inferred_type = "Régression"
else:
    inferred_type = "Classification"

user_confirmed = st.radio("Problem type", ["Régression", "Classification"],
                           index=0 if inferred_type == "Régression" else 1)
```
**Action:** Auto-infer problem type based on target cardinality, but always let user confirm.

---

### 5.5 **Data Template Export for Prediction Input**

**Feature:** After Step 1 (Typing), ML Studio auto-generates a downloadable data template (CSV + Excel) with final columns (post-deletion) and inferred types.

**Implementation details:**
- **Phase 1 (m1_chargement.py):** After user validates typing, generate template with:
  - Columns from `df_courant` (final columns after any user deletions)
  - Types deduced from `type_mapping`: "Texte", "Catégorie", "Numérique", "Date", "Booléen"
  - Files saved: `trame_donnees.csv` + `trame_donnees.xlsx` (2 sheets: "Données" + "Types")
  - Metadata stored in rapport: `trame_colonnes`, `trame_types`, `trame_csv_path`, `trame_xlsx_path`

- **Phase 2 (m4_entrainement.py):** When exporting model to `.mlmodel`, include template metadata:
  - Add `trame_colonnes` and `trame_types` to `export_data` dict
  - Allows prediction page to display exact expected columns & types

- **Phase 3 (app_prediction.py):** In prediction step 2 "Load data":
  - After model upload, if `model_meta` contains `trame_colonnes` → show expander "📥 Download Template"
  - Display table: Colonne | Type | Exemple (empty)
  - Buttons: CSV download + Excel download (2 sheets)
  - Fallback: if openpyxl unavailable, show caption in Excel column

**When implementing template features:**
- Always generate template AFTER typing (ensures final columns after user edits)
- Include type inference logic: extract from `type_mapping` and normalize names
- Support both CSV (simple) and Excel (with Types reference sheet)
- Store template metadata in rapport for later export
- Error handling: wrap openpyxl operations in try/except (graceful fallback)

**User workflow:**
1. Type data in Step 1 → trame auto-generated
2. Train model + export → trame embedded in `.mlmodel`
3. Load model in prediction page → trame displayed
4. Download trame → fill with new data → upload → get predictions

---

### 6. **Legacy Code (app_old.py)**
**Problem:** Old 3983-line monolithic file still in repo. Can confuse teammates.
**Action:** Archive `app_old.py` to `/notebooks/archive/` or document its deprecation status in README.

---

## When to Invoke This Guide

This instruction file auto-loads when:
- Editing `src/models.py`, `src/feature_engineering.py`, `src/evaluation.py`
- Editing `modules/m4_entrainement.py`, `modules/m5_evaluation.py`, `modules/m6_prediction.py`
- Writing new ML wrapper functions or refactoring existing training code

**Triggered by keywords:** model training, feature engineering, hyperparameter tuning, cross-validation, overfitting, data leakage, quality validation, preprocessing.

---

## Key Files Reference

| File | Purpose | Key Functions |
|------|---------|---|
| [src/models.py](src/models.py) | Core ML training | `get_model()`, `train_model()`, `train_multiple()`, `optimize_model()`, `split_data_stratified()` |
| [src/rules_engine.py](src/rules_engine.py) | Business rules engine | `recommend_split_strategy()`, `recommend_metrics()`, `evaluate_stage_gates()`, `infer_problem_type()` |
| [src/rules_engine.py](src/rules_engine.py) | Leakage/compliance heuristics | `detect_leakage_suspects()`, `detect_compliance_risks()` |
| [src/validators.py](src/validators.py) | Stage validation | `validate_production_readiness()`, `validate_data_quality()`, `validate_model_scores()` |
| [src/feature_engineering.py](src/feature_engineering.py) | Feature creation | `combine_columns()`, `transform_column()`, `discretize_column()` |
| [src/evaluation.py](src/evaluation.py) | Result analysis | `results_table()`, metric calculations |
| [modules/m4_entrainement.py](modules/m4_entrainement.py) | Training UI | Step 7 Streamlit interface, model selection, auto split policy |
| [modules/m5_evaluation.py](modules/m5_evaluation.py) | Results UI | Step 8 Streamlit interface, confusion matrices, ROC curves |
| [modules/m6_prediction.py](modules/m6_prediction.py) | Prediction & exports UI | Step 9 predictions, API export, anomaly cases CSV/HTML exports |
| [config.py](config.py) | Constants & thresholds | DEFAULT_PARAM_GRIDS, validation rules, model lists |

---

## Quick Reference — Metrics Interpretation

| Metric | Range | Interpretation |
|--------|-------|---|
| **R² (Regression)** | [0, 1] | % of variance explained. 0.80+ = good, 0.60–0.80 = okay, <0.60 = poor |
| **RMSE (Regression)** | [0, ∞) | Root mean squared error. Lower is better; in same units as target |
| **MAE (Regression)** | [0, ∞) | Mean absolute error. Robuster to outliers; interpret directly as avg error |
| **Accuracy (Classification)** | [0, 1] | % correct predictions. Can be misleading on imbalanced data |
| **F1-Score (Classification)** | [0, 1] | Harmonic mean of precision & recall. Balanced metric for imbalance |
| **AUC-ROC (Classification)** | [0, 1] | Area under receiver-operating-characteristic. 0.5=random, 1.0=perfect |
| **Overfit %)** | [0, ∞) | |abs(train_score - test_score)| × 100. >10% = likely overfitting |

---

## Integration with .instructions.md System

This file is a **workspace instruction file** recognized by GitHub Copilot. It:
- Auto-loads when you edit ML-related files in `src/` and `modules/`
- Provides shared context across coding sessions (project structure, best practices, gotchas)
- Does NOT require slash-command activation (`/ml-studio-guide`) — applies automatically
- Can be updated without restarting VS Code

To disable locally: Comment out `applyTo` patterns in YAML frontmatter, or move file out of workspace.

---

## Stabilization Pass Protocol

When asked to run a full project stabilization pass, apply this sequence before feature work:

1. **Global diagnostics first**
- Run workspace error scan (`get_errors`) and Python compile check (`python -m compileall app.py app_pipeline.py modules src utils`).
- If diagnostics are clean, continue with runtime stability checks instead of refactoring blindly.

2. **Runtime smoke checks**
- Verify imports of orchestrator + stage modules (`app_pipeline`, `modules/m1..m6`, `src/models`, `src/rules_engine`, `src/validators`).
- Boot Streamlit briefly to catch startup regressions.

3. **State/navigation safety**
- Never rely on dynamic labels for step navigation state.
- Prefer stable integer keys (`nav_step_idx`) and explicit forward targets (`_pending_step` + `nav_step_idx`).
- Avoid navigation-side effects that can trigger rerun loops.

4. **Persistence robustness**
- Autosave must never block UI navigation.
- Wrap disk save operations in safe try/except and expose non-blocking warnings.
- Keep manual save always available as fallback.

5. **Anomaly mode resilience**
- In anomaly workflows, metrics/UX must not assume supervised labels.
- Guard calls to `decision_function`/`score_samples` with safe fallback.
- Keep anomaly exports deterministic: full CSV, anomaly-only CSV, HTML summary.

6. **Definition of done**
- No new diagnostics errors.
- Compile/import checks pass.
- Streamlit startup smoke check passes.
- Document stabilization deltas in final response.

---

**Last updated:** March 21, 2026  
**Project:** ML Studio (Webapp_Processus_LM)  
**Maintained by:** Data Science team
