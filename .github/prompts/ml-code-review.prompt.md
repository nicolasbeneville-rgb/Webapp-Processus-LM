---
name: ml-code-review
description: "Use when: reviewing ML code for completeness, best practices, quality gates, and data validation. Analyzes model training, feature engineering, hyperparameter tuning, and error handling. Outputs structured feedback with checklist status and recommendations for ML Studio project."
---

# /ml-review — Comprehensive ML Code Review

## Overview

This prompt invokes a complete ML code review based on the ML Studio best practices and quality standards from `copilot-instructions.md`.

## When to Use

Use this prompt when:
- Writing new model training code and want to verify completeness
- Reviewing feature engineering implementations
- Validating hyperparameter tuning logic
- Ensuring data validation and quality gates are present
- Checking for common pitfalls (data leakage, overfitting, silent failures)

**Command in VS Code chat:**
```
/ml-review [function/file description]
```

Example:
```
/ml-review Review my train_model() function for completeness and best practices
/ml-review Check this feature engineering pipeline for data leakage
/ml-review Is my hyperparameter grid appropriate for Gradient Boosting?
```

---

## Review Checklist

The review will evaluate code against these criteria:

### 1. **Model Selection & Instantiation**
- [ ] Uses `get_model()` registry pattern or equivalent safe instantiation
- [ ] Validates parameters before passing to model constructor
- [ ] Sets `random_state=DEFAULT_RANDOM_STATE` for reproducibility
- [ ] Handles special cases (Pipelines, custom preprocessing) explicitly
- [ ] Problem type correctly inferred ("Régression" vs "Classification" vs "Série temporelle" vs "Détection d'anomalies")

### 2. **Data Preparation**
- [ ] Train-test split done AFTER all preprocessing (prevents leakage)
- [ ] `recommend_split_strategy()` from `src/rules_engine.py` was called or its rules manually applied:
  - Chronological for time series (mandatory — block if overridden)
  - Stratified (`split_data_stratified()`) for imbalanced classification (ratio >10:1)
  - Group split warning if ID column detected in features
  - Cross-validation preferred if <100 rows
- [ ] Scaler/encoder fitted on train data only, then applied to both train & test
- [ ] Minimum validation passed: ≥50 rows, ≥2 features, quality ≥60%, missing <20%

### 3. **Training & Metrics**
- [ ] Model fitted with `.fit(X_train, y_train)`, predictions on separate test set
- [ ] Scores calculated: train_score + test_score (detect overfitting)
- [ ] Problem-specific metrics included:
  - Regression: R², RMSE, MAE
  - Classification: Accuracy, F1-Score, AUC-ROC
- [ ] Cross-validation enabled (cv_folds ≥ 5) for generalization estimate
- [ ] Execution time tracked for UX feedback

### 4. **Hyperparameter Tuning**
- [ ] Uses DEFAULT_PARAM_GRIDS patterns or documented justification
- [ ] Grid size appropriate (GridSearchCV for <20 combinations, RandomizedSearchCV for larger)
- [ ] Cross-validation used during tuning (cv ≥ 5)
- [ ] Stopping criteria or iteration limit set (min improvement ≥1%, max iterations ≤200)

### 5. **Feature Engineering**
- [ ] New features created with causal domain logic, not overfitting to target
- [ ] No future data or target leakage in derived features
- [ ] Leakage heuristics checked via `detect_leakage_suspects()`
- [ ] Compliance heuristics checked via `detect_compliance_risks()` for PII-like columns
- [ ] Multicollinearity checked: features correlated >0.80 identified and handled
- [ ] Transformations (log, sqrt) handle edge cases (negative values, zeros)
- [ ] Feature selection done on train data only, with cv validation

### 6. **Error Handling & Reporting**
- [ ] Exceptions caught and returned in result dict (not raised silently)
- [ ] User-facing errors reported via `st.error()` or equivalent UI
- [ ] File save operations verified and user notified of success/failure
- [ ] Edge cases handled: empty inputs, invalid parameters, division by zero

### 7. **Quality Validation Gates**
- [ ] Quality score ≥60% enforced before training
- [ ] Missing data <20% enforced before training
- [ ] Outliers detected (IQR × 1.5) and handled (capping or removal)
- [ ] Quasi-constant features removed (>95% same value)
- [ ] Class imbalance flagged if ratio >10:1 (classification)

### 8. **Overfitting & Generalization**
- [ ] Overfitting % calculated: |train_score - test_score| × 100
- [ ] Warning triggered if overfitting >10%
- [ ] Regularization applied if overfitting detected (L1/L2, depth limits, etc.)
- [ ] Cross-validation stability checked: std(cv_scores) reasonable

### 9. **Code Quality**
- [ ] Functions have docstrings with args, returns, exceptions
- [ ] Consistent naming: `col` vs `col_` vs `column_name` standardized
- [ ] Return types explicit: dict, DataFrame, ndarray, list clearly documented
- [ ] `.copy()` used to avoid SettingWithCopyWarning on DataFrames
- [ ] No hardcoded values: use config.py constants (DEFAULT_TEST_SIZE, CORRELATION_THRESHOLD, etc.)

### 10. **Documentation & Traceability**
- [ ] Comments explain "why", not "what" (code should be readable enough)
- [ ] Complex logic (tuning, diagnosis) has inline comments
- [ ] Result dict keys documented (especially custom metrics)
- [ ] Test data assumptions documented (e.g., "assumes chronological order for time series")

### 11. **Production Readiness**
- [ ] `validate_production_readiness()` from `src/validators.py` called before export/deploy
- [ ] Score test ≥ metric threshold (business requirement, not just DEFAULT_MIN_SCORE)
- [ ] Overfit < 15% — hard block gate (>15% = not deployable)
- [ ] Model saved (joblib) and project path present in rapport
- [ ] Model card metadata present: model name, problem type, test score, training date
- [ ] `evaluate_stage_gates("production", ctx)` from `src/rules_engine.py` returns no blocking issues

### 12. **Anomaly Detection Specific**
- [ ] Uses dedicated model(s) such as Isolation Forest with `problem_type="Détection d'anomalies"`
- [ ] Does not require a supervised target label for training
- [ ] Reports anomaly rate and stability train/test, not misleading Accuracy/R²
- [ ] Explains contamination and threshold implications to users
- [ ] Provides anomaly exports: full predictions CSV, anomaly cases CSV, and HTML summary

---

## Expected Review Output

When you use `/ml-review`, you'll receive:

```
## ML Code Review — [Function/File Name]

### ✅ Strengths
- [Positive findings, patterns followed well]

### ⚠️ Issues Found
- [Specific issues with line references and severity]

### ❌ Critical Issues
- [Must-fix items: data leakage, silent failures, incorrect metrics]

### 📋 Checklist Summary
| Category | Status | Notes |
|----------|--------|-------|
| Model Selection | ✅ | Registry pattern used correctly |
| Data Preparation | ⚠️ | Scaler fitted on full data (potential leakage) |
| Metrics | ✅ | All required metrics calculated |
| ... | ... | ... |

### 🛠️ Recommendations
1. [Priority 1: Critical fixes]
2. [Priority 2: Important improvements]
3. [Priority 3: Nice-to-have enhancements]

### 📚 Reference
- Best practices checklist: `copilot-instructions.md` § Best Practices
- Code patterns: `copilot-instructions.md` § Code Patterns
- Pitfalls: `copilot-instructions.md` § Common Pitfalls
```

---

## Integration

This prompt is stored in `.github/prompts/ml-code-review.prompt.md` and:
- Appears as `/ml-review` slash command in VS Code chat
- Auto-includes `copilot-instructions.md` context
- Applies to `src/` and `modules/` code

No manual setup required — just type `/ml-review` in the chat and provide your code question.

---

## Example Usage

### Example 1: Review a training function
```
/ml-review 
Here's my train_model() implementation. Does it follow ML Studio patterns and test for overfitting?

def train_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    return {"train": train_score, "test": test_score}
```

**Expected Feedback:** Missing metrics (RMSE/MAE for regression, F1 for classification), no error handling, no timing, no problem_type parameter to determine scoring metric, overfitting not calculated explicitly.

---

### Example 2: Validate hyperparameter grid
```
/ml-review
Is this Random Forest hyperparameter grid appropriate? How many combinations will be tested?

grid = {
    "n_estimators": [50, 100, 200, 500],
    "max_depth": [3, 5, 7, 10, 15],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}
```

**Expected Feedback:** 4×5×3×4 = 240 combinations. Recommend RandomizedSearchCV with n_iter=50. Compare to DEFAULT_PARAM_GRIDS in config (smaller, validated set).

---

### Example 3: Check for data leakage
```
/ml-review
I'm scaling my features in one step before training. Any issues?

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Full data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y)
model.fit(X_train, y_train)
```

**Expected Feedback:** CRITICAL ISSUE — data leakage. Scaler sees entire dataset (leaks test statistics). Fix: fit scaler on X_train only.

---

## Commands Reference

| Command | Purpose |
|---------|---------|
| `/ml-review [description]` | Standard comprehensive review |
| `/ml-review check [function]` | Quick checklist-only review (no detailed feedback) |
| `/ml-review validate-grid [grid definition]` | Specific hyperparameter grid review |
| `/ml-review detect-leakage [code snippet]` | Focus on data leakage risks |
| `/ml-review metrics [problem_type, predictions]` | Validate metric calculation |

---

**Last updated:** March 19, 2026  
**Project:** ML Studio (Webapp_Processus_LM)
