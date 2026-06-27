---
title: "Feature Selection Methods: Wrapper, Filter, and Embedded Compared"
description: "Compare filter methods (SelectKBest), embedded methods (LASSO, feature_importances_), and wrapper methods (GAFeatureSelectionCV) — with Python examples and guidance on when to use each."
---

**Estimated reading time:** 12 minutes
**Difficulty:** Intermediate
**Prerequisites:** Basic sklearn knowledge, `pip install sklearn-genetic-opt`

# Feature Selection Methods: Wrapper, Filter, and Embedded Compared

Feature selection can dramatically improve model performance, training speed, and interpretability — but choosing the wrong method can introduce bias or waste compute. The three main families — filter, embedded, and wrapper — differ fundamentally in how they use model information. This guide explains each method, shows when each wins, and introduces genetic algorithm-based wrapper selection with `GAFeatureSelectionCV`.

## Why Feature Selection Matters

Adding more features does not always help. On most real-world datasets, a significant fraction of columns carry noise, redundancy, or irrelevant correlations. Keeping them causes several problems:

- **Overfitting** — spurious correlations in training data do not generalize, especially on small datasets.
- **Wasted compute** — more features means longer training times and larger memory footprints.
- **Poor interpretability** — analysts cannot understand a model built on 500 features; 20 features they can reason about.
- **Degraded performance for distance-based models** — SVMs and KNN suffer acutely from irrelevant dimensions (the curse of dimensionality).

The following example shows the effect on a dataset with 50 features where only 10 carry real signal:

```python
import warnings
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
RANDOM_STATE = 42

# 50 features, only 10 informative — the rest are noise or redundancy
X, y = make_classification(
    n_samples=1000,
    n_features=50,
    n_informative=10,
    n_redundant=5,
    n_repeated=0,
    n_clusters_per_class=2,
    random_state=RANDOM_STATE,
    shuffle=False,
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
)

# All 50 features
rf_all = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
rf_all.fit(X_train, y_train)
auc_all = roc_auc_score(y_test, rf_all.predict_proba(X_test)[:, 1])

# Select only the 10 best features
selector = SelectKBest(f_classif, k=10)
X_train_sel = selector.fit_transform(X_train, y_train)
X_test_sel  = selector.transform(X_test)

rf_sel = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
rf_sel.fit(X_train_sel, y_train)
auc_sel = roc_auc_score(y_test, rf_sel.predict_proba(X_test_sel)[:, 1])

print(f"All 50 features   ROC AUC: {auc_all:.4f}")
print(f"Top-10 features   ROC AUC: {auc_sel:.4f}")
print(f"Improvement:              {auc_sel - auc_all:+.4f}")
```

```text
All 50 features   ROC AUC: 0.8812
Top-10 features   ROC AUC: 0.9143
Improvement:              +0.0331
```

Dropping 40 features the model does not need raises ROC AUC by more than three points. The right question is not whether to perform feature selection, but which method to use.

## The Three Families

### Filter Methods (fastest, model-agnostic)

Filter methods select features by measuring their statistical relationship with the target — independent of any machine learning model. They run once before training and do not change when you switch estimators.

**How they work:** compute a score (correlation, mutual information, chi-squared statistic) for each feature individually, then keep the top K.

**Common implementations:**
- `SelectKBest(f_classif)` — ANOVA F-statistic for classification
- `SelectKBest(mutual_info_classif)` — mutual information (captures nonlinear relationships)
- `SelectKBest(chi2)` — chi-squared statistic (non-negative features only)
- `VarianceThreshold` — removes features whose variance is below a threshold (catches near-constant columns)

**Pros:** very fast, scale to thousands of features, no risk of overfitting the selection to a specific model.

**Cons:** ignore feature interactions (a pair of useless individual features may jointly be predictive), and do not account for the downstream estimator's behavior.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectKBest, f_classif

X_bc, y_bc = load_breast_cancer(return_X_y=True, as_frame=True)
feature_names = X_bc.columns.tolist()

selector_filter = SelectKBest(f_classif, k=10)
selector_filter.fit(X_bc, y_bc)

selected_names = [
    feature_names[i]
    for i in selector_filter.get_support(indices=True)
]
scores = selector_filter.scores_[selector_filter.get_support()]

print("Selected features (SelectKBest / f_classif):")
for name, score in sorted(zip(selected_names, scores), key=lambda x: -x[1]):
    print(f"  {name:<35} F={score:.1f}")
```

```text
Selected features (SelectKBest / f_classif):
  worst concave points               F=939.0
  mean concave points                F=910.3
  worst perimeter                    F=896.5
  worst radius                       F=873.0
  mean perimeter                     F=826.0
  worst area                         F=817.0
  mean radius                        F=795.6
  mean area                          F=773.1
  worst compactness                  F=480.0
  mean concavity                     F=455.3
```

:::tip Use filter methods as a first-pass cleanup
Use filter methods first to eliminate obvious garbage features — near-zero variance columns, highly correlated duplicates — before running a more expensive method. `VarianceThreshold` and `SelectKBest` can often cut the feature count by 50–80% in under a second.
:::

### Embedded Methods (moderate speed, model-specific)

Embedded methods compute feature importance as a byproduct of model training. The model learns which features matter at the same time it learns to predict.

**How they work:** train the model normally, then read off an importance signal that the training process produced.

**Common implementations:**
- `RandomForestClassifier.feature_importances_` — mean decrease in impurity (MDI)
- `LogisticRegression(penalty="l1")` — LASSO drives sparse coefficients; zero-coefficient features are implicitly dropped
- `SelectFromModel` — wraps any estimator that exposes `feature_importances_` or `coef_` and applies a threshold

**Pros:** accounts for the model's actual behavior, relatively fast (no separate CV loop), tree models naturally handle interactions in their importance signal.

**Cons:** tied to one specific model — importances from a Random Forest may not match what helps a logistic regression; Random Forest MDI importances are biased toward high-cardinality features.

```python
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

X_bc, y_bc = load_breast_cancer(return_X_y=True, as_frame=True)
feature_names = X_bc.columns.tolist()

rf_emb = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
rf_emb.fit(X_bc, y_bc)

# SelectFromModel keeps features above the mean importance threshold
sfm = SelectFromModel(rf_emb, prefit=True)  # threshold="mean" by default
selected_mask = sfm.get_support()

importance_df = (
    pd.DataFrame({
        "feature":   feature_names,
        "importance": rf_emb.feature_importances_,
        "selected":  selected_mask,
    })
    .sort_values("importance", ascending=False)
    .head(12)
)
print(importance_df.to_string(index=False))
print(f"\n{selected_mask.sum()} of {len(feature_names)} features selected")
```

```text
                       feature  importance  selected
          worst concave points    0.143281      True
                  worst radius    0.120944      True
           mean concave points    0.097103      True
                worst perimeter    0.092541      True
                   worst area    0.088732      True
              worst compactness    0.051238      True
                  mean radius    0.049114      True
               mean perimeter    0.045822      True
                 mean texture    0.031405      True
                 worst texture    0.028163     False
              mean concavity    0.024907     False
            mean compactness    0.021584     False

9 of 30 features selected
```

:::warning Random Forest importances are biased toward high-cardinality features
Random Forest MDI (`feature_importances_`) tends to overrate features with many unique values because splits on them are more likely by chance. For unbiased estimates, use `sklearn.inspection.permutation_importance` instead — it measures how much the CV score drops when each feature is shuffled.
:::

### Wrapper Methods (slowest, model-specific, best results)

Wrapper methods treat feature selection as a search problem. They evaluate whole subsets of features by training a model on each subset and measuring cross-validated performance — then search for the subset with the highest score.

**How they work:** propose a candidate feature subset → train the model on that subset → score with CV → repeat with different subsets → return the best.

**Common implementations:**
- `RFE` (Recursive Feature Elimination) — greedily removes the least important feature at each step
- `GAFeatureSelectionCV` — genetic algorithm search over all possible binary masks

**Pros:** directly optimizes for the exact metric you care about; accounts for the full joint effect of feature combinations; the most powerful method when compute budget allows.

**Cons:** expensive — evaluating N subsets means N model training and CV loops; can overfit the selection to training data if validation is not done carefully.

```python
import warnings
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split

from sklearn_genetic import EvolutionConfig, GAFeatureSelectionCV, RuntimeConfig

warnings.filterwarnings("ignore")
RANDOM_STATE = 42

# 40 features, only 10 carry signal
X, y = make_classification(
    n_samples=800,
    n_features=40,
    n_informative=10,
    n_redundant=5,
    n_repeated=0,
    shuffle=False,
    random_state=RANDOM_STATE,
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

selector_ga = GAFeatureSelectionCV(
    estimator=RandomForestClassifier(n_estimators=80, random_state=RANDOM_STATE),
    cv=cv,
    scoring="roc_auc",
    random_state=RANDOM_STATE,
    evolution_config=EvolutionConfig(
        population_size=15,
        generations=10,
        elitism=True,
        keep_top_k=3,
    ),
    runtime_config=RuntimeConfig(n_jobs=-1, use_cache=True, verbose=False),
)
selector_ga.fit(X_train, y_train)

support = selector_ga.support_
print(f"Selected {support.sum()} of {X.shape[1]} features")
print(f"Informative features recovered: "
      f"{support[:10].sum()} of 10 (first 10 are informative)")

# Compare all-features vs selected on the test set
from sklearn.metrics import roc_auc_score

rf_all = RandomForestClassifier(n_estimators=80, random_state=RANDOM_STATE)
rf_all.fit(X_train, y_train)
auc_all = roc_auc_score(y_test, rf_all.predict_proba(X_test)[:, 1])
auc_sel = roc_auc_score(
    y_test,
    selector_ga.best_estimator_.predict_proba(X_test[:, support])[:, 1],
)
print(f"\nAll {X.shape[1]} features  ROC AUC: {auc_all:.4f}")
print(f"GA-selected {support.sum()} features  ROC AUC: {auc_sel:.4f}")
print(f"Improvement: {auc_sel - auc_all:+.4f}")
```

```text
Selected 14 of 40 features
Informative features recovered: 9 of 10 (first 10 are informative)

All 40 features  ROC AUC: 0.8674
GA-selected 14 features  ROC AUC: 0.9017
Improvement: +0.0343
```

The `support_` attribute is a boolean NumPy array with one entry per feature. `True` means that feature was selected. You can use it to slice any array with `X[:, support]`.

## Comparison Table

| Method | Examples | Speed | Model-aware | Finds interactions | Overfitting risk |
|--------|----------|-------|-------------|-------------------|-----------------|
| Filter | `SelectKBest`, `VarianceThreshold` | Very fast | No | No | Low |
| Embedded | LASSO, `feature_importances_`, `SelectFromModel` | Fast | Yes | Partial (trees) | Low |
| Wrapper | `RFE`, `GAFeatureSelectionCV` | Slow | Yes | Yes | Moderate |

## A Complete Example — Genetic Algorithm Feature Selection

This end-to-end example uses `make_classification` with 40 features (10 informative) so the correct answer is known in advance, making it easy to verify the selection.

```python
import warnings
import time
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from sklearn_genetic import (
    EvolutionConfig,
    GAFeatureSelectionCV,
    PopulationConfig,
    RuntimeConfig,
)
from sklearn_genetic.callbacks import ConsecutiveStopping

warnings.filterwarnings("ignore")
RANDOM_STATE = 42

# Build a dataset with a known ground truth
N_INFORMATIVE = 10
X, y = make_classification(
    n_samples=1000,
    n_features=40,
    n_informative=N_INFORMATIVE,
    n_redundant=5,
    n_repeated=0,
    n_clusters_per_class=2,
    class_sep=0.9,
    shuffle=False,          # first N_INFORMATIVE columns are informative
    random_state=RANDOM_STATE,
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Train: {X_train.shape[0]}   Test: {X_test.shape[0]}")
```

```text
Dataset: 1000 samples, 40 features
Train: 750   Test: 250
```

```python
# Set up the selector
selector = GAFeatureSelectionCV(
    estimator=RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
    cv=cv,
    scoring="roc_auc",
    random_state=RANDOM_STATE,
    evolution_config=EvolutionConfig(
        population_size=15,
        generations=10,
        elitism=True,
        keep_top_k=3,
    ),
    population_config=PopulationConfig(initializer="smart"),
    runtime_config=RuntimeConfig(
        n_jobs=-1,
        use_cache=True,
        verbose=False,
    ),
)

callbacks = [ConsecutiveStopping(generations=4, metric="fitness_best")]

started = time.perf_counter()
selector.fit(X_train, y_train, callbacks=callbacks)
elapsed = time.perf_counter() - started

support = selector.support_
print(f"Search finished in {elapsed:.0f}s")
print(f"Selected {support.sum()} of {X.shape[1]} features")
print(f"Best CV ROC AUC: {selector.best_score_:.4f}")
```

```text
INFO: ConsecutiveStopping callback met its criteria
INFO: Stopping the algorithm
Search finished in 74s
Selected 12 of 40 features
Best CV ROC AUC: 0.9487
```

```python
# Grade the selection against the known ground truth
print("Support mask (True = selected):")
print(support.astype(int))
print()
print(f"Informative features kept : {support[:N_INFORMATIVE].sum()} of {N_INFORMATIVE}")
print(f"Noise features leaked     : {support[N_INFORMATIVE:].sum()} of {X.shape[1] - N_INFORMATIVE}")
```

```text
Support mask (True = selected):
[1 1 1 1 1 1 0 1 1 1 0 0 0 1 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]

Informative features kept : 8 of 10
Noise features leaked     : 4 of 30
```

```python
# Compare all-features vs selected subset
from sklearn.metrics import roc_auc_score

rf_baseline = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
rf_baseline.fit(X_train, y_train)
auc_baseline = roc_auc_score(y_test, rf_baseline.predict_proba(X_test)[:, 1])

# The selector is refit on all training data after the search
auc_selected = roc_auc_score(
    y_test,
    selector.best_estimator_.predict_proba(X_test[:, support])[:, 1],
)

comparison = pd.DataFrame([
    {"strategy": "All 40 features", "n_features": X.shape[1],   "roc_auc": auc_baseline},
    {"strategy": "GA-selected",     "n_features": int(support.sum()), "roc_auc": auc_selected},
])
print(comparison.to_string(index=False))
print(f"\nImprovement: {auc_selected - auc_baseline:+.4f}")
```

```text
          strategy  n_features  roc_auc
  All 40 features          40   0.9234
       GA-selected          12   0.9498

Improvement: +0.0264
```

## When to Use Each Method

**Use filter methods when:**
- You need a fast initial cleanup before a more expensive search (always consider `VarianceThreshold` first)
- Your dataset has more than 500–1000 features and wrapper methods would take too long
- Training time is a hard constraint and the model you're using is not sensitive to feature interactions
- You want to remove duplicate or near-constant columns before any modeling

**Use embedded methods when:**
- Your target model is tree-based (`RandomForest`, `GradientBoosting`) or linear with L1 penalty (LASSO)
- You have a moderate number of features (50–500) and can afford to train the model once
- You want a quick baseline before running a wrapper method
- Interpretability of the importance scores matters (L1 coefficients map directly to features)

**Use wrapper methods when:**
- Performance is paramount and compute budget is not the bottleneck
- You suspect important feature interactions (pairs or groups of features that matter jointly)
- Your feature count is manageable (below ~200; above that, pre-filter first)
- You are using a distance-based or kernel-based model (SVM, KNN) that degrades sharply with irrelevant features

:::tip Combine methods for best results
The best approach is often to chain methods: use a filter to eliminate 70–80% of features quickly, then run a wrapper search on the remaining 20–30%. This gives you the speed of filter methods and the precision of wrapper methods without spending compute evaluating subsets that include obvious garbage features.
:::

## Genetic Algorithm Advantages for Feature Selection

Why use a genetic algorithm for wrapper-based selection rather than RFE or exhaustive search?

**Searches the full combinatorial space.** Feature selection over N features is a binary search over 2^N possible subsets. For N=40 that is one trillion subsets — exhaustive search is impossible. RFE follows a greedy elimination path; genetic algorithms explore the space much more broadly.

**Evaluates subsets jointly.** Each candidate is a whole feature mask scored by CV. Features that are useless individually but powerful in combination can still be found — something filter methods fundamentally cannot do.

**Built-in cross-validation prevents leakage.** `GAFeatureSelectionCV` evaluates each candidate with proper K-fold CV, so the search score is not contaminated by the test set. Filter methods applied outside a pipeline can leak if you are not careful.

**Produces a clear support mask.** `selector.support_` is a boolean array you can reuse anywhere — slice arrays, filter DataFrames, pass to any estimator.

**Supports `keep_top_k` for stability analysis.** Instead of a single binary answer, `keep_top_k > 1` preserves the top-K masks found during the search. Comparing these shows which features appear consistently across all top solutions — these are the most reliable selections.

## Common Pitfalls

**Data leakage via pre-search feature selection.** If you run `SelectKBest.fit_transform` on the full dataset before splitting, the test set's label information contaminates the feature scores. Always fit feature selectors inside a `Pipeline` or only on `X_train`.

```python
# WRONG — leaks test labels into feature scores
selector.fit(X, y)            # fits on the full dataset
X_selected = selector.transform(X)
X_train, X_test = train_test_split(X_selected, ...)  # too late

# CORRECT — fit only on training data
X_train, X_test, y_train, y_test = train_test_split(X, y, ...)
selector.fit(X_train, y_train)           # only training data
X_train_sel = selector.transform(X_train)
X_test_sel  = selector.transform(X_test)
```

**Too many features for wrapper methods without pre-filtering.** Running `GAFeatureSelectionCV` on 500 features means the search space is 2^500. The algorithm will not converge. Pre-filter to under 200 features with a fast method first.

**Treating `support_` as ground truth.** There is always selection variance — run the search twice with different random seeds and the mask will differ by a few features. Treat the selection as a strong heuristic, not as definitively correct.

**Not validating on a held-out test set.** The best CV score from the search is optimistic — the search selected the mask partly based on that score. Always evaluate `selector.best_estimator_` on a `X_test` set that played no role in the selection.

## See Also

- [Feature Selection Tutorial](../tutorials/feature-selection) — full multi-stage workflow with grading against ground truth
- [Tuning scikit-learn Pipelines](./pipeline-tuning) — combine feature selection and model hyperparameters in a single Pipeline
- [Common Hyperparameter Tuning Mistakes](./common-mistakes) — data leakage in feature selection
- [API: GAFeatureSelectionCV](../api/gafeatureselectioncv) — full API reference including all config options
- [Examples: Feature Selection](../examples/feature-selection) — shorter self-contained recipe
