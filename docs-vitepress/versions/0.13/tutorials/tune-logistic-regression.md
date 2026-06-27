---
title: "Logistic Regression Hyperparameter Tuning with scikit-learn and Genetic Algorithms"
description: "Tune LogisticRegression hyperparameters — C, solver, penalty, max_iter — using GASearchCV, with a comparison to GridSearchCV and practical guidance on solver–penalty compatibility."
---

**Estimated reading time:** 10 minutes
**Difficulty:** Beginner–Intermediate
**Prerequisites:** `pip install sklearn-genetic-opt`

# Logistic Regression Hyperparameter Tuning with scikit-learn and Genetic Algorithms

`LogisticRegression` has fewer hyperparameters than tree-based models, but the interactions between `solver`, `penalty`, and `C` still matter. Getting them wrong leads to convergence warnings, suboptimal regularization, and slow training. This tutorial covers which parameters matter, their recommended ranges, and a full genetic search example — including a side-by-side comparison with `GridSearchCV`.

## Which Hyperparameters Matter in Logistic Regression?

Logistic Regression's tunable surface is small, but each parameter has a meaningful effect.

| Hyperparameter | Type | Default | Recommended Range | Why it matters |
|---|---|---|---|---|
| `C` | float | `1.0` | `Continuous(1e-3, 1e3, distribution="log-uniform")` | Inverse of regularization strength. Smaller = stronger regularization. Critical for high-dimensional or noisy feature sets. |
| `penalty` | str | `"l2"` | `Categorical(["l1", "l2", "elasticnet"])` | `l1` drives sparse solutions; `l2` is standard ridge; `elasticnet` mixes both. Not all solvers support all penalties. |
| `solver` | str | `"lbfgs"` | `Categorical(["lbfgs", "liblinear", "saga"])` | Determines the optimization algorithm. Compatibility with `penalty` is a hard constraint — see the section below. |
| `max_iter` | int | `100` | `Integer(100, 2000)` | Maximum optimization iterations. Low values on large or poorly-scaled datasets trigger `ConvergenceWarning`. |
| `l1_ratio` | float | `None` | `Continuous(0.0, 1.0)` | Elastic-net mixing ratio. Only active when `penalty="elasticnet"`. Has no effect for other penalties. |
| `class_weight` | str/None | `None` | `Categorical([None, "balanced"])` | Upweights the minority class by its inverse frequency. Almost always helpful on imbalanced datasets. |

The two parameters with the biggest impact on most datasets are **`C`** and **`penalty`**. `max_iter` matters mainly to avoid convergence failures; the others provide finer control.

## Solver–Penalty Compatibility

:::danger Incompatible solver + penalty combinations raise ValueError
Combining a solver with a penalty it does not support will raise `ValueError` at fit time — not at `GASearchCV` construction time. The search will crash mid-run if you let the algorithm freely combine incompatible values. Always constrain solver and penalty together.
:::

This is the most common mistake when tuning `LogisticRegression`. Different solvers implement different regularization algorithms:

| Solver | Supported penalties |
|---|---|
| `lbfgs` | `l2`, `None` |
| `liblinear` | `l1`, `l2` |
| `saga` | `l1`, `l2`, `elasticnet`, `None` |
| `newton-cg` | `l2`, `None` |

**The practical recommendation:** if you want to tune `penalty` as a hyperparameter — especially to include `l1` or `elasticnet` — fix `solver="saga"` in the estimator constructor and exclude `solver` from the search space. `saga` is the only solver that supports the full penalty range.

```python
# Safe: fix solver="saga", search over penalty freely
from sklearn.linear_model import LogisticRegression

estimator = LogisticRegression(solver="saga", random_state=42)
```

If you only want to tune `C` and `max_iter` (keeping `penalty="l2"`), any solver works and you can include `solver` in the search.

## Recommended Search Spaces

### Option 1 — Simple (l2 only, any solver)

Use this when you want a fast, safe search over regularization strength and iteration budget. No solver compatibility issues arise because `l2` is supported everywhere.

```python
from sklearn_genetic.space import Categorical, Continuous, Integer

param_grid = {
    "C":            Continuous(1e-3, 1e3, distribution="log-uniform"),
    "max_iter":     Integer(100, 1000),
    "class_weight": Categorical([None, "balanced"]),
}
```

### Option 2 — Full (multi-penalty with saga)

Use this when you want the genetic algorithm to discover whether sparsity (`l1`) or mixed regularization (`elasticnet`) outperforms the standard `l2`. Fix `solver="saga"` in the constructor to avoid compatibility errors.

```python
from sklearn_genetic.space import Categorical, Continuous, Integer

param_grid = {
    "C":        Continuous(1e-3, 1e3, distribution="log-uniform"),
    "penalty":  Categorical(["l1", "l2", "elasticnet"]),
    "l1_ratio": Continuous(0.0, 1.0),   # only active when penalty="elasticnet"
    "max_iter": Integer(200, 2000),
}
```

:::tip l1_ratio is always sampled, but only acts for elasticnet
The genetic algorithm will sample `l1_ratio` for every candidate regardless of `penalty`. When `penalty != "elasticnet"`, `l1_ratio` has no effect — sklearn ignores it. This is fine: the search discovers that only `elasticnet` candidates are sensitive to `l1_ratio` and adjusts accordingly.
:::

## Step 1 — Establish a Baseline

Train `LogisticRegression` with its defaults and record test-set ROC-AUC. This is the number to beat.

```python
import warnings
import time

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

warnings.filterwarnings("ignore")
RANDOM_STATE = 42

# 569 samples, 30 features, binary target (malignant / benign)
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Train: {X_train.shape[0]}   Test: {X_test.shape[0]}")
print(f"Class balance (train): {y_train.mean():.2%} positive")
```

```text
Dataset: 569 samples, 30 features
Train: 426   Test: 143
Class balance (train): 62.68% positive
```

```python
def evaluate(name, estimator):
    """Return a metrics dict for a fitted estimator."""
    proba = estimator.predict_proba(X_test)[:, 1]
    pred  = estimator.predict(X_test)
    return {
        "model":             name,
        "accuracy":          round(accuracy_score(y_test, pred), 4),
        "balanced_accuracy": round(balanced_accuracy_score(y_test, pred), 4),
        "roc_auc":           round(roc_auc_score(y_test, proba), 4),
    }


# Default LogisticRegression — solver="lbfgs", penalty="l2", C=1.0, max_iter=100
baseline = LogisticRegression(random_state=RANDOM_STATE)
baseline.fit(X_train, y_train)
baseline_metrics = evaluate("LR defaults", baseline)
print(baseline_metrics)
```

```text
{'model': 'LR defaults', 'accuracy': 0.958, 'balanced_accuracy': 0.9524, 'roc_auc': 0.9941}
```

:::info ConvergenceWarning on unscaled data
If you run this with `warnings.filterwarnings("default")` and skip feature scaling, sklearn often emits `ConvergenceWarning: lbfgs failed to converge`. On `breast_cancer` features range over very different scales (e.g. area vs fractal dimension). Scaling with `StandardScaler` — or raising `max_iter` — resolves this. The genetic search will discover that higher `max_iter` values improve reliability.
:::

## Step 2 — Run the Genetic Search

We use the simple search space (Option 1: `C`, `max_iter`, `class_weight`) with `penalty="l2"` fixed in the estimator so any solver is safe.

```python
import time

from sklearn_genetic import (
    EvolutionConfig,
    GASearchCV,
    PopulationConfig,
    RuntimeConfig,
)
from sklearn_genetic.callbacks import ConsecutiveStopping
from sklearn_genetic.space import Categorical, Continuous, Integer

param_grid = {
    "C":            Continuous(1e-3, 1e3, distribution="log-uniform"),
    "max_iter":     Integer(100, 1000),
    "class_weight": Categorical([None, "balanced"]),
}

ga_search = GASearchCV(
    estimator=LogisticRegression(
        penalty="l2",
        solver="lbfgs",
        random_state=RANDOM_STATE,
    ),
    random_state=RANDOM_STATE,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=cv,
    evolution_config=EvolutionConfig(
        population_size=15,
        generations=10,
        elitism=True,
        keep_top_k=3,
    ),
    population_config=PopulationConfig(
        initializer="smart",
        warm_start_configs=[{
            "C": 1.0,
            "max_iter": 100,
            "class_weight": None,
        }],
    ),
    runtime_config=RuntimeConfig(
        n_jobs=-1,
        parallel_backend="population",
        use_cache=True,
        verbose=False,
    ),
)

callbacks = [ConsecutiveStopping(generations=4, metric="fitness_best")]

started = time.perf_counter()
ga_search.fit(X_train, y_train, callbacks=callbacks)
elapsed = time.perf_counter() - started

print(f"Best CV ROC AUC : {ga_search.best_score_:.4f}   (search took {elapsed:.0f}s)")
print("Best parameters :")
for key, value in ga_search.best_params_.items():
    print(f"  {key}: {value}")
```

```text
INFO: ConsecutiveStopping callback met its criteria
INFO: Stopping the algorithm
Best CV ROC AUC : 0.9959   (search took 18s)
Best parameters :
  C: 12.347
  max_iter: 632
  class_weight: None
```

:::tip Logistic Regression is fast — use larger populations
Each `LogisticRegression.fit()` call typically takes milliseconds on tabular data. This means you can afford a larger `population_size` (20–30) and more `generations` (15–25) than you would with a tree ensemble. The overall search time stays short while coverage of the parameter space improves.
:::

## Step 3 — Compare with GridSearchCV

To give the genetic search a fair comparison, we run `GridSearchCV` with a small but reasonable grid over the same parameters — capping the total number of candidates to a similar budget.

```python
from sklearn.model_selection import GridSearchCV

grid_param = {
    "C":            [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
    "max_iter":     [100, 300, 600, 1000],
    "class_weight": [None, "balanced"],
}

grid_search = GridSearchCV(
    LogisticRegression(penalty="l2", solver="lbfgs", random_state=RANDOM_STATE),
    param_grid=grid_param,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1,
    refit=True,
)

started_grid = time.perf_counter()
grid_search.fit(X_train, y_train)
elapsed_grid = time.perf_counter() - started_grid

print(f"GridSearchCV best CV ROC AUC : {grid_search.best_score_:.4f}  "
      f"({len(grid_search.cv_results_['params'])} candidates, {elapsed_grid:.0f}s)")
print("Best parameters:")
for key, value in grid_search.best_params_.items():
    print(f"  {key}: {value}")
```

```text
GridSearchCV best CV ROC AUC : 0.9953  (56 candidates, 12s)
Best parameters:
  C: 10.0
  max_iter: 100
  class_weight: None
```

```python
# Collect test-set results for all three methods
ga_metrics   = evaluate("GASearchCV (tuned)", ga_search)
grid_metrics = evaluate("GridSearchCV",       grid_search)

comparison = pd.DataFrame([baseline_metrics, grid_metrics, ga_metrics])
comparison["best_cv_auc"] = [
    None,
    round(grid_search.best_score_, 4),
    round(ga_search.best_score_, 4),
]
comparison["n_candidates"] = [
    None,
    len(grid_search.cv_results_["params"]),
    ga_search.fit_stats_["unique_candidates"],
]
print(comparison.to_string(index=False))
```

```text
              model  accuracy  balanced_accuracy  roc_auc  best_cv_auc  n_candidates
        LR defaults     0.958            0.9524   0.9941          NaN           NaN
       GridSearchCV     0.972            0.9637   0.9957       0.9953          56.0
 GASearchCV (tuned)     0.972            0.9637   0.9958       0.9959          42.0
```

The genetic search matches GridSearchCV's test-set performance with fewer candidates evaluated. On this small, well-behaved search space the two methods are broadly equivalent — the genetic algorithm's advantage grows with the number of interacting parameters and the size of the search space.

## Practical Notes

**Always set `max_iter` high enough.** The default of 100 is frequently too low, especially on unscaled data. Setting it to at least 500–1000 prevents `ConvergenceWarning` from corrupting the CV scores. If you cannot scale features, raise the lower bound of the `Integer` range.

**Fix `solver="saga"` when tuning `penalty`.** If you include `penalty` in the search space, hard-code `solver="saga"` in the estimator constructor. This is the only solver that supports all penalty types. Attempting to search over solver + penalty jointly without constraints will produce `ValueError` at fit time.

**`l1_ratio` has no effect outside `elasticnet`.** When `penalty != "elasticnet"`, sklearn silently ignores `l1_ratio`. The genetic algorithm will still sample it for non-elasticnet candidates — this is harmless but means `l1_ratio` is effectively a free dimension for those candidates. If this bothers you, reduce the problem to a two-stage search: first find the best `penalty`, then tune `l1_ratio` only if `elasticnet` wins.

**`class_weight="balanced"` on imbalanced data.** If your positive-class rate is below 20%, always include `Categorical([None, "balanced"])` in the search space — and almost always the search will find `"balanced"` is better. See [Hyperparameter Tuning for Imbalanced Datasets](./imbalanced-classification) for a full example.

## When Should You Use Logistic Regression?

Logistic Regression is a strong first choice in several situations:

- **Linear decision boundary problems** — text classification, sentiment, many biomedical datasets
- **High-dimensional sparse features** — TF-IDF vectors, bag-of-words, one-hot encoded categorical data
- **Probability calibration required** — logistic regression outputs well-calibrated probabilities by default
- **Fast inference at scale** — a single dot product per sample; far faster than any ensemble at predict time
- **Interpretability** — coefficients map directly to feature importance; regularization (via `C`) controls stability

## When Is Logistic Regression Not Enough?

Switch to a tree-based model when:

- **Non-linear relationships** dominate — decision boundaries that cannot be approximated by a hyperplane
- **Interacting features** — effects that only appear in combination (e.g. `feature_a AND feature_b > threshold`)
- **Mixed feature types** — numerical and ordinal features with irregular distributions that trees handle natively
- **Many missing values** — `HistGradientBoostingClassifier` handles `NaN` natively; logistic regression requires imputation

In those situations, see [Random Forest Hyperparameter Tuning](./tune-random-forest) or [Gradient Boosting Hyperparameter Tuning](./tune-gradient-boosting).

## See Also

- [Random Forest Hyperparameter Tuning](./tune-random-forest)
- [Gradient Boosting Hyperparameter Tuning](./tune-gradient-boosting)
- [XGBoost Hyperparameter Tuning](./tune-xgboost)
- [Hyperparameter Tuning for Imbalanced Datasets](./imbalanced-classification)
- [Choosing the Right Search Space](../guide/choosing-search-spaces)
- [Grid Search vs Genetic Algorithms](../comparisons/grid-search-vs-genetic-algorithms)
