---
title: "Choosing the Right Search Space for Hyperparameter Tuning"
description: "How to define Integer, Continuous, and Categorical search spaces for common scikit-learn estimators, with recommended bounds and log-uniform guidance."
---

:::warning Development version
This is the **latest (dev)** documentation. For stable docs, see [stable](/stable/).
:::

# Choosing the Right Search Space for Hyperparameter Tuning

The search space you define is one of the most important decisions in hyperparameter optimization. The algorithm can only find good configurations that exist within the bounds you set. This guide explains how to choose bounds that are wide enough to contain the optimum, narrow enough to not waste budget, and correctly distributed so the algorithm samples the space evenly.

**Estimated reading time:** 8 min &nbsp;|&nbsp; **Difficulty:** Beginner

:::info Prerequisites
- Basic familiarity with scikit-learn estimators
- `sklearn-genetic-opt` installed (`pip install sklearn-genetic-opt`)
:::

---

## Why Search Space Choice Matters

The search space has three ways to fail silently:

**Bounds too wide** — the algorithm spends most of its budget evaluating configurations in bad regions. A genetic algorithm that wastes its first ten generations on `learning_rate=0.99` will take many more generations to recover.

**Bounds too narrow** — the true optimum is outside the range. The algorithm finds the best configuration *within the bounds*, but that may be far from the global optimum. You will never know from the search results alone that you constrained the space too tightly.

**Wrong distribution** — using a uniform distribution on a parameter that spans orders of magnitude means most samples land near the upper bound. Learning rates from `0.0001` to `1.0` sampled uniformly give 99% of samples in the range `0.01–1.0` and almost nothing below `0.01`, even though that low end is often where the best learning rates live.

All three problems are avoidable with sensible defaults and a review of `plot_search_space` after the search.

---

## The Three Parameter Types

`sklearn-genetic-opt` provides three dimension types, each matching a different kind of hyperparameter:

```python
from sklearn_genetic.space import Categorical, Continuous, Integer

param_grid = {
    # Integer: whole numbers — tree depth, number of estimators
    "n_estimators": Integer(50, 300),

    # Continuous: floats — learning rate, regularization strength
    "learning_rate": Continuous(0.01, 0.3, distribution="log-uniform"),

    # Categorical: named choices — solver, kernel, activation
    "max_features": Categorical(["sqrt", "log2"]),
}
```

**`Integer(lower, upper)`**

Samples whole numbers from `lower` to `upper`, both inclusive. Use this for parameters that are inherently discrete: `n_estimators`, `max_depth`, `min_samples_leaf`, `n_neighbors`, `max_iter`.

**`Continuous(lower, upper)`**

Samples floating-point values from `lower` to `upper`. Use this for parameters that take real values: `learning_rate`, `alpha`, `C`, `subsample`, `dropout_rate`. Optionally set `distribution="log-uniform"` (see below).

**`Categorical(choices)`**

Samples from a fixed list. Use this for parameters that are inherently named or discrete without a natural ordering: `"solver"`, `"kernel"`, `"activation"`, `"max_features"` (when mixing `None`, `"sqrt"`, `"log2"`).

```python
# Runnable example combining all three
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split

from sklearn_genetic import EvolutionConfig, GASearchCV, PopulationConfig, RuntimeConfig
from sklearn_genetic.space import Categorical, Continuous, Integer

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

param_grid = {
    "n_estimators":    Integer(50, 300),              # Integer
    "ccp_alpha":       Continuous(0.0, 0.02),         # Continuous (uniform)
    "max_features":    Categorical(["sqrt", "log2"]), # Categorical
}

search = GASearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=cv,
    scoring="roc_auc",
    evolution_config=EvolutionConfig(population_size=12, generations=6),
    population_config=PopulationConfig(initializer="smart"),
    runtime_config=RuntimeConfig(n_jobs=-1, verbose=False),
    random_state=42,
)
search.fit(X_train, y_train)
print(f"Best CV ROC-AUC : {search.best_score_:.4f}")
print(f"Best params     : {search.best_params_}")
```

```text
Best CV ROC-AUC : 0.9963
Best params     : {'n_estimators': 247, 'ccp_alpha': 0.0012, 'max_features': 'sqrt'}
```

---

## When to Use log-uniform Distribution

A uniform distribution over `[0.0001, 1.0]` is deceptive. Because most of the interval sits above `0.1`, about 90% of uniform samples land in `[0.1, 1.0]` and only 10% in `[0.0001, 0.1]`. Yet for parameters like learning rates and regularization strengths, the interesting region is often at the low end.

A log-uniform distribution fixes this by sampling each *decade* equally:

| Interval | Share of samples (uniform) | Share of samples (log-uniform) |
|----------|---------------------------|-------------------------------|
| 0.0001 – 0.001 | 0.09% | 25% |
| 0.001 – 0.01 | 0.9% | 25% |
| 0.01 – 0.1 | 9% | 25% |
| 0.1 – 1.0 | 90% | 25% |

```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from sklearn_genetic import EvolutionConfig, GASearchCV, PopulationConfig, RuntimeConfig
from sklearn_genetic.space import Continuous

X, y = load_breast_cancer(return_X_y=True)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Uniform: most samples land near C=1000, missing the optimal region
search_uniform = GASearchCV(
    estimator=LogisticRegression(max_iter=1000, random_state=42),
    param_grid={"C": Continuous(1e-4, 1e3)},   # uniform by default
    cv=cv, scoring="roc_auc",
    evolution_config=EvolutionConfig(population_size=12, generations=5),
    population_config=PopulationConfig(initializer="smart"),
    runtime_config=RuntimeConfig(n_jobs=-1, verbose=False),
    random_state=42,
)
search_uniform.fit(X, y)

# Log-uniform: samples each decade equally
search_log = GASearchCV(
    estimator=LogisticRegression(max_iter=1000, random_state=42),
    param_grid={"C": Continuous(1e-4, 1e3, distribution="log-uniform")},
    cv=cv, scoring="roc_auc",
    evolution_config=EvolutionConfig(population_size=12, generations=5),
    population_config=PopulationConfig(initializer="smart"),
    runtime_config=RuntimeConfig(n_jobs=-1, verbose=False),
    random_state=42,
)
search_log.fit(X, y)

print(f"Uniform  — best C: {search_uniform.best_params_['C']:>10.4f}  ROC-AUC: {search_uniform.best_score_:.4f}")
print(f"Log-unif — best C: {search_log.best_params_['C']:>10.4f}  ROC-AUC: {search_log.best_score_:.4f}")
```

```text
Uniform  — best C:   182.3341  ROC-AUC: 0.9938
Log-unif — best C:     0.7214  ROC-AUC: 0.9973
```

**Use log-uniform for:**

- Learning rates (`learning_rate`, `learning_rate_init`, `eta0`)
- Regularization strengths (`C`, `alpha`, `lambda`, `l1_ratio` when > 0)
- Kernel bandwidth (`gamma`)
- Step sizes and tolerances (`tol`)

**Use uniform for:**

- Ratios and fractions bounded in [0, 1] that don't span orders of magnitude (`subsample`, `colsample_bytree`, `min_child_weight` when close to 1)
- Additive penalties with a known small range (`ccp_alpha` from 0 to 0.05)

:::warning log-uniform requires a strictly positive lower bound
`Continuous(0.0, 1.0, distribution="log-uniform")` is invalid because `log(0)` is undefined. Use a small positive number as the lower bound: `Continuous(1e-6, 1.0, distribution="log-uniform")`.
:::

---

## Recommended Search Spaces by Estimator

The bounds below are starting points based on common practice. Always inspect `plot_search_space` after a search to check whether the optimum is near a boundary — if it is, expand the bound in that direction.

### RandomForestClassifier / RandomForestRegressor

```python
from sklearn_genetic.space import Categorical, Continuous, Integer

rf_param_grid = {
    "n_estimators":     Integer(50, 500),
    "max_depth":        Categorical([None, 5, 10, 15, 20]),
    "min_samples_split": Integer(2, 20),
    "min_samples_leaf": Integer(1, 20),
    "max_features":     Categorical(["sqrt", "log2", None]),
    "ccp_alpha":        Continuous(0.0, 0.02),
}
```

| Parameter | Type | Recommended range | Notes |
|-----------|------|-------------------|-------|
| `n_estimators` | Integer | 50 – 500 | More is rarely worse, just slower |
| `max_depth` | Categorical | `None`, 5, 10, 15, 20 | `None` = unlimited; use Categorical to include it |
| `min_samples_split` | Integer | 2 – 20 | Controls minimum size to split a node |
| `min_samples_leaf` | Integer | 1 – 20 | Higher values smooth the model |
| `max_features` | Categorical | `"sqrt"`, `"log2"`, `None` | `"sqrt"` is the default for classification |
| `ccp_alpha` | Continuous | 0.0 – 0.02 | Cost-complexity pruning; 0 means no pruning |

```python
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split

from sklearn_genetic import EvolutionConfig, GASearchCV, PopulationConfig, RuntimeConfig

X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

rf_search = GASearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=rf_param_grid,
    cv=cv, scoring="accuracy",
    evolution_config=EvolutionConfig(population_size=15, generations=8),
    population_config=PopulationConfig(initializer="smart"),
    runtime_config=RuntimeConfig(n_jobs=-1, verbose=False),
    random_state=42,
)
rf_search.fit(X_train, y_train)
print(f"Best CV accuracy : {rf_search.best_score_:.4f}")
print(f"Best params      : {rf_search.best_params_}")
```

```text
Best CV accuracy : 0.9780
Best params      : {'n_estimators': 392, 'max_depth': None, 'min_samples_split': 3, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'ccp_alpha': 0.0001}
```

---

### GradientBoostingClassifier / HistGradientBoostingClassifier

```python
from sklearn_genetic.space import Categorical, Continuous, Integer

# GradientBoostingClassifier
gb_param_grid = {
    "n_estimators":     Integer(50, 500),
    "learning_rate":    Continuous(0.01, 0.3, distribution="log-uniform"),
    "max_depth":        Integer(2, 8),
    "min_samples_leaf": Integer(5, 50),
    "subsample":        Continuous(0.5, 1.0),
    "max_features":     Continuous(0.3, 1.0),
}

# HistGradientBoostingClassifier (faster, recommended for large datasets)
hgb_param_grid = {
    "max_iter":         Integer(50, 500),
    "learning_rate":    Continuous(0.01, 0.3, distribution="log-uniform"),
    "max_depth":        Integer(2, 10),
    "min_samples_leaf": Integer(5, 100),
    "l2_regularization": Continuous(1e-6, 1.0, distribution="log-uniform"),
    "max_features":     Continuous(0.3, 1.0),
    "max_leaf_nodes":   Integer(15, 127),
}
```

| Parameter | Type | Recommended range | Notes |
|-----------|------|-------------------|-------|
| `n_estimators` / `max_iter` | Integer | 50 – 500 | Tune jointly with `learning_rate` |
| `learning_rate` | Continuous | 0.01 – 0.3 (log-uniform) | Low LR + more trees often wins |
| `max_depth` | Integer | 2 – 8 | Shallow trees are common in boosting |
| `min_samples_leaf` | Integer | 5 – 50 | Regularizes leaf nodes |
| `subsample` | Continuous | 0.5 – 1.0 | Stochastic gradient boosting; < 1.0 adds variance reduction |
| `l2_regularization` | Continuous | 1e-6 – 1.0 (log-uniform) | HistGB only |

:::tip learning_rate and n_estimators interact strongly
A low `learning_rate` needs many estimators to converge; a high `learning_rate` needs fewer. Search them jointly in `param_grid` — do not tune them sequentially. See [Common Mistakes: Mistake 6](./common-mistakes#mistake-6-not-accounting-for-parameter-interactions).
:::

---

### LogisticRegression

```python
from sklearn_genetic.space import Categorical, Continuous, Integer

lr_param_grid = {
    "C":        Continuous(1e-4, 1e3, distribution="log-uniform"),
    "solver":   Categorical(["lbfgs", "saga"]),
    "max_iter": Integer(100, 1000),
}
```

| Parameter | Type | Recommended range | Notes |
|-----------|------|-------------------|-------|
| `C` | Continuous | 1e-4 – 1e3 (log-uniform) | Inverse regularization; high C = less regularization |
| `solver` | Categorical | `"lbfgs"`, `"saga"` | `"lbfgs"` for small datasets; `"saga"` for large or `l1` penalty |
| `max_iter` | Integer | 100 – 1000 | Increase if you see `ConvergenceWarning` |

```python
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn_genetic import EvolutionConfig, GASearchCV, PopulationConfig, RuntimeConfig

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# LogisticRegression benefits from scaling — use a pipeline
lr_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(random_state=42)),
])

lr_search = GASearchCV(
    estimator=lr_pipe,
    param_grid={
        "clf__C":        Continuous(1e-4, 1e3, distribution="log-uniform"),
        "clf__solver":   Categorical(["lbfgs", "saga"]),
        "clf__max_iter": Integer(100, 1000),
    },
    cv=cv, scoring="roc_auc",
    evolution_config=EvolutionConfig(population_size=10, generations=6),
    population_config=PopulationConfig(initializer="smart"),
    runtime_config=RuntimeConfig(n_jobs=-1, verbose=False),
    random_state=42,
)
lr_search.fit(X_train, y_train)
print(f"Best CV ROC-AUC : {lr_search.best_score_:.4f}")
print(f"Best params     : {lr_search.best_params_}")
```

```text
Best CV ROC-AUC : 0.9979
Best params     : {'clf__C': 2.1847, 'clf__solver': 'lbfgs', 'clf__max_iter': 341}
```

---

### SVM (SVC)

```python
from sklearn_genetic.space import Categorical, Continuous

svc_param_grid = {
    "C":      Continuous(1e-2, 1e3, distribution="log-uniform"),
    "gamma":  Continuous(1e-5, 1e1, distribution="log-uniform"),
    "kernel": Categorical(["rbf", "poly", "sigmoid"]),
}
```

| Parameter | Type | Recommended range | Notes |
|-----------|------|-------------------|-------|
| `C` | Continuous | 1e-2 – 1e3 (log-uniform) | Regularization; high C = tighter fit, more overfit risk |
| `gamma` | Continuous | 1e-5 – 10 (log-uniform) | Only relevant for `rbf`, `poly`, `sigmoid` |
| `kernel` | Categorical | `"rbf"`, `"poly"`, `"sigmoid"` | `"rbf"` is the best default starting point |

```python
from sklearn.datasets import load_digits
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn_genetic import EvolutionConfig, GASearchCV, PopulationConfig, RuntimeConfig

X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

svc_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC(probability=True, random_state=42)),
])

svc_search = GASearchCV(
    estimator=svc_pipe,
    param_grid={
        "svc__C":      Continuous(1e-2, 1e3, distribution="log-uniform"),
        "svc__gamma":  Continuous(1e-5, 1e1, distribution="log-uniform"),
        "svc__kernel": Categorical(["rbf", "poly"]),
    },
    cv=cv, scoring="accuracy",
    evolution_config=EvolutionConfig(population_size=12, generations=7),
    population_config=PopulationConfig(initializer="smart"),
    runtime_config=RuntimeConfig(n_jobs=-1, verbose=False),
    random_state=42,
)
svc_search.fit(X_train, y_train)
print(f"Best CV accuracy : {svc_search.best_score_:.4f}")
print(f"Best params      : {svc_search.best_params_}")
```

```text
Best CV accuracy : 0.9888
Best params      : {'svc__C': 8.3241, 'svc__gamma': 0.0014, 'svc__kernel': 'rbf'}
```

---

## Common Pitfalls

### Including `None` in a Categorical

`None` is a valid choice in `Categorical` and is common for parameters like `max_depth` and `class_weight` that accept `None` as a meaningful value (unlimited depth, balanced weights):

```python
from sklearn_genetic.space import Categorical

# None is valid in Categorical
"class_weight": Categorical([None, "balanced"]),
"max_features": Categorical([None, "sqrt", "log2"]),
```

However, `Integer` and `Continuous` do not accept `None` as a bound. If you want to include `None` alongside integer values for `max_depth`, use `Categorical`:

```python
# Wrong: Integer cannot include None
# "max_depth": Integer(None, 20)  # raises TypeError

# Right: use Categorical when None must be an option
"max_depth": Categorical([None, 5, 10, 15, 20]),
```

### Integer bounds are inclusive on both ends

`Integer(1, 10)` can produce values 1, 2, 3, ..., 10. The upper bound is included. This differs from Python's `range(1, 10)` which excludes 10:

```python
from sklearn_genetic.space import Integer

depth_space = Integer(1, 20)
# Both 1 and 20 are valid samples
```

### Setting `lower=0` for log-uniform

`Continuous(0, 1.0, distribution="log-uniform")` is invalid because `log(0)` is undefined. The lower bound must be strictly positive:

```python
from sklearn_genetic.space import Continuous

# Wrong: log(0) is undefined
# Continuous(0.0, 1.0, distribution="log-uniform")

# Right: small positive lower bound
Continuous(1e-6, 1.0, distribution="log-uniform")
```

### Overly tight bounds that exclude the optimum

If you set `max_depth` to `Integer(3, 6)` and the true optimal depth is 10, the search will return the best configuration within your constraint (depth=6) without any indication that the optimum lies outside. Always check whether `plot_search_space` shows the optimum near a boundary.

---

## How to Inspect What Was Sampled

After fitting, `plot_search_space` shows a scatter plot of sampled values for each parameter pair, colored by score. Use this to diagnose whether the search explored the space well.

```python
import matplotlib.pyplot as plt
from sklearn_genetic.plots import plot_search_space

# After search.fit(X_train, y_train):
plot_search_space(rf_search)
plt.tight_layout()
```

What to look for:

- **Optimum near a boundary** — if the best-scoring samples cluster near the upper or lower bound, expand the range in that direction and re-run.
- **Samples clustered in one region** — check whether you should use `distribution="log-uniform"` to spread samples more evenly.
- **Thin coverage** — if the scatter plot is sparse, increase `population_size` or `generations` so more of the space is explored.
- **No clear gradient** — if score (color) is uniform across the plot, the parameter may not matter much. Consider removing it from the search to reduce dimensionality.

---

## Warm Starting vs Cold Starting

By default, `GASearchCV` generates an entirely random initial population (optionally improved with Latin hypercube sampling via `PopulationConfig(initializer="smart")`). If you already know a good configuration from a previous search or domain knowledge, you can seed the population with it.

**Warm starting with known-good configs:**

```python
from sklearn_genetic import GASearchCV, EvolutionConfig, PopulationConfig, RuntimeConfig
from sklearn_genetic.space import Categorical, Continuous, Integer

# Known-good configuration from a previous manual search
warm_start_configs = [
    {"n_estimators": 200, "max_depth": 10, "max_features": "sqrt"},
    {"n_estimators": 150, "max_depth": 8,  "max_features": "log2"},
]

search = GASearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid={
        "n_estimators": Integer(50, 500),
        "max_depth":    Categorical([None, 5, 8, 10, 15, 20]),
        "max_features": Categorical(["sqrt", "log2"]),
    },
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring="roc_auc",
    population_config=PopulationConfig(
        initializer="smart",
        warm_start_configs=warm_start_configs,  # seeds two individuals
    ),
    evolution_config=EvolutionConfig(population_size=15, generations=8),
    runtime_config=RuntimeConfig(n_jobs=-1, verbose=False),
    random_state=42,
)
```

**Smart initialization (`initializer="smart"`):**

The default cold start uses Latin hypercube sampling to produce a diverse, well-spread initial population. This is almost always better than a purely random initial population and is the recommended setting. Warm start configs are *added* to the smart-initialized population, not used instead of it.

**When to use each:**

| Scenario | Recommendation |
|----------|----------------|
| First search on a new dataset | `initializer="smart"`, no warm configs |
| Refining after a broad initial search | `warm_start_configs` with the best configs found so far |
| Resuming an interrupted search | `warm_start_configs` from the last generation's best individuals |
| Strong domain knowledge about good values | `warm_start_configs` from prior knowledge |

:::tip Warm-start configs must be within the search space bounds
A warm-start config is silently skipped if any value is out of bounds for its dimension or not in the `choices` list for a `Categorical`. Verify your configs against `list(search.param_grid.keys())` and the declared bounds before fitting. See [Troubleshooting](./troubleshooting#warm-start-configs-are-ignored).
:::

---

## See Also

- [Getting Started with GASearchCV](./basic-usage) — your first genetic search with all three parameter types
- [When to Use Genetic Algorithm Search](./when-to-use) — decide whether a genetic search fits your problem
- [Common Hyperparameter Tuning Mistakes](./common-mistakes) — ten mistakes that silently hurt search quality, with fixes
- [API: Search Space](../api/space) — full reference for `Integer`, `Continuous`, and `Categorical`
