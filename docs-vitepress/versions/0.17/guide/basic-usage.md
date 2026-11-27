---
title: "Start Hyperparameter Tuning With GASearchCV"
description: "Step-by-step introduction to GASearchCV for hyperparameter optimization and GAFeatureSelectionCV for feature selection, with complete runnable examples."
---

# Start Hyperparameter Tuning With GASearchCV

**Estimated reading time:** 15 minutes. **Difficulty:** Beginner.

This page walks through the two workflows you will use most:

- **Hyperparameter tuning** with `GASearchCV`
- **Feature selection** with `GAFeatureSelectionCV`

Every code block below is executed to build this page, so what you copy is exactly what ran — including the outputs and figures.

This walkthrough uses the default DEAP engine. To keep the same estimator and CV workflow while changing optimizers, follow [Switch optimization backends](../tutorials/optimizer-backends).

## Prerequisites

- `sklearn-genetic-opt` installed (`pip install sklearn-genetic-opt`)
- Optional plotting extra for the figures: `pip install sklearn-genetic-opt[plot]`
- Basic familiarity with Python, pip, `train_test_split`, and classifiers' `fit` / `predict` API

## Hyperparameter Tuning

We tune an `MLPClassifier` on the
[handwritten digits dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)
(10 classes, 64 pixel features). First, the imports and data:

```python
import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier

from sklearn_genetic import EvolutionConfig, GASearchCV, PopulationConfig, RuntimeConfig
from sklearn_genetic.space import Categorical, Continuous, Integer

RANDOM_STATE = 42

X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, stratify=y, random_state=RANDOM_STATE
)
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
print(f"train={X_train.shape}  test={X_test.shape}  classes={len(np.unique(y))}")
```

```text
train=(1203, 64)  test=(594, 64)  classes=10
```

### Define the search space

The keys in `param_grid` are estimator parameter names. Each value is a
*space* the genetic algorithm samples from:

- `Integer` — whole numbers in a range
- `Continuous` — floats in a range (optionally `distribution="log-uniform"`)
- `Categorical` — a fixed list of choices

```python
param_grid = {
    "hidden_layer_sizes": Categorical([(32,), (64,), (32, 16)]),
    "alpha": Continuous(1e-5, 1e-1, distribution="log-uniform"),
    "learning_rate_init": Continuous(1e-4, 1e-1, distribution="log-uniform"),
    "activation": Categorical(["relu", "tanh"]),
    "batch_size": Integer(32, 256),
}
```

### Configure and run the search

`GASearchCV` is configured with small config objects. `EvolutionConfig`
controls the population and generations; `PopulationConfig(initializer="smart")`
builds a diverse, well-spread starting population; `RuntimeConfig` controls
parallelism and logging.

```python
clf = MLPClassifier(max_iter=150, early_stopping=True, random_state=RANDOM_STATE)

search = GASearchCV(
    estimator=clf,
    random_state=RANDOM_STATE,
    cv=cv,
    scoring="accuracy",
    param_grid=param_grid,
    evolution_config=EvolutionConfig(population_size=8, generations=6),
    population_config=PopulationConfig(initializer="smart"),
    runtime_config=RuntimeConfig(n_jobs=-1, verbose=False),
)

search.fit(X_train, y_train)

print(f"Best CV accuracy : {search.best_score_:.4f}")
print(f"Best parameters  : {search.best_params_}")
```

```text
Best CV accuracy : 0.9634
Best parameters  : {'hidden_layer_sizes': (64,), 'alpha': 5.664055981885377e-05, 'learning_rate_init': 0.007059082353376892, 'activation': 'relu', 'batch_size': 81}
```

After fitting, `GASearchCV` behaves like any fitted scikit-learn estimator —
it has already refit the best configuration on all of `X_train`:

```python
y_pred = search.predict(X_test)
print(f"Holdout accuracy : {accuracy_score(y_test, y_pred):.4f}")
```

```text
Holdout accuracy : 0.9495
```

### Reading the evolution

Each generation is logged. Inspect the full history as a DataFrame — the
columns explain how the population improved and how diverse it stayed:

| Column | Meaning |
|--------|---------|
| `gen` | generation number |
| `fitness` | mean CV score in the generation |
| `fitness_best` | best CV score found so far |
| `genotype_diversity` | Mean probability that two distinct population members disagree at a coordinate |
| `unique_individual_ratio` | fraction of distinct configurations |
| `stagnation_generations` | consecutive generations without improvement |

```python
import pandas as pd

history = pd.DataFrame(search.history)
print(history[["gen", "fitness", "fitness_best", "genotype_diversity",
               "unique_individual_ratio", "stagnation_generations"]].to_string(index=False))
```

```text
 gen  fitness  fitness_best  genotype_diversity  unique_individual_ratio  stagnation_generations
   0 0.833437      0.949293            0.864286                    1.000                       0
   1 0.916667      0.950956            0.742857                    0.750                       0
   2 0.951372      0.951787            0.321429                    0.375                       0
   3 0.949190      0.959268            0.507143                    0.625                       0
   4 0.955528      0.960931            0.485714                    0.750                       0
   5 0.959892      0.963425            0.328571                    0.375                       0
   6 0.957918      0.963425            0.321429                    0.500                       1
```

`fit_stats_` summarizes what the search actually spent — useful for spotting
wasted effort (e.g. many `skipped_invalid_candidates`):

```python
for key, value in search.fit_stats_.items():
    print(f"{key}: {value}")
```

```text
evaluated_candidates: 104
unique_candidates: 100
cross_validate_calls: 100
cache_hits: 4
duplicate_candidates: 0
skipped_invalid_candidates: 0
population_parallel_batches: 7
population_serial_batches: 0
random_immigrants: 0
local_refinement_candidates: 0
```

### Visualize convergence

With the `[plot]` extra installed, `plot_fitness_evolution` shows the best
score climbing over generations, and `plot_search_overview` gives a compact
diagnostic dashboard.

```python
from sklearn_genetic.plots import plot_search_overview

overview = plot_search_overview(search, top_k=6)
```

![Search overview dashboard: convergence, diversity, optimizer events, and top candidates](/images/basic_usage_mlp_search_overview.png)

```python
from sklearn_genetic.plots import plot_parameter_evolution

evolution = plot_parameter_evolution(search, parameters=["alpha", "learning_rate_init", "batch_size"])
```

![How sampled values of alpha, learning rate, and batch size evolved over evaluations](/images/basic_usage_mlp_parameter_evolution.png)

## Feature Selection

`GAFeatureSelectionCV` searches for the most useful subset of columns. To make
the task concrete we take the Iris dataset and bolt on 10 columns of pure
noise; a good selector should keep the real measurements and discard the noise.

```python
from sklearn.datasets import load_iris
from sklearn.svm import SVC

from sklearn_genetic import GAFeatureSelectionCV

iris = load_iris()
rng = np.random.default_rng(RANDOM_STATE)
noise = rng.uniform(0, 10, size=(iris.data.shape[0], 10))
X_fs = np.hstack([iris.data, noise])
feature_names = list(iris.feature_names) + [f"noise_{i}" for i in range(10)]
y_fs = iris.target

Xtr, Xte, ytr, yte = train_test_split(
    X_fs, y_fs, test_size=0.33, stratify=y_fs, random_state=RANDOM_STATE
)

selector = GAFeatureSelectionCV(
    estimator=SVC(gamma="auto"),
    random_state=RANDOM_STATE,
    cv=3,
    scoring="accuracy",
    evolution_config=EvolutionConfig(population_size=10, generations=8,
                                     elitism=True, keep_top_k=2),
    population_config=PopulationConfig(initializer="smart"),
    runtime_config=RuntimeConfig(n_jobs=-1, verbose=False),
)
selector.fit(Xtr, ytr)

selected = [name for name, keep in zip(feature_names, selector.support_) if keep]
print(f"Selected {len(selected)} of {len(feature_names)} features:")
for name in selected:
    print(f"  - {name}")
print(f"Holdout accuracy : {accuracy_score(yte, selector.predict(Xte)):.4f}")
```

```text
Selected 2 of 14 features:
  - sepal length (cm)
  - petal length (cm)
Holdout accuracy : 0.9800
```

`support_` is a boolean mask (`True` = kept). The search keeps the informative
Iris measurements and drops most of the noise columns. Visualize the mask:

```python
from sklearn_genetic.plots import plot_feature_selection

mask_plot = plot_feature_selection(selector, feature_names=feature_names)
```

![Selected-feature mask over the four Iris features plus ten noise columns](/images/basic_usage_iris_mask.png)

## Tips & Gotchas

- Pass the training targets to `search.fit(X_train, y_train)` and
  `selector.fit(Xtr, ytr)`. Models that require targets raise an
  error before cross-validation when the target is omitted. For
  [unlabeled outlier detection](./outliers#score-unlabeled-samples), call
  `fit(X_train)`.
- Set `RuntimeConfig(verbose=True)` to watch the per-generation log live.
- `PopulationConfig(initializer="smart")` is strongly recommended — it usually
  finds better solutions faster than a purely random start.
- If `accuracy` is already near 1.0 on your data, switch to a more
  discriminative metric (e.g. `roc_auc`, `balanced_accuracy`).
- Check `fit_stats_["skipped_invalid_candidates"]` after fitting — a non-zero
  value counts candidates dropped before CV. For feature selection, use
  `cv_results_["skipped_invalid_mask"]` to locate the affected rows.

## See Also

- [How Hyperparameter Optimization Works](./how-hyperparameter-optimization-works) — understand the theory behind different search methods
- [Choosing the Right Search Space](./choosing-search-spaces) — pick optimal parameter bounds
- [Common Hyperparameter Tuning Mistakes](./common-mistakes) — avoid the most frequent pitfalls
- [Cross-Validation in Hyperparameter Search](./understand-cv) — understand generation logs and convergence
- [Random Forest Hyperparameter Tuning](../tutorials/tune-random-forest) — first real tutorial
- [Troubleshooting](./troubleshooting) — if something doesn't work
