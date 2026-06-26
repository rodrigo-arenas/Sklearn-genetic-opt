---
title: Basic Usage
description: Your first hyperparameter search with GASearchCV and your first feature selection with GAFeatureSelectionCV — a complete, runnable walkthrough.
---

:::warning Development version
This is the **latest (dev)** documentation. It may contain unreleased features or breaking changes. For the stable release, use [version 0.13](/versions/0.13/).
:::

# Basic Usage

This page walks through the two workflows you will use most:

- **Hyperparameter tuning** with `GASearchCV`
- **Feature selection** with `GAFeatureSelectionCV`

Every code block below is executed to build this page, so what you copy is exactly what ran — including the outputs and figures.

## Prerequisites

- `sklearn-genetic-opt` installed (`pip install sklearn-genetic-opt`)
- Optional plotting extra for the figures: `pip install sklearn-genetic-opt[plot]`
- Basic familiarity with scikit-learn's `fit` / `predict` API

## Hyperparameter Tuning

We tune an `MLPClassifier` on the
[handwritten digits dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)
(10 classes, 64 pixel features). First, the imports and data:

```python
import random

import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier

from sklearn_genetic import EvolutionConfig, GASearchCV, PopulationConfig, RuntimeConfig
from sklearn_genetic.space import Categorical, Continuous, Integer

RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

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
    cv=cv,
    scoring="accuracy",
    param_grid=param_grid,
    evolution_config=EvolutionConfig(population_size=10, generations=8),
    population_config=PopulationConfig(initializer="smart"),
    runtime_config=RuntimeConfig(n_jobs=-1, verbose=False),
)

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
search.fit(X_train, y_train)

print(f"Best CV accuracy : {search.best_score_:.4f}")
print(f"Best parameters  : {search.best_params_}")
```

```text
Best CV accuracy : 0.9659
Best parameters  : {'hidden_layer_sizes': (64,), 'alpha': 0.0009444574254983562, 'learning_rate_init': 0.0036999724314638123, 'activation': 'relu', 'batch_size': 46}
```

After fitting, `GASearchCV` behaves like any fitted scikit-learn estimator —
it has already refit the best configuration on all of `X_train`:

```python
y_pred = search.predict(X_test)
print(f"Holdout accuracy : {accuracy_score(y_test, y_pred):.4f}")
```

```text
Holdout accuracy : 0.9613
```

### Reading the evolution

Each generation is logged. Inspect the full history as a DataFrame — the
columns explain how the population improved and how diverse it stayed:

| Column | Meaning |
|--------|---------|
| `gen` | generation number |
| `fitness` | mean CV score in the generation |
| `fitness_best` | best CV score found so far |
| `genotype_diversity` | 1.0 = diverse, 0.0 = converged |
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
   0 0.817872      0.942643            0.666667                      1.0                       0
   1 0.938487      0.950125            0.444444                      0.8                       0
   2 0.948545      0.953450            0.200000                      0.5                       0
   3 0.939817      0.953450            0.444444                      0.8                       1
   4 0.952702      0.954281            0.288889                      0.6                       0
   5 0.954032      0.963425            0.200000                      0.6                       0
   6 0.951704      0.963425            0.222222                      0.6                       1
   7 0.950873      0.963425            0.066667                      0.4                       2
   8 0.944140      0.963425            0.288889                      0.7                       3
```

`fit_stats_` summarizes what the search actually spent — useful for spotting
wasted effort (e.g. many `skipped_invalid_candidates`):

```python
for key, value in search.fit_stats_.items():
    print(f"{key}: {value}")
```

```text
evaluated_candidates: 170
unique_candidates: 166
cross_validate_calls: 166
cache_hits: 4
duplicate_candidates: 0
skipped_invalid_candidates: 0
population_parallel_batches: 9
population_serial_batches: 0
random_immigrants: 8
local_refinement_candidates: 0
```

### Visualize convergence

With the `[plot]` extra installed, `plot_fitness_evolution` shows the best
score climbing over generations, and `plot_search_overview` gives a compact
diagnostic dashboard.

```python
import matplotlib.pyplot as plt
from sklearn_genetic.plots import plot_search_overview

plot_search_overview(search, top_k=6)
plt.tight_layout()
```

![Search overview dashboard: convergence, diversity, optimizer events, and top candidates](/images/basic_usage_search_overview.png)

```python
from sklearn_genetic.plots import plot_parameter_evolution

plot_parameter_evolution(search, parameters=["alpha", "learning_rate_init", "batch_size"])
plt.tight_layout()
```

![How sampled values of alpha, learning rate, and batch size evolved over evaluations](/images/basic_usage_parameter_evolution.png)

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
    cv=3,
    scoring="accuracy",
    evolution_config=EvolutionConfig(population_size=10, generations=8,
                                     elitism=True, keep_top_k=2),
    population_config=PopulationConfig(initializer="smart"),
    runtime_config=RuntimeConfig(n_jobs=-1, verbose=False),
)
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
selector.fit(Xtr, ytr)

selected = [name for name, keep in zip(feature_names, selector.support_) if keep]
print(f"Selected {len(selected)} of {len(feature_names)} features:")
for name in selected:
    print(f"  - {name}")
print(f"Holdout accuracy : {accuracy_score(yte, selector.predict(Xte)):.4f}")
```

```text
Selected 3 of 14 features:
  - sepal length (cm)
  - sepal width (cm)
  - petal length (cm)
Holdout accuracy : 0.9800
```

`support_` is a boolean mask (`True` = kept). The search keeps the informative
Iris measurements and drops most of the noise columns. Visualize the mask:

```python
from sklearn_genetic.plots import plot_feature_selection

plot_feature_selection(selector, feature_names=feature_names)
plt.tight_layout()
```

![Selected-feature mask over the four Iris features plus ten noise columns](/images/basic_usage_feature_selection.png)

## Tips & Gotchas

- Set `RuntimeConfig(verbose=True)` to watch the per-generation log live.
- `PopulationConfig(initializer="smart")` is strongly recommended — it usually
  finds better solutions faster than a purely random start.
- If `accuracy` is already near 1.0 on your data, switch to a more
  discriminative metric (e.g. `roc_auc`, `balanced_accuracy`).
- Check `fit_stats_["skipped_invalid_candidates"]` after fitting — a non-zero
  value means some sampled configurations raised errors during `fit`.

## Next Steps

- [When to Use](./when-to-use) — is a genetic search the right tool for your problem?
- [Understanding Cross-Validation](./understand-cv) — read the generation log in depth
- [Pipeline Tuning](./pipeline-tuning) — tune a scikit-learn `Pipeline` with `step__param` names
- [Callbacks](./callbacks) — early stopping, progress bars, and checkpoints
- [Plotting Gallery](../examples/plotting-gallery) — every diagnostic plot, explained
