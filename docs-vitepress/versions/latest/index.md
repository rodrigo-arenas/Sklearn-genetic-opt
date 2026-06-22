---
title: sklearn-genetic-opt (latest / dev)
description: Development documentation for sklearn-genetic-opt — tracks the master branch and may contain unreleased features.
---

# sklearn-genetic-opt

`sklearn-genetic-opt` adds evolutionary optimization tools to the scikit-learn workflow. It can tune hyperparameters with `GASearchCV` and select feature subsets with `GAFeatureSelectionCV` using algorithms powered by DEAP.

:::warning Development version
You are reading the **latest (development)** docs. This version tracks the `master` branch and may contain unreleased features or breaking changes. For stable documentation, see [0.13 (stable)](/versions/0.13/).
:::

## Highlights

- `GASearchCV` for hyperparameter search across classification, regression, and supported outlier-detection estimators.
- `GAFeatureSelectionCV` for wrapper-based feature selection with cross-validation.
- Search spaces for integer, continuous, and categorical parameters.
- Grouped configuration objects: `EvolutionConfig`, `PopulationConfig`, `RuntimeConfig`, and `OptimizationConfig`.
- Smart initial populations with `PopulationConfig(initializer="smart")`.
- Adaptive mutation and crossover schedules.
- Optional local search, diversity control, random immigrants, and fitness sharing.
- Parallel candidate evaluation with `n_jobs` and `parallel_backend`.
- Evaluation caching, optimizer telemetry through `history`, and fit-cost counters through `fit_stats_`.
- Callbacks for early stopping, progress reporting, checkpoints, TensorBoard, and custom logic.
- Plotting helpers plus MLflow 3 logging support.

## Installation (dev)

Install from the master branch:

```bash
pip install git+https://github.com/rodrigo-arenas/Sklearn-genetic-opt.git@master
```

Or install the released version:

```bash
pip install sklearn-genetic-opt
```

## Quick Start

```python
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score

from sklearn_genetic import EvolutionConfig, GASearchCV, PopulationConfig, RuntimeConfig
from sklearn_genetic.space import Categorical, Continuous, Integer

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

param_grid = {
    "n_estimators": Integer(50, 250),
    "max_depth": Integer(2, 14),
    "min_samples_split": Integer(2, 12),
    "min_samples_leaf": Integer(1, 8),
    "max_features": Categorical(["sqrt", "log2", None]),
    "ccp_alpha": Continuous(0.0, 0.03),
}

search = GASearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
    scoring="roc_auc",
    evolution_config=EvolutionConfig(population_size=20, generations=12),
    population_config=PopulationConfig(initializer="smart"),
    runtime_config=RuntimeConfig(n_jobs=-1, parallel_backend="auto", use_cache=True),
)

search.fit(X_train, y_train)
print(search.best_params_)
```

## Next Steps

- [When to Use](./guide/when-to-use)
- [Basic Usage](./guide/basic-usage)
- [Troubleshooting](./guide/troubleshooting)
