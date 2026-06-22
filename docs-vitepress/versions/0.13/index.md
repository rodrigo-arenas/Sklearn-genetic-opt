---
title: sklearn-genetic-opt 0.13
description: Documentation for sklearn-genetic-opt version 0.13 — evolutionary hyperparameter tuning and feature selection for scikit-learn.
---

# sklearn-genetic-opt

`sklearn-genetic-opt` adds evolutionary optimization tools to the scikit-learn workflow. It can tune hyperparameters with `GASearchCV` and select feature subsets with `GAFeatureSelectionCV` using algorithms powered by DEAP.

The project is useful when a search space is mixed, irregular, expensive, or not well served by an exhaustive grid. It follows familiar scikit-learn patterns: define an estimator, define a search space, call `fit`, inspect `best_params_` or `support_`, and use the fitted object for prediction.

:::info Version
You are reading the **0.13 (stable)** docs. Looking for the development version? See [latest](/versions/latest/).
:::

## Highlights

- `GASearchCV` for hyperparameter search across classification, regression, and supported outlier-detection estimators.
- `GAFeatureSelectionCV` for wrapper-based feature selection with cross-validation.
- Search spaces for integer, continuous, and categorical parameters.
- Grouped configuration objects for readable advanced setups: `EvolutionConfig`, `PopulationConfig`, `RuntimeConfig`, and `OptimizationConfig`.
- Smart initial populations with `PopulationConfig(initializer="smart")`, including warm-start seeds, estimator defaults, Latin-hypercube numeric coverage, stratified categorical coverage, and duplicate avoidance.
- Adaptive mutation and crossover schedules.
- Optional local search, diversity control, random immigrants, and fitness sharing.
- Parallel candidate evaluation with `n_jobs` and `parallel_backend`.
- Evaluation caching, optimizer telemetry through `history`, and fit-cost counters through `fit_stats_`.
- Callbacks for early stopping, progress reporting, checkpoints, TensorBoard, and custom logic.
- Plotting helpers plus MLflow 3 logging support.

## Installation

```bash
pip install sklearn-genetic-opt
```

Or with conda:

```bash
conda install -c conda-forge sklearn-genetic-opt
```

Install optional plotting, MLflow, and TensorBoard integrations:

```bash
pip install sklearn-genetic-opt[all]
```

## Requirements

| Package | Minimum version |
|---------|----------------|
| Python | 3.12 |
| scikit-learn | 1.9.0 |
| NumPy | 2.4.6 |
| DEAP | 1.4.4 |
| tqdm | 4.68.3 |

Optional extras: Seaborn ≥ 0.13.2 for plots, MLflow ≥ 3.14.0 for experiment logging, TensorFlow ≥ 2.21.0 + TensorBoard ≥ 2.20.0 for TensorBoard logging.

## Quick Start

This example tunes a `RandomForestClassifier` across six hyperparameters on the breast cancer dataset.

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
print("CV score:", round(search.best_score_, 4))

y_prob = search.predict_proba(X_test)[:, 1]
print("Holdout ROC-AUC:", round(roc_auc_score(y_test, y_prob), 4))

print(search.fit_stats_)
```

## Next Steps

- Not sure if GA search is right for your problem? Start with [When to Use](./guide/when-to-use).
- New to the library? [Basic Usage](./guide/basic-usage) walks through the full workflow.
- Tuning a scikit-learn `Pipeline`? See [Pipeline Tuning](./guide/pipeline-tuning).
- Something not working? See [Troubleshooting](./guide/troubleshooting).
