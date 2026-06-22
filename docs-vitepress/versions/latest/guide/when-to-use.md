---
title: When to Use sklearn-genetic-opt
description: Decide whether a genetic algorithm search fits your tuning problem, and see a realistic example.
---

:::warning Development version
You are reading the **latest (dev)** docs. For stable documentation, see [0.13](/versions/0.13/guide/when-to-use).
:::

# When to Use sklearn-genetic-opt

This page helps you decide whether a genetic search is the right tool for your tuning problem and shows what a realistic setup looks like.

## Prerequisites

- Basic familiarity with scikit-learn's `GridSearchCV` or `RandomizedSearchCV`
- `sklearn-genetic-opt` installed (`pip install sklearn-genetic-opt`)

## Choosing a Search Method

| Method | Best for | Weakness | Typical space size |
|--------|----------|----------|--------------------|
| `GridSearchCV` | Small, fully discrete grids | Candidate count multiplies with each dimension | ≤ 3 parameters |
| `RandomizedSearchCV` | Continuous spaces, large grids | Treats every parameter independently — misses interactions | 3–6 parameters |
| `GASearchCV` | Mixed or large spaces with parameter interactions | Adds overhead on trivially small spaces | 5+ parameters |

The key limitation of random search is **independence**: it samples each parameter as if the others do not exist. A genetic algorithm recombines *complete configurations* that performed well, so it naturally gravitates toward combinations that work together.

## Signs That GA Will Help

- **Five or more hyperparameters.** The search space grows exponentially.
- **Known or suspected parameter interactions.** `learning_rate` × number of estimators, regularization strength × solver.
- **Mixed parameter types.** Integers, floats, and categoricals in one search.
- **Expensive evaluations.** GA caches every evaluated candidate.
- **You want to narrow the space iteratively.** `plot_search_space` shows where the algorithm sampled.

## Signs That GA Will Not Help

- One or two continuous parameters.
- Very fast evaluations with a large budget.
- A fully discrete grid with few values.

## Example: Gradient Boosting With Seven Parameters

```python
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score

from sklearn_genetic import EvolutionConfig, GASearchCV, PopulationConfig, RuntimeConfig
from sklearn_genetic.space import Categorical, Continuous, Integer

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

param_grid = {
    "learning_rate": Continuous(0.01, 0.3, distribution="log-uniform"),
    "max_iter": Integer(50, 300),
    "max_depth": Integer(2, 8),
    "min_samples_leaf": Integer(5, 50),
    "l2_regularization": Continuous(1e-6, 1.0, distribution="log-uniform"),
    "max_features": Continuous(0.3, 1.0),
    "max_leaf_nodes": Integer(15, 127),
}

search = GASearchCV(
    estimator=HistGradientBoostingClassifier(random_state=42, early_stopping=False),
    param_grid=param_grid,
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
    scoring="roc_auc",
    evolution_config=EvolutionConfig(population_size=20, generations=15),
    population_config=PopulationConfig(initializer="smart"),
    runtime_config=RuntimeConfig(n_jobs=-1, parallel_backend="auto", use_cache=True),
)

search.fit(X_train, y_train)
print("Best CV score:", round(search.best_score_, 4))
print("Best parameters:", search.best_params_)
```

## Next Steps

- [Basic Usage](./basic-usage)
- [Understanding Cross-Validation](./understand-cv)
- [Pipeline Tuning](./pipeline-tuning)
- [Advanced Optimizer Control](./advanced-optimizer-control)
