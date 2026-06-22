---
title: When to Use sklearn-genetic-opt
description: Decide whether a genetic algorithm search fits your tuning problem, and see a realistic example.
---

# When to Use sklearn-genetic-opt

This page helps you decide whether a genetic search is the right tool for your tuning problem and shows what a realistic setup looks like.

## Prerequisites

- Basic familiarity with scikit-learn's `GridSearchCV` or `RandomizedSearchCV`
- `sklearn-genetic-opt` installed (`pip install sklearn-genetic-opt`)

## Choosing a Search Method

scikit-learn ships three search strategies. Each targets a different situation:

| Method | Best for | Weakness | Typical space size |
|--------|----------|----------|--------------------|
| `GridSearchCV` | Small, fully discrete grids | Candidate count multiplies with each dimension | ≤ 3 parameters |
| `RandomizedSearchCV` | Continuous spaces, large grids | Treats every parameter independently — misses interactions | 3–6 parameters |
| `GASearchCV` | Mixed or large spaces with parameter interactions | Adds overhead on trivially small spaces | 5+ parameters |

The key limitation of random search is **independence**: it samples each parameter as if the others do not exist. When two parameters interact — for example, `learning_rate` and `n_estimators` in a gradient boosting model — random search is as likely to pair a low learning rate with few estimators (underfit) as to pair a low learning rate with many estimators (good). A genetic algorithm recombines *complete configurations* that performed well, so it naturally gravitates toward combinations that work together.

## Signs That GA Will Help

- **Five or more hyperparameters.** The search space grows exponentially. GA explores it with a population of complete solutions rather than independently sampling each axis.
- **Known or suspected parameter interactions.** `learning_rate` × number of estimators, regularization strength × solver, kernel bandwidth × `C` in SVMs.
- **Mixed parameter types in the same space.** Integers, floats, and categoricals in one search.
- **Expensive evaluations.** GA caches every evaluated candidate and reuses its score when the same configuration appears again.
- **You want to narrow the space iteratively.** `plot_search_space` shows where the algorithm sampled. You can tighten ranges between runs.

## Signs That GA Will Not Help

- **One or two continuous parameters.** Random search covers this well and is faster to configure.
- **Very fast evaluations with a large budget.** If you can afford ten thousand random candidates in seconds, more candidates beat a smarter search.
- **A fully discrete grid with few values.** `GridSearchCV` is exhaustive and easier to reason about.

## Example: Gradient Boosting With Seven Parameters

`HistGradientBoostingClassifier` has a well-known interaction between `learning_rate` and `max_iter`: a low learning rate needs more iterations, and a high learning rate converges faster. Random search samples these independently. The genetic algorithm recombines configurations that worked and tends to find consistent (learning rate, iteration count) pairs.

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

y_prob = search.predict_proba(X_test)[:, 1]
print("Holdout ROC-AUC:", round(roc_auc_score(y_test, y_prob), 4))
```

After fitting, inspect cache efficiency and diversity telemetry:

```python
print(search.fit_stats_)

import pandas as pd
history = pd.DataFrame(search.history)
print(history[["gen", "fitness_best", "genotype_diversity", "stagnation_generations"]].tail())
```

## Minimum Recommended Configuration

The essentials for any production run:

```python
from sklearn_genetic import EvolutionConfig, GASearchCV, PopulationConfig, RuntimeConfig

search = GASearchCV(
    estimator=your_estimator,
    param_grid=your_param_grid,
    cv=your_cv_strategy,
    scoring="your_metric",
    evolution_config=EvolutionConfig(
        population_size=20,   # start here; increase for larger spaces
        generations=15,       # 10–20 is a reasonable default
    ),
    population_config=PopulationConfig(initializer="smart"),
    runtime_config=RuntimeConfig(n_jobs=-1, use_cache=True),
)

search.fit(X_train, y_train)

print(search.best_params_)
print(search.best_score_)
```

`population_size` and `generations` control the total evaluation budget. With population 20 and 15 generations, approximately 620 candidate configurations are generated — a reasonable budget for a 7-parameter space.

## Next Steps

- [Basic Usage](./basic-usage) — full workflow for hyperparameter tuning and feature selection with plots.
- [Understanding Cross-Validation](./understand-cv) — how the genetic algorithm evaluates candidates and what the generation log means.
- [Pipeline Tuning](./pipeline-tuning) — how to tune a scikit-learn `Pipeline`.
- [Advanced Optimizer Control](./advanced-optimizer-control) — diversity control, local search, and fitness sharing.
