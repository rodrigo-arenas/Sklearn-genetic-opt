---
title: "Seed a Search with Known-Good Hyperparameters (Warm Start)"
description: "Recipe to use warm_start_configs to initialize GASearchCV population with known-good hyperparameter values, avoiding a cold start."
---

# Seed a Search with Known-Good Params (Warm Start)

**Time:** 5 min | **Difficulty:** Intermediate

## What This Solves

Without seeding, the first generation is randomly initialized — the search may spend several generations discovering the obvious "known good" region. `warm_start_configs` injects configurations you already know are reasonable, so generation 1 starts from a better baseline.

## Recipe

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier

from sklearn_genetic import GASearchCV, EvolutionConfig, PopulationConfig, RuntimeConfig
from sklearn_genetic.space import Continuous, Integer

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

param_grid = {
    "n_estimators": Integer(50, 400),
    "max_depth":    Integer(2, 10),
    "learning_rate": Continuous(0.01, 0.3, distribution="log-uniform"),
    "subsample":    Continuous(0.5, 1.0),
    "colsample_bytree": Continuous(0.4, 1.0),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

ga = GASearchCV(
    estimator=XGBClassifier(
        tree_method="hist", eval_metric="logloss",
        random_state=42, n_jobs=1,
    ),
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
        warm_start_configs=[
            # Seed with XGBoost's well-known defaults
            {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.3,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
            },
            # Also seed with a conservative, well-regularized config
            {
                "n_estimators": 200,
                "max_depth": 4,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
            },
        ],
    ),
    runtime_config=RuntimeConfig(n_jobs=-1, parallel_backend="cv", verbose=True),
    random_state=42,
)
ga.fit(X_train, y_train)

print("Best CV ROC AUC:", round(ga.best_score_, 4))
print("Best params:", ga.best_params_)
```

## Key Points

- **`PopulationConfig(warm_start_configs=[...])`**: Each dict in the list seeds one individual in the first population. Must match all keys in `param_grid`.
- **`initializer="smart"`**: Combines warm-start seeds with Latin hypercube sampling for the remaining population slots.
- **Seed count ≤ `population_size`**: If you provide more seeds than the population size, excess seeds are ignored.
- **Values must be in range**: Each seeded value must be within the `param_grid` bounds. Out-of-bounds values raise an error.
- **Skip cold start**: Without warm start, the first generation can waste 10–20% of your budget rediscovering that, for example, `learning_rate=0.001` is too slow.

## Multiple Seeds from Prior Runs

If you've run a previous search, you can seed the next one with its top results:

```python
import pandas as pd

# After a previous run
top_k = pd.DataFrame(old_ga.cv_results_).nlargest(3, "mean_test_score")
warm_configs = top_k[[f"param_{p}" for p in param_grid]].rename(
    columns=lambda c: c.replace("param_", "")
).to_dict("records")

ga = GASearchCV(..., population_config=PopulationConfig(warm_start_configs=warm_configs), ...)
```

## See Also

- [Adaptive Crossover & Mutation Schedules](../../guide/adapters) — anneal exploration after warm start
- [Resume a Stopped Search](./checkpointing) — full resume vs warm start
- [XGBoost Hyperparameter Tuning](../../tutorials/tune-xgboost) — warm start in a full tutorial
