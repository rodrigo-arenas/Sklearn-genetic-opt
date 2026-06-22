---
title: Reproducibility
description: Make sklearn-genetic-opt results reproducible across runs by seeding all random sources.
---

# Reproducibility

Genetic algorithms are stochastic — results vary between runs unless every random source is seeded. This page shows a complete reproducibility setup.

## Prerequisites

- Completed [Basic Usage](./basic-usage)

## Seed All Random Sources

```python
import random
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split

from sklearn_genetic import EvolutionConfig, GASearchCV, PopulationConfig, RuntimeConfig
from sklearn_genetic.space import Categorical, Continuous, Integer

# 1. Seed Python's built-in random module
random.seed(42)

# 2. Seed NumPy
np.random.seed(42)

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42  # 3. Seed the split
)

search = GASearchCV(
    estimator=RandomForestClassifier(random_state=42),  # 4. Seed the estimator
    param_grid={
        "n_estimators": Integer(50, 250),
        "max_depth": Integer(2, 14),
        "max_features": Categorical(["sqrt", "log2"]),
        "ccp_alpha": Continuous(0.0, 0.03),
    },
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),  # 5. Seed CV
    scoring="roc_auc",
    evolution_config=EvolutionConfig(population_size=20, generations=12),
    population_config=PopulationConfig(initializer="smart"),
    runtime_config=RuntimeConfig(n_jobs=1),  # 6. n_jobs=1 for full determinism
)

search.fit(X_train, y_train)
```

## Why `n_jobs=1`?

Parallel execution with `n_jobs > 1` uses `joblib`, which may introduce non-determinism when multiple workers race to write results. For fully reproducible results, use `n_jobs=1`. For production runs where speed matters, accept that results may vary slightly between runs.

## Checkpointing

For long runs, save the logbook periodically so you can inspect progress even if the search is interrupted:

```python
from sklearn_genetic.callbacks import LogbookSaver

search.fit(X_train, y_train, callbacks=[
    LogbookSaver(checkpoint_path="./logbook.pkl")
])

# Load later:
import pickle
with open("./logbook.pkl", "rb") as f:
    logbook = pickle.load(f)
```

## Tips & Gotchas

- Seed `random`, `np.random`, the CV splitter, and every estimator that accepts `random_state`.
- If using `PopulationConfig(warm_start_configs=[...])`, the warm-start configs are deterministic by definition.
- DEAP's internal RNG is seeded through Python's `random` module — seeding `random` before `fit` is sufficient.

## Next Steps

- [Troubleshooting](./troubleshooting#reproducibility) — debugging non-deterministic results.
- [Callbacks](./callbacks) — use `LogbookSaver` for checkpointing.
