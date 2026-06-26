---
title: Reproducibility
description: Make sklearn-genetic-opt results reproducible across runs by seeding all random sources.
---

# Reproducibility

Genetic algorithms are stochastic — results vary between runs unless the
randomness is seeded. The simplest, recommended way is the single
`random_state` parameter on `GASearchCV` / `GAFeatureSelectionCV`.

## Prerequisites

- Completed [Basic Usage](./basic-usage)

## One Seed Controls the Search

Pass `random_state` to the estimator. At `fit` time it seeds **every** stochastic
part of the search from that one value — population initialization (including the
Latin hypercube sampler), mutation, crossover, and random immigrants — so
repeated fits give identical results. You no longer need to seed the global
`random` / `numpy` RNGs yourself.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split

from sklearn_genetic import EvolutionConfig, GASearchCV, PopulationConfig, RuntimeConfig
from sklearn_genetic.space import Categorical, Continuous, Integer

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42  # seed the split
)

search = GASearchCV(
    estimator=RandomForestClassifier(random_state=42),  # seed the estimator
    random_state=42,                                    # seed the whole search
    param_grid={
        "n_estimators": Integer(50, 250),
        "max_depth": Integer(2, 14),
        "max_features": Categorical(["sqrt", "log2"]),
        "ccp_alpha": Continuous(0.0, 0.03),
    },
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),  # seed CV
    scoring="roc_auc",
    evolution_config=EvolutionConfig(population_size=20, generations=12),
    population_config=PopulationConfig(initializer="smart"),
    runtime_config=RuntimeConfig(n_jobs=1),  # n_jobs=1 for full determinism
)

search.fit(X_train, y_train)
```

The four `random_state=42` values above seed four independent things: the
train/test split, the estimator, the cross-validation splitter, and — the new
one — the genetic search itself. Set all four for an end-to-end reproducible run.

:::tip Leave it unset for variety
`random_state=None` (the default) keeps the search non-deterministic, matching
scikit-learn's convention — useful when you want to explore different runs.
:::

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
