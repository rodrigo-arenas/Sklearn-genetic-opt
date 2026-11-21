---
title: "Checkpointing and Persistence"
description: "Write per-generation checkpoints with the ModelCheckpoint callback, save the logbook with LogbookSaver, and round-trip a fully fitted GASearchCV with save / load — verified against real, captured output."
---

# Checkpointing and Persistence

This example uses the default DEAP engine and its historical checkpoint format. See the [backend tutorial](../tutorials/optimizer-backends#select-features-and-inspect-progress) for a worked example of resuming feature selection with pymoo. Long-running searches should be able to write intermediate checkpoints, persist the fitted search object, and reload it later for inspection or prediction. This page runs a small real search and shows both persistence mechanisms end to end: a checkpoint callback that fires every generation, and `save` / `load` for the finished search.

## The Two Persistence Mechanisms

| Mechanism | When to use |
|-----------|-------------|
| `ModelCheckpoint` callback | Progress recovery and audit trails **during** a fit |
| `LogbookSaver` callback | Persist just the parameter logbook each generation |
| `search.save()` / `search.load()` | Reusing a fully fitted search object **after** training |

## Setup

We tune a `RandomForestClassifier` on the breast-cancer dataset. Artifacts are
written to a temporary directory outside the project tree so nothing is left in
the repo.

```python
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from sklearn_genetic import (
    EvolutionConfig,
    GASearchCV,
    OptimizationConfig,
    PopulationConfig,
    RuntimeConfig,
)
from sklearn_genetic.callbacks import ConsecutiveStopping, LogbookSaver, ModelCheckpoint
from sklearn_genetic.space import Categorical, Continuous, Integer

warnings.filterwarnings("ignore")

data = load_breast_cancer(as_frame=True)
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

artifacts = tempfile.TemporaryDirectory()
artifact_dir = Path(artifacts.name)
checkpoint_path = artifact_dir / "rf_checkpoint.pkl"
logbook_path = artifact_dir / "rf_logbook.pkl"
saved_search_path = artifact_dir / "rf_search.pkl"
```

## Search Configuration

`ModelCheckpoint` writes a checkpoint after every generation; `LogbookSaver`
persists just the parameter logbook each generation; `ConsecutiveStopping`
halts the run early if the best fitness stops improving.

```python
param_grid = {
    "n_estimators": Integer(30, 90),
    "max_depth": Integer(2, 12),
    "min_samples_leaf": Integer(1, 8),
    "max_features": Continuous(0.25, 1.0),
    "criterion": Categorical(["gini", "entropy"]),
    "class_weight": Categorical([None, "balanced"]),
}

callbacks = [
    ModelCheckpoint(checkpoint_path),     # writes estimator_state + logbook each generation
    LogbookSaver(logbook_path),           # writes just the parameter logbook each generation
    ConsecutiveStopping(generations=6, metric="fitness_best"),
]

search = GASearchCV(
    estimator=RandomForestClassifier(random_state=42, n_jobs=1),
    random_state=42,
    cv=cv,
    scoring="roc_auc",
    param_grid=param_grid,
    evolution_config=EvolutionConfig(
        population_size=10,
        generations=6,
        crossover_probability=0.85,
        mutation_probability=0.15,
        keep_top_k=4,
    ),
    population_config=PopulationConfig(initializer="random"),
    runtime_config=RuntimeConfig(use_cache=True, n_jobs=1, verbose=False),
    optimization_config=OptimizationConfig(
        diversity_control=True,
        fitness_sharing=True,
    ),
    refit=True,
)
```

## Fit With Checkpointing

Each generation updates the checkpoint. After fitting, inspect the selected
parameters and their cross-validation score.

```python
search.fit(X_train, y_train, callbacks=callbacks)
print()
print("Best params   :", search.best_params_)
print("Best CV ROC AUC:", round(search.best_score_, 4))
```

```text

Best params   : {'n_estimators': 81, 'max_depth': 9, 'min_samples_leaf': 4, 'max_features': 0.43187019562301654, 'criterion': 'entropy', 'class_weight': None}
Best CV ROC AUC: 0.9893
```

## Evaluate on the Test Set

```python
y_pred = search.predict(X_test)
y_proba = search.predict_proba(X_test)[:, 1]

pd.Series({
    "accuracy": accuracy_score(y_test, y_pred),
    "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
    "roc_auc": roc_auc_score(y_test, y_proba),
}).round(4).to_frame("test_score")
```

```text
                   test_score
accuracy               0.9580
balanced_accuracy      0.9512
roc_auc                0.9966
```

## Inspect the Checkpoint Contents

`ModelCheckpoint.load()` returns the search configuration, generation log and
runtime state. The runtime state includes candidate records, evaluation
counters and cached scores used when resuming.

```python
checkpoint = ModelCheckpoint(checkpoint_path).load()

print("checkpoint keys      :", sorted(checkpoint.keys()))
print("generations in logbook:", len(checkpoint["logbook"]))
print("Candidate CV calls:", checkpoint["runtime_state"]["fit_stats_"]["cross_validate_calls"])
```

```text
checkpoint keys      : ['estimator_state', 'logbook', 'runtime_state']
generations in logbook: 7
Candidate CV calls: 127
```

The `estimator_state` is intentionally small — it captures the search
configuration so a run can be reconstructed, not the trained estimator. For the
full fitted object, use `save` / `load` below.

## Inspect the LogbookSaver Output

`LogbookSaver` persists only the parameter chapter of the logbook using joblib,
which is handy when you want the per-candidate parameter records without the
rest of the checkpoint.

```python
import joblib

saved_logbook = joblib.load(logbook_path)
print("records saved by LogbookSaver:", len(saved_logbook))
print("fields per record           :", sorted(saved_logbook[0].keys()))
```

```text
records saved by LogbookSaver: 130
fields per record           : ['class_weight', 'criterion', 'cv_scores', 'fit_time', 'index', 'max_depth', 'max_features', 'min_samples_leaf', 'n_estimators', 'score', 'score_time', 'test_score']
```

## Save and Reload the Fitted Search

`save` pickles the full fitted search (dropping only volatile internals like the
DEAP toolbox and population). `load` restores it into a fresh `GASearchCV` that
predicts identically.

```python
search.save(saved_search_path)

restored_search = GASearchCV(
    estimator=RandomForestClassifier(random_state=42, n_jobs=1),
    random_state=42,
    cv=cv,
    scoring="roc_auc",
    param_grid=param_grid,
)
restored_search.load(saved_search_path)

restored_pred = restored_search.predict(X_test)
identical = bool((restored_pred == y_pred).all())
print("Predictions identical to original:", identical)
print("Restored best CV ROC AUC         :", round(restored_search.best_score_, 4))
```

```text
Predictions identical to original: True
Restored best CV ROC AUC         : 0.9893
```

The restored search reproduces the original predictions exactly.

```python
artifacts.cleanup()
print("Cleaned up:", not artifact_dir.exists())
```

```text
Cleaned up: True
```

## Practical Notes

- Resume from the most recent successful write using the [checkpoint recipe](../recipes/advanced/checkpointing).
- Reach for `LogbookSaver` when you only need the per-candidate parameter records.
- Use `save` / `load` for fitted search objects that need to be reused for
  prediction or later analysis — they carry the fitted estimator, not just the
  configuration.
- Store checkpoints **outside** temporary notebook directories for long runs so
  they survive process restarts.
- Keep `random_state` fixed across the estimator, splitter, and inputs to
  produce repeatable artifacts.

## See Also

- [Callbacks](../guide/callbacks) — `ModelCheckpoint`, `LogbookSaver`, and friends
- [Reproducibility](../guide/reproducibility) — fixing seeds for repeatable runs
- [Callbacks API](../api/callbacks) — `ModelCheckpoint` parameter reference
