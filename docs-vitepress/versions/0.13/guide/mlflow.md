---
title: MLflow Integration
description: Log sklearn-genetic-opt experiments to MLflow 3 using MLflowCallback.
---

# MLflow Integration

`sklearn-genetic-opt` integrates with MLflow 3 to log hyperparameter search experiments — including per-generation metrics, the best parameters, and fitted model artifacts.

## Prerequisites

- MLflow installed: `pip install sklearn-genetic-opt[mlflow]`
- An MLflow tracking server or the local file store

## Quick Start

```python
import mlflow
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split

from sklearn_genetic import EvolutionConfig, GASearchCV, PopulationConfig, RuntimeConfig
from sklearn_genetic.callbacks import MLflowCallback
from sklearn_genetic.space import Categorical, Continuous, Integer

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

mlflow.set_experiment("sklearn-genetic-opt-demo")

with mlflow.start_run():
    search = GASearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid={
            "n_estimators": Integer(50, 200),
            "max_depth": Integer(2, 10),
            "min_samples_leaf": Integer(1, 8),
            "max_features": Categorical(["sqrt", "log2"]),
            "ccp_alpha": Continuous(0.0, 0.02),
        },
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        scoring="roc_auc",
        evolution_config=EvolutionConfig(population_size=20, generations=12),
        population_config=PopulationConfig(initializer="smart"),
        runtime_config=RuntimeConfig(n_jobs=-1, use_cache=True),
    )

    search.fit(X_train, y_train, callbacks=[MLflowCallback()])

    mlflow.log_params(search.best_params_)
    mlflow.log_metric("best_roc_auc", search.best_score_)
```

## What Gets Logged

`MLflowCallback` logs per-generation metrics as MLflow step metrics:

- `fitness` — mean population fitness
- `fitness_best` — best fitness found so far
- `fitness_std` — standard deviation of fitness
- `genotype_diversity` — population diversity
- `stagnation_generations` — consecutive stagnant generations

After the run, you can view these in the MLflow UI with `mlflow ui`.

## Viewing the Experiment

```bash
mlflow ui
# Open http://127.0.0.1:5000
```

## Tips & Gotchas

- Start the `mlflow.start_run()` context before calling `fit` so all generation metrics fall inside the same run.
- For long searches, use `LogbookSaver` alongside `MLflowCallback` as a local backup.
- MLflow 3 is required — older versions are not supported.

## Next Steps

- [Callbacks](./callbacks) — combine MLflowCallback with early stopping.
- [Reproducibility](./reproducibility) — log seeds alongside metrics for fully reproducible runs.
