---
title: MLflow 3 Experiment Tracking
description: Log a GASearchCV run to MLflow 3 — nested candidate runs via MLflowConfig, parent run with dataset inputs and holdout metrics, and logged-model lifecycle management.
---
# MLflow 3 Experiment Tracking

This example combines `sklearn-genetic-opt`'s built-in `MLflowConfig` integration with MLflow 3 tracking features: dataset inputs, logged models, model tags, and searchable run metadata.

## Logging Architecture

| Level | What is logged |
|-------|----------------|
| Parent run | Dataset inputs, optimizer settings, best parameters, holdout metrics, `fit_stats_`, fitted best estimator |
| Nested candidate run (per candidate) | Hyperparameter values, cross-validation score (via `MLflowConfig`) |

## Setup

```python
from pprint import pprint
import warnings

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from sklearn_genetic import (
    EvolutionConfig, GASearchCV, OptimizationConfig, PopulationConfig, RuntimeConfig,
)
from sklearn_genetic.callbacks import ConsecutiveStopping, DeltaThreshold, TimerStopping
from sklearn_genetic.mlflow_log import MLflowConfig
from sklearn_genetic.schedules import ExponentialAdapter, InverseAdapter
from sklearn_genetic.space import Categorical, Continuous, Integer

warnings.filterwarnings("ignore", category=UserWarning)

RANDOM_STATE = 42
TRACKING_URI = "sqlite:///mlflow3_tracking.db"
EXPERIMENT_NAME = "sklearn-genetic-opt-mlflow3"

data = load_breast_cancer(as_frame=True)
X = data.data
y = data.target.rename("target")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
)
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
```

## Create a Local MLflow Experiment

A SQLite tracking URI is convenient for local tutorials. Replace `TRACKING_URI` with your remote server address for production use.

```python
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# Log dataset inputs for reproducibility
train_dataset = mlflow.data.from_pandas(
    pd.concat([X_train, y_train], axis=1),
    targets="target",
    name="breast-cancer-train",
)
test_dataset = mlflow.data.from_pandas(
    pd.concat([X_test, y_test], axis=1),
    targets="target",
    name="breast-cancer-test",
)
```

## Configure the Search

`MLflowConfig` is passed to `log_config` — it automatically creates nested candidate runs during `search.fit`.

```python
param_grid = {
    "n_estimators": Integer(40, 120),
    "max_depth": Integer(2, 10),
    "min_samples_split": Integer(2, 12),
    "min_samples_leaf": Integer(1, 8),
    "max_features": Categorical(["sqrt", "log2", None]),
    "ccp_alpha": Continuous(0.0, 0.03),
}

mlflow_config = MLflowConfig(
    tracking_uri=TRACKING_URI,
    experiment=EXPERIMENT_NAME,
    run_name="candidate-random-forest",
    save_models=False,   # omit candidate artifacts to save storage
)

search = GASearchCV(
    estimator=RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
    param_grid=param_grid,
    scoring="roc_auc",
    cv=cv,
    evolution_config=EvolutionConfig(
        population_size=12,
        generations=8,
        crossover_probability=ExponentialAdapter(initial_value=0.8, end_value=0.4, adaptive_rate=0.15),
        mutation_probability=InverseAdapter(initial_value=0.25, end_value=0.08, adaptive_rate=0.25),
        tournament_size=3,
        elitism=True,
        keep_top_k=3,
    ),
    population_config=PopulationConfig(
        initializer="smart",
        warm_start_configs=[{
            "n_estimators": 80, "max_depth": 6,
            "min_samples_split": 4, "min_samples_leaf": 2,
            "max_features": "sqrt", "ccp_alpha": 0.0,
        }],
    ),
    runtime_config=RuntimeConfig(
        n_jobs=-1,
        parallel_backend="auto",
        use_cache=True,
        verbose=True,
        return_train_score=False,
    ),
    optimization_config=OptimizationConfig(
        local_search=True, local_search_top_k=2, local_search_steps=1, local_search_radius=0.20,
        diversity_control=True, diversity_threshold=0.30,
        diversity_stagnation_generations=3, diversity_mutation_boost=1.8,
        random_immigrants_fraction=0.10,
        fitness_sharing=True, sharing_radius=0.40,
    ),
    log_config=mlflow_config,   # <-- attach MLflow logging
)
```

## Run Inside a Parent MLflow Run

The parent run holds the overall search summary. Nested candidate runs are created automatically by `MLflowConfig` during `search.fit`.

```python
callbacks = [
    DeltaThreshold(threshold=0.0005, generations=5, metric="fitness_best"),
    ConsecutiveStopping(generations=7, metric="fitness_best"),
    TimerStopping(total_seconds=120),
]

with mlflow.start_run(run_name="ga-random-forest-search") as parent_run:
    mlflow.set_tags({
        "project": "sklearn-genetic-opt",
        "run_level": "parent",
        "optimizer": "GASearchCV",
    })
    mlflow.log_input(train_dataset, context="training")
    mlflow.log_input(test_dataset, context="holdout")
    mlflow.log_params({
        "population_size": search.population_size,
        "generations": search.generations,
        "local_search": search.local_search,
        "diversity_control": search.diversity_control,
        "fitness_sharing": search.fitness_sharing,
    })

    # Initialize logged-model record before fitting
    logged_model = mlflow.initialize_logged_model(
        name="ga-random-forest-best-model",
        source_run_id=parent_run.info.run_id,
        model_type="classifier",
        tags={"stage": "candidate", "owner": "sklearn-genetic-opt"},
    )

    search.fit(X_train, y_train, callbacks=callbacks)

    # Log holdout metrics and best parameters
    predictions = search.predict(X_test)
    probabilities = search.predict_proba(X_test)[:, 1]
    holdout_metrics = {
        "holdout_accuracy": accuracy_score(y_test, predictions),
        "holdout_balanced_accuracy": balanced_accuracy_score(y_test, predictions),
        "holdout_roc_auc": roc_auc_score(y_test, probabilities),
    }
    mlflow.log_metrics(holdout_metrics)
    mlflow.log_metric("best_cv_roc_auc", search.best_score_)
    mlflow.log_params({f"best__{k}": v for k, v in search.best_params_.items()})
    mlflow.log_metrics({
        f"fit_stats_{k}": v
        for k, v in search.fit_stats_.items()
        if isinstance(v, (int, float))
    })

    # Log the best estimator and finalize the model record
    mlflow.sklearn.log_model(
        sk_model=search.best_estimator_,
        name="best_estimator",
        model_id=logged_model.model_id,
        input_example=X_test.head(5),
        params=search.best_params_,
        tags={"optimizer": "GASearchCV", "dataset": "breast_cancer"},
        model_type="classifier",
    )
    mlflow.set_logged_model_tags(
        logged_model.model_id,
        {
            "stage": "validated",
            "best_cv_roc_auc": f"{search.best_score_:.4f}",
            "holdout_roc_auc": f"{holdout_metrics['holdout_roc_auc']:.4f}",
        },
    )
    mlflow.finalize_logged_model(logged_model.model_id, status="READY")

print("Parent run ID:", parent_run.info.run_id)
print("Best CV ROC AUC:", round(search.best_score_, 4))
pprint(search.best_params_)
```

## Inspect Results

```python
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

# List runs — parent run has summary metrics, candidate runs have per-candidate scores
runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["attributes.start_time DESC"],
)
print(runs[["tags.mlflow.runName", "metrics.score", "metrics.best_cv_roc_auc"]].head(10))

# List logged models
logged_models = mlflow.search_logged_models(
    experiment_ids=[experiment.experiment_id],
    order_by=[{"field_name": "creation_time", "ascending": False}],
    output_format="list",
)
print([(m.model_id, m.name, m.status) for m in logged_models[:5]])
```

## Open the MLflow UI

```bash
mlflow ui --backend-store-uri sqlite:///mlflow3_tracking.db
# Open http://127.0.0.1:5000 in your browser
```

In the UI you will see:
- A parent run for the full search with optimizer settings and holdout metrics
- A nested child run for **each evaluated candidate** with its parameters and CV score

## Practical Notes

- Use a parent run for the overall search and nested runs for candidate-level details.
- Log datasets with `mlflow.log_input` so future readers know which data context produced the model.
- Keep `save_models=False` in `MLflowConfig` if candidate-level model artifacts are too heavy; log only the final `best_estimator_` from the parent run.
- Use logged-model tags for lifecycle metadata such as `stage`, validation metrics, owner, and optimizer settings.
- For remote tracking, replace `TRACKING_URI` with your MLflow tracking server URI.

## See Also

- [MLflow Integration Guide](../guide/mlflow) — setup and full configuration reference
- [MLflow API](../api/mlflow) — `MLflowConfig` parameter reference
- [Callbacks](../guide/callbacks) — combine MLflow logging with early stopping
