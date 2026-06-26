---
title: MLflow Integration
description: Log sklearn-genetic-opt experiments to MLflow using MLflowConfig — track hyperparameter search runs, compare experiments, and save fitted models.
---

:::warning Development version
You are reading the **latest (dev)** docs. For the stable version, see [stable](/stable/).
:::

**Estimated reading time:** 10 minutes  
**Difficulty:** Intermediate  
**Prerequisites:** [Getting Started with GASearchCV](./basic-usage), MLflow installed (`pip install sklearn-genetic-opt[mlflow]`)

# MLflow Integration

`sklearn-genetic-opt` has built-in integration with MLflow via `MLflowConfig`. It logs each evaluated candidate as a nested child run within a parent experiment run, making it easy to compare hyperparameter configurations in the MLflow UI.

## Prerequisites

- MLflow installed: `pip install sklearn-genetic-opt[mlflow]`
- An MLflow tracking server running, or the default local file store

## Configuration

Import and configure `MLflowConfig` with your tracking server details:

```python
from sklearn_genetic.mlflow_log import MLflowConfig

mlflow_config = MLflowConfig(
    tracking_uri="http://localhost:5000",
    experiment="my-experiment",
    run_name="RandomForest search",
    save_models=True,
    tags={"team": "ml-team", "version": "0.13"},
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tracking_uri` | str | — | Address of the MLflow tracking server (e.g., `"http://localhost:5000"`) |
| `experiment` | str | — | Name of the MLflow experiment. Created if it doesn't exist |
| `run_name` | str | — | Name for the parent run |
| `save_models` | bool | `False` | If `True`, log the fitted estimator as an MLflow artifact for each candidate |
| `registry_uri` | str | `None` | Address of the MLflow model registry server |
| `tags` | dict | `None` | Tags to apply to the run |

Then pass `mlflow_config` to `GASearchCV` via the `log_config` parameter:

## Full Example

```python
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier

from sklearn_genetic import EvolutionConfig, GASearchCV, PopulationConfig, RuntimeConfig
from sklearn_genetic.mlflow_log import MLflowConfig
from sklearn_genetic.space import Categorical, Continuous, Integer

data = load_digits()
X, y = data["data"], data["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

mlflow_config = MLflowConfig(
    tracking_uri="http://localhost:5000",
    experiment="Digits-sklearn-genetic-opt",
    run_name="Decision Tree",
    save_models=True,
    tags={"team": "sklearn-genetic-opt"},
)

param_grid = {
    "min_weight_fraction_leaf": Continuous(0, 0.5),
    "criterion": Categorical(["gini", "entropy"]),
    "max_depth": Integer(2, 20),
    "max_leaf_nodes": Integer(2, 30),
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

evolved_estimator = GASearchCV(
    estimator=DecisionTreeClassifier(),
    cv=cv,
    scoring="accuracy",
    param_grid=param_grid,
    evolution_config=EvolutionConfig(
        population_size=10,
        generations=8,
        tournament_size=3,
        elitism=True,
        crossover_probability=0.9,
        mutation_probability=0.05,
    ),
    population_config=PopulationConfig(initializer="smart"),
    runtime_config=RuntimeConfig(n_jobs=-1, verbose=True),
    log_config=mlflow_config,
)

evolved_estimator.fit(X_train, y_train)
y_pred = evolved_estimator.predict(X_test)
print(evolved_estimator.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred))
```

## Starting the MLflow UI

Start the local tracking server:

```bash
mlflow ui
# Open http://127.0.0.1:5000
```

In the MLflow UI you will see:

- A parent run for the full `GASearchCV.fit()` call with your `run_name` and `tags`
- A nested child run for **each evaluated candidate**, showing its parameters and cross-validation score
- If `save_models=True`, each child run has the fitted estimator attached as an artifact

If you run `fit` again on the same experiment, a second parent run appears — click the "+" symbol on each parent to expand its children and compare runs.

## What Gets Logged

| Level | What is logged |
|-------|---------------|
| Parent run | Experiment metadata, `run_name`, tags |
| Child run (per candidate) | Hyperparameter values, cross-validation score |
| Artifact (if `save_models=True`) | The fitted estimator for that candidate |

## Tips & Gotchas

- Each `GASearchCV.fit()` call creates one parent run and one child run per evaluated candidate. With `population_size=20` and `generations=10`, expect up to 200 child runs (fewer with caching).
- Use `RuntimeConfig(use_cache=True)` (the default) to avoid logging duplicate candidates.
- The `save_models=True` option can create many artifacts if the population is large. Use it selectively for the best candidates or disable it to save storage.
- If the tracking server is not running, `MLflowConfig` will raise a connection error. Start `mlflow ui` before calling `fit`.

## Viewing the Experiment

```bash
# Start the local server
mlflow ui

# Or point to a remote tracking URI in MLflowConfig:
mlflow_config = MLflowConfig(
    tracking_uri="http://your-server:5000",
    experiment="...",
    run_name="...",
)
```

## See Also

- [Early Stopping with Callbacks](./callbacks) — combine MLflow logging with early stopping callbacks
- [Reproducibility & Checkpointing](./reproducibility) — checkpoint and resume alongside MLflow tracking
- [Multi-Metric Optimization](./multi-metric) — log multiple metrics to MLflow
- [Examples: MLflow Experiment Tracking](../examples/mlflow-tracking) — complete end-to-end example
- [API: MLflow](../api/mlflow) — MLflowConfig reference
