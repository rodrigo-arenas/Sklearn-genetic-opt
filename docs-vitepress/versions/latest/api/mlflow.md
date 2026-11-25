---
title: MLflow API
description: API reference for MLflowConfig — log sklearn-genetic-opt experiments to MLflow.
---

:::warning Development version
You are reading the **latest (dev)** docs. For the stable version, see [stable](/stable/).
:::

# MLflow API

```python
from sklearn_genetic.mlflow_log import MLflowConfig
```

Requires MLflow: `pip install sklearn-genetic-opt[mlflow]`.

## MLflowConfig

Configures MLflow logging for a `GASearchCV` or `GAFeatureSelectionCV` run. Pass an instance to the `log_config` parameter of the search estimator.

The logger opens a run for each fresh candidate evaluation and writes candidate parameters and the CV objective as `score`. The runs are nested only when an active parent is present; `fit` does not itself open a parent.

```python
MLflowConfig(
    tracking_uri,
    experiment,
    run_name,
    save_models=False,
    registry_uri=None,
    tags=None,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tracking_uri` | str | — | Address of the MLflow tracking server (e.g., `"http://localhost:5000"`) |
| `experiment` | str | — | Name of the MLflow experiment. Created if it doesn't exist |
| `run_name` | str | — | Name used for each candidate run |
| `save_models` | bool | `False` | Log the configured candidate estimator, not one of the fitted CV-fold clones |
| `registry_uri` | str | None | Address of the MLflow model registry server |
| `tags` | dict | None | Tags applied to the active run when the configuration is constructed; with no active run, MLflow can start one |

### Usage

```python
from sklearn_genetic.mlflow_log import MLflowConfig

mlflow_config = MLflowConfig(
    tracking_uri="http://localhost:5000",
    experiment="my-experiment",
    run_name="RF hyperparameter search",
    save_models=False,
)

search = GASearchCV(
    estimator=your_estimator,
    param_grid=your_param_grid,
    cv=your_cv,
    scoring="roc_auc",
    log_config=mlflow_config,  # <-- pass here
)

search.fit(X_train, y_train)
```

To group candidate runs, explicitly start a parent run in the same experiment around `fit`. To save a fitted model, log `search.best_estimator_` after a successful refit. The [runnable recipe](../recipes/integrations/mlflow-logging) uses a local SQLite database.

## See Also

- [MLflow Integration Guide](../guide/mlflow) — full tutorial with setup instructions
- [Callbacks](./callbacks) — combine MLflow logging with early stopping
