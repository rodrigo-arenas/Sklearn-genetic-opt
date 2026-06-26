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

Each call to `fit` creates:
- One **parent run** for the full search (with `run_name` and `tags`)
- One **child run** per evaluated candidate (with parameters and CV score)

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
| `run_name` | str | — | Name for the parent run (stored as a `mlflow.runName` tag) |
| `save_models` | bool | `False` | Log the fitted estimator as an artifact for each candidate run |
| `registry_uri` | str | None | Address of the MLflow model registry server |
| `tags` | dict | None | Dictionary of tag name → value applied to the parent run |

### Usage

```python
from sklearn_genetic.mlflow_log import MLflowConfig

mlflow_config = MLflowConfig(
    tracking_uri="http://localhost:5000",
    experiment="my-experiment",
    run_name="RF hyperparameter search",
    save_models=True,
    tags={"team": "ml", "dataset": "breast_cancer"},
)

search = GASearchCV(
    estimator=your_estimator,
    param_grid=your_param_grid,
    cv=your_cv,
    scoring="roc_auc",
    evolution_config=...,
    log_config=mlflow_config,  # <-- pass here
)

search.fit(X_train, y_train)
```

## See Also

- [MLflow Integration Guide](../guide/mlflow) — full tutorial with setup instructions
- [Callbacks](./callbacks) — combine MLflow logging with early stopping
