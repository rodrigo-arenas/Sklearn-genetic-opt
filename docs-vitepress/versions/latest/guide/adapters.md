---
title: Adapters
description: Use adapters to integrate sklearn-genetic-opt with other scikit-learn utilities like feature selection pipelines.
---

:::warning Development version
You are reading the **latest (dev)** docs. For the stable version, see [0.13](/versions/0.13/).
:::

# Adapters

Adapters let you use the output of `GAFeatureSelectionCV` in contexts that expect a sklearn transformer, such as inside a `Pipeline` or with `SelectFromModel`.

## Prerequisites

- Completed [Basic Usage](./basic-usage)

## Feature Selection as a Transformer

After fitting `GAFeatureSelectionCV`, wrap it in an adapter to use the selected feature mask in a downstream pipeline:

```python
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from sklearn_genetic import EvolutionConfig, GAFeatureSelectionCV, PopulationConfig, RuntimeConfig
from sklearn_genetic.mlflow import GAFeatureSelectionCVAdapter
import numpy as np

# Build a feature-selection search
X, y = load_iris(return_X_y=True)
noise = np.random.uniform(0, 10, size=(X.shape[0], 10))
X = np.hstack((X, noise))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

selector = GAFeatureSelectionCV(
    estimator=RandomForestClassifier(random_state=42),
    cv=3,
    scoring="accuracy",
    evolution_config=EvolutionConfig(population_size=20, generations=10),
    population_config=PopulationConfig(initializer="smart"),
    runtime_config=RuntimeConfig(n_jobs=-1),
)

selector.fit(X_train, y_train)

print("Selected features:", selector.support_)
```

## Tips & Gotchas

- `GAFeatureSelectionCV` already exposes `transform` and `predict` directly — use it as a drop-in estimator for prediction tasks.
- For pure feature selection followed by a different estimator, extract the `support_` mask and apply it yourself: `X_selected = X[:, selector.support_]`.

## Next Steps

- [Pipeline Tuning](./pipeline-tuning) — combine feature selection with hyperparameter tuning.
- [MLflow Integration](./mlflow) — log feature selection results to MLflow.
