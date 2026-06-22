---
title: Search Space
description: API reference for Integer, Continuous, and Categorical search space dimensions.
---

# Search Space

The `sklearn_genetic.space` module provides three dimension types for defining the hyperparameter search space.

## Integer

Samples integer values from a range `[lower, upper]`.

```python
from sklearn_genetic.space import Integer

Integer(lower, upper, distribution="uniform")
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lower` | int | — | Minimum value (inclusive) |
| `upper` | int | — | Maximum value (inclusive) |
| `distribution` | str | `"uniform"` | Sampling distribution: `"uniform"` |

**Example:**

```python
"n_estimators": Integer(50, 500),
"max_depth": Integer(1, 20),
```

## Continuous

Samples floating-point values from a range `[lower, upper]`.

```python
from sklearn_genetic.space import Continuous

Continuous(lower, upper, distribution="uniform")
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lower` | float | — | Minimum value |
| `upper` | float | — | Maximum value |
| `distribution` | str | `"uniform"` | Sampling distribution: `"uniform"` or `"log-uniform"` |

Use `distribution="log-uniform"` for parameters that span orders of magnitude (learning rates, regularization strengths):

```python
"learning_rate": Continuous(1e-4, 1e-1, distribution="log-uniform"),
"alpha": Continuous(1e-6, 1.0, distribution="log-uniform"),
```

## Categorical

Samples from a fixed list of choices.

```python
from sklearn_genetic.space import Categorical

Categorical(choices)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `choices` | list | — | List of valid values. Can include `None` |

**Example:**

```python
"max_features": Categorical(["sqrt", "log2", None]),
"solver": Categorical(["lbfgs", "sgd", "adam"]),
"activation": Categorical(["relu", "tanh", "logistic"]),
```

## Complete Example

```python
from sklearn_genetic.space import Categorical, Continuous, Integer

param_grid = {
    "n_estimators": Integer(50, 300),
    "max_depth": Integer(2, 15),
    "learning_rate": Continuous(0.01, 0.3, distribution="log-uniform"),
    "subsample": Continuous(0.5, 1.0),
    "max_features": Categorical(["sqrt", "log2"]),
    "min_samples_leaf": Integer(1, 20),
}
```

## See Also

- [Basic Usage](../guide/basic-usage) — tutorial using all three dimension types
- [GASearchCV](./gasearchcv) — the search estimator that consumes `param_grid`
