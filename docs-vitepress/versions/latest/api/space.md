---
title: Search Space
description: API reference for Integer, Continuous, and Categorical search space dimensions.
---

:::warning Development version
You are reading the **latest (dev)** docs. For the stable version, see [stable](/stable/).
:::

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

## Convert sklearn/scipy-style spaces

`from_sklearn_space` converts common `RandomizedSearchCV`-style dictionaries into
native `sklearn-genetic-opt` dimensions:

```python
from scipy import stats

from sklearn_genetic.space import from_sklearn_space

param_grid = from_sklearn_space({
    "n_estimators": stats.randint(50, 300),
    "learning_rate": stats.loguniform(1e-3, 1e-1),
    "max_features": stats.uniform(0.2, 0.8),
    "criterion": ["gini", "entropy"],
})
```

Conversion rules:

| Input value | Output dimension |
|-------------|------------------|
| `list`, `tuple`, `set`, `range`, numpy array | `Categorical([...])` |
| `scipy.stats.randint(low, high)` | `Integer(low, high - 1)` |
| `scipy.stats.uniform(loc, scale)` | `Continuous(loc, loc + scale)` |
| `scipy.stats.loguniform(a, b)` / `reciprocal(a, b)` | `Continuous(a, b, distribution="log-uniform")` |
| Existing `Integer`, `Continuous`, `Categorical` | returned unchanged |

Unsupported scipy distributions raise an actionable `ValueError` that names the
distribution and suggests defining the corresponding `Integer`, `Continuous`, or
`Categorical` dimension manually.

## See Also

- [Basic Usage](../guide/basic-usage) — tutorial using all three dimension types
- [Presets](./presets) — starter spaces for common scikit-learn estimators
- [GASearchCV](./gasearchcv) — the search estimator that consumes `param_grid`
