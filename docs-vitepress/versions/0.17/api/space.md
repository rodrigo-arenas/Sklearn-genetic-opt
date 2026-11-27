---
title: Search Space
description: "API reference for the Integer, Continuous and Categorical search space dimensions, and for converting scikit-learn and SciPy-style parameter spaces."
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

Candidate repair and local refinement preserve integer values exactly,
including adjacent values above `2**53`. Pass large integer warm-start
values as Python or NumPy integers to preserve their identity.
Equal bounds hold an integer parameter fixed. Sampling that dimension returns
the exact Python integer, including values outside the signed 64-bit range.
To evaluate a few scales early while keeping a full integer range, see the
[integer warm-start example](../guide/choosing-search-spaces#start-an-integer-search-at-several-scales).

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

Equal bounds hold a continuous parameter fixed. For example,
`Continuous(0.01, 0.01, distribution="log-uniform")` always proposes 0.01
while other dimensions evolve. Bounds must be finite and within the float
range. A fixed value must be exactly representable as a float. A variable
interval must contain at least one representable float and uniform sampling
requires a finite span. Use `Integer` when you need to sample exact integers
beyond float precision. Log-uniform bounds must both be positive. If distinct bounds
round to the same logarithm, sampling uses the interval in the original
scale at float precision.

Use `distribution="log-uniform"` for parameters that span orders of magnitude (learning rates, regularization strengths):

```python
"learning_rate": Continuous(1e-4, 1e-1, distribution="log-uniform"),
"alpha": Continuous(1e-6, 1.0, distribution="log-uniform"),
```

## Categorical

Values that compare equal in Python can configure an estimator differently.
For example, `Categorical([1, 1.0])` keeps two distinct per-split feature limits
for a decision tree's `max_features`. The integer uses one feature and the float
uses the full feature count. The cache, coverage
count and finalist shortlist use the declared types to distinguish them.
When an older checkpoint may have merged their scores, start the search
with a new checkpoint file.

Samples from a fixed list of choices.

```python
from sklearn_genetic.space import Categorical

Categorical(choices, priors=None, distribution="choice", random_state=None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `choices` | list | — | List of valid values. Can include `None` |
| `priors` | list of floats | `None` | Sampling probabilities in the same order as `choices`. They must be nonnegative and sum to one. |
| `distribution` | str | `"choice"` | Distribution used to sample this dimension. |
| `random_state` | int or None | `None` | Seed for direct sampling of this dimension. |

For example, `Categorical(["gini", "entropy"], priors=[0.75, 0.25])` assigns three quarters of the sampling probability to Gini. Smart initialization stratifies the weighted draws across the available population slots. With four slots this allocation gives three Gini draws and one entropy draw before duplicate removal. With fewer slots, the search seed controls which categories receive the fractional slots. Warm starts and estimator defaults are added separately.

When all positive weights are equal, the stratified draws use the same seeded sample as `priors=None` over the eligible choices. If fewer slots remain than eligible choices, any subset of the requested size can be drawn.

**Example:**

```python
"max_features": Categorical(["sqrt", "log2", None]),
"solver": Categorical(["lbfgs", "sgd", "adam"]),
"activation": Categorical(["relu", "tanh", "logistic"]),
```

For [preconfigured estimator choices](../examples/conditional-pipeline-choices),
the search matches a copied estimator to a choice by its constructor
parameters. An exact match in values and types takes priority over a choice
whose values merely compare equal.
Object arrays in constructor parameters compare by shape, dtype and cell
contents, including nested arrays and dictionaries. For example, an integer
`1` in an object-array cell prefers the declared integer choice when a
Boolean `True` choice also compares equal.

If a pipeline step accepts missing-value markers, you can include them in its
search space. For a step named `imputer` that uses `SimpleImputer` with numeric
data, you can use the following dimension.

```python
import numpy as np
import pandas as pd
from sklearn_genetic.space import Categorical

param_grid = {"imputer__missing_values": Categorical([np.nan, pd.NA])}
```

Both markers are separate choices and can use the cache. With markers whose
equality is ambiguous or non-reflexive, the search skips the early stopping
check for full finite-space coverage. Other stopping conditions still apply.

For a floating-point NaN listed directly in `choices`, warm starts and
checkpoints match the declared type and bit pattern. This applies to Python
`float` and NumPy `float16`, `float32`, and `float64`. Use the same scalar
type in `warm_start_configs` as in `choices`.

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
native `sklearn-genetic-opt` dimensions. Use it when migrating an existing
`param_distributions` mapping; when defining a new space, prefer the native
dimensions directly:

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

- [Migrating from RandomizedSearchCV](../guide/choosing-search-spaces#migrating-from-randomizedsearchcv) — copyable conversion recipe and bound semantics
- [Basic Usage](../guide/basic-usage) — tutorial using all three dimension types
- [Presets](./presets) — starter spaces for common scikit-learn estimators
- [GASearchCV](./gasearchcv) — the search estimator that consumes `param_grid`
