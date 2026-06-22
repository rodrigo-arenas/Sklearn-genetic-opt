---
title: Plots
description: API reference for plot_fitness_evolution and plot_search_space.
---

# Plots

```python
from sklearn_genetic.plots import plot_fitness_evolution, plot_search_space
```

Requires seaborn: `pip install sklearn-genetic-opt[all]`.

## plot_fitness_evolution

Plot the fitness score over generations.

```python
plot_fitness_evolution(estimator, metric="fitness")
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `estimator` | fitted `GASearchCV` or `GAFeatureSelectionCV` | — | The fitted search estimator |
| `metric` | str | `"fitness"` | Which fitness column to plot: `"fitness"`, `"fitness_best"`, `"fitness_std"` |

Returns a `matplotlib.axes.Axes`.

**Example:**

```python
import matplotlib.pyplot as plt
from sklearn_genetic.plots import plot_fitness_evolution

plot_fitness_evolution(search, metric="fitness_best")
plt.title("Best fitness over generations")
plt.show()
```

## plot_search_space

Plot the distribution of sampled hyperparameter values as a pair plot.

```python
plot_search_space(estimator, features=None, figsize=(12, 12))
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `estimator` | fitted `GASearchCV` | — | The fitted search estimator |
| `features` | list of str | `None` | Parameter names to include. `None` includes all |
| `figsize` | tuple | `(12, 12)` | Figure size |

Returns a `seaborn.PairGrid`.

**Example:**

```python
from sklearn_genetic.plots import plot_search_space

plot_search_space(search, features=["learning_rate", "max_depth", "n_estimators"])
plt.show()
```

## See Also

- [Basic Usage](../guide/basic-usage) — tutorial with both plot types
