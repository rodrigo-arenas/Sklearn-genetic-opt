---
title: Plotting Gallery
description: Walkthrough of all plot helpers in sklearn-genetic-opt — fitness evolution, history telemetry, and search-space visualization.
---

# Plotting Gallery

`sklearn-genetic-opt` ships three plot functions. This gallery shows each one with its main options.

## Setup

```python
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from sklearn_genetic import GASearchCV
from sklearn_genetic.plots import plot_fitness_evolution, plot_history, plot_search_space
from sklearn_genetic.space import Categorical, Continuous, Integer

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

search = GASearchCV(
    DecisionTreeRegressor(random_state=42),
    cv=2,
    scoring="r2",
    population_size=8,
    generations=10,
    tournament_size=3,
    elitism=True,
    crossover_probability=0.9,
    mutation_probability=0.05,
    param_grid={
        "ccp_alpha": Continuous(0, 1),
        "criterion": Categorical(["squared_error", "absolute_error"]),
        "max_depth": Integer(2, 20),
        "min_samples_split": Integer(2, 30),
    },
    n_jobs=1,
)

search.fit(X_train, y_train)
```

## Fitness Plots

`plot_fitness_evolution` shows how fitness changes across generations.

### Default View

```python
plot_fitness_evolution(search)
plt.show()
```

### Multi-Metric With Smoothing

```python
plot_fitness_evolution(
    search,
    metrics=["fitness_best", "fitness", "fitness_max"],
    window=2,
    kind="line",
    title="Fitness comparison with smoothing",
)
plt.show()
```

| Parameter | Description |
|-----------|-------------|
| `metrics` | List of history fields to plot. Default: `["fitness_best"]` |
| `window` | Rolling average window size. Default: `1` (no smoothing) |
| `kind` | `"line"` or `"bar"`. Default: `"line"` |
| `title` | Chart title |

## History Plots

`plot_history` can plot any field from `history` or `logbook`. Use it to inspect fitness signals, diversity indicators, or optimizer-control events.

### History Fields

```python
plot_history(
    search,
    fields=["fitness_best", "fitness", "unique_individual_ratio", "genotype_diversity"],
    kind="line",
    subplots=True,
    title="Optimizer history overview",
)
plt.show()
```

### Logbook Fields

```python
plot_history(
    search,
    fields=["score", "fit_time", "score_time"],
    source="logbook",
    kind="area",
    title="Logbook fields from candidate evaluations",
)
plt.show()
```

| Parameter | Description |
|-----------|-------------|
| `fields` | List of column names to plot |
| `source` | `"history"` (generation stats) or `"logbook"` (candidate evaluations) |
| `kind` | `"line"`, `"bar"`, or `"area"` |
| `subplots` | `True` to draw each field in its own subplot |

## Search-Space Plots

The search-space plots show how the optimizer explored the parameter space.

### Pair Plot

Shows relationships between sampled parameters. Points are colored by a category column when `hue` is set.

```python
plot_search_space(
    search,
    features=["ccp_alpha", "max_depth", "min_samples_split", "criterion"],
    hue="criterion",
    kind="pair",
)
plt.show()
```

### Heatmap

Quick correlation view of sampled numeric parameters.

```python
plot_search_space(
    search,
    features=["ccp_alpha", "max_depth", "min_samples_split"],
    kind="heatmap",
)
plt.show()
```

| Parameter | Description |
|-----------|-------------|
| `features` | Parameter names to include. Omit to use all numeric parameters |
| `hue` | Categorical column to use for color coding (pair plot only) |
| `kind` | `"pair"` or `"heatmap"` |

## Reading History Directly

All plot functions read from `search.history`. You can also work with it directly as a DataFrame for custom reporting.

```python
history = pd.DataFrame(search.history)

# Fields available in history
telemetry_columns = [
    "gen",
    "fitness_best",
    "fitness",
    "fitness_max",
    "unique_individual_ratio",
    "genotype_diversity",
    "mutation_probability",
    "selection_pressure",
    "random_immigrants",
    "duplicate_replacements",
    "local_refinements",
]

available = [c for c in telemetry_columns if c in history.columns]
print(history[available].tail())
```

## When to Use Each Plot

| Plot | Use when |
|------|----------|
| `plot_fitness_evolution` | Quick fitness trend overview |
| `plot_history` | Inspecting telemetry: diversity, stagnation, control events |
| `plot_search_space` with `kind="pair"` | Understanding parameter interactions |
| `plot_search_space` with `kind="heatmap"` | Spotting correlations between numeric parameters |

## See Also

- [Plots API](../api/plots) — full parameter reference for all three functions
- [Understanding Cross-Validation](../guide/understand-cv) — what the generation log columns mean
- [Advanced Optimizer Control](../guide/advanced-optimizer-control) — interpreting diversity and stagnation signals
