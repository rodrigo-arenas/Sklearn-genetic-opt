---
title: Plotting Gallery
description: Walkthrough of all plot helpers in sklearn-genetic-opt — overview dashboards, fitness evolution, history telemetry, search-space visualization, and feature-selection masks.
---
:::warning Development version
This is the **latest (dev)** documentation. It may contain unreleased features or breaking changes. For the stable release, use [version 0.13](/versions/0.13/).
:::


# Plotting Gallery

`sklearn-genetic-opt` ships diagnostic plot functions that read the metadata saved after `.fit(...)`. This gallery shows how to answer the main post-search questions: what was trained, what space was explored, what decisions the optimizer made, how it converged, and whether good solutions were found.

## Setup

```python
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from sklearn_genetic import GASearchCV
from sklearn_genetic.plots import (
    SearchPlotter,
    plot_candidate_rankings,
    plot_convergence,
    plot_cv_scores,
    plot_diversity,
    plot_feature_selection,
    plot_fitness_evolution,
    plot_history,
    plot_optimizer_events,
    plot_parameter_evolution,
    plot_search_overview,
    plot_search_space,
    plot_score_landscape,
)
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

## Overview Dashboard

`plot_search_overview` is the fastest way to inspect a fitted search. It combines convergence, diversity, optimizer decisions, and the strongest evaluated candidates.

```python
plot_search_overview(search, top_k=6)
plt.show()
```

![Search overview dashboard](/images/plotting_gallery_search_overview.png)

You can also keep a fitted search wrapped in a small plotting facade:

```python
plotter = SearchPlotter(search)
plotter.convergence()
plotter.score_landscape("ccp_alpha", "max_depth")
plotter.cv_scores(top_k=5)
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

## Focused Diagnostics

Advanced users often want one figure per question instead of a dense dashboard.

### Convergence

```python
plot_convergence(search)
plt.show()
```

### Diversity

```python
plot_diversity(search)
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

## Parameter Exploration Over Time

`plot_parameter_evolution` shows the sampled value for each parameter in evaluation order. Points are colored by the selected score, which helps spot whether good candidates clustered in a narrow range.

```python
plot_parameter_evolution(
    search,
    parameters=["ccp_alpha", "max_depth", "min_samples_split"],
)
plt.show()
```

![Parameter evolution over candidate evaluations](/images/plotting_gallery_parameter_evolution.png)

## Optimizer Events

`plot_optimizer_events` shows interventions as a timeline. This is easier to scan than overlapping many step lines when you only care about when the optimizer changed behavior.

```python
plot_optimizer_events(search)
plt.show()
```

![Optimizer event timeline](/images/plotting_gallery_optimizer_events.png)

## Score Landscapes

`plot_score_landscape` highlights promising regions in a two-parameter slice of the explored search space.

```python
plot_score_landscape(search, x="ccp_alpha", y="max_depth")
plt.show()
```

![Score landscape scatter plot](/images/plotting_gallery_score_landscape.png)

Dense numeric spaces can be aggregated with hexbins:

```python
plot_score_landscape(search, x="ccp_alpha", y="max_depth", kind="hexbin")
plt.show()
```

![Score landscape hexbin plot](/images/plotting_gallery_score_landscape_hexbin.png)

## Candidate Rankings

`plot_candidate_rankings` compares top candidates with mean CV score and standard deviation.

```python
plot_candidate_rankings(search, top_k=8)
plt.show()
```

![Candidate ranking with CV uncertainty](/images/plotting_gallery_candidate_rankings.png)

## CV Robustness

`plot_cv_scores` shows fold-level scores for the strongest candidates, which helps detect winners that are not robust across splits.

```python
plot_cv_scores(search, top_k=5)
plt.show()
```

![Fold-level CV scores for top candidates](/images/plotting_gallery_cv_scores.png)

## Feature-Selection Plots

For `GAFeatureSelectionCV`, `plot_feature_selection` shows the final boolean support mask. `plot_search_overview` also works with feature-selection estimators and replaces the candidate panel with the selected-feature mask.

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.svm import SVC

from sklearn_genetic import GAFeatureSelectionCV

iris = load_iris()
X_fs, y_fs = iris.data, iris.target
noise = np.random.default_rng(42).uniform(0, 10, size=(X_fs.shape[0], 6))
X_fs = np.hstack((X_fs, noise))
feature_names = list(iris.feature_names) + [f"noise_{i}" for i in range(noise.shape[1])]

selector = GAFeatureSelectionCV(
    SVC(gamma="auto"),
    cv=3,
    scoring="accuracy",
    population_size=12,
    generations=8,
    max_features=6,
    n_jobs=1,
)
selector.fit(X_fs, y_fs)
```

```python
plot_feature_selection(selector, feature_names=feature_names)
plt.show()
```

![Selected feature mask](/images/plotting_gallery_feature_selection.png)

```python
plot_search_overview(selector)
plt.show()
```

![Feature-selection overview dashboard](/images/plotting_gallery_feature_overview.png)

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
| `plot_search_overview` | One-call diagnostic dashboard after `.fit(...)` |
| `SearchPlotter` | Repeated diagnostics from a fitted search object |
| `plot_convergence` | Inspecting fitness progress without event clutter |
| `plot_diversity` | Checking diversity collapse and stagnation |
| `plot_fitness_evolution` | Quick fitness trend overview |
| `plot_history` | Inspecting telemetry: diversity, stagnation, control events |
| `plot_search_space` with `kind="pair"` | Understanding parameter interactions |
| `plot_search_space` with `kind="heatmap"` | Spotting correlations between numeric parameters |
| `plot_parameter_evolution` | Seeing how parameter values changed across evaluations |
| `plot_optimizer_events` | Explaining optimizer-control interventions as a timeline |
| `plot_score_landscape` | Finding promising parameter regions |
| `plot_candidate_rankings` | Comparing top solutions with CV uncertainty |
| `plot_cv_scores` | Checking fold-level robustness |
| `plot_feature_selection` | Inspecting the selected support mask |

## See Also

- [Plots API](../api/plots) — full parameter reference for all plot functions
- [Understanding Cross-Validation](../guide/understand-cv) — what the generation log columns mean
- [Advanced Optimizer Control](../guide/advanced-optimizer-control) — interpreting diversity and stagnation signals
