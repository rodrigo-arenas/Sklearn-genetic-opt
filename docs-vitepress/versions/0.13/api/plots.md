---
title: Plots
description: API reference for genetic-search diagnostic plots.
---

# Plots

```python
from sklearn_genetic.plots import (
    SearchPlotter,
    plot_candidate_rankings,
    plot_candidate_scores,
    plot_convergence,
    plot_cv_scores,
    plot_diversity,
    plot_feature_selection,
    plot_fitness_evolution,
    plot_history,
    plot_optimizer_events,
    plot_parameter_evolution,
    plot_search_decisions,
    plot_search_overview,
    plot_search_space,
    plot_score_landscape,
)
```

Requires seaborn: `pip install sklearn-genetic-opt[all]`.

## Quick diagnostic overview

Use `plot_search_overview` after `.fit(...)` when you want one compact answer to:

- how did the search converge?
- did diversity collapse?
- did optimizer controls intervene?
- which candidates or features won?

```python
from sklearn_genetic.plots import plot_search_overview

plot_search_overview(search)
```

Returns a 2x2 array of `matplotlib.axes.Axes`.

## Object-oriented API

Use `SearchPlotter` when you want a compact, sklearn-style object around a fitted search.

```python
plotter = SearchPlotter(search)

plotter.overview()
plotter.convergence()
plotter.diversity()
plotter.optimizer_events()
plotter.parameter_evolution(["max_depth", "ccp_alpha"])
plotter.score_landscape("max_depth", "min_samples_split")
plotter.candidate_rankings(top_k=15)
plotter.cv_scores(top_k=5)
```

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

## Search diagnostics

### plot_convergence

Plot fitness convergence without mixing diversity or event signals.

```python
plot_convergence(search)
```

### plot_diversity

Plot diversity ratios on the main axis and stagnation on a secondary axis.

```python
plot_diversity(search)
```

### plot_parameter_evolution

Plot the explored values for each parameter across evaluation order, colored by score.

```python
plot_parameter_evolution(search, parameters=["learning_rate", "max_depth"])
```

### plot_search_decisions

Plot optimizer-control telemetry such as mutation probability, selection pressure,
diversity-control triggers, duplicate replacements, random immigrants, local refinements,
and fitness sharing.

```python
plot_search_decisions(search)
```

### plot_optimizer_events

Show optimizer interventions as a timeline instead of overlapping many step lines.

```python
plot_optimizer_events(search)
```

### plot_score_landscape

Inspect promising regions in a two-parameter slice of the search space.

```python
plot_score_landscape(search, x="max_depth", y="min_samples_split")
plot_score_landscape(search, x="max_depth", y="min_samples_split", kind="hexbin")
```

### plot_candidate_scores

Show the top evaluated candidates from `cv_results_`.

```python
plot_candidate_scores(search, top_k=10)
```

### plot_candidate_rankings

Compare top candidates with mean CV score and standard deviation.

```python
plot_candidate_rankings(search, top_k=15)
plot_candidate_rankings(search, top_k=15, label_params=["learning_rate", "max_depth"])
```

### plot_cv_scores

Inspect fold-level robustness for top candidates.

```python
plot_cv_scores(search, top_k=5, kind="box")
plot_cv_scores(search, top_k=5, kind="violin")
plot_cv_scores(search, top_k=5, label_params=["learning_rate", "max_depth"])
```

Candidate labels are compact by default: numeric parameters are prioritized, float
values are rounded to a few significant digits, and hidden parameters are summarized
with `+N more`.

### plot_feature_selection

For `GAFeatureSelectionCV`, show the selected feature mask.

```python
plot_feature_selection(selector, feature_names=X.columns)
```

## Which plot answers what?

| Question | Function |
|----------|----------|
| What did I train? | `plot_candidate_rankings`, `plot_candidate_scores`, `plot_feature_selection` |
| What space did the algorithm explore? | `plot_search_space`, `plot_parameter_evolution`, `plot_score_landscape` |
| What decisions did it make? | `plot_optimizer_events`, `plot_search_decisions` |
| How did it converge? | `plot_convergence`, `plot_fitness_evolution`, `plot_search_overview` |
| Did diversity collapse? | `plot_diversity`, `plot_search_overview` |
| Are top solutions robust? | `plot_cv_scores`, `plot_candidate_rankings` |

## See Also

- [Basic Usage](../guide/basic-usage) — tutorial with both plot types
