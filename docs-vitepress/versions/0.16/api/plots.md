---
title: Plots
description: API reference for genetic-search diagnostic plots.
---

# Plots

```python
from sklearn_genetic.plots import (
    LivePlot,
    SearchPlotter,
    plot_candidate_rankings,
    plot_candidate_scores,
    plot_convergence,
    plot_cv_scores,
    plot_diversity,
    plot_feature_selection,
    plot_fitness_evolution,
    plot_generation_metrics,
    plot_history,
    plot_optimizer_events,
    plot_parameter_evolution,
    plot_search_decisions,
    plot_search_overview,
    plot_search_space,
    plot_score_landscape,
)
```

Requires the plotting extra: `pip install sklearn-genetic-opt[plot]`.

## During fitting

`LivePlot` works during fitting with both estimators and all five [optimization backends](./optimizer-backends). It updates the same lines for the requested series after generation 0 and each completed generation. See the [callback guide](../guide/callbacks) for Matplotlib display setup and checkpoint behavior. With a graphical Matplotlib backend the chart
updates on screen; `save_path` also rewrites a PNG for headless monitoring.

```python
from sklearn_genetic.plots import LivePlot

live_plot = LivePlot(metrics=("fitness_best_observed", "fitness"), save_path="training-progress.png")
search.fit(X, y, callbacks=live_plot)
```

For a custom callback, `plot_generation_metrics(logbook)` draws directly from
the in-progress generation logbook. Pass `metrics` to choose fitness fields or
`ax` to reuse a Matplotlib axis. After fitting, use the diagnostics below to
explore all evaluated candidates and the selected solution.

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

The convergence, diversity and optimizer-event panels use the generation numbers in
`search.history["gen"]`, including gaps in a filtered history. To see how many
candidate cross-validations the search actually ran, inspect
`search.fit_stats_["cross_validate_calls"]`. Repeated candidates can reuse a cached
score, so successive generations can have different evaluation costs.

To plot the best score against the cumulative number of candidate CV evaluations,
use the aligned entries in the fitted history:

```python
import matplotlib.pyplot as plt

plt.step(search.history["cross_validate_calls"], search.history["fitness_best_observed"], where="post")
plt.xlabel("Candidate cross-validations")
plt.ylabel("Best search score")
```

Each call evaluates one candidate over its search folds. Cache hits add zero.
The last generation includes local refinement. Finalist validation and the
full-data refit happen after the search. When resuming an older checkpoint,
generation records saved without this counter have `None` at those positions.

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
plot_fitness_evolution(estimator, metric="fitness_best")
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `estimator` | fitted `GASearchCV` or `GAFeatureSelectionCV` | — | The fitted search estimator |
| `metric` | str | `"fitness_best"` | Best archived fitness by default. Use `"fitness_best_observed"` for the best finite evaluation so far or `"fitness"` for the population mean. |

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

`plot_candidate_scores`, `plot_candidate_rankings` and `plot_cv_scores` order
candidates by the selected metric. With `criteria="min"`, the lowest mean CV
score appears first. With `criteria="max"`, the highest appears first. All
three omit non-finite means and preserve evaluation order among exact ties.
Use `plot_candidate_scores(search, sort=False)` to see the finite candidates
in evaluation order.

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
| What is happening during fitting? | `LivePlot`, `plot_generation_metrics` |
| Did diversity collapse? | `plot_diversity`, `plot_search_overview` |
| Are top solutions robust? | `plot_cv_scores`, `plot_candidate_rankings` |

## See Also

- [Basic Usage](../guide/basic-usage) — tutorial with both plot types
