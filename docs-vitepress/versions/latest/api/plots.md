---
title: Plots
description: API reference for genetic-search diagnostic plots.
---

:::warning Development version
You are reading the **latest (dev)** docs. For the stable version, see [stable](/stable/).
:::

# Plots

```python
from sklearn_genetic.plots import (
    LivePlot,
    SearchPlotter,
    plot_candidate_rankings,
    plot_candidate_scores,
    plot_convergence,
    plot_cv_fold_scores,
    plot_cv_scores,
    plot_diversity,
    plot_feature_selection,
    plot_finalist_scores,
    plot_fit_work,
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
other recorded telemetry, and `ax` to reuse a Matplotlib axis. Both functions
preserve each generation number and leave a gap where a metric is missing.
Each requested metric must appear in at least one generation. See the
[checkpoint example](../tutorials/optimizer-backends#plot-generations-with-missing-counts).
After fitting, use the diagnostics below to
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

For a saved search with incomplete history, the overview displays the
available plots. A panel with missing telemetry shows **No data recorded**.
An event history containing only zero counts shows **No optimizer events recorded**.
See the [plotting gallery example](../examples/plotting-gallery#overview-with-incomplete-history).

For a hyperparameter search whose requested metric has no finite mean CV
score, the candidate panel shows up to `top_k` candidates in evaluation order.
The heatmap displays recorded fold scores and marks non-finite values with
crosses. If split scores are unavailable, the panel reports that no finite
means were found. See the [fold score example](../examples/plotting-gallery#overview-without-finite-cv-means).

The convergence, diversity and optimizer-event panels use the generation numbers in
`search.history["gen"]`, including gaps in a filtered history. To see how many
candidate cross-validations the search actually ran, inspect
`search.fit_stats_["cross_validate_calls"]`. Repeated candidates can reuse a cached
score, so successive generations can have different evaluation costs.

To plot the best score against the cumulative number of candidate CV evaluations,
pass `x="cv_calls"` to [plot_fitness_evolution](#plot-fitness-evolution).

```python
from sklearn_genetic.plots import plot_fitness_evolution

plot_fitness_evolution(search, metric="fitness_best_observed", x="cv_calls", kind="step")
```

Each call evaluates one candidate over its search folds. Cache hits add zero.
The last generation includes local refinement. Finalist validation and the
full-data refit happen after the search. After resuming an older checkpoint,
the plot includes the generations with recorded counts. If all counts are
missing, use the generation axis.

## Object-oriented API

Use `SearchPlotter` when you want a compact, sklearn-style object around a fitted search.

```python
plotter = SearchPlotter(search)

plotter.overview()
plotter.convergence()
plotter.fit_work()
plotter.diversity()
plotter.optimizer_events()
plotter.parameter_evolution(["max_depth", "ccp_alpha"])
plotter.score_landscape("max_depth", "min_samples_split")
plotter.candidate_rankings(top_k=15)
plotter.cv_scores(top_k=5)
plotter.cv_fold_scores(top_k=5)
plotter.finalist_scores()  # after fitting with finalist validation
```

## plot_finalist_scores

Compare search CV and final validation for each candidate on the shortlist.
Fit `GASearchCV` or `GAFeatureSelectionCV` with refitting enabled and
`final_selection=True` or `final_selection="reuse_folds"` before plotting.

```python
ax = plot_finalist_scores(search)
```

Row numbers index `cv_results_`. An open circle represents search CV, a square
represents finalist CV, and a star identifies the finalist selected for refit.
The plot also works after a validation error when at least one finalist
evaluation completed. It shows the available scores and leaves the winner
unmarked when `selected_index` is `None`.
If finalist validation completed before a refit error, you can still plot its
scores. Check `fit_counts_["refit_fits"]` to see whether the full-data fit
completed. The chart labels the search winner and any unavailable scores.
Pass `ax=` to draw on an existing axis. See the
[finalist score comparison](../examples/plotting-gallery#finalist-score-comparison)
for a feature-selection example.

## plot_fit_work

Show how many estimator fits occurred during the latest call to `fit`.
Search and finalist CV each count one fit per trained fold. The full-data refit
has its own bar. The hatched bar counts finalist fold scores reused from the
search without training another model.

```python
ax = plot_fit_work(search)
elapsed_ax = plot_fit_work(search, unit="seconds")
```

Use `unit="seconds"` to show [recorded phase durations](./optimizer-backends#fit-time-by-phase)
for search, finalist validation, and refit. A phase with no duration appears as
`No timing`, for example when refit is disabled or when a saved search predates
`search_time_`. If the optimizer raises, seconds mode can still chart
`search_time_` from the interrupted run. `SearchPlotter(search).fit_work(unit="seconds")`
accepts the same option.

Pass `ax=` to reuse a Matplotlib axis. The function returns that axis or the
new axis it creates. It reads `fit_counts_` from either fitted estimator,
including with `refit=False` or after resuming a checkpoint. Cached and
restored search scores add zero fresh search fits. Use
[`plot_fitness_evolution(x="cv_calls")`](#plot-fitness-evolution) for cumulative
candidate CV across generations. See the [worked example](../examples/plotting-gallery#fit-work-by-phase).

## plot_fitness_evolution

Plot fitness over generations or cumulative candidate cross-validation calls.

```python
plot_fitness_evolution(estimator, metric="fitness_best")
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `estimator` | fitted `GASearchCV` or `GAFeatureSelectionCV` | — | The fitted search estimator |
| `metric` | str | `"fitness_best"` | Best archived fitness by default. Use `"fitness_best_observed"` for the best finite evaluation so far or `"fitness"` for the population mean. |
| `x` | `"generation"` or `"cv_calls"` | `"generation"` | Horizontal axis. Generations with the same cumulative CV cost share an x position. |
| `kind` | `"line"`, `"bar"`, `"area"` or `"step"` | `"line"` | Style. Bars at the same CV cost form a chronological group around that cost. |

Non-finite scores appear as gaps, including when smoothing with `window`.
Hollow diamonds mark finite plotted values from rows with `partial=True`,
including groups whose CV cost repeats. See the
[checkpoint resume example](../tutorials/optimizer-backends#find-partial-generations-before-and-after-resume).
The CV axis still shows the cost of generations with unavailable scores.
When a saved generation has no recorded CV count, lines, steps and filled
areas break across that part of the history. Bars omit those rows. On the CV
axis a rolling window uses rows with a recorded count. See the
[saved history example](../examples/plotting-gallery#missing-costs-in-a-saved-history).
When several generations report at the same CV count, `kind="bar"` places
their bars side by side in generation order. A change within that group
can reflect different survivors or scores reused from the cache. The group
shares the same number of fresh CV calls.
Compare `finite_fitness_count` with `population_size` using `plot_history`
to see how many population scores contribute to its summary.

Returns a `matplotlib.axes.Axes`. Both axes are available after a fit with
`refit=False`.

**Example:**

```python
import matplotlib.pyplot as plt
from sklearn_genetic.plots import plot_fitness_evolution

plot_fitness_evolution(search, metric="fitness_best")
plt.title("Best fitness over generations")
plt.show()
```

## plot_search_space

Plot sampled hyperparameters as a pair plot or a correlation heatmap.

```python
plot_search_space(estimator, height=2, s=25, features=None, kind="pair", hue=None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `estimator` | fitted `GASearchCV` | — | The fitted search estimator |
| `height` | float | `2` | Height of each pair plot facet |
| `s` | float | `25` | Marker size for the pair plot |
| `features` | list of str | `None` | Columns from `cv_results_` or the candidate logbook |
| `kind` | str | `"pair"` | `"pair"` for a pair plot or `"heatmap"` for Pearson correlations |
| `hue` | str | `None` | Optional column used to color the pair plot |

The default view shows numeric parameters and the optimization metric in
`score`. When parameter names overlap with diagnostics, it uses `param_*`
labels. For explicit selections, columns in `cv_results_` take precedence
over logbook fields of the same name.

Returns a `seaborn.PairGrid` for a pair plot or a Matplotlib `Axes` for a heatmap.

```python
from sklearn_genetic.plots import plot_search_space

plot_search_space(search, features=["learning_rate", "max_depth", "n_estimators"])
plot_search_space(search, features=["param_fit_time", "mean_fit_time"], kind="heatmap")
```

See the [example with overlapping parameter names](../tutorials/optimizer-backends#inspect-parameters-with-names-shared-by-cv-diagnostics).

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
plot_cv_scores(search, candidate_indices=[2, 0], kind="strip")
```

Use `candidate_indices` to select zero-based `cv_results_` rows in display
order, including candidates with a non-finite mean. Only finite fold scores
contribute to the distribution. The label reads **All fits failed** for rows
flagged by `cv_results_["all_fits_failed"]`. Other incomplete rows show their
finite fold counts, including zero when every score is undefined. Older
results without the flag also use the finite fold count.
Use [`plot_cv_fold_scores`](#plot_cv_fold_scores) to inspect which folds
are missing or display a selection with no finite scores. See the
[incomplete score example](../examples/plotting-gallery#inspect-incomplete-fold-distributions).

Candidate labels are compact by default: numeric parameters are prioritized, float
values are rounded to a few significant digits, and hidden parameters are summarized
with `+N more`.

### plot_cv_fold_scores

Compare individual test folds in a heatmap. Ranked rows keep their original
positions in `cv_results_` as labels. With `top_k`, tied ranks keep
evaluation order, including ties at the cutoff. Outlined squares mark candidates whose
estimator failed in every fold, based on `cv_results_["all_fits_failed"]`.
Crosses mark other `NaN` and infinite scores. For older results without the
failure flag, crosses mark all non-finite values.
Pass zero-based `candidate_indices` to inspect specific rows in the order
supplied, including rows with undefined mean scores.

```python
plot_cv_fold_scores(search, top_k=5)
plot_cv_fold_scores(search, candidate_indices=search.best_index_)
```

Pass `metric="f1"` to inspect a named test metric in a multi-metric search.
The map uses the saved search-fold scores, including when a finalist round
used different folds. See the [fold heatmap example](../examples/plotting-gallery#compare-scores-by-fold).

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
| How many fresh fits did it run and how long did each phase take? | `plot_fit_work` |
| Which finalist did the search refit? | `plot_finalist_scores` |
| Did diversity collapse? | `plot_diversity`, `plot_search_overview` |
| Are top solutions robust? | `plot_cv_scores`, `plot_candidate_rankings` |
| Which folds explain a candidate's score? | `plot_cv_fold_scores` |

## See Also

- [Basic Usage](../guide/basic-usage) — tutorial with both plot types
