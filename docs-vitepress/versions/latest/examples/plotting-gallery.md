---
title: "Plotting Gallery"
description: "Plot scikit-learn genetic search results, including convergence, candidate scores, fit counts, optimizer events, and selected features."
---

:::warning Development version
This is the **latest (dev)** documentation. It may contain unreleased features or breaking changes. For the stable release, use [stable](/stable/).
:::

# Plotting Gallery

`sklearn-genetic-opt` ships diagnostic plot functions that read the metadata stored on a fitted search. This gallery fits real searches and uses the recorded results to draw each figure. Follow the examples to inspect candidate scores, convergence, diversity, fit work, and the selected features.

## Setup: One Search to Plot

We tune a `RandomForestClassifier` on the breast-cancer dataset with several
hyperparameters of mixed type (integers, a continuous fraction, and two
categoricals). Diversity control, fitness sharing, and local search are
enabled so the diversity, optimizer-event, and decision plots actually have
something to show. The run is small (population 10 x 10 generations) so the
whole gallery builds in well under a minute.

```python
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split

from sklearn_genetic import (
    EvolutionConfig,
    GASearchCV,
    OptimizationConfig,
    PopulationConfig,
    RuntimeConfig,
)
from sklearn_genetic.plots import (
    LivePlot,
    SearchPlotter,
    plot_candidate_scores,
    plot_candidate_rankings,
    plot_convergence,
    plot_cv_fold_scores,
    plot_cv_scores,
    plot_diversity,
    plot_fitness_evolution,
    plot_history,
    plot_optimizer_events,
    plot_parameter_evolution,
    plot_search_decisions,
    plot_search_overview,
    plot_search_space,
    plot_score_landscape,
)
from sklearn_genetic.space import Categorical, Continuous, Integer

warnings.filterwarnings("ignore")
RANDOM_STATE = 42

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

search = GASearchCV(
    estimator=RandomForestClassifier(random_state=42, n_jobs=1),
    random_state=RANDOM_STATE,
    cv=cv,
    scoring="roc_auc",
    param_grid={
        "n_estimators": Integer(30, 90),
        "max_depth": Integer(2, 16),
        "min_samples_split": Integer(2, 24),
        "max_features": Continuous(0.2, 1.0),
        "criterion": Categorical(["gini", "entropy"]),
        "class_weight": Categorical([None, "balanced"]),
    },
    evolution_config=EvolutionConfig(
        population_size=10,
        generations=10,
        crossover_probability=0.9,
        mutation_probability=0.1,
        elitism=True,
        keep_top_k=4,
    ),
    population_config=PopulationConfig(initializer="random"),
    runtime_config=RuntimeConfig(n_jobs=1, use_cache=True, verbose=False),
    optimization_config=OptimizationConfig(
        diversity_control=True,
        fitness_sharing=True,
        local_search=True,
        local_search_top_k=2,
    ),
)

live = LivePlot(metrics=("fitness_best_observed", "fitness"))
search.fit(X_train, y_train, callbacks=live)
print("Best CV ROC AUC:", round(search.best_score_, 4))
print("Best params    :", search.best_params_)
```

```text
Best CV ROC AUC: 0.9893
Best params    : {'n_estimators': 83, 'max_depth': 15, 'min_samples_split': 3, 'max_features': 0.27267402874991875, 'criterion': 'entropy', 'class_weight': None}
```

![Cumulative best ROC AUC and population mean at the end of the fit](/images/generation_gallery_live.png)

*Final frame of the generation callback. The mean can fall while the best stays level.*

## Overview Dashboard

`plot_search_overview` is the fastest way to inspect a fitted search. It packs
convergence, diversity, optimizer events, and the strongest candidates into a
single 2x2 figure.

```python
plot_search_overview(search, top_k=6)
```

![Search overview dashboard with convergence, diversity, events, and candidates](/images/generation_gallery_search_overview.png)

*Four diagnostics at a glance: convergence (top-left), diversity (top-right), optimizer events (bottom-left), and the best evaluated candidates (bottom-right).*

**What to look for:** a rising best-fitness curve that flattens (converged),
diversity that stays above zero (no premature collapse), and a tight cluster
of strong candidates at the top of the ranking panel.

You can also keep a fitted search wrapped in a small plotting facade and call
the same plots as methods:

```python
plotter = SearchPlotter(search)
type(plotter).__name__
```

```text
'SearchPlotter'
```

## Overview with incomplete history

A saved search may have candidate scores and convergence history from
before diversity and optimizer events were recorded. The overview keeps
the available plots and labels the panels with missing telemetry.
This copy contains only the generation numbers and best fitness.

```python
from copy import copy

older_search = copy(search)
older_search.history = {
    name: search.history[name] for name in ("gen", "fitness_best")
}
plot_search_overview(older_search, top_k=6)
```

![Search overview showing convergence and ranked candidates with missing data in the other two panels](/images/generation_gallery_incomplete_overview.png)

*The saved fitness curve and candidate rankings remain visible.*

## Overview without finite CV means

A non-finite fold score leaves the candidate mean unavailable. The
overview shows the recorded folds in evaluation order when every mean
is unavailable. Here the scorer reports a value on one of the two
validation folds. A candidate with `max_depth=0` also enters the search
and fails to fit on both folds.

```python
from sklearn.tree import DecisionTreeClassifier

X_small, y_small = X_train[:101], y_train[:101]
folds = list(StratifiedKFold(2, shuffle=True, random_state=7).split(X_small, y_small))
larger_fold = max(len(test) for _, test in folds)

def score_with_missing_fold(estimator, X_validation, y_validation):
    if len(y_validation) == larger_fold:
        return np.nan
    return estimator.score(X_validation, y_validation)

missing_fold = GASearchCV(
    DecisionTreeClassifier(random_state=7),
    param_grid={
        "max_depth": Categorical([0, 2, 4, 6]),
        "min_samples_leaf": Integer(1, 5),
    },
    warm_start_configs=[
        {"max_depth": depth, "min_samples_leaf": 2} for depth in (2, 0, 4, 6)
    ],
    scoring=score_with_missing_fold,
    cv=folds,
    refit=False,
    error_score=np.nan,
    n_jobs=1,
    random_state=7,
    population_size=4,
    generations=1,
    verbose=False,
).fit(X_small, y_small)
plot_search_overview(missing_fold, top_k=4)
```

![Search overview with a fold-score heatmap, crosses for missing scores, and squares for a failed candidate](/images/generation_gallery_nonfinite_overview.png)

*Outlined squares identify the candidate that failed to fit every fold. Crosses mark the other missing scores.*

## Fitness Evolution

`fitness_best_observed` includes every finite candidate evaluation.
`fitness_best` follows the archive, which may retain an earlier score for a repeated configuration.

```python
plot_fitness_evolution(search, metric="fitness_best_observed")
```

![Best observed fitness across generations](/images/generation_gallery_fitness_evolution.png)

*Best-so-far ROC AUC climbing across generations.*

Plot several metrics together with light smoothing:

```python
plot_fitness_evolution(
    search,
    metrics=["fitness_best", "fitness", "fitness_max"],
    window=2,
    kind="line",
    title="Fitness comparison with smoothing",
)
```

![Best, mean, and max fitness with a rolling window](/images/generation_gallery_fitness_multi.png)

*best vs. population mean vs. per-generation max, smoothed with a window of 2.*

**What to look for:** the gap between `fitness_best` and `fitness` (mean) shows how far the current mean trails the best seen anywhere in the run. A mean that races up to meet the best can signal the
population converging — cross-check with the diversity plot below.

| Parameter | Description |
|-----------|-------------|
| `metric` / `metrics` | One field, or a list of `history` fields to overlay |
| `window` | Rolling-average window (default: no smoothing) |
| `kind` | `"line"`, `"bar"`, `"area"`, or `"step"` |
| `title` | Chart title |

### Missing costs in a saved history

A saved generation may have a fitness value and lack its cumulative CV
count. Here we omit one count in a copy of the recorded history. The CV
curve resumes at the next known cost. Use the generation axis to see
the fitness of the generation whose cost is unavailable.

```python
missing_cost = copy(search)
costs = list(search.history["cross_validate_calls"])
costs[len(costs) // 2] = None
missing_cost.history = {**search.history, "cross_validate_calls": costs}

plot_fitness_evolution(missing_cost, metric="fitness_best_observed", x="cv_calls")
```

![Fitness curve with a gap between two recorded cumulative cross-validation costs](/images/generation_gallery_missing_cv_cost.png)

*The break marks a generation whose cumulative CV count is unavailable.*

## History and Decisions

`plot_history` plots any fields from `history` (generation stats) or `logbook`
(per-candidate evaluations).

```python
plot_history(
    search,
    fields=["fitness_best", "fitness", "unique_individual_ratio", "genotype_diversity"],
    kind="line",
    subplots=True,
    title="Optimizer history overview",
)
```

![Fitness and diversity history fields in stacked subplots](/images/generation_gallery_history.png)

*Fitness and diversity telemetry, one field per subplot.*

`plot_search_decisions` focuses only on the optimizer-control fields (mutation
probability, selection pressure, immigrants, local refinements, ...) as a stack
of step plots.

```python
plot_search_decisions(search)
```

![Optimizer-control decisions over generations as step plots](/images/generation_gallery_search_decisions.png)

*When the optimizer adjusted mutation, injected random immigrants, or ran local refinements.*

**What to look for:** spikes in `random_immigrants` or `duplicate_replacements`
mean diversity control kicked in; steps in `mutation_probability` show an
adaptive schedule responding to stagnation.

## Focused Convergence and Diversity

When you want one figure per question instead of the dashboard:

```python
plot_convergence(search)
```

![Convergence of fitness summary fields across generations](/images/generation_gallery_convergence.png)

*Best, mean, max, and min fitness on a single axis.*

**What to look for:** the spread between max and min narrowing over time is healthy convergence; a flat-from-the-start best curve suggests the search space was too easy or too small.

```python
plot_diversity(search)
```

![Population diversity ratios with stagnation on a secondary axis](/images/generation_gallery_diversity.png)

*Unique-individual ratio and genotype diversity; stagnant generations on the right axis.*

**What to look for:** if diversity collapses toward zero early and stagnation
climbs, the population has converged prematurely — enable `fitness_sharing`,
raise `random_immigrants_fraction`, or lower `diversity_threshold` before
simply adding generations.

## Search-Space Exploration

`plot_search_space` shows how the sampled parameters relate. The pair plot
colors points by a categorical column when `hue` is set.

```python
plot_search_space(
    search,
    features=["n_estimators", "max_depth", "min_samples_split", "max_features"],
    kind="pair",
)
```

![Pairwise relationships between sampled hyperparameters](/images/generation_gallery_search_space_pair.png)

*Pairwise scatter of the numeric parameters the search actually sampled.*

A correlation heatmap is a compact alternative for numeric parameters:

```python
plot_search_space(
    search,
    features=["n_estimators", "max_depth", "min_samples_split", "max_features"],
    kind="heatmap",
)
```

![Correlation heatmap of sampled numeric parameters](/images/generation_gallery_search_space_heatmap.png)

*Pearson correlations between the four displayed parameters.*

**What to look for:** clusters of sampled points reveal where the optimizer
concentrated its effort; a strong correlation in the heatmap hints at a
parameter that drove the score.

| Parameter | Description |
|-----------|-------------|
| `features` | Parameter names to include (omit for all numeric params) |
| `hue` | Categorical column for color coding (pair plot only) |
| `kind` | `"pair"` or `"heatmap"` |

## Parameter Exploration Over Time

`plot_parameter_evolution` plots each parameter's sampled value in evaluation
order, colored by the score, so you can see whether strong candidates clustered
in a narrow range.

```python
plot_parameter_evolution(
    search,
    parameters=["n_estimators", "max_depth", "min_samples_split"],
)
```

![Parameter values across candidate evaluations colored by score](/images/generation_gallery_parameter_evolution.png)

*Each parameter's sampled value over evaluation order; brighter points scored higher.*

**What to look for:** if the brightest points concentrate in a band, the
optimizer found a productive region for that parameter; values scattered with
no color pattern mean the parameter mattered little.

## Optimizer Events

`plot_optimizer_events` shows interventions as a timeline — easier to scan than
overlapping step lines when you only care *when* the optimizer changed behavior.

```python
plot_optimizer_events(search)
```

![Optimizer interventions as a generation timeline](/images/generation_gallery_optimizer_events.png)

*Each row is an intervention type; markers mark the generations where it fired.*

**What to look for:** events bunched in the late generations usually mean the
optimizer is fighting stagnation; an empty timeline means the search converged
smoothly without needing diversity interventions.

## Score Landscapes

`plot_score_landscape` highlights promising regions in a two-parameter slice.

```python
plot_score_landscape(search, x="max_depth", y="min_samples_split")
```

![Score landscape scatter for two parameters](/images/generation_gallery_score_landscape.png)

*Scatter of evaluated candidates; color is the CV score, marker size encodes CV std.*

Dense numeric spaces aggregate cleanly with hexbins:

```python
plot_score_landscape(search, x="max_depth", y="min_samples_split", kind="hexbin")
```

![Score landscape hexbin for two parameters](/images/generation_gallery_score_landscape_hexbin.png)

*Hexbin aggregation of the same slice; each cell is the mean score of the candidates it contains.*

**What to look for:** the brightest region is where the search found its best
scores — a useful sanity check that `best_params_` sits inside it rather than
on a lonely edge point.

## Candidate Rankings and CV Robustness

`plot_candidate_rankings` compares the top candidates with their mean score and
CV standard deviation as error bars.

```python
plot_candidate_rankings(search, top_k=8)
```

![Top candidates ranked by mean CV score with standard-deviation error bars](/images/generation_gallery_candidate_rankings.png)

*Top candidates ranked by mean CV score; horizontal bars are the CV standard deviation.*

```python
plot_candidate_scores(search, top_k=5)
```

![Mean cross-validation score of the five highest-ranked evaluations](/images/generation_gallery_candidate_scores.png)

*Each bar is one evaluation, including candidates with the same displayed parameters.*

`plot_cv_scores` shows the fold-level scores for the strongest candidates so you can spot a winner that is not robust across splits.

```python
plot_cv_scores(search, top_k=5)
```

![Fold-level CV score distributions for the top candidates](/images/generation_gallery_cv_scores.png)

*Per-fold scores for the top candidates; a wide box means a candidate is fragile across splits.*

**What to look for:** prefer a candidate with a slightly lower mean but a tight
fold distribution over a high-mean candidate whose folds are all over the place.

### Compare scores by fold

Use a heatmap when you need to identify the fold behind a low score.
The row labels are the original candidate positions from `cv_results_`.
Each column is the same validation fold for every candidate. With `top_k`,
tied test ranks keep evaluation order at the cutoff. Crosses mark
non-finite scores, and outlined squares mark rows flagged as
`all_fits_failed`. The [overview example](#overview-without-finite-cv-means)
shows both markers.

```python
plot_cv_fold_scores(search, top_k=5)
```

![Heatmap with candidate row labels and a column for each cross-validation fold](/images/generation_gallery_cv_fold_scores.png)

*Read down a column to compare candidates on the same validation fold.*

Pass `candidate_indices=search.best_index_` to show only the selected row,
or pass a list of candidate positions to compare specific rows.

### Inspect incomplete fold distributions

The distribution plot also accepts `candidate_indices`. Select rows
from the earlier search with unavailable mean scores. A partially scored
candidate shows its finite fold count. A candidate flagged by
`all_fits_failed` carries that status below its label.

```python
failed_fits = missing_fold.cv_results_["all_fits_failed"]
recorded = np.flatnonzero(~failed_fits)[0]
failed = np.flatnonzero(failed_fits)[0]
plot_cv_scores(
    missing_fold, candidate_indices=[recorded, failed],
    kind="strip", label_params=["max_depth"],
)
```

![Fold distribution for one partly scored candidate and one candidate that failed every fold](/images/generation_gallery_incomplete_cv_distribution.png)

*The first candidate contributes one finite score. The second is marked All fits failed.*

## Feature-Selection Plots

`plot_feature_selection` draws the boolean support mask chosen by
`GAFeatureSelectionCV`, and `plot_search_overview` swaps its candidate panel for
that mask when given a feature-selection estimator. We run a small selection
search on iris padded with noise columns.

```python
from sklearn.datasets import load_iris
from sklearn.svm import SVC

from sklearn_genetic import GAFeatureSelectionCV
from sklearn_genetic.plots import plot_feature_selection

iris = load_iris()
X_fs, y_fs = iris.data, iris.target
rng = np.random.default_rng(42)
noise = rng.uniform(0, 10, size=(X_fs.shape[0], 6))
X_fs = np.hstack((X_fs, noise))
feature_names = list(iris.feature_names) + [f"noise_{i}" for i in range(noise.shape[1])]

selector = GAFeatureSelectionCV(
    estimator=SVC(gamma="auto"),
    random_state=RANDOM_STATE,
    cv=3,
    scoring="accuracy",
    population_size=12,
    generations=8,
    max_features=6,
    n_jobs=1,
    verbose=False,
)
selector.fit(X_fs, y_fs)
print("Selected", int(selector.best_features_.sum()), "of", len(feature_names), "features")
```

```text
Selected 4 of 10 features
```

```python
plot_feature_selection(selector, feature_names=feature_names)
```

![Selected feature mask for the GAFeatureSelectionCV search](/images/generation_gallery_feature_selection.png)

*The four real iris features survive; most injected noise columns are rejected.*

**What to look for:** the genuine features kept and the noise columns dropped.

```python
plot_search_overview(selector)
```

![Feature-selection overview dashboard](/images/generation_gallery_feature_overview.png)

*Same dashboard layout, but the candidate panel is replaced by the selected-feature mask.*

## Fit Work by Phase

Use `plot_fit_work` to see how much training the latest `fit` performed.
In this example, three finalists reuse two search folds each and train on
one new fold each. The hatched bar counts the six scores they reuse.

```python
from sklearn.base import clone

from sklearn_genetic.callbacks import CVBudgetStopping
from sklearn_genetic.plots import plot_fit_work

search_folds = list(StratifiedKFold(3, shuffle=True, random_state=42).split(X_fs, y_fs))
extra_folds = list(StratifiedKFold(3, shuffle=True, random_state=71).split(X_fs, y_fs))
work_example = clone(selector).set_params(
    cv=search_folds,
    error_score="raise",
    final_selection="reuse_folds",
    final_selection_cv=[search_folds[0], search_folds[1], extra_folds[2]],
    final_selection_top_k=3,
)
work_example.fit(X_fs, y_fs, callbacks=CVBudgetStopping(6))
work_ax = plot_fit_work(work_example)
```

![Bar chart showing 18 search fold fits, 3 finalist fold fits, 1 full-data refit and 6 reused finalist scores](/images/generation_gallery_fit_work.png)

*The first three bars count new estimator fits. The hatched bar shows scores reused during finalist validation.*

Switch to `unit="seconds"` to see the time each phase took on this run.
The [timing reference](../api/optimizer-backends#fit-time-by-phase) describes
which work each duration includes.

```python
elapsed_ax = plot_fit_work(work_example, unit="seconds")
```

![Bar chart of elapsed seconds for search, finalist validation, and full-data refit](/images/generation_gallery_fit_time.png)

*Elapsed times for one run of the feature-selection example above.*

Leave `final_selection_cv=None` when the finalists use the exact search
folds. With `final_selection="reuse_folds"`, finalist validation can
reuse all the fold scores in this example.

```python
same_folds = clone(work_example).set_params(final_selection_cv=None)
same_folds.fit(X_fs, y_fs, callbacks=CVBudgetStopping(6))
print("Fresh finalist fold fits", same_folds.fit_counts_["finalist_fold_fits"])
print("Reused finalist fold scores", same_folds.fit_counts_["finalist_reused_folds"])
```

```text
INFO: Stopping the algorithm
Fresh finalist fold fits 0
Reused finalist fold scores 9
```

## Finalist Score Comparison

Use `plot_finalist_scores` after fitting with finalist validation.
Each row compares the search score and the finalist score for the same
candidate. The star marks the candidate selected for the full-data refit.
The example uses the search with partial fold reuse from the previous section.

```python
from sklearn_genetic.plots import plot_finalist_scores

finalist_ax = plot_finalist_scores(work_example)
finalists = work_example.final_selection_results_
print("Search winner row", finalists["original_best_index"])
print("Refitted row", work_example.best_index_)
```

```text
Search winner row 4
Refitted row 4
```

![Paired dot plot comparing search and finalist CV scores, with a star on the refitted row](/images/generation_gallery_finalist_scores.png)

*The open circle marks the search score. Squares and the star show the scores from finalist validation.*

## Reading History Directly

Generation summaries come from `search.history`. Candidate plots use per-evaluation records or `cv_results_`. You can also work with it as a
DataFrame for custom reporting.

```python
history = pd.DataFrame(search.history)
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
print(history[available].tail().to_string(index=False))
```

```text
 gen  fitness_best  fitness  fitness_max  unique_individual_ratio  genotype_diversity  mutation_probability  selection_pressure  random_immigrants  duplicate_replacements  local_refinements
   6      0.989223 0.986980     0.989223                      0.9            0.688889                   0.1                 3.0                  0                       5                  0
   7      0.989259 0.987609     0.989259                      0.8            0.474074                   0.1                 3.0                  0                       3                  0
   8      0.989259 0.987503     0.989259                      1.0            0.629630                   0.1                 3.0                  0                      10                  0
   9      0.989259 0.987771     0.989259                      0.9            0.622222                   0.1                 3.0                  0                       5                  0
  10      0.989259 0.987891     0.989259                      1.0            0.540741                   0.1                 3.0                  0                       6                  2
```

## When to Use Each Plot

| Plot | Use when |
|------|----------|
| `plot_search_overview` | One-call diagnostic dashboard after `.fit(...)` |
| `SearchPlotter` | Repeated diagnostics from a fitted search object |
| `LivePlot` / `plot_generation_metrics` | Follow or snapshot recorded fields during fitting |
| `plot_fitness_evolution` | Quick fitness trend, optionally with multiple fields |
| `plot_fit_work` | Checking counts and elapsed times by fit phase |
| `plot_finalist_scores` | Comparing search and finalist scores and locating the refitted candidate |
| `plot_convergence` | Fitness progress without event clutter |
| `plot_diversity` | Checking diversity collapse and stagnation |
| `plot_history` | Inspecting any telemetry field from history or logbook |
| `plot_search_decisions` | Just the optimizer-control fields as step plots |
| `plot_search_space` (`pair`) | Understanding parameter interactions |
| `plot_search_space` (`heatmap`) | Spotting correlations between numeric params |
| `plot_parameter_evolution` | How parameter values changed across evaluations |
| `plot_optimizer_events` | Explaining interventions as a timeline |
| `plot_score_landscape` | Finding promising parameter regions |
| `plot_candidate_scores` / `plot_candidate_rankings` | Comparing top evaluations by mean score or fold spread |
| `plot_cv_scores` | Checking fold-level robustness |
| `plot_cv_fold_scores` | Locating low or non-finite scores in a specific fold |
| `plot_feature_selection` | Inspecting the selected support mask |

## Practical Notes

- Single-axis helpers that expose `ax=` let you
  compose them into your own figures.
- The post-fit plots use `seaborn`, included in the `sklearn-genetic-opt[plot]`
  extra.
- If `plot_diversity` shows an early collapse, reach for `fitness_sharing`,
  `random_immigrants_fraction`, and `diversity_control` before adding generations.
- `plot_search_space`, `plot_score_landscape`, and `plot_parameter_evolution` are
  `GASearchCV`-only; the feature-selection estimator uses
  `plot_feature_selection` and the overview dashboard instead.

## See Also

- [Plots API](../api/plots) — full parameter reference for every plot function
- [Feature Selection](./feature-selection) — the search behind the mask plots
- [Advanced Optimizer Control](../guide/advanced-optimizer-control) — interpreting diversity and stagnation signals
