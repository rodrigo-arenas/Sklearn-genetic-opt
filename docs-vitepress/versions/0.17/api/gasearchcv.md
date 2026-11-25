---
title: GASearchCV
description: API reference for GASearchCV — genetic algorithm hyperparameter search for scikit-learn estimators.
---

# GASearchCV

Genetic algorithm hyperparameter search for scikit-learn estimators.

`GASearchCV` implements the scikit-learn estimator interface and follows the same patterns as `GridSearchCV` and `RandomizedSearchCV`. After a successful search, inspect `best_params_` and `cv_results_`. With refitting enabled, use `score` and the prediction methods supported by the wrapped estimator.

## Common constructor arguments

```python
import numpy as np
from sklearn_genetic import GASearchCV

GASearchCV(
    estimator,
    cv=3,
    param_grid=param_grid,
    scoring=None,
    refit=True,
    verbose=True,
    error_score=np.nan,
    return_train_score=False,
    random_state=None,
    evolution_config=None,
    population_config=None,
    runtime_config=None,
    optimization_config=None,
    optimizer_backend="deap",
    objective_metric=None,
)
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `optimizer_backend` | str or callable, keyword-only | `"deap"` | Select `"deap"`, `"scipy"`, `"nevergrad"`, `"cmaes"`, `"pymoo"` or a plugin. See [backend settings and limits](./optimizer-backends) and the [tutorial](../tutorials/optimizer-backends). |
| `estimator` | estimator | — | A scikit-learn estimator with a `fit` method |
| `param_grid` | dict | — | Mapping from parameter name to `Integer`, `Continuous`, or `Categorical` |
| `scoring` | str, callable, list, tuple or dict | `None` | Metric(s) to evaluate. `None` uses default scoring. A standalone callable returns one numeric score per fold. Use [named scorers](../guide/multi-metric) for multiple metrics. |
| `cv` | int, CV splitter, iterable or `None` | `3` | Cross-validation strategy, including for [outlier detection](../guide/outliers#choose-validation-folds). See [explicit fold validation](../tutorials/optimizer-backends#validate-explicit-cv-splits). |
| `refit` | bool, str or callable | `True` | Refit the selected candidate. With multiple metrics, pass a metric name or a [refit function](../guide/advanced-optimizer-control#choose-a-refit-candidate-with-a-function) and `objective_metric`. |
| `objective_metric` | str or `None`, keyword-only | `None` | Search objective for multiple metrics with `refit=False` or callable `refit` |
| `verbose` | bool | `True` | Print the generation log |
| `error_score` | float or `"raise"` | `np.nan` | Replacement score for failed CV folds. With a numeric setting, candidates that [fail to fit every fold](../guide/troubleshooting#all-fits-failed) receive `NaN` and rank last. |
| `return_train_score` | bool | `False` | Include training scores in `cv_results_` |
| `random_state` | int, RandomState or `None` | `None` | Seeds the whole search (population init, mutation, crossover, immigrants) at `fit` time for reproducible runs. `None` is non-deterministic |
| `evolution_config` | `EvolutionConfig` | `None` | Controls population size, generations, crossover/mutation rates, elitism |
| `population_config` | `PopulationConfig` | `None` | Controls initialization strategy, warm starts, diversity |
| `runtime_config` | `RuntimeConfig` | `None` | Controls parallelism, caching, verbosity |
| `optimization_config` | `OptimizationConfig` | `None` | Finalist selection; DEAP-specific local search and fitness sharing |

Under the default DEAP engine, standard mutation excludes numeric dimensions with equal bounds and
single-choice categorical dimensions. `GASearchCV` chooses uniformly among
the remaining dimensions and draws from the chosen distribution. A discrete draw may still
repeat the parent value. With caching enabled, fewer repeated proposals can
increase the number of distinct models fitted in a generation.

Local refinement (`local_search=True`) uses the same variable dimensions when
choosing a neighbor of an elite candidate. Each step proposes one neighbor per
selected elite. Clipping and previously visited configurations can reduce the
number of distinct evaluations.

With `algorithm="eaDifferentialEvolution"`, binomial crossover always takes
one coordinate from the donor trial, even when `crossover_probability=0`.
If the grid has any variable dimensions, the algorithm chooses this coordinate
among them. A trial can still match its parent after rounding, clipping, or
drawing the same categorical value.

## Search correctness

Each fresh `fit` recomputes candidate scores for the supplied data and scoring settings.
Caching avoids repeated evaluations within that fit; explicit checkpoint resumes retain
saved evaluations. With DEAP, both elitist and non-elitist tournament selection support signed scores, so
negative scores and `criteria="min"` remain meaningful. Scikit-learn's negative loss
scorers normally use the default `criteria="max"`.

Final selection's `top_k` counts distinct parameter configurations, even when the
candidate log contains repeated cache hits. Among candidates with the same
search rank, the shortlist promotes distinct configurations in evaluation
order.

## Attributes After `fit`

| Attribute | Description |
|-----------|-------------|
| `best_index_` | Row in `cv_results_` for the selected candidate, including with `refit=False` |
| `best_params_` | Parameters from the selected candidate row |
| `best_score_` | Mean CV objective score for the selected candidate. Uses the finalist score when finalist validation runs and retains extended NumPy precision when available. Not set with callable refit; use the objective column of the row at `best_index_`. |
| `best_estimator_` | Estimator fitted with `best_params_` on the full training data after a successful refit |
| `search_time_` | [Elapsed search seconds until the optimizer returned or raised](./optimizer-backends#fit-time-by-phase) |
| `search_interrupted_` | Whether the optimizer raised and its [recorded search work](./optimizer-backends#fit-work-by-phase) is a lower bound |
| `refit_time_` | Elapsed seconds of the successful full-data refit |
| `refit_failure_time_` | [Elapsed seconds of the full-data refit phase when it raises](./optimizer-backends#fit-time-by-phase) |
| `cv_results_` | Dict with per-candidate results, compatible with `pd.DataFrame` |
| `history` | Dictionary of lists aligned by generation. Pair `cross_validate_calls` with `fitness_best_observed` for the best finite score evaluated so far (including discarded candidates), or `fitness_best` for the best archived score. |
| `logbook` | Compatibility view of per-candidate records, materialized on access for shared-runtime engines, in `logbook.chapters["parameters"]` |
| `fit_stats_` | Dict with evaluation counters (cache hits, skipped candidates, etc.) |
| `fit_counts_` | [Fresh fits and reused finalist folds by phase](./optimizer-backends#fit-work-by-phase) in the most recent call to `fit` |
| `search_space_exhausted_` | Whether the engine detected full coverage of a finite space with caching enabled. See [backend stopping rules](./optimizer-backends). |

See [plotting score against evaluation cost](./plots#quick-diagnostic-overview) to compare runs with different cache-hit rates.

## Methods

| Method | Description |
|--------|-------------|
| `fit(X, y=None, callbacks=None, groups=None, **fit_params)` | Run the genetic search |
| `cv_results_dataframe(*, preserve_index=False, columns=None, top_k=None)` | Return ranked candidates as a pandas DataFrame |
| `predict(X)` | Predict using `best_estimator_` |
| `predict_proba(X)` | Predict class probabilities (if estimator supports it) |
| `score(X, y=None, **params)` | Evaluate the refitted estimator using the configured scorer, or the search objective with multiple scorers. Scorer metadata requires metadata routing. |
| `get_params()` / `set_params()` | Standard sklearn estimator interface |

Pass `y` if the estimator or a step in a scikit-learn `Pipeline` requires a
target. The search raises a `ValueError` before evaluating candidates when a
required target is missing. When the estimator accepts an omitted target,
call `fit(X)`. CV, refit and `score(X)` omit the target.
See [custom scoring without labels](../examples/unlabeled-outlier-scoring).

For an estimator that declares a precomputed kernel or distance input,
pass the square matrix over training samples to `fit`. For predictions, pass the
matrix whose rows are new samples and whose columns are the training samples
in their original order. See the [precomputed SVC example](../examples/precomputed-kernel-search).

With metadata routing enabled, `fit` can receive
[separate weights for estimator fitting and scoring](../guide/understand-cv#separate-training-and-validation-weights).
Use `set_fit_request(groups=True)` or a string alias to receive group labels
from a pipeline or outer cross-validation. The search forwards these labels
to its explicit splitters, including `final_selection_cv`. With a string
alias, an outer splitter can still receive its own labels through `groups`.
See
[group labels in nested validation](../guide/understand-cv#group-labels-in-pipelines-and-nested-validation).
A custom `split(X, y)` method can supply folds for one stage when the
other active splitter accepts `groups`.

After `set_params(param_grid=new_space)`, the next `fit` validates and uses
`new_space`, including changed parameter names. An explicit checkpoint resume
restores the saved search space.

Each evaluation configures a cloned estimator. Immutable scalar parameters can
be passed directly. Mutable choices, including dictionaries and estimator-valued
pipeline steps, follow scikit-learn's cloning rules before configuration.
A grid can contain one parameter, including a pipeline step with
[preconfigured estimator choices](../examples/conditional-pipeline-choices).

`cv_results_["params"]` and the `param_*` columns retain the values passed to
the estimator, including names shared with diagnostics. Parameter columns and
each `params` dictionary follow the order of `param_grid`. Each completed
proposal has its own dictionary, including proposals with cached scores.
For an estimator parameter named `fit_time`, read `param_fit_time` for the parameter and
`mean_fit_time` for the measured fold time. See the
[example with overlapping names](../tutorials/optimizer-backends#inspect-parameters-with-names-shared-by-cv-diagnostics).

`cv_results_dataframe()` uses the selected `refit` or `objective_metric` for multi-metric
searches and keeps evaluation order for tied ranks. It leaves `cv_results_`
unchanged and also works after a fit with `refit=False`.
The Boolean column `all_fits_failed` identifies candidates whose estimator
failed to fit every search fold. See the
[CV troubleshooting guide](../guide/troubleshooting#all-fits-failed).
With numeric `error_score`, the table and `history` remain available when
every candidate fails, even though `fit` raises.
Pass `columns` as a name or a list of names to build only those columns,
in the requested order. The table stays sorted by the same test rank even
when the rank column is omitted. Unknown column names raise `KeyError`.

Pass a positive integer as `top_k` to return up to that many rows. Tied
candidates retain their evaluation order at the cutoff.
For a short list from a long history, the exporter can select a rank cutoff
and sort only the leading rows.

With `preserve_index=True`, the sorted table uses positions in `cv_results_`
as labels in the `candidate_index` index. Retrieve the selected row with
`search.cv_results_dataframe(preserve_index=True).loc[search.best_index_]`.

To [inspect a search without refitting](../guide/advanced-optimizer-control#inspect-the-selected-candidate-without-refitting),
set `refit=False`. Read the selected row, parameters and CV objective score
through `best_index_`, `best_params_` and `best_score_`. For multiple scorers,
set `objective_metric` to the one you want to optimize.

## Example

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn_genetic import EvolutionConfig, GASearchCV
from sklearn_genetic.space import Categorical, Integer

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

search = GASearchCV(
    estimator=DecisionTreeClassifier(random_state=42),
    param_grid={
        "max_depth": Integer(2, 12),
        "criterion": Categorical(["gini", "entropy"]),
    },
    cv=3,
    scoring="roc_auc",
    evolution_config=EvolutionConfig(population_size=6, generations=2),
    random_state=42,
)

search.fit(X_train, y_train)
print("Selected parameters", search.best_params_)
print("Search CV evaluations", search.history["cross_validate_calls"])
print("Held-out ROC AUC", search.score(X_test, y_test))
print(search.cv_results_dataframe(columns=["params", "mean_test_score"], top_k=5))
```

## See Also

- [Basic Usage](../guide/basic-usage) — step-by-step tutorial
- [Config Objects](./config) — `EvolutionConfig`, `PopulationConfig`, `RuntimeConfig`, `OptimizationConfig`
- [Search Space](./space) — `Integer`, `Continuous`, `Categorical`
- [GAFeatureSelectionCV](./gafeatureselectioncv) — feature selection variant
