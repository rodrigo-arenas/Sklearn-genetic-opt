---
title: Troubleshooting
description: Diagnose failed CV fits, slow hyperparameter searches, parallelism settings, repeated candidates, and convergence in sklearn-genetic-opt.
---

# Troubleshooting

This page covers the most common problems, organized by symptom.

## Common Errors at a Glance

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `... is not a valid Sklearn classifier, regressor, or outlier detector` | An unsupported estimator was passed to `GASearchCV` | Use `GASearchCV` with a classifier, regressor, or outlier detector; for feature selection use `GAFeatureSelectionCV` |
| `Invalid param_grid entry for '...'` (value is not a space object) | A raw `list`/`tuple`/`range` was used instead of a space object | Wrap choices in `Categorical([...])` and ranges in `Integer(...)` / `Continuous(...)` |
| `ImportError` for `mlflow` or `seaborn` | An optional dependency is not installed | `pip install "sklearn-genetic-opt[all]"` (or the specific extra) |
| `TensorBoard` asks for TensorFlow | A legacy callback is in use | Use `LivePlot` from `sklearn_genetic.plots`, or install TensorFlow separately for the old callback |
| `InvalidParameterError` mentioning `'max_error'` | recent scikit-learn no longer accepts `max_error` as a scoring string | Use `scoring="neg_max_error"` |
| Fit is much slower than expected | Expensive CV folds, repeated proposals, or nested parallelism | [Measure fit work and choose a parallelism layer](#search-is-slow) |

Each row is explained in more detail in the sections below.

## Parameter Errors

**`ValueError: parameter X is not a valid parameter for estimator Y`**

The keys in `param_grid` must exactly match sklearn's parameter names for the estimator. For plain estimators, check `estimator.get_params().keys()`. For pipelines, the pattern is `stepname__paramname`:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

pipe = Pipeline([("scaler", StandardScaler()), ("clf", RandomForestClassifier())])
print(list(pipe.get_params().keys()))
# -> ['scaler', 'clf', 'scaler__copy', ..., 'clf__n_estimators', ...]
```

If the pipeline step is named `"clf"` then the key must be `"clf__n_estimators"`, not `"n_estimators"`.

---

**`KeyError` or `unexpected keyword argument` during fit**

A parameter value from `param_grid` is invalid for the estimator in that configuration. Check:

- Integer ranges do not include values the estimator rejects (e.g., `max_depth=0` is invalid for most tree models; use `Integer(1, ...)`)
- Categorical choices are all valid strings or `None`, not mixed types the estimator cannot handle

---

**`Invalid param_grid entry for 'X': expected a space object ...`**

Every value in `param_grid` must be a space object, not a raw list or range:

```python
from sklearn_genetic.space import Categorical, Integer

# ❌ Wrong — raw list / range
param_grid = {"kernel": ["rbf", "linear"], "max_depth": range(2, 20)}

# ✅ Correct — space objects
param_grid = {"kernel": Categorical(["rbf", "linear"]), "max_depth": Integer(2, 20)}
```

## Precomputed kernels and distances

`GAFeatureSelectionCV` selects original feature columns. The columns of
a precomputed kernel or distance matrix represent training samples. Pass
the original features to the selector with an estimator that computes its
kernel or distances during fitting. A pipeline can also compute the matrix
from the selected columns inside each training fold.

Use [GASearchCV with a precomputed matrix](../examples/precomputed-kernel-search)
to tune an estimator's hyperparameters on a kernel or distances you have
already computed.

## All Candidates Score the Same

**Every generation shows the same `fitness` and `fitness_best`**

Possible causes:

1. **The scoring metric is saturated.** On a small dataset a simple model can already achieve 100% accuracy. Try a harder dataset or a more discriminative metric.

2. **The search space is too narrow.** If all parameter combinations produce similar models, expand the ranges or add more parameters.

3. **The cross-validation strategy is deterministic across runs.** Set `shuffle=True` and a `random_state` on your CV splitter.

4. **The search encountered fold failures.** Check for `FitFailedWarning`
   and inspect the split scores in `cv_results_`.

## All fits failed

When an estimator cannot fit any fold for a particular hyperparameter
configuration or feature subset, scikit-learn reports `All the ... fits failed`.
With numeric `error_score`, the search emits this diagnostic as a warning,
records `NaN` scores for that candidate and ranks it below candidates that fit.
If every candidate fails to fit, the search raises before selecting a winner.
With numeric `error_score`, it keeps `cv_results_`, `history` and
`fit_counts_` available for inspection. Failed fits still count toward the
CV budget. In feature selection, this also applies when every attempted mask
fails and the remaining masks were skipped. The `skipped_invalid_mask` column
identifies the masks that never reached CV.

Filter `cv_results_["all_fits_failed"]` to find the affected configurations or
masks by row. Partial fit failures and non-finite scores from fitted estimators
leave this flag false. Repeated proposals that reuse a failed evaluation carry
the same flag. See the
[failed candidate example](../tutorials/optimizer-backends#inspect-candidates-that-fail-all-cv-folds).

The [fold score heatmap](../api/plots#plot_cv_fold_scores) marks a candidate's
row with outlined squares when this flag is true. Pass the affected row
positions as `candidate_indices` to inspect them alongside successful rows.
When every fit fails, the heatmap can still show the affected rows and folds.
The [fold score distribution](../api/plots#plot_cv_scores) labels those
candidates **All fits failed** alongside the rows with finite scores.

For candidates that fit at least one fold, scikit-learn can use the requested
replacement score on failed folds. Set `error_score="raise"` on your search
to get the original exception and identify the parameter combination, feature
subset or data constraint that caused the failure. If the exception ends the
optimizer run, `search_time_` records the elapsed seconds up to that point.
`fit_counts_` includes completed CV results already returned by an
interrupted batch. With `search_interrupted_=True`, those values are lower
bounds. Checkpoints continue to track the search work that has been recorded.
See [how to read the duration after an error](../tutorials/optimizer-backends#inspect-an-optimizer-error).

## Finalist validation ends without a winner

When all finalist scores are non-finite, `fit` raises before the final refit.
`final_selection_results_["candidates"]` still contains each completed
finalist evaluation and its per-fold scores. `selected_index` is `None`.
For any failure in automatic finalist validation, check
`final_selection_results_["failed"]` and `final_selection_results_["time_seconds"]`.
Use `plot_fit_work(search, unit="seconds")` to see how long the failed
phase ran. If the error occurred while preparing the folds, the search
scores are still available in `cv_results_`.
If validation raises during a candidate, earlier completed evaluations
remain available. Inspect `final_selection_results_["failed_candidate"]`
for the interrupted row, parameters, CV calls, and known fold fits.
`fit_counts_["finalist_cv_calls"]` includes the interrupted call. The known
fold count is a lower bound because the interrupted call can have run some
folds before raising.
Use `plot_finalist_scores(search)` to see the completed evaluations and the
interrupted row. It can show the search score even if the first finalist
raises. Use `plot_fit_work(search)` for the fold counts. See the
[interrupted validation example](../tutorials/optimizer-backends#inspect-an-interrupted-finalist).

## Refit fails after cross-validation

The selected estimator can fail on the full training set after completing its
cross-validation folds. `fit` propagates the original error. The search
keeps its `cv_results_` and any completed finalist scores so you can inspect
the selected configuration. `fit_counts_["refit_fits"]` is zero and
`best_estimator_` becomes available only after a successful refit.

Read `refit_failure_time_` for the elapsed seconds before the error, or use
`plot_fit_work(search, unit="seconds")` to see it alongside the other phases.
See the [failed-refit example](../tutorials/optimizer-backends#inspect-a-failed-full-data-refit).

If finalist validation completed, use
[`plot_finalist_scores(search)`](../api/plots#plot-finalist-scores) to compare
the candidate chosen for refit with the other finalists. The plot reads saved
CV scores, so it works even after a refit error.

Use [a search with refit disabled](./advanced-optimizer-control#inspect-the-selected-candidate-without-refitting)
when you want to inspect the CV results and fit the selected model separately.

## No candidate has a finite objective score

The warning names the metric that guides optimization. Inspect its
`split*_test_<metric>` columns in `cv_results_` for non-finite values.
Check whether the scorer is defined on each validation fold. For example,
ROC AUC requires both classes in a binary validation fold. A finite
secondary metric is still available for reporting or callable refit.
With the usual refit rule, the selected row can have a non-finite
`best_score_`. See the
[undefined objective example](../tutorials/optimizer-backends#diagnose-an-undefined-objective).
Use the [fold score heatmap](../api/plots#plot_cv_fold_scores) to locate
non-finite cells and compare the same fold across candidates.

## Search Is Slow

Start with elapsed `fit` time and `fit_counts_["search_fold_fits"]` to compare
total runtime and search workload. Use `mean_fit_time` and `mean_score_time`
in `cv_results_` to inspect per-candidate fold timings in seconds.
`fit_counts_` also reports finalist and refit work separately. With caching
enabled, `fit_stats_["cache_hits"]` counts proposals reused from earlier batches
and `fit_stats_["duplicate_candidates"]` counts repeats that share a fresh
evaluation in their batch. The
[cache example](../tutorials/optimizer-backends#read-both-cache-counters-in-one-batch)
shows how to interpret both counters together. Compare
settings with the same CV splits, seed, and fresh candidate budget. See
[fit work by phase](../api/optimizer-backends#fit-work-by-phase)
and [`CVBudgetStopping`](../api/callbacks#cvbudgetstopping).

For pipelines that repeatedly fit an expensive transformer, try
[caching fitted preprocessing](./pipeline-tuning#cache-fitted-preprocessing-with-pipeline-memory)
and compare total `fit` time at the same number of fold fits.

### Check the cost of training scores

`return_train_score=True` scores each fitted fold model on its training
rows as well as its validation rows. This adds scoring work without increasing
`fit_counts_["search_fold_fits"]`. The `mean_score_time` column measures
validation scoring, so use elapsed `fit` time to compare this setting.

Both search classes default to `return_train_score=False` and still report
validation scores for every configured metric. Enable training scores when you
need the `mean_train_<metric>` and `split<N>_train_<metric>` columns, for
example to compare training and validation performance across candidates.

### Choose a parallelism layer

Start by allocating multiple workers to one layer. The search-level `n_jobs`
and the estimator's worker count are independent.

| Work to parallelize | Search settings | Estimator settings |
| --- | --- | --- |
| Different candidates | `n_jobs=2`, `parallel_backend="population"` or `"auto"` | Set `n_jobs=1` or the estimator's equivalent, if available |
| CV folds for one candidate | `n_jobs=2`, `parallel_backend="cv"` | Set `n_jobs=1` or the estimator's equivalent, if available |
| Training within one fold | `n_jobs=1` | Set the estimator's worker count, for example `RandomForestClassifier(n_jobs=2)` |

`parallel_backend="cv"` controls fold scheduling. A forest with its own
`n_jobs=2` can still request multiple workers inside each fold. With
`"population"`, cached proposals require no fold work, so many repeated
candidates can leave candidate workers idle. See the [joblib parallelism
recipe](../recipes/integrations/joblib-parallel).

Native OpenMP and BLAS libraries may also use threads. Check their limits
separately if CPU usage stays high or adding workers slows the run. See
[scikit-learn's guide to parallelism and oversubscription](https://scikit-learn.org/stable/computing/parallelism.html#oversubscription-spawning-too-many-threads).

## Population Converges Too Fast

**`genotype_diversity` drops to zero within a few generations**

The population has converged prematurely. Remedies:

- Check that `diversity_control=True` is active (it is by default as of 0.13.0). Verify via `search.diversity_control`.
- Lower `diversity_threshold` if it is higher than the observed `genotype_diversity` floor, or raise it if diversity control is not triggering early enough.
- Increase `random_immigrants_fraction` to inject more fresh individuals when triggered.
- Reduce `tournament_size` to lower selection pressure.
- Increase `population_size`. A larger population maintains diversity naturally.

Inspect the history to diagnose:

```python
import pandas as pd

history = pd.DataFrame(search.history)
print(history[["gen", "genotype_diversity", "unique_individual_ratio",
               "stagnation_generations", "diversity_control_triggered"]])
```

---

**`stagnation_generations` keeps growing**

The best score is not improving. This may mean:

- The search has found a genuine optimum. Check if the score is already close to what you expect.
- The population has converged around a local optimum. See the diversity section above.
- The scoring metric is too noisy for the CV fold count. Increase `cv` splits to reduce variance.

Use a callback to stop early and avoid wasting evaluations:

```python
from sklearn_genetic.callbacks import ConsecutiveStopping

search.fit(X_train, y_train, callbacks=[
    ConsecutiveStopping(generations=8, metric="fitness_best"),
])
```

## Understanding `fit_stats_`

Each call to `fit` clears the previous search's counters and candidate
records. If validation fails before the search begins, the previous
`fit_stats_`, `logbook` and `hof` are unavailable on that instance.

After fitting, `search.fit_stats_` is a dictionary with evaluation counters:

```python
{
    "evaluated_candidates":        420,  # total individuals presented
    "unique_candidates":           310,  # fresh candidate evaluations
    "cross_validate_calls":        310,  # actual CV calls made
    "cache_hits":                  110,  # scores reused from cache
    "duplicate_candidates":          0,  # repeats coalesced within a batch
    "skipped_invalid_candidates":    0,  # invalid candidates skipped before CV
    "population_parallel_batches":  21,
    "population_serial_batches":     0,
    "random_immigrants":            12,  # injected by diversity control
    "local_refinement_candidates":   2,
}
```

Key ratios to check:

- With caching enabled, add `cache_hits` and `duplicate_candidates` and divide
  by `evaluated_candidates` to see the share of proposals that reused a score.
- `skipped_invalid_candidates` counts candidates skipped before starting CV.
  If a feature mask is skipped, `cv_results_["skipped_invalid_mask"]`
  identifies its row. A failed fold handled by `error_score` still belongs
  to a CV call.
- `random_immigrants > 0` — diversity control was triggered at least once.

`fit_stats_` includes the search history restored from a checkpoint. For work
done by the latest call to `fit`, including finalist validation and refitting,
read [`fit_counts_`](../api/optimizer-backends#fit-work-by-phase).

## Reproducibility

**Results differ between runs with the same code**

Set `random_state` on the search — it seeds every stochastic part of the run
(population initialization, mutation, crossover, random immigrants) from one
place at `fit` time, so you don't need to seed the global `random`/`numpy` RNGs
by hand:

```python
search = GASearchCV(
    estimator=estimator,
    param_grid=param_grid,
    random_state=42,   # makes the whole search reproducible
    # ...
)
search.fit(X_train, y_train)
```

Also seed the CV splitter and any estimator that accepts `random_state`. See [Reproducibility](./reproducibility) for a complete example.

## Warm-Start Config Errors

**`ValueError` mentioning `warm_start_configs`**

Warm-start configs in `PopulationConfig(warm_start_configs=[...])` are checked at fit time, so a mistake raises a clear error instead of being silently ignored. You'll get one if a config:

- is not a dict
- uses a key that isn't in `param_grid` (e.g. a typo like `max_depths` instead of `max_depth`)
- gives a value outside an `Integer`/`Continuous` dimension's bounds, or not in a `Categorical` dimension's `choices`

Missing keys are fine — they're filled by sampling. Check your config keys against `list(search.param_grid.keys())` and the bounds of each dimension.

## Multi-Metric: `best_params_` Is Not What I Expected

First distinguish the metric guiding the search from the rule selecting its winner:

- With `refit="roc_auc"`, the search optimizes that named metric. With `refit=False` or a callable, set `objective_metric` to the metric to optimize.
- `best_params_` describes the selected row at `best_index_`. A callable can pick a different row from the one ranked first on the objective. In this case there is no `best_score_`; inspect, for example, `search.cv_results_["mean_test_roc_auc"][search.best_index_]`.
- When finalist validation is enabled, `best_score_` comes from that validation and can differ from the search score recorded in `cv_results_`.

To inspect what the best configuration would be for a different metric:

```python
import pandas as pd

results = pd.DataFrame(search.cv_results_)
best_by_f1 = results.sort_values("rank_test_f1").iloc[0]
print(best_by_f1["params"])
```

See [Multi-Metric Optimization](./multi-metric) for a full example.

## Plots Show Nothing or Raise Errors

`plot_fitness_evolution` and `plot_search_space` require seaborn:

```bash
python -m pip install "sklearn-genetic-opt[plot]"
```

Post-fit plots read generation history, candidate records or fitted results, depending on the plot. Setting `verbose=False` only changes printed output. Use [LivePlot](../api/plots#during-fitting) to see progress during fitting.

## Optimizer selection or checkpoint errors

| Symptom | What to check |
| --- | --- |
| Unknown optimizer backend | Check the installed package version and [install the engine extra](./installation#optimization-backends) in the interpreter running the search. Use the exact [backend name](../api/optimizer-backends), such as `"cmaes"` for CatCMAwM. A third-party name must be registered in `sklearn_genetic.optimizers`. |
| Import error naming an extra | From that checkout, install the named extra in the same environment, for example `python -m pip install -e ".[nevergrad]"`. |
| A control is DEAP-only | Remove custom mutation, tournament, elitism or other DEAP settings before selecting another engine. Consult its [supported settings](../api/optimizer-backends). |
| A progress snapshot has `resumable=False` | Use `ModelCheckpoint.load()` to inspect it. Earlier progress-only snapshots and snapshots from plugins without continuation cannot restart a solver. Give a new run a fresh path. |
| A catalog checkpoint reports incompatible settings or environment | Recreate the same training inputs, search settings, solver options and software environment. Pass the same callback path to `fit`. See the [resume recipe](../recipes/advanced/checkpointing). |

## Unsupported Estimator Type

**`... is not a valid Sklearn classifier, regressor, or outlier detector`**

`GASearchCV` tunes the hyperparameters of a classifier, regressor, or outlier
detector. If you want to select features instead, use `GAFeatureSelectionCV`:

```python
from sklearn_genetic import GAFeatureSelectionCV

selector = GAFeatureSelectionCV(estimator=clf, cv=3, scoring="accuracy")
selector.fit(X, y)
```

## Optional Dependencies

**`ImportError` / `ModuleNotFoundError` for `mlflow` or `seaborn`**

These power optional experiment logging and plots. Install the extra you need:

```bash
pip install "sklearn-genetic-opt[all]"   # plotting and MLflow
pip install "sklearn-genetic-opt[mlflow]" # MLflow logging only
pip install seaborn                        # plotting only
```

## `max_error` Scoring String Rejected

**`InvalidParameterError` mentioning `'max_error'`**

Recent scikit-learn versions don't accept `"max_error"` as a scoring string.
Use its negated form, which works as a maximization objective like the other
`neg_*` scorers:

```python
GASearchCV(estimator=reg, scoring="neg_max_error", param_grid=param_grid)
```

## Getting More Help

- [Understanding Cross-Validation](./understand-cv) — detailed explanation of the generation log and evaluation process.
- [Advanced Optimizer Control](./advanced-optimizer-control) — guidance on diversity control, fitness sharing, and local search telemetry.
- Open an issue at [github.com/rodrigo-arenas/Sklearn-genetic-opt/issues](https://github.com/rodrigo-arenas/Sklearn-genetic-opt/issues) and include the output of `search.fit_stats_`, the relevant section of `pd.DataFrame(search.history)`, and a minimal reproducible example.
