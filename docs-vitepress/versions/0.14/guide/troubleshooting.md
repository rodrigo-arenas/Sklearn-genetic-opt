---
title: Troubleshooting
description: Common problems encountered with sklearn-genetic-opt, organized by symptom.
---

# Troubleshooting

This page covers the most common problems, organized by symptom.

## Common Errors at a Glance

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `... is not a valid Sklearn classifier, regressor, or outlier detector` | An unsupported estimator was passed to `GASearchCV` | Use `GASearchCV` with a classifier, regressor, or outlier detector; for feature selection use `GAFeatureSelectionCV` |
| `Invalid param_grid entry for '...'` (value is not a space object) | A raw `list`/`tuple`/`range` was used instead of a space object | Wrap choices in `Categorical([...])` and ranges in `Integer(...)` / `Continuous(...)` |
| `ImportError` for `mlflow`, `tensorflow`, or `seaborn` | An optional dependency is not installed | `pip install "sklearn-genetic-opt[all]"` (or the specific extra) |
| `InvalidParameterError` mentioning `'max_error'` | recent scikit-learn no longer accepts `max_error` as a scoring string | Use `scoring="neg_max_error"` |
| Fit is much slower than expected | CPU oversubscription from nested parallelism | See [Search Is Slow](#search-is-slow) and the `parallel_backend` table |

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

## All Candidates Score the Same

**Every generation shows the same `fitness` and `fitness_best`**

Possible causes:

1. **The scoring metric is saturated.** On a small dataset a simple model can already achieve 100% accuracy. Try a harder dataset or a more discriminative metric.

2. **The search space is too narrow.** If all parameter combinations produce similar models, expand the ranges or add more parameters.

3. **The cross-validation strategy is deterministic across runs.** Set `shuffle=True` and a `random_state` on your CV splitter.

4. **`error_score` is masking failures.** If the estimator raises exceptions for some configurations, `error_score=nan` (default) replaces scores with `nan`, which can appear as a flat line. Check `fit_stats_["skipped_invalid_candidates"]`:

```python
print(search.fit_stats_)
# skipped_invalid_candidates > 0 means some configs raised exceptions
```

Switch to `RuntimeConfig(error_score="raise")` temporarily to see the actual exception.

## Search Is Slow

**Fit takes much longer than expected**

1. **Nested parallelism.** The default `parallel_backend="auto"` parallelizes across unique candidates in a generation. If the estimator itself uses parallelism (e.g., `RandomForestClassifier(n_jobs=-1)`), you will oversubscribe the CPU. Either set the estimator's `n_jobs=1`, or switch parallelism to the CV level:

```python
runtime_config=RuntimeConfig(parallel_backend="cv", n_jobs=-1)
```

This keeps candidate evaluation serial but parallelizes the cross-validation splits for each candidate.

**Choosing `parallel_backend` vs estimator `n_jobs`.** Parallelize in exactly one layer to avoid oversubscribing the CPU:

| Your estimator | Recommended setting |
|----------------|---------------------|
| Pure sklearn estimator with no internal threading (e.g. `SVC`, `LogisticRegression`) | `parallel_backend="population"` (or `"auto"`); leave estimator `n_jobs` unset |
| sklearn estimator already using all cores (e.g. `RandomForestClassifier(n_jobs=-1)`) | Set the estimator's `n_jobs=1`, **or** use `parallel_backend="cv"` |
| XGBoost / LightGBM / CatBoost | Set the model's thread count to 1 (`n_jobs=1` / `thread_count=1`) and use `parallel_backend="cv"` |

Rule of thumb: parallelize either across candidates/CV (`parallel_backend`) **or** inside the estimator (`n_jobs`) — not both. See the [joblib parallelism recipe](../recipes/integrations/joblib-parallel) and [`RuntimeConfig.parallel_backend`](../api/config).

2. **Too many unique candidates.** Check `fit_stats_["unique_candidates"]`. If it equals `fit_stats_["evaluated_candidates"]` and `cache_hits` is zero, the cache is not helping. This is normal on the first run.

3. **Population is too large relative to the space.** On a small space with many duplicate candidates, reduce `population_size` and add more generations instead.

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

After fitting, `search.fit_stats_` is a dictionary with evaluation counters:

```python
{
    "evaluated_candidates":        420,  # total individuals presented
    "unique_candidates":           310,  # distinct configs cross-validated
    "cross_validate_calls":        310,  # actual CV calls made
    "cache_hits":                  110,  # scores reused from cache
    "duplicate_candidates":          0,  # within-generation duplicates
    "skipped_invalid_candidates":    0,  # configs that raised exceptions
    "population_parallel_batches":  21,
    "population_serial_batches":     0,
    "random_immigrants":            12,  # injected by diversity control
    "local_refinement_candidates":   2,
}
```

Key ratios to check:

- `cache_hits / evaluated_candidates` — cache efficiency; above 20% is good for many-generation searches.
- `skipped_invalid_candidates > 0` — some parameter combinations caused exceptions.
- `random_immigrants > 0` — diversity control was triggered at least once.

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

With a multi-metric scoring dict, `GASearchCV` optimizes the `refit` metric during the evolutionary search. `best_params_` and `best_score_` always refer to the `refit` metric.

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
pip install sklearn-genetic-opt[all]
# or just seaborn:
pip install seaborn
```

The plotting helpers operate on the fitted estimator's `logbook` attribute. If `verbose=False` was set during fit, the logbook is still populated — the flag only controls printed output.

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

**`ImportError` / `ModuleNotFoundError` for `mlflow`, `tensorflow`, or `seaborn`**

These power optional features (experiment logging, the TensorBoard callback, and
plots). Install the extra you need:

```bash
pip install "sklearn-genetic-opt[all]"   # everything
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
