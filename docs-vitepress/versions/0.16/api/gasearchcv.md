---
title: GASearchCV
description: API reference for GASearchCV — genetic algorithm hyperparameter search for scikit-learn estimators.
---

# GASearchCV

Genetic algorithm hyperparameter search for scikit-learn estimators.

`GASearchCV` implements the scikit-learn estimator interface and follows the same patterns as `GridSearchCV` and `RandomizedSearchCV`. After calling `fit`, it exposes `best_params_`, `best_score_`, `cv_results_`, `predict`, `predict_proba`, and `score`.

## Common constructor arguments

```python
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
)
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `optimizer_backend` | str or callable, keyword-only | `"deap"` | Select `"deap"`, `"scipy"`, `"nevergrad"`, `"cmaes"`, `"pymoo"` or a plugin. See [backend settings and limits](./optimizer-backends) and the [tutorial](../tutorials/optimizer-backends). |
| `estimator` | estimator | — | A scikit-learn estimator with a `fit` method |
| `param_grid` | dict | — | Mapping from parameter name to `Integer`, `Continuous`, or `Categorical` |
| `scoring` | str or dict | `None` | Metric(s) to evaluate. `None` uses the estimator's default scorer |
| `cv` | int or CV splitter | `3` | Cross-validation strategy |
| `refit` | bool or str | `True` | Metric to use when refitting the best model. If `scoring` is a dict, pass the metric name |
| `verbose` | bool | `True` | Print the generation log |
| `error_score` | float or `"raise"` | `np.nan` | Score to use when a candidate raises an exception |
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
saved evaluations. Both elitist and non-elitist searches use tournament selection, so
negative scores and `criteria="min"` remain meaningful. Scikit-learn's negative loss
scorers normally use the default `criteria="max"`.

Final selection's `top_k` counts distinct parameter configurations, even when the
candidate log contains repeated cache hits.

## Attributes After `fit`

| Attribute | Description |
|-----------|-------------|
| `best_params_` | Parameter setting that gave the best mean cross-validated score |
| `best_score_` | Mean cross-validated score of the best estimator |
| `best_estimator_` | Estimator fitted with `best_params_` on the full training data (if `refit=True`) |
| `cv_results_` | Dict with per-candidate results, compatible with `pd.DataFrame` |
| `history` | Dictionary of lists aligned by generation. At position `i`, `history["fitness_best"][i]` is the best search score after `history["cross_validate_calls"][i]` cumulative candidate evaluations. |
| `logbook` | Compatibility view of per-candidate records, materialized on access for shared-runtime engines, in `logbook.chapters["parameters"]` |
| `fit_stats_` | Dict with evaluation counters (cache hits, skipped candidates, etc.) |

See [plotting score against evaluation cost](./plots#quick-diagnostic-overview) to compare runs with different cache-hit rates.

## Methods

| Method | Description |
|--------|-------------|
| `fit(X, y, callbacks=None)` | Run the genetic search |
| `cv_results_dataframe()` | Return candidate parameters and scores as a pandas DataFrame sorted by test rank |
| `predict(X)` | Predict using `best_estimator_` |
| `predict_proba(X)` | Predict class probabilities (if estimator supports it) |
| `score(X, y)` | Score using `best_estimator_` |
| `get_params()` / `set_params()` | Standard sklearn estimator interface |

After `set_params(param_grid=new_space)`, the next `fit` validates and uses
`new_space`, including changed parameter names. An explicit checkpoint resume
restores the saved search space.

`cv_results_dataframe()` uses the selected `refit` metric for multi-metric
searches and keeps evaluation order for tied ranks. It leaves `cv_results_`
unchanged and also works after a single-metric fit with `refit=False`.

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
print(search.cv_results_dataframe()[["params", "mean_test_score", "rank_test_score"]].head())
```

## See Also

- [Basic Usage](../guide/basic-usage) — step-by-step tutorial
- [Config Objects](./config) — `EvolutionConfig`, `PopulationConfig`, `RuntimeConfig`, `OptimizationConfig`
- [Search Space](./space) — `Integer`, `Continuous`, `Categorical`
- [GAFeatureSelectionCV](./gafeatureselectioncv) — feature selection variant
