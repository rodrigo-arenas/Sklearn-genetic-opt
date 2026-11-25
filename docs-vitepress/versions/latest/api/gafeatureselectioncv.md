---
title: GAFeatureSelectionCV
description: API reference for GAFeatureSelectionCV — genetic algorithm feature selection for scikit-learn estimators.
---

:::warning Development version
You are reading the **latest (dev)** docs. For the stable version, see [stable](/stable/).
:::

# GAFeatureSelectionCV

Genetic algorithm wrapper-based feature selection with cross-validation.

`GAFeatureSelectionCV` selects a subset of columns from the input data that maximises the cross-validation score of the given estimator. After fitting, it exposes `support_` (a boolean mask of selected features) and behaves as a fitted scikit-learn estimator for prediction.

## Common constructor arguments

```python
from sklearn_genetic import GAFeatureSelectionCV

GAFeatureSelectionCV(
    estimator,
    cv=3,
    scoring=None,
    refit=True,
    verbose=True,
    keep_top_k=1,
    elitism=True,
    error_score=np.nan,
    return_train_score=False,
    random_state=None,
    evolution_config=None,
    population_config=None,
    runtime_config=None,
    optimization_config=None,
    optimizer_backend="deap",
    preserve_dataframe=False,
)
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `optimizer_backend` | str or callable, keyword-only | `"deap"` | Select `"deap"`, `"scipy"`, `"nevergrad"`, `"cmaes"`, `"pymoo"` or a plugin. See [backend settings and limits](./optimizer-backends) and the [tutorial](../tutorials/optimizer-backends). |
| `estimator` | estimator | — | A scikit-learn estimator with a `fit` method |
| `cv` | int, CV splitter, iterable or `None` | `3` | Cross-validation strategy, including for [outlier detection](../guide/outliers#choose-validation-folds). See [explicit fold validation](../tutorials/optimizer-backends#validate-explicit-cv-splits). |
| `scoring` | str, callable, list, tuple or dict | `None` | Metric(s) to evaluate. `None` uses default scoring. A standalone callable returns one numeric score per fold. Use [named scorers](../guide/multi-metric) for multiple metrics. |
| `refit` | bool, str or callable | `True` | Refit the selected mask. With multiple metrics, pass a metric name or a [refit function](../guide/advanced-optimizer-control#choose-a-refit-candidate-with-a-function) and `objective_metric`. |
| `objective_metric` | str or `None`, keyword-only | `None` | Search objective for multiple metrics with `refit=False` or callable `refit` |
| `preserve_dataframe` | bool, keyword-only | `False` | Pass selected pandas columns with their names and dtypes to the wrapped estimator during search, finalist validation, refit and prediction. |
| `verbose` | bool | `True` | Print the generation log |
| `keep_top_k` | int | `1` | Number of hall-of-fame individuals to keep across generations |
| `elitism` | bool | `True` | DEAP control for carrying an elite into the next generation |
| `error_score` | float or `"raise"` | `np.nan` | Replacement score for failed CV folds. With a numeric setting, masks that [fail to fit every fold](../guide/troubleshooting#all-fits-failed) receive `NaN` and rank last. |
| `random_state` | int, RandomState or `None` | `None` | Seeds the whole search at `fit` time for reproducible runs. `None` is non-deterministic |
| `population_initializer` | `"smart"`, `"random"` or `"spread"` | `"smart"` | Strategy for [initial feature masks](#initial-mask-order-under-a-cv-budget) |
| `evolution_config` | `EvolutionConfig` | `None` | Population size and generations; variation settings depend on the backend |
| `population_config` | `PopulationConfig` | `None` | Initialization strategy and optional warm-start feature masks |
| `runtime_config` | `RuntimeConfig` | `None` | Parallelism, caching, verbosity |
| `optimization_config` | `OptimizationConfig` | `None` | Finalist selection; DEAP-specific local search and fitness sharing |

## Search correctness

Each fresh `fit` recomputes candidate scores for the supplied data and scoring settings.
Caching avoids repeated evaluations within that fit; explicit checkpoint resumes retain
saved evaluations. Both elitist and non-elitist searches use tournament selection, so
negative scores and `criteria="min"` remain meaningful. Scikit-learn's negative loss
scorers normally use the default `criteria="max"`.

Smart initialization spreads target mask sizes across the allowed range. It
samples column indices without replacement, drawing the omitted columns for
dense masks. This reduces initialization cost on wide data. Within the same
version, set `random_state` to reproduce the initial masks. If caching is enabled
and the population exceeds the number of possible masks, smart initialization
reuses duplicate draws to fill the remaining slots with independent individuals.

`max_features` must be `None` or an integer from 1 through the number of input
columns. Initialisation, crossover and mutation repair masks to keep at least one
feature and respect that upper bound.

The built-in initializers prepare masks within these bounds during setup.
Random sampling repairs each mask as it is drawn. Smart, spread and warm
starts build valid masks directly.
DEAP sends those masks directly to the first evaluation batch when
the search has no callbacks or uses only the built-in `CVBudgetStopping`.
Custom startup callbacks can change the population before evaluation, so
their initial masks go through the usual repair.

### Initial mask order under a CV budget

`GAFeatureSelectionCV` accepts `population_initializer="spread"` or
`PopulationConfig(initializer="spread")`. This uses the same masks drawn by
`"smart"`. It evaluates a mask near the middle of the sampled size range first,
then takes the remaining mask whose size is farthest from the nearest size
already in the order.

Use this option with [CVBudgetStopping](./callbacks#cvbudgetstopping) when the
search may stop before evaluating the initial population. Early candidates
cover a wider range of subset sizes. The selected mask can have more features,
and fitting larger masks can take longer at the same evaluation budget. Compare
validation scores, held-out scores and fit time for your estimator and data.
See the [paired initialization example](../tutorials/optimizer-backends#compare-initial-mask-order-at-a-fixed-cv-budget).

### Start from known feature subsets

Supply `PopulationConfig(warm_start_masks=[previous.get_support()])` to
give a new search the Boolean mask from a fitted selector. Keep the same
input columns in the same order. Each mask must select at least one column,
respect `max_features` and have one entry per input column.

All five built-in engines evaluate distinct seeds in input order, up to
`population_size`. Entries after the population fills are ignored.
Invalid entries encountered during initialization raise `ValueError`.
The initializer fills the remaining slots, and `"spread"` orders those
generated masks by size. A fresh fit scores the seeds on its current CV
splits. Use masks chosen independently of the rows used to validate the
search. See the [warm-start example](../examples/warm-start-feature-selection).

## Attributes After `fit`

| Attribute | Description |
|-----------|-------------|
| `support_` | Boolean mask — `True` for selected features |
| `best_index_` | Row in `cv_results_` for the selected mask, including with `refit=False` when that row is available |
| `best_score_` | Mean CV objective score of the selected mask. Uses the finalist score when finalist validation runs and retains extended NumPy precision when available. With callable refit, read the score from the selected row. |
| `best_estimator_` | Estimator fitted with the selected features on the full training data when refitting is enabled |
| `search_time_` | [Elapsed search seconds until the optimizer returned or raised](./optimizer-backends#fit-time-by-phase) |
| `search_interrupted_` | Whether the optimizer raised and its [recorded search work](./optimizer-backends#fit-work-by-phase) is a lower bound |
| `refit_time_` | Elapsed seconds of the successful refit on the selected columns |
| `refit_failure_time_` | [Elapsed seconds of the full-data refit phase when it raises](./optimizer-backends#fit-time-by-phase) |
| `classes_` | Class labels when the selected estimator is a refitted classifier |
| `cv_results_` | Dict with per-candidate results |
| `history` | Dictionary of lists aligned by generation. Pair `history["cross_validate_calls"]` with `history["fitness_best"]` to track the best search score against the number of mask evaluations. |
| `logbook` | Compatibility view of per-candidate records, materialized on access for shared-runtime engines, in `logbook.chapters["parameters"]` |
| `fit_stats_` | Evaluation counters |
| `fit_counts_` | [Fresh fits and reused finalist folds by phase](./optimizer-backends#fit-work-by-phase) in the most recent call to `fit` |
| `search_space_exhausted_` | Whether the engine detected full coverage of a finite space with caching enabled. See [backend stopping rules](./optimizer-backends). |
| `n_features_` | Number of selected features in `support_`, available after a search with or without refit |
| `n_features_in_` | Number of input features before selection |
| `feature_names_in_` | Feature names seen during `fit` when `X` has string column names |

See [plotting score against evaluation cost](./plots#quick-diagnostic-overview) to compare runs with different cache-hit rates.

## Methods

| Method | Description |
|--------|-------------|
| `fit(X, y=None, callbacks=None, groups=None, **fit_params)` | Run the genetic feature selection |
| `cv_results_dataframe(preserve_index=False, columns=None, top_k=None)` | Return ranked feature masks as a pandas DataFrame |
| `transform(X)` | Return `X` with only the selected features |
| `inverse_transform(X)` | Restore the original feature count by inserting zeros in removed columns |
| `get_support(indices=False)` | Return the mask or indices of selected features |
| `get_feature_names_out(input_features=None)` | Return the selected feature names |
| `predict(X)` | Predict using `best_estimator_` on selected features |
| `predict_proba(X)` | Predict class probabilities |
| `score_samples(X)` | Return per-sample scores when the refitted estimator exposes that method |
| `score(X, y=None, *, sample_weight=None)` | Score the selected features with the configured scorer, the default anomaly scorer, or the search objective for multi-metric scoring. Requires refitting. |

Pass `y` if the estimator or a step in a scikit-learn `Pipeline` requires a
target. The selector raises a `ValueError` before evaluating masks when a
required target is missing. When the estimator accepts an omitted target,
call `fit(X)`. Lists, NumPy arrays and pandas DataFrames can supply the
feature columns in this mode. CV, refit and `score(X)` omit the target.
See [custom scoring without labels](../examples/unlabeled-outlier-scoring).

For multi-output regression or multilabel classification, pass `y` with shape
`(n_samples, n_outputs)` and choose an estimator that supports that target.
The selector applies one feature mask to `X` and evaluates it against all
outputs using a scorer that returns one value per fold. With `cv=None` or an
integer, multi-output regression and multilabel classification use `KFold`.
See the [multi-output regression example](../examples/multi-output-feature-selection).

For raw categorical columns, use an estimator that accepts two-dimensional
categorical input or a pipeline that starts with a compatible encoder.
The search passes each candidate subset to the pipeline as an array of
selected columns. See [feature selection with OneHotEncoder](../examples/categorical-feature-selection).

Feature selection requires the original input columns. If the estimator
declares a precomputed kernel or distance input, `fit` raises a
`ValueError` before evaluating masks. See
[precomputed kernels and distances](../guide/troubleshooting#precomputed-kernels-and-distances).

With a sparse-compatible estimator, you can pass a SciPy sparse matrix directly to
`fit`. The selector keeps CSR and CSC training matrices in their original formats
and converts other sparse formats, including COO, to CSC before selecting columns.
Prediction and scoring keep CSR and CSC inputs in their original formats. They
convert other sparse inputs to CSR before selecting columns. Choose the format
your wrapped estimator accepts for inference. See the
[sparse feature selection tutorial](../tutorials/sparse-feature-selection).
The selector reports the wrapped estimator's sparse input support through
scikit-learn's estimator tags, including when it is a step in a pipeline.
`transform` returns CSR for sparse inputs. For large CSC inputs that retain
at most half the columns, it validates the entire input and converts only
the selected columns to CSR.
See the [sparse pipeline example](../tutorials/sparse-feature-selection#use-the-selector-in-a-sparse-pipeline).
With caching enabled, standard evaluation on sparse inputs with at least
4,096 columns can build an independently writable Boolean mask from the
tuple already prepared for the cache lookup.
For wide sparse training matrices, see how to
[compare CSR and CSC at the same CV budget](../tutorials/sparse-feature-selection#compare-training-layouts-at-the-same-cv-budget).

With refitting enabled, scikit-learn recognizes the selector as the type of
estimator it wraps. For a classifier, an integer `cv` in external validation
uses stratified folds and scorers such as `"roc_auc"` use the refitted class
labels. With `refit=False`, the selector can be used as a transformer in a
pipeline whose final step makes predictions.

For an estimator that advertises `allow_nan` through scikit-learn's tags,
the selector preserves `NaN` in feature values during fit, transformation,
prediction and scoring. The estimator still validates the selected data.
For a scikit-learn `Pipeline`, the selector also checks the first active
step's tag. This lets a step such as `SimpleImputer` handle missing values
inside each cross-validation fit before passing the data to the classifier.
See the [sparse example with imputation](../tutorials/sparse-feature-selection#impute-missing-values-inside-feature-selection).
Supervised targets must contain valid values.
`inverse_transform` preserves these values in the selected columns and
inserts zeros for the removed columns. Sparse inputs produce a CSC result.
For CSR and CSC inputs, the selector reconstructs the sparse column index
directly. The result follows scikit-learn's configured sparse interface
when that setting is available.

`set_output(transform="pandas")` controls the output of `transform`. By
default, prediction and scoring pass selected columns as an array or sparse
matrix. The selector checks string column names and their order before
predicting or scoring.

Set `preserve_dataframe=True` when the wrapped estimator needs pandas column
names, dtypes or row indices. Supply a pandas DataFrame at fit and prediction
time. The selector validates its shape and column names, and the wrapped
estimator validates the selected values. Use a callable column selector when
preprocessing must adapt to the columns in each mask. A fixed list of
required names can fail if a candidate drops one of those columns. See the
[dtype-based preprocessing example](../examples/categorical-feature-selection#route-columns-by-dtype).

`score(X, y)` evaluates the refitted estimator on new data. Read
`cv_results_["features"][best_index_]` to find the selected mask in the search
results. Without finalist validation, standard refit and `refit=False`
follow the archived mask and score, including when several rows tie or
repeat that mask. When several rows match a finite archived score for the
same mask, `best_index_` identifies the first matching row. With a custom scorer
that includes a feature-count penalty, `score(X, y)` includes it.
Calculate an unpenalized test metric separately when needed.

With a callable `refit`, `final_selection_results_["original_best_index"]`
identifies the row the standard refit would select when the archived mask is
available in the search results. `selected_index` identifies the callable's
choice and `changed` compares those row indices.
See the [example that refits the best observed mask](../guide/advanced-optimizer-control#refit-the-best-observed-feature-mask).

Pass `sample_weight` to `score` to weight the evaluation rows when the
selected scorer accepts it. With metadata routing enabled, request an
alias such as `selector.set_score_request(sample_weight="outer_weight")`.
Then `cross_val_score(selector, X, y, scoring=None, params={"outer_weight": weights})`
passes the weights for each outer validation fold. When sharing a metadata
name across fit and score, set each estimator and scorer request explicitly.
For the search itself, `fit` can receive
[separate weights for estimator fitting and scoring](../guide/understand-cv#separate-training-and-validation-weights).
Pipelines also forward those requests to the selector when they call
`fit` or `fit_transform`.

With metadata routing enabled, groups reach the selector when its search
splitter or active finalist splitter requests them. Finalist validation
runs when refitting and `final_selection` are enabled. See
[grouped finalist validation](../tutorials/optimizer-backends#route-groups-to-feature-finalists)
for an example.

`cv_results_dataframe()` uses the selected `refit` or `objective_metric` for multi-metric
searches and keeps evaluation order for tied ranks. Its `params` column contains
each candidate's feature mask. It leaves `cv_results_` unchanged and also works
after a fit with `refit=False`.
In long histories with frequent cache reuse, the exporter can convert a
shared Boolean mask once. Each `params` row still has its own editable list.
The Boolean column `all_fits_failed` identifies masks whose estimator
failed to fit every search fold. See the
[CV troubleshooting guide](../guide/troubleshooting#all-fits-failed).
With numeric `error_score`, the table and `history` remain available when
every mask fails, even though `fit` raises.
If evaluation skips an empty mask or one outside `max_features`,
`cv_results_["skipped_invalid_mask"]` identifies the skipped rows. Their CV
scores are `NaN` and they are ineligible for finalist selection and refit.
The search raises with the diagnostics available when every mask is skipped.
When at least one mask was evaluated, the search raises if every evaluated mask
failed to fit every fold. The two Boolean columns distinguish skipped masks
from failed evaluations.
Use `cv_results_["n_features"]` for the size of each evaluated mask.
`rank_n_features` is one plus the number of candidate rows with fewer
selected features. Rows with the same size share the same rank.
For long histories with a bounded range of sizes, the selector computes
these ranks by counting rows at each size. When cached rows share a Boolean
mask, the selector can count its selected features once. Every completed
proposal has its own row and contributes to the ranks.

Pass `columns` as a name or a list of names to build only those columns,
in the requested order. Sorting still uses the selected test rank, and
unknown column names raise `KeyError`. For wide data, request columns such
as `n_features` and `mean_test_score` to inspect subset sizes and scores
without converting each mask into a list for `params`.

Pass a positive integer as `top_k` for a ranked shortlist. Tied rows retain
their evaluation order at the cutoff. For a short list from a long history,
the exporter can sort only the leading rows. If you request `params`, the
selector converts only the masks in the returned rows.

With `preserve_index=True`, the sorted table uses positions in `cv_results_`
as labels in the `candidate_index` index. When
`best_index_` is available, retrieve the selected row with
`selector.cv_results_dataframe(preserve_index=True).loc[selector.best_index_]`.

To [inspect a search without refitting](../guide/advanced-optimizer-control#inspect-the-selected-candidate-without-refitting),
set `refit=False`. Read `support_` for the mask and `best_score_` for its CV
objective score. For multiple scorers, set `objective_metric` to the one you
want to optimize.

`get_support()`, `transform()`, `inverse_transform()` and
`get_feature_names_out()` work after any successful fit. The selector's
prediction and scoring methods require its refitted estimator.

## Example

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from sklearn_genetic import EvolutionConfig, GAFeatureSelectionCV

X, y = load_iris(return_X_y=True)
rng = np.random.default_rng(42)
noise = rng.uniform(0, 10, size=(X.shape[0], 10))
X = np.hstack((X, noise))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

selector = GAFeatureSelectionCV(
    estimator=SVC(gamma="auto"),
    cv=3,
    scoring="accuracy",
    evolution_config=EvolutionConfig(population_size=6, generations=2),
    random_state=42,
)

selector.fit(X_train, y_train)

print("Selected features:", selector.support_)
print("Number of selected features:", selector.n_features_)
print("Search CV evaluations:", selector.history["cross_validate_calls"])
print("Selected mask's search row:", selector.best_index_)
print("Selected mask's CV accuracy:", round(selector.best_score_, 4))
print("Test accuracy:", round(selector.score(X_test, y_test), 4))
print(selector.cv_results_dataframe(columns=["n_features", "mean_test_score"], top_k=5))
```

## Use the selector in a pipeline without refitting

Set `refit=False` when the next pipeline step trains the final predictor.
The selector uses cross-validation to choose the columns. The final step
fits on those columns from the full training set.

For repeated observations, use a splitter such as `GroupKFold` to keep
each group together during feature selection. With metadata routing enabled,
pass `groups` to `pipeline.fit` when the selector's `cv` requests them. See
the [nested grouped pipeline example](../tutorials/optimizer-backends#select-features-inside-a-grouped-pipeline).
A custom `split(X, y)` method can supply folds for one stage when the other
active splitter accepts `groups`. See the
[custom finalist splitter example](../tutorials/optimizer-backends#combine-grouped-search-with-a-custom-finalist-splitter).

```python
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier

from sklearn_genetic import GAFeatureSelectionCV

X, y = load_iris(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)
selector = GAFeatureSelectionCV(
    DecisionTreeClassifier(random_state=42),
    scoring="accuracy",
    cv=3,
    population_size=6,
    generations=2,
    refit=False,
    n_jobs=1,
    random_state=42,
    verbose=False,
).set_output(transform="pandas")
pipeline = make_pipeline(selector, LogisticRegression(max_iter=500)).fit(X_train, y_train)
print(selector.get_feature_names_out())
print(selector.fit_counts_["refit_fits"])  # 0
print(pipeline.score(X_test, y_test))
```

## See Also

- [Basic Usage](../guide/basic-usage) — feature selection tutorial
- [Config Objects](./config) — configuration objects
- [GASearchCV](./gasearchcv) — hyperparameter tuning variant
