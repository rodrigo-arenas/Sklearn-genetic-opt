---
title: "How GASearchCV Uses Cross-Validation"
description: How cross-validation interacts with evolutionary search, how to read the generation log, and how to interpret convergence plots.
---

:::warning Development version
You are reading the **latest (dev)** docs. For the stable version, see [stable](/stable/).
:::

**Estimated reading time:** 8 minutes  
**Difficulty:** Intermediate  
**Prerequisites:** [Getting Started with GASearchCV](./basic-usage), basic cross-validation knowledge

# Cross-Validation in Genetic Algorithm Hyperparameter Search

This tutorial explains how `GASearchCV` evaluates candidate hyperparameters and how cross-validation fits into the evolutionary search process.

## Prerequisites

- Completed [Basic Usage](./basic-usage)
- Basic familiarity with scikit-learn cross-validation

## Key Parameters

`cv`
: Use a fold count, a scikit-learn splitter or an iterable of explicit train/test pairs. Each pair can use positional indices or Boolean masks. Explicit splits must include at least one pair. Lists and tuples of folds are checked for malformed pairs and invalid row selectors before candidate evaluation. See the [split validation example](../tutorials/optimizer-backends#validate-explicit-cv-splits).

`scoring`
: The metric used to evaluate each candidate. For classification: `"accuracy"`, `"roc_auc"`, `"f1"`. For regression: `"r2"`, `"neg_root_mean_squared_error"`. See [scikit-learn model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html) for the full list.

## How searches use the folds

Both search estimators materialize the folds before evaluating candidates.
A fresh search keeps the independent arrays returned by standard `KFold`,
`StratifiedKFold`, and `GroupKFold` splitters. The search copies each fold
from a custom splitter before requesting the next one, so its buffers can
be reused safely. The same rules apply to an explicit
`final_selection_cv`.

## Evolutionary Algorithm Background

A genetic algorithm is a metaheuristic optimization method inspired by natural selection. In sklearn-genetic-opt:

- **Individual** — one candidate solution (one set of hyperparameters).
- **Population** — a group of individuals evaluated in the same generation.
- **Generation** — one iteration of the evolutionary process.
- **Fitness value** — the cross-validation score used to compare individuals.

Each generation, individuals are selected by fitness, crossed over to create offspring, and mutated to introduce variation. The process repeats until the configured number of generations is reached or a callback stops the search.

## The Generation Log

When `RuntimeConfig(verbose=True)` is set, each generation prints a log row:

```
gen  nevals  fitness  fitness_std  fitness_best  fitness_max  fitness_min  div    unique  stag  events
1    20      0.932    0.012        0.961         0.961        0.901        0.87   1.00    0     
2    18      0.944    0.008        0.972         0.972        0.923        0.81   0.95    0     dup=2
```

| Column | Meaning |
|--------|---------|
| `gen` | Generation number |
| `nevals` | Candidates processed by the engine this generation; cached proposals need not trigger a fresh cross-validation call. Read `cross_validate_calls` for cumulative fresh candidate evaluations |
| `fitness` | Average CV score across the population |
| `fitness_std` | Standard deviation of the finite objective scores across candidates in the population |
| `fitness_best` | Best score retained in the optimizer archive. Use `history["fitness_best_observed"]` for the best finite score seen, even if the candidate was discarded |
| `fitness_max` | Highest finite population score (best only when maximizing) |
| `fitness_min` | Lowest finite population score (best when minimizing) |
| `div` | Mean probability that two distinct population members disagree at a coordinate |
| `unique` | Unique individual ratio — fraction of distinct individuals in the population |
| `stag` | Stagnation generations — consecutive generations without improvement in `fitness_best` |
| `events` | Optimizer interventions: `div` (diversity control), `imm=N` (N immigrants injected), `dup=N` (N duplicates replaced), `share` (fitness sharing applied) |

## How Candidates Are Evaluated

1. The optimizer proposes a batch of parameter configurations or feature masks.
2. With caching enabled, the evaluator looks for candidates already in the fitness cache and groups repeated proposals within the batch.
3. Fresh candidates can run in parallel when `n_jobs > 1` and `parallel_backend` is `"auto"` or `"population"`.
4. With population-level parallelism, each candidate runs its CV folds sequentially. Set `parallel_backend="cv"` to parallelize folds.
5. With caching enabled, the evaluator stores scores for reuse and records the completed candidates in proposal order.

Both searches make private candidate copies as they feed the evaluator. Serial
evaluation prepares one candidate at a time. The standard population worker
pool keeps a bounded number of candidates in flight, which limits temporary
memory for large populations and wide feature masks.

## Visualizing the CV Process

```python
import pandas as pd
from sklearn_genetic.plots import plot_fitness_evolution

# After fitting:
history = pd.DataFrame(evolved_estimator.history)
print(history[["gen", "fitness", "fitness_best", "genotype_diversity", "stagnation_generations"]])

plot_fitness_evolution(evolved_estimator)
```

The `plot_fitness_evolution` chart shows:
- The mean fitness (blue line) — average quality of the population over generations.
- The best fitness (orange line) — the best solution found so far.

A healthy search shows the mean and best fitness rising together, then plateauing as the population converges.

## Passing Metadata (groups & sample_weight)

With metadata routing disabled, both `GASearchCV` and `GAFeatureSelectionCV`
forward estimator fit parameters through cross-validation and the final refit:

```python
from sklearn.model_selection import GroupKFold

evolved_estimator = GASearchCV(clf, cv=GroupKFold(n_splits=5), param_grid=param_grid)

# groups drives the CV splitter; sample_weight reaches the estimator's fit
evolved_estimator.fit(X, y, groups=groups, sample_weight=weights)
```

- `groups` is passed to group-aware splitters (`GroupKFold`, `LeaveOneGroupOut`, ...)
  when materializing the folds.
- `sample_weight` (and any other `fit` metadata) reaches the wrapped estimator during
  candidate evaluation, final-selection scoring, and the final refit. Sample-aligned
  values are sliced per fold by scikit-learn.

With routing disabled, supply arguments accepted by the wrapped estimator's
`fit`. For a `Pipeline`, use a step prefix such as `clf__sample_weight`.

### Group labels in pipelines and nested validation

With metadata routing enabled, a search splitter such as `GroupKFold`
requests `groups` through `cv` for either estimator, including when
`GAFeatureSelectionCV` is a transformer in a pipeline. Use
`set_fit_request(groups=True)` when only `final_selection_cv` needs groups,
or pass a string to request the group labels under an alias.
For example, with `set_fit_request(groups="inner_groups")`, the search and
finalist splitters receive `inner_groups` and the outer splitter can request
`groups` separately.
With a direct call to the search's `fit`, pass `groups` as the named argument.

If a custom splitter defines only `split(X, y)`, you can still use it in
one stage when the other active splitter accepts the `groups` keyword.
The custom splitter defines its own train/test boundaries. Use folds
that keep related rows together at every stage where you want to
measure performance on unseen groups. See the
[custom finalist splitter example](../tutorials/optimizer-backends#combine-grouped-search-with-a-custom-finalist-splitter).

For nested validation on repeated observations, use group-aware splitters
for the outer evaluation, the candidate search, and any finalist validation.
The outer evaluator slices the group labels to the training rows before
fitting each search. The
[grouped validation tutorial](../tutorials/optimizer-backends#route-groups-in-nested-cross-validation)
shows how to route separate names to the outer splitter and the inner search.

### Separate training and validation weights

Use scikit-learn metadata requests to give the estimator and the scorer
different weight arrays in a direct call to either search's `fit`. Each array
has one value per row in `X`. The final refit receives the full training weights.

```python
from sklearn import config_context
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.tree import DecisionTreeClassifier
from sklearn_genetic import GAFeatureSelectionCV

with config_context(enable_metadata_routing=True):
    tree = DecisionTreeClassifier(random_state=42).set_fit_request(
        sample_weight="training_weight"
    )
    scorer = make_scorer(accuracy_score).set_score_request(
        sample_weight="validation_weight"
    )
    selector = GAFeatureSelectionCV(
        tree, scoring=scorer, cv=3, random_state=42,
        n_jobs=2, parallel_backend="population",
    )
    selector.fit(
        X, y, training_weight=training_weights, validation_weight=validation_weights
    )
```

Each candidate and finalist receives the training weights for its training rows,
and the scorer receives the validation weights for its validation rows.
Keep the call to `fit` inside `config_context` so candidate workers also use
the active routing configuration.
When `GAFeatureSelectionCV` is a step in a pipeline, pass the same aliases
to `pipeline.fit`. The selector exposes both sets of requests, including
when it acts as a transformer with `refit=False`.

## Tips & Gotchas

- Use `StratifiedKFold` for classification tasks to keep class balance across folds.
- Use `history["fitness_std"]` to see how spread out the population's scores are.
  To see how a single candidate's score varies across folds, read its row in
  `cv_results_["std_test_score"]` or `cv_results_["std_test_<metric>"]` for a named metric.
  The mean and standard deviation use the scorer's units, including for custom
  scorers with very large or very small finite values.
  The standard deviation takes the square root of the mean squared deviation
  from the fold mean (`ddof=0`). Compare it across candidates on the same splits.
- Use `cv_results_["rank_test_score"]` or `cv_results_["rank_test_<metric>"]`
  to compare positions, starting at 1. Tied scores share the first position in
  their group. For a maximized metric, scores of 0.9, 0.9, and 0.8 get ranks 1, 1,
  and 3. Repeated candidate rows count toward the next rank, including cache hits.
- The `use_cache=True` option in `RuntimeConfig` is strongly recommended — it avoids re-evaluating previously seen configurations.

## See Also

- [Common Hyperparameter Tuning Mistakes](./common-mistakes) — avoid CV-related mistakes
- [Early Stopping with Callbacks](./callbacks) — stop search when CV plateaus
- [Advanced Optimizer Control](./advanced-optimizer-control) — diversity and stagnation telemetry
- [Reproducibility & Checkpointing](./reproducibility) — seed your CV splitter
- [API: GASearchCV](../api/gasearchcv) — full parameter reference
