---
title: "Switch optimization backends in a scikit-learn search"
description: "Compare optimization backends at a fixed cross-validation budget, select features and resume from a generation checkpoint."
---

:::warning Development version
This is the **latest (dev)** documentation. It may contain unreleased features or breaking changes. For the stable release, use [stable](/stable/).
:::

# Switch optimization backends in a scikit-learn search

Use the same folds and scorer when changing engines. This walkthrough uses all five named backends and keeps a test partition for the final model.

## Install the engines

Follow the [installation guide for your version](../guide/installation#optimization-backends).
To run every example, select the `scipy`, `nevergrad`, `cmaes` and `pymoo` extras.
For a single engine, select just its extra. The fitness chart also uses the `plot` extra.
The examples use built-in scikit-learn data. The parallel examples request two
candidate workers.

```python
from sklearn.base import clone
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn_genetic import GAFeatureSelectionCV, GASearchCV
from sklearn_genetic.space import Categorical, Continuous, Integer

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.25, random_state=7
)
model = DecisionTreeClassifier(random_state=7)
folds = StratifiedKFold(3, shuffle=True, random_state=7)
settings = dict(cv=folds, scoring="balanced_accuracy", refit=True, random_state=7,
                population_size=8, generations=2, n_jobs=1, verbose=False)
search = GASearchCV(
    model,
    param_grid={"max_depth": Integer(2, 10), "criterion": Categorical(["gini", "entropy"]),
                "ccp_alpha": Continuous(1e-6, 0.1, distribution="log-uniform")},
    **settings,
)
```

## Change the engine

Clone the template to start a fresh search. Inspect the best CV score and the number
of real CV calls. A cache hit reuses a score, and DEAP's default algorithm can generate
more offspring per generation. Each real CV call here fits three folds.
The engines share candidate preparation. Scalar search parameters such as the
tree depth and criterion can be passed directly into a cloned estimator.
Mutable choices such as dictionaries or pipeline steps are cloned using
scikit-learn's rules before they are applied.

```python
fitted = {}
for engine in ("deap", "scipy", "nevergrad", "cmaes", "pymoo"):
    run = clone(search).set_params(optimizer_backend=engine).fit(X_train, y_train)
    fitted[engine] = run
    print(engine, round(run.best_score_, 3), run.fit_stats_["cross_validate_calls"])
```

```text
deap 0.941 39
scipy 0.941 23
nevergrad 0.941 23
cmaes 0.941 24
pymoo 0.928 20
```

Inspect one fitted model on the held-out data. The choice of SciPy here is fixed in
advance. Use `cv_results_` for individual candidates and `history` for generations.
SciPy proposes values for the log-uniform pruning parameter in logarithmic
coordinates. The candidate table shows each value in the tree's original units.
For a performance comparison across datasets and seeds, see the
[matched-budget benchmark](../benchmarks/optimizer-catalog).

```python
run = fitted["scipy"]
print("candidate rows", len(run.cv_results_["mean_test_score"]))
pruning = run.cv_results_["param_ccp_alpha"]
print("pruning range", round(min(pruning), 7), round(max(pruning), 7))
print("CV calls by generation", list(run.history["cross_validate_calls"]))
print("test balanced accuracy", round(balanced_accuracy_score(y_test, run.predict(X_test)), 3))
```

```text
candidate rows 24
pruning range 1.4e-06 0.0522366
CV calls by generation [8, 16, 23]
test balanced accuracy 0.917
```

## Keep an integer parameter fixed

Set both integer bounds to the same value to keep a parameter fixed in the
search space. Here smart initialization varies the tree depth and keeps two
training samples per leaf as the minimum. The fixed value appears in every
candidate record. The pruning parameter uses log-uniform sampling inside
its positive bounds.

```python
from sklearn_genetic.callbacks import CVBudgetStopping

fixed = clone(search).set_params(
    param_grid={
        "max_depth": Integer(2, 8),
        "min_samples_leaf": Integer(2, 2),
        "ccp_alpha": Continuous(1e-6, 1e-2, distribution="log-uniform"),
    },
    use_cache=False,
)
fixed.fit(X_train, y_train, callbacks=CVBudgetStopping(4))
print("Sampled depths", fixed.cv_results_["param_max_depth"])
print("Fixed leaf minimums", fixed.cv_results_["param_min_samples_leaf"])
print("Sampled pruning strengths", [round(value, 6) for value in fixed.cv_results_["param_ccp_alpha"]])
```

```text
INFO: Stopping the algorithm
Sampled depths [7, 5, 5, 4]
Fixed leaf minimums [2, 2, 2, 2]
Sampled pruning strengths [0.009061, 4.1e-05, 0.003083, 1.8e-05]
```

For a grid made entirely of built-in integer dimensions, the standard
evaluator can reuse the repaired values when dispatching each candidate.
Here the leaf minimum stays fixed and the depth varies.

```python
integer_only = clone(fixed).set_params(
    param_grid={"max_depth": Integer(2, 8), "min_samples_leaf": Integer(2, 2)},
).fit(X_train, y_train, callbacks=CVBudgetStopping(4))
print("Candidate integers", integer_only.cv_results_["params"])
print("Search CV calls", integer_only.fit_counts_["search_cv_calls"])
```

```text
INFO: Stopping the algorithm
Candidate integers [{'max_depth': 8, 'min_samples_leaf': 2}, {'max_depth': 3, 'min_samples_leaf': 2}, {'max_depth': 5, 'min_samples_leaf': 2}, {'max_depth': 7, 'min_samples_leaf': 2}]
Search CV calls 4
```

## Select features and inspect progress

Pass the same keyword to `GAFeatureSelectionCV`. pymoo evolves binary masks and
repairs proposals as needed to keep at least one column and respect `max_features`.
The estimator applies the selected mask for prediction. A stopping callback checks
each reported generation. Here the callback stops after generation 1 so we can
resume the original two-generation budget and measure additional CV calls.
To inspect CV scores and subset sizes without converting the masks to lists,
request those columns and keep the original candidate positions as labels.
Pass `top_k` when you only need the first few ranked candidates.

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from sklearn_genetic.callbacks import BaseCallback, ModelCheckpoint

class StopAfterOne(BaseCallback):
    def on_step(self, record=None, logbook=None, estimator=None):
        return record["gen"] == 1

selector = GAFeatureSelectionCV(model, optimizer_backend="pymoo", max_features=8, **settings)
with TemporaryDirectory() as directory:
    checkpoint = ModelCheckpoint(Path(directory) / "progress.pkl")
    clone(selector).fit(X_train, y_train, callbacks=[StopAfterOne(), checkpoint])
    saved = checkpoint.load()
    selector.fit(X_train, y_train, callbacks=[checkpoint])

print("checkpoint", saved["backend"], saved["logbook"][-1]["gen"], saved["resumable"])
print("final generation", selector.history["gen"][-1])
print("additional CV calls",
      selector.fit_stats_["cross_validate_calls"] - saved["fit_stats"]["cross_validate_calls"])
mask_sizes = selector.cv_results_["n_features"]
print("evaluated subset sizes", min(mask_sizes), "to", max(mask_sizes))
print("selected columns", selector.get_support(indices=True).tolist())
print("test balanced accuracy", round(balanced_accuracy_score(y_test, selector.predict(X_test)), 3))
ranked = selector.cv_results_dataframe(
    columns=["n_features", "mean_test_score"], preserve_index=True, top_k=3
)
print(ranked.round(3).reset_index().to_string(index=False))
```

```text
checkpoint pymoo 1 True
final generation 2
additional CV calls 6
evaluated subset sizes 1 to 8
selected columns [6, 11, 12, 15, 23, 27]
test balanced accuracy 0.901
 candidate_index  n_features  mean_test_score
              11           6            0.934
              17           6            0.934
              18           6            0.934
```

SciPy and pymoo verify the recorded proposals while reconstructing the solver,
using saved results for generations already completed. Nevergrad and CatCMAwM
restore native solver state. Use the same data, search settings and environment
when resuming. Work in a partially completed generation can require CV again.
The [checkpoint recipe](../recipes/advanced/checkpointing) also covers DEAP's
historical format and older progress-only files. Third-party engines can implement
continuation through the [optimizer protocol](../api/algorithms#callbacks-and-checkpoints).

## Diagnose an invalid checkpoint counter

A resumable optimizer snapshot includes the cumulative counters in
`fit_stats`, including cache hits and worker batches. `fit` checks those
counters before evaluating candidates. This example writes a separate temporary
copy with an invalid cache-hit count to show the error.

```python
import pickle

with TemporaryDirectory() as directory:
    invalid_path = Path(directory) / "invalid-counter.pkl"
    invalid = {**saved, "fit_stats": {**saved["fit_stats"], "cache_hits": -1}}
    invalid_path.write_bytes(pickle.dumps(invalid))

    try:
        clone(selector).fit(X_train, y_train, callbacks=ModelCheckpoint(invalid_path))
    except ValueError as error:
        print(str(error).replace(str(invalid_path), invalid_path.name))
    else:
        raise AssertionError("Expected invalid checkpoint counter to fail")
```

```text
Invalid search checkpoint at invalid-counter.pkl: fit_stats.cache_hits must be a nonnegative integer. Use a new checkpoint path to start a search.
```

## Compare feature selection at the same CV budget

Use `CVBudgetStopping` to cap fresh candidate evaluations even when a generation
has more proposals than your budget. This example disables caching so each
evaluated mask uses one of the four available CV calls. Each call fits three
folds, and the final refit has its own count.

```python
from sklearn_genetic.callbacks import CVBudgetStopping

for engine in ("scipy", "pymoo"):
    limited = clone(selector).set_params(optimizer_backend=engine, use_cache=False)
    limited.fit(X_train, y_train, callbacks=CVBudgetStopping(4))
    costs = limited.fit_counts_
    masks = limited.cv_results_["features"]
    print(engine, "CV calls", costs["search_cv_calls"],
          "fold fits", costs["search_fold_fits"], "refits", costs["refit_fits"],
          "recorded masks", len(masks),
          "within feature limit", all(1 <= mask.sum() <= limited.max_features for mask in masks))
```

```text
scipy CV calls 4 fold fits 12 refits 1 recorded masks 4 within feature limit True
pymoo CV calls 4 fold fits 12 refits 1 recorded masks 4 within feature limit True
```

The budget limits fresh search CV. The selector still constructs its initial
population, and optional finalist validation and refit have their own costs.
Built-in initialization prepares feature masks within the requested bounds
and reuses those masks during setup.
DEAP also uses the prepared masks for its first evaluation batch with no
callbacks or when the built-in CV budget callback is the only callback.
SciPy and pymoo propose binary feature masks. The selector validates their
width and selected column count and can evaluate integer masks directly after
that check. The recorded masks in `cv_results_` are Boolean arrays.
The batch evaluator prepares private masks as candidates are dispatched.
With caching enabled and at least 4,096 sparse input columns, the native
evaluator can build that writable mask from the already prepared tuple key.
Serial evaluation prepares one row at a time, and the standard population
worker pool keeps a bounded number of rows in flight.
For fresh feature selection runs, pymoo, Nevergrad and CatCMAwM can reuse
the immutable initial masks until dispatch.
With a CV budget and no timer, it checks cache keys only through the first
proposal that would need another CV call. Reusable scores before that
proposal still appear in order. With caching disabled, there are no cache
lookups. The recorded masks reflect the proposals completed.
When comparing runtime, keep the data, folds, seeds and worker settings fixed.
Compare the observed CV counts and record held-out scores separately.
`cv_results_["mean_fit_time"]` and `cv_results_["mean_score_time"]` summarize
recorded fold timings in seconds. Time the full `fit` call to include optimizer
work, candidate dispatch, summary preparation, and refit.
To visualize fresh fits and reused scores, see the
[fit-work example](../examples/plotting-gallery#fit-work-by-phase).

### Compare initial mask order at a fixed CV budget

The `"spread"` initializer visits a wider range of mask sizes early in the
same population drawn by `"smart"`. Compare both orders when your CV budget is
smaller than `population_size`. The example also fits both orders with enough
CV calls for the full initial population.

```python
for budget in (4, 8):
    for initializer in ("smart", "spread"):
        ordered = clone(selector).set_params(
            population_initializer=initializer, population_size=8, generations=1,
            use_cache=True,
        )
        ordered.fit(X_train, y_train, callbacks=CVBudgetStopping(budget))
        holdout = balanced_accuracy_score(y_test, ordered.predict(X_test))
        print(initializer, "CV calls", ordered.fit_counts_["search_cv_calls"],
              "sizes", [int(size) for size in ordered.cv_results_["n_features"]],
              "selected", int(ordered.get_support().sum()),
              "CV", round(ordered.best_score_, 3), "holdout", round(holdout, 3))
```

```text
smart CV calls 4 sizes [1, 2, 3, 4] selected 4 CV 0.915 holdout 0.951
spread CV calls 4 sizes [4, 8, 1, 6] selected 6 CV 0.93 holdout 0.899
smart CV calls 8 sizes [1, 2, 3, 4, 5, 6, 7, 8] selected 6 CV 0.93 holdout 0.899
spread CV calls 8 sizes [4, 8, 1, 6, 2, 3, 5, 7] selected 6 CV 0.93 holdout 0.899
```

Use the actual call counts to compare the two orders. Check the CV and
holdout scores separately and inspect the selected subset size. Larger
masks can raise fit cost even at equal CV counts. Run the comparison across
several seeds and splits before choosing an order for your workload.

### Choose a CV budget at startup

A `CVBudgetStopping` subclass can set a positive integer limit in
`on_start` after inspecting the materialized search folds. Call the base
method so the callback counts fresh work in each fit, including resumes.
With several budget callbacks, the smallest initialized limit applies.

```python
class FoldAwareBudget(CVBudgetStopping):
    def on_start(self, estimator=None):
        super().on_start(estimator)
        self.max_cv_calls = 4 if estimator.n_splits_ >= 5 else 8

adaptive = clone(search).set_params(cv=5, use_cache=False).fit(
    X_train, y_train, callbacks=[CVBudgetStopping(6), FoldAwareBudget(12)],
)
print("Search CV calls", adaptive.fit_counts_["search_cv_calls"])
print("Search fold fits", adaptive.fit_counts_["search_fold_fits"])
```

```text
INFO: Stopping the algorithm
Search CV calls 4
Search fold fits 20
```

## Inspect actual work for each finalist

Use the same three search folds and compare fresh, partial, and full reuse
for a deterministic estimator. `cv_calls` counts actual calls for a finalist,
and `fold_fits` counts each new estimator fit in those calls. `reused_folds`
counts the search scores that contributed to the final score. If a partial
evaluation requires a retry, `fold_fits` includes both attempts.

```python
search_folds = list(folds.split(X_train, y_train))
extra_folds = list(StratifiedKFold(3, shuffle=True, random_state=23).split(X_train, y_train))
for label, mode, final_cv in (
    ("Fresh", True, None),
    ("Partial reuse", "reuse_folds", [search_folds[0], search_folds[1], extra_folds[2]]),
    ("Full reuse", "reuse_folds", None),
):
    finalist_run = clone(search).set_params(
        cv=search_folds, final_selection=mode, final_selection_cv=final_cv,
        final_selection_top_k=2, use_cache=False, error_score="raise",
        population_size=4, generations=0,
    ).fit(X_train, y_train)
    print(label, [(row["cv_calls"], row["fold_fits"], row["reused_folds"])
                  for row in finalist_run.final_selection_results_["candidates"]])
```

```text
Fresh [(1, 3, 0), (1, 3, 0)]
Partial reuse [(1, 1, 2), (1, 1, 2)]
Full reuse [(0, 0, 3), (0, 0, 3)]
```

Each tuple gives CV calls, new fold fits, and reused scores for one finalist.
Read `fit_counts_` for totals across finalists and the rest of the fit.
For hyperparameter search, tied search ranks promote distinct configurations
in evaluation order. Read each finalist's `index` to locate its row in
`cv_results_`.
In feature selection, scikit-learn clones the estimator for each new fold fit
during search and finalist validation. The final refit uses its own clone of
the estimator on the selected columns.

Read the elapsed seconds for each phase of the last fit in the loop.
Compare these durations alongside `fit_counts_` when diagnosing fit cost.
See [fit time by phase](../api/optimizer-backends#fit-time-by-phase) for the
work measured by each field.

```python
phase_seconds = {
    "search": finalist_run.search_time_,
    "finalist_validation": finalist_run.final_selection_results_["time_seconds"],
    "refit": finalist_run.refit_time_,
}
```

### Preserve repeated folds in the same order

When the search and finalist fold sequences match in order, each finalist
score comes from the corresponding search position. This includes repeated
train/test pairs. The controlled scorer below gives different values to
two occurrences of the same split so you can see which scores were reused.

```python
repeated_folds = [search_folds[0], search_folds[1], search_folds[0]]
repeated_scores = iter((0.25, 0.5, 1.0))

def controlled_score(estimator, X, y):
    return next(repeated_scores) if estimator.max_depth == 2 else 0.5

repeat_run = GASearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid={"max_depth": Integer(2, 3), "criterion": Categorical(["gini"])},
    cv=repeated_folds, scoring=controlled_score, final_selection="reuse_folds",
    final_selection_top_k=2, population_size=6, generations=0,
    random_state=42, error_score="raise", verbose=False,
).fit(X_train, y_train)
best = repeat_run.final_selection_results_["candidates"][0]
print("Winner depth", repeat_run.best_params_["max_depth"])
print("Fold scores in search order", best["cv_scores"])
print("New finalist fold fits", repeat_run.fit_counts_["finalist_fold_fits"])
```

```text
Winner depth 2
Fold scores in search order [0.25, 0.5, 1.0]
New finalist fold fits 0
```

### Reuse folds defined by Boolean masks

A fold can also identify its rows with a Boolean mask per partition.
The two masks have one entry per training row. A mask selects rows in
dataset order. Deterministic scores can be reused across masks and positional
indices when both identify the same rows in the same order.
Negative positions count from the end of the training dataset.

```python
import numpy as np

masked_folds = []
for train_rows, validation_rows in search_folds:
    train_mask = np.zeros(len(y_train), dtype=bool)
    validation_mask = np.zeros(len(y_train), dtype=bool)
    train_mask[train_rows] = True
    validation_mask[validation_rows] = True
    masked_folds.append((train_mask, validation_mask))

negative_folds = [(train - len(y_train), test - len(y_train)) for train, test in search_folds]
for label, final_cv in (
    ("Same masks", None),
    ("Positional indices", search_folds),
    ("Negative indices", negative_folds),
):
    masked_search = clone(search).set_params(
        cv=masked_folds, final_selection="reuse_folds", final_selection_top_k=2,
        final_selection_cv=final_cv,
        population_size=4, generations=0, error_score="raise",
    ).fit(X_train, y_train)
    counts = masked_search.fit_counts_
    print(label, "finalist fold fits", counts["finalist_fold_fits"],
          "reused scores", counts["finalist_reused_folds"])
```

```text
Same masks finalist fold fits 0 reused scores 6
Positional indices finalist fold fits 0 reused scores 6
Negative indices finalist fold fits 0 reused scores 6
```

## Validate explicit CV splits

Supply at least one train/test pair when setting `cv` or `final_selection_cv`
to an explicit collection of folds. An empty search collection raises before
candidate evaluation. Before evaluating candidates, the search checks that
each explicit list, tuple, or array fold provides a training selector and a
validation selector. These checks apply to lists and tuples supplied to
`cv` and `final_selection_cv`. The same check identifies
out-of-range positions and wrong-length Boolean masks in one-dimensional
selectors. The error names the fold that needs correction.

```python
for invalid_splits in (
    {"cv": []},
    {"cv": [(search_folds[0][0], [len(y_train)])]},
    {"final_selection": True, "final_selection_cv": []},
    {"final_selection": True, "final_selection_cv": [(search_folds[0][0],)]},
    {"final_selection": True,
     "final_selection_cv": [(search_folds[0][0], [len(y_train)])]},
):
    probe = clone(search).set_params(**invalid_splits)
    try:
        probe.fit(X_train, y_train, callbacks=CVBudgetStopping(1))
    except (ValueError, IndexError) as error:
        print(error)
    else:
        raise AssertionError("Expected an empty fold list to raise")

one_fold = list(folds.split(X_train, y_train))[:1]
single = clone(search).set_params(cv=one_fold)
single.fit(X_train, y_train, callbacks=CVBudgetStopping(1))
print("Search CV calls", single.fit_counts_["search_cv_calls"],
      "fold fits", single.fit_counts_["search_fold_fits"])
```

```text
cv must provide at least one train/test split.
cv fold 1 validation indices must be between -426 and 425
final_selection_cv must provide at least one train/test split.
final_selection_cv fold 1 must contain exactly two selectors (training and validation), got 1
final_selection_cv fold 1 validation indices must be between -426 and 425
INFO: Stopping the algorithm
Search CV calls 1 fold fits 1
```

## Repeat a seeded DEAP feature search

DEAP's uniform crossover draws once per shared column when it combines two
parent masks. Keep the search seed, estimator seed and folds fixed to reproduce
the sequence. Compare the masks in `cv_results_` to check every evaluated
candidate, including those evaluated after the initial population.

```python
import numpy as np

seeded = clone(selector).set_params(optimizer_backend="deap", use_cache=False)
first = clone(seeded).fit(X_train, y_train, callbacks=CVBudgetStopping(16))
second = clone(seeded).fit(X_train, y_train, callbacks=CVBudgetStopping(16))
candidate_pairs = zip(
    first.cv_results_["features"], second.cv_results_["features"], strict=True
)
print("Same masks", all(np.array_equal(a, b) for a, b in candidate_pairs))
print("CV calls", first.fit_counts_["search_cv_calls"], second.fit_counts_["search_cv_calls"])
```

```text
INFO: Stopping the algorithm
INFO: Stopping the algorithm
Same masks True
CV calls 16 16
```

## Check worker use at a small CV budget

A population batch that starts with one remaining CV call runs that candidate
on the calling thread.
The `population_serial_batches` and `population_parallel_batches` counters show
which dispatch path was used. When you need to parallelize the folds of a single
candidate, select `parallel_backend="cv"`.

```python
for limit in (1, 2):
    probe = clone(search).set_params(
        n_jobs=2, parallel_backend="population", use_cache=False
    )
    probe.fit(X_train, y_train, callbacks=CVBudgetStopping(limit))
    print("CV calls", probe.fit_counts_["search_cv_calls"],
          "serial batches", probe.fit_stats_["population_serial_batches"],
          "parallel batches", probe.fit_stats_["population_parallel_batches"])
```

```text
INFO: Stopping the algorithm
CV calls 1 serial batches 1 parallel batches 0
INFO: Stopping the algorithm
CV calls 2 serial batches 0 parallel batches 1
```

Use `n_jobs=1` for serial execution. A zero worker count raises before CV,
including when the budget allows only one candidate.

```python
invalid_workers = clone(search).set_params(n_jobs=0)
try:
    invalid_workers.fit(X_train, y_train, callbacks=CVBudgetStopping(1))
except ValueError as error:
    print(error)
else:
    raise AssertionError("Expected a zero worker count to fail")
```

```text
n_jobs must be nonzero. Use n_jobs=1 for serial execution.
```

## Stop DEAP with local refinement enabled

If a callback stops DEAP at a generation boundary, the search skips local
refinement. The model can still validate finalists and refit. Here a spread
threshold of 1.0 deliberately triggers on the first pair of generation
scores because balanced accuracy ranges from zero to one.

```python
from sklearn_genetic.callbacks import DeltaThreshold

early = clone(search).set_params(
    optimizer_backend="deap",
    generations=3,
    local_search=True,
    local_search_top_k=2,
    local_search_steps=2,
    final_selection=True,
    final_selection_top_k=2,
)
early.fit(X_train, y_train, callbacks=DeltaThreshold(1.0, generations=2))
print("Reported generations", early.history["gen"])
print("Local refinement candidates", early.fit_stats_["local_refinement_candidates"])
print("Finalist CV calls", early.fit_counts_["finalist_cv_calls"])
print("Refit fits", early.fit_counts_["refit_fits"])
```

```text
INFO: DeltaThreshold callback met its criteria
INFO: Stopping the algorithm
Reported generations [0, 1]
Local refinement candidates 0
Finalist CV calls 2
Refit fits 1
```

## Compare cached evaluations with Nevergrad

In feature selection, Nevergrad's default OnePlusOne strategy starts by
proposing the best initial mask again. With `use_cache=True`, the selector can
reuse the score for that mask. With caching disabled, evaluating a repeated
mask uses a fresh CV call. This example keeps the initial population, folds,
seeds and CV budget the same and reports how much evaluation work was reused.

```python
for use_cache in (False, True):
    nevergrad = clone(selector).set_params(
        optimizer_backend="nevergrad", generations=4, use_cache=use_cache
    )
    nevergrad.fit(X_train, y_train, callbacks=CVBudgetStopping(16))
    sizes = nevergrad.cv_results_["n_features"]
    print("use_cache", use_cache,
          "search CV calls", nevergrad.fit_counts_["search_cv_calls"],
          "recorded candidates", len(sizes),
          "cache hits", nevergrad.fit_stats_["cache_hits"],
          "same-batch duplicates", nevergrad.fit_stats_["duplicate_candidates"],
          "largest mask", max(sizes))
```

```text
use_cache False search CV calls 16 recorded candidates 16 cache hits 0 same-batch duplicates 0 largest mask 8
use_cache True search CV calls 16 recorded candidates 17 cache hits 1 same-batch duplicates 0 largest mask 8
```

The batch decoder repairs empty or oversized masks before evaluation, so
the largest evaluated mask respects the selector's `max_features` setting.
Use `cv_results_["n_features"]` to read the recorded subset sizes.

Use the observed counts when comparing runs. Caching can change how many
proposals fit within a CV budget, and the generation limit still applies.
Each completed proposal has its own row in `cv_results_`. Keep the folds and
estimator randomness fixed when you compare repeated evaluations.
In long histories with frequent cache reuse, the selector can count the
features in a shared mask once and report that count in each matching row.

## Read both cache counters in one batch

A custom optimizer first evaluates mask A, then proposes A, B, B and A in a
second batch. Both occurrences of A in that batch use the previously cached
score. The first B uses a new CV call and the second B shares its result.
The five candidate rows therefore include two fresh calls, two cache hits
and one duplicate coalesced within a batch. Both searches reuse a key for
equivalent proposals in a batch and record every completed proposal in order.
For hyperparameter search, the cache key uses sorted parameter names.
The candidate table preserves the parameter order from the search grid.

```python
mask_a = (1, 0) + (0,) * (X_train.shape[1] - 2)
mask_b = (0, 1) + (0,) * (X_train.shape[1] - 2)

def repeat_masks(problem, evaluate, report):
    report([handle for _, handle in evaluate([mask_a])])
    report([handle for _, handle in evaluate([mask_a, mask_b, mask_b, mask_a])])

cached = clone(selector).set_params(
    optimizer_backend=repeat_masks, population_size=4, generations=1, use_cache=True
).fit(X_train, y_train)

stats = cached.fit_stats_
print("Candidate rows", len(cached.cv_results_["features"]))
print("Search CV calls", cached.fit_counts_["search_cv_calls"])
print("Cache hits", stats["cache_hits"])
print("Same-batch duplicates", stats["duplicate_candidates"])
```

```text
Candidate rows 5
Search CV calls 2
Cache hits 2
Same-batch duplicates 1
```

## Stop when a categorical space is covered

A small imputation search can include floating-point `NaN` as a category.
Here the two equivalent `NaN` entries share a cache key. Warm starts cover
the four distinct configurations, so the engine can stop at generation zero.

```python
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

marker = "simpleimputer__missing_values"
leaf_size = "decisiontreeclassifier__min_samples_leaf"
markers = [float("nan"), -999.0, float("nan")]
imputation_search = GASearchCV(
    make_pipeline(SimpleImputer(), DecisionTreeClassifier(random_state=7)),
    param_grid={marker: Categorical(markers), leaf_size: Integer(1, 2)},
    warm_start_configs=[
        {marker: value, leaf_size: size} for value in markers[:2] for size in (1, 2)
    ],
    optimizer_backend="pymoo", cv=folds, scoring="balanced_accuracy",
    population_size=8, generations=4, use_cache=True,
    random_state=7, n_jobs=1, verbose=False, error_score="raise",
).fit(X_train, y_train)
print("Space exhausted", imputation_search.search_space_exhausted_)
print("Search CV calls", imputation_search.fit_counts_["search_cv_calls"])
print("Generations", imputation_search.history["gen"])
```

```text
Space exhausted True
Search CV calls 4
Generations [0]
```

### Compare a CV limit with complete coverage

Give the same four-configuration space two different CV limits.
A limit of two interrupts the starting population. A limit of ten leaves
room to score the entire space, and the search stops when it has covered
all four configurations. Compare the observed CV calls and the last row's
`partial` flag with `search_space_exhausted_`.

```python
for limit in (2, 10):
    probe = clone(imputation_search).fit(
        X_train, y_train, callbacks=CVBudgetStopping(limit)
    )
    print("CV limit", limit,
          "fresh calls", probe.fit_counts_["search_cv_calls"],
          "partial", probe.history.get("partial", [False])[-1],
          "space exhausted", probe.search_space_exhausted_)
```

```text
CV limit 2 fresh calls 2 partial True space exhausted False
CV limit 10 fresh calls 4 partial False space exhausted True
```

## Find partial generations before and after resume

Here the budget allows eight calls for initialization and four more for the
next generation. The last row in memory describes the partial generation.
pymoo's checkpoint retains the last completed generation for exact replay.

```python
from matplotlib import pyplot as plt
from sklearn_genetic.plots import plot_fitness_evolution

template = clone(selector).set_params(use_cache=False, generations=4)
with TemporaryDirectory() as directory:
    checkpoint = ModelCheckpoint(Path(directory) / "budget.pkl")
    stopped = clone(template).fit(
        X_train, y_train, callbacks=[CVBudgetStopping(12), checkpoint]
    )
    print("Fresh calls before stopping", stopped.fit_counts_["search_cv_calls"])
    print("Calls retained for resume", checkpoint.resumable_cv_calls_)
    print("Calls outside the checkpoint", checkpoint.unsaved_cv_calls_)
    saved_calls = checkpoint.load()["fit_stats"]["cross_validate_calls"]
    resumed = clone(template).fit(
        X_train, y_train, callbacks=[CVBudgetStopping(12), checkpoint]
    )

fresh_calls = resumed.fit_counts_["search_cv_calls"]
cumulative_calls = resumed.fit_stats_["cross_validate_calls"]
print("Fresh calls during resume", fresh_calls)
print("Restored calls", cumulative_calls - fresh_calls, "saved", saved_calls)
print("New calls retained for another resume", checkpoint.resumable_cv_calls_)
print("Calls in the resumed history", resumed.history["cross_validate_calls"][-1])
```

```text
Fresh calls before stopping 12
Calls retained for resume 8
Calls outside the checkpoint 4
Fresh calls during resume 12
Restored calls 8 saved 8
New calls retained for another resume 8
Calls in the resumed history 20
```

```python
fig, axes = plt.subplots(1, 2, figsize=(10, 4.1), sharex=True, sharey=True, layout="constrained")
for ax, run, title in zip(axes, (stopped, resumed), ("Stopped fit", "Resumed fit")):
    plot_fitness_evolution(
        run, metric="fitness_best_observed", x="cv_calls", kind="step", ax=ax, title=title
    )
axes[1].set_ylabel("")
```

![Two fitness charts show hollow diamonds at 12 calls in the stopped fit and 20 calls after resume](/images/optimizer_budget_partial_generations.png)

*Hollow diamonds identify the partial generation in each fit. The resumed history starts from eight saved CV calls.*

Use the diamond to identify a point based on a partial generation. The resumed
history includes the saved calls and the new calls from the second fit.
The checkpoint counters count fresh work in the most recent fit.
Add the two `fit_counts_["search_cv_calls"]` values to obtain the work performed
across both fits, including the work from the first partial generation.
If the callback warns about serialization, the search continues from its
current state. Only successfully written resumable checkpoints count as
retained work.
An existing file with an empty or unrecognized snapshot stops a new fit
before candidate evaluation. Start an independent search with a new path.
For an incomplete resumable optimizer snapshot, the error identifies the
checkpoint path and missing fields.

### Resume with shared Boolean folds

The search and finalist validation can share the same list of Boolean folds.
A cloned estimator can hold separate copies of those arrays. The checkpoint
compares their contents and order, so a resume can reuse the saved search
results with either copy.

```python
masked_template = clone(template).set_params(
    cv=masked_folds, final_selection_cv=masked_folds,
    final_selection="reuse_folds", final_selection_top_k=2, error_score="raise",
)
with TemporaryDirectory() as directory:
    checkpoint = ModelCheckpoint(Path(directory) / "masked-folds.pkl")
    clone(masked_template).fit(
        X_train, y_train, callbacks=[CVBudgetStopping(8), checkpoint]
    )
    saved_calls = checkpoint.load()["fit_stats"]["cross_validate_calls"]
    mask_resume = masked_template.fit(
        X_train, y_train, callbacks=[CVBudgetStopping(8), checkpoint]
    )
print("Saved search CV calls", saved_calls)
print("Fresh search CV calls", mask_resume.fit_counts_["search_cv_calls"])
print("Reused finalist scores", mask_resume.fit_counts_["finalist_reused_folds"])
```

```text
Saved search CV calls 8
Fresh search CV calls 8
Reused finalist scores 6
```

### Resume an exhausted DEAP search

Once a verified DEAP checkpoint holds scores for the entire finite search
space, the search can restore the saved history and proceed with refitting.
This example leaves finalist validation disabled and keeps the score cache.

```python
deap_template = clone(imputation_search).set_params(optimizer_backend="deap")
with TemporaryDirectory() as directory:
    checkpoint = ModelCheckpoint(Path(directory) / "exhausted.pkl")
    first_fit = clone(deap_template).fit(X_train, y_train, callbacks=checkpoint)
    resumed_fit = clone(deap_template).fit(X_train, y_train, callbacks=checkpoint)
print("Search CV calls on resume", resumed_fit.fit_counts_["search_cv_calls"])
print("Generations before and after", first_fit.history["gen"], resumed_fit.history["gen"])
print("Full-data refits on resume", resumed_fit.fit_counts_["refit_fits"])
```

```text
Search CV calls on resume 0
Generations before and after [0] [0]
Full-data refits on resume 1
```

### Evaluate fresh proposals after resuming a covered space

A custom `on_start` callback can disable caching before evaluation starts.
The saved candidates remain available. Coverage of their finite space
allows the resumed engine to stop only while the cache is active.
With caching disabled, the engine can score fresh proposals until another
stopping rule applies.

```python
from sklearn_genetic.callbacks import BaseCallback

class DisableCacheAtStart(BaseCallback):
    def on_start(self, estimator):
        estimator.use_cache = False

covered_template = clone(imputation_search)
with TemporaryDirectory() as directory:
    checkpoint = ModelCheckpoint(Path(directory) / "covered-space.pkl")
    clone(covered_template).fit(X_train, y_train, callbacks=checkpoint)
    fresh_resume = clone(covered_template).fit(
        X_train, y_train,
        callbacks=[DisableCacheAtStart(), CVBudgetStopping(10), checkpoint],
    )
print("Fresh search CV calls", fresh_resume.fit_counts_["search_cv_calls"])
print("Stopped on cached coverage", fresh_resume.search_space_exhausted_)
```

```text
Fresh search CV calls 10
Stopped on cached coverage False
```

## Plot generations with missing counts

Some older checkpoints have generation rows without a cumulative CV count.
`LivePlot(metrics=("cross_validate_calls",))` can follow a resumed search as
soon as at least one generation supplies the count. The generation axis
still includes earlier rows with missing values.

Use `plot_generation_metrics` for the same view of saved rows. Here the
first row omits its count to illustrate a checkpoint written before that
field was recorded.

```python
from sklearn_genetic.plots import plot_generation_metrics

rows = [
    {"gen": generation, "cross_validate_calls": count}
    for generation, count in zip(
        resumed.history["gen"], resumed.history["cross_validate_calls"], strict=True
    )
]
rows[0].pop("cross_validate_calls")
plot_generation_metrics(rows, metrics=("cross_validate_calls",))
```

![Cumulative CV calls across generations 0 to 2, with a gap at generation 0](/images/optimizer_historical_cv_counts.png)

*Generation 0 stays on the axis. The curve starts when the first CV count is available.*

## Configure a solver

The [backend API](../api/optimizer-backends) lists supported settings. The following
examples create configured estimators. Call `fit` as above to run them.

```python
from functools import partial
from importlib.metadata import entry_points

engines = entry_points(group="sklearn_genetic.optimizers")
cma = clone(search).set_params(optimizer_backend=partial(engines["cmaes"].load(), sigma=0.15))
ng = clone(search).set_params(optimizer_backend=partial(engines["nevergrad"].load(), strategy="TwoPointsDE"))
ga = clone(selector).set_params(crossover_probability=0.7)
```

## Run binary feature selection with CatCMAwM

CatCMAwM uses two categories per feature. The adapter builds the starting
probabilities from the best initial masks and decodes each proposal in bulk.
This example spends eight CV calls on initialization and eight on one new
generation. It disables caching to make the observed budget explicit.

```python
cma_selector = clone(selector).set_params(optimizer_backend="cmaes", use_cache=False)
cma_selector.fit(X_train, y_train, callbacks=CVBudgetStopping(16))
print("CV calls by generation", cma_selector.history["cross_validate_calls"])
print("Search fold fits", cma_selector.fit_counts_["search_fold_fits"])
```

```text
CV calls by generation [8, 16]
Search fold fits 48
```

Use the [CatCMAwM reference](../api/algorithms#catcma-with-margin) for supported
coordinates and checkpoint behavior. When timing a wider feature search, keep the
splits, seed and observed number of CV calls fixed.

## Route training and scoring weights to parallel candidates

Enable scikit-learn metadata routing while declaring the requests and calling
`fit`. Each candidate worker inherits that configuration and receives the
weights for its fold. Use separate requests when training and validation need
different weights.

```python
import numpy as np
from sklearn import config_context
from sklearn.metrics import accuracy_score, make_scorer

train_weights = np.where(y_train == 0, 2.0, 1.0)
score_weights = np.where(y_train == 0, 1.5, 1.0)
with config_context(enable_metadata_routing=True):
    weighted_model = clone(model).set_fit_request(sample_weight="train_weights")
    weighted_score = make_scorer(accuracy_score).set_score_request(
        sample_weight="score_weights"
    )
    weighted_search = clone(search).set_params(
        estimator=weighted_model, scoring=weighted_score, n_jobs=2,
        parallel_backend="population", generations=0,
    )
    weighted_search.fit(
        X_train, y_train, train_weights=train_weights, score_weights=score_weights
    )
print("Parallel batches", weighted_search.fit_stats_["population_parallel_batches"])
print("Weighted CV accuracy", round(weighted_search.best_score_, 3))
```

```text
Parallel batches 1
Weighted CV accuracy 0.924
```

See the [cross-validation guide](../guide/understand-cv#separate-training-and-validation-weights)
for the same setup with feature selection and pipelines.

## Route groups in nested cross-validation

Keep related observations together in the outer evaluation, the search and
finalist validation. This example creates simulated groups for the built-in
dataset. For repeated measurements, use the grouping units that match your
evaluation, such as patients for both stages or sites for the outer split and
patients for an inner split. Here the outer splitter requests its usual
`groups`. The alias `inner_groups` provides the labels for the search and
finalist splitters.

```python
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.pipeline import make_pipeline

rows = np.arange(len(y_train))
groups = rows % 9
inner_groups = groups + 9 * (rows // 9 % 2)
with config_context(enable_metadata_routing=True):
    outer_cv = GroupKFold(3)
    grouped_search = clone(search).set_params(
        cv=GroupKFold(2), final_selection=True, final_selection_top_k=2,
        final_selection_cv=GroupKFold(3), population_size=4, generations=0,
    ).set_fit_request(groups="inner_groups")
    nested = cross_validate(
        make_pipeline(grouped_search), X_train, y_train, cv=outer_cv,
        scoring="balanced_accuracy",
        params={"groups": groups, "inner_groups": inner_groups},
        return_estimator=True, error_score="raise",
    )
print("Outer held-out scores", np.round(nested["test_score"], 3).tolist())
print("Finalist fold fits per outer split", [
    pipeline.steps[-1][1].fit_counts_["finalist_fold_fits"]
    for pipeline in nested["estimator"]
])
```

```text
Outer held-out scores [0.863, 0.947, 0.917]
Finalist fold fits per outer split [6, 6, 6]
```

The outer scores evaluate the search and refit procedure on held-out groups.
The finalist scores used to choose a configuration come from the outer
training rows. Both `GASearchCV` and `GAFeatureSelectionCV` accept the group
request shown here.

### Select features inside a grouped pipeline

A feature selector can use `GroupKFold` as its search splitter and pass
the selected columns to the final predictor. With metadata routing enabled,
both the outer splitter and the selector's search splitter request `groups`.
The outer validation passes only its training rows and their group labels to
each pipeline fit.

```python
grouped_selector = GAFeatureSelectionCV(
    model, cv=GroupKFold(2), scoring="balanced_accuracy", refit=False,
    population_size=4, generations=0, max_features=5,
    random_state=7, n_jobs=1, verbose=False,
)
with config_context(enable_metadata_routing=True):
    selected = cross_validate(
        make_pipeline(grouped_selector, model), X_train, y_train,
        cv=GroupKFold(3), scoring="balanced_accuracy",
        params={"groups": groups}, return_estimator=True, error_score="raise",
    )
print("Outer held-out scores", np.round(selected["test_score"], 3).tolist())
print("Search folds per outer split", [
    pipeline.steps[0][1].n_splits_ for pipeline in selected["estimator"]
])
```

```text
Outer held-out scores [0.895, 0.88, 0.917]
Search folds per outer split [2, 2, 2]
```

### Route groups to feature finalists

Set a group-aware `final_selection_cv` when only the finalist stage needs
group labels. With metadata routing enabled, the ordinary `groups`
parameter reaches both the outer splitter and the selector's finalist
splitter. The search uses its own CV policy.

```python
finalist_selector = GAFeatureSelectionCV(
    model, cv=2, scoring="balanced_accuracy",
    final_selection=True, final_selection_cv=GroupKFold(2),
    final_selection_top_k=2, population_size=4, generations=0,
    max_features=5, random_state=7, n_jobs=1, verbose=False,
)
with config_context(enable_metadata_routing=True):
    grouped_finalists = cross_validate(
        make_pipeline(finalist_selector, clone(model)), X_train, y_train,
        cv=GroupKFold(3), scoring="balanced_accuracy", params={"groups": groups},
        return_estimator=True, error_score="raise",
    )
print("Outer held-out scores", np.round(grouped_finalists["test_score"], 3).tolist())
print("Finalist fold fits per outer split", [
    pipeline.steps[0][1].fit_counts_["finalist_fold_fits"]
    for pipeline in grouped_finalists["estimator"]
])
```

```text
Outer held-out scores [0.895, 0.88, 0.917]
Finalist fold fits per outer split [4, 4, 4]
```

### Combine grouped search with a custom finalist splitter

If your custom splitter implements `split(X, y)`, it can supply folds for
one stage while a group-aware splitter receives `groups` in the other.
In this example, the finalist splitter returns group-disjoint folds
prepared from the training partition before fitting.

```python
class PresetFolds:
    def __init__(self, folds):
        self.folds = folds

    def split(self, X, y):
        return iter(self.folds)

finalist_folds = list(GroupKFold(3).split(X_train, y_train, groups=groups))
mixed_search = clone(search).set_params(
    cv=GroupKFold(2), final_selection=True, final_selection_top_k=2,
    final_selection_cv=PresetFolds(finalist_folds),
    population_size=4, generations=0,
).fit(X_train, y_train, groups=groups)
print("Search folds", mixed_search.n_splits_)
print("Finalist fold fits", mixed_search.fit_counts_["finalist_fold_fits"])
```

```text
Search folds 2
Finalist fold fits 6
```

## Inspect candidates that fail all CV folds

This search tries one valid tree depth and one invalid depth. With numeric
`error_score`, the invalid candidate issues a warning and the search can still
return a fitted model. Select the Boolean `all_fits_failed` column to identify
rows whose estimator failed to fit every fold.

```python
import warnings
from sklearn.exceptions import FitFailedWarning

depths = clone(search).set_params(
    param_grid={"max_depth": Categorical([0, 3]), "criterion": Categorical(["gini"])},
    warm_start_configs=[{"max_depth": 0, "criterion": "gini"},
                        {"max_depth": 3, "criterion": "gini"}],
    population_size=2, generations=0, n_jobs=1, error_score=float("nan"),
)
with warnings.catch_warnings(record=True) as recorded:
    warnings.simplefilter("always", FitFailedWarning)
    depths.fit(X_train, y_train)
print("Emitted FitFailedWarning:",
      any(issubclass(item.category, FitFailedWarning) for item in recorded))
rows = depths.cv_results_dataframe(
    columns=["param_max_depth", "mean_test_score", "all_fits_failed"],
    preserve_index=True,
)
print(rows.loc[rows["all_fits_failed"]].reset_index().to_string(index=False))
```

```text
Emitted FitFailedWarning: True
 candidate_index  param_max_depth  mean_test_score  all_fits_failed
               0                0              NaN             True
```

A partial fit failure or a scorer returning `NaN` after fitting leaves
`all_fits_failed` false. Use the split score columns and warnings to investigate
those cases. The flag describes each candidate row, including reused results.

With numeric `error_score`, the search keeps the report even if every
candidate fails. Catch the search error to inspect the parameters and
fold diagnostics. Both failed rows share rank 1 because they tie.

```python
invalid_depths = clone(depths).set_params(
    param_grid={"max_depth": Categorical([0, -1])},
    warm_start_configs=[{"max_depth": 0}, {"max_depth": -1}],
)
with warnings.catch_warnings(record=True) as recorded:
    warnings.simplefilter("always", FitFailedWarning)
    try:
        invalid_depths.fit(X_train, y_train)
    except ValueError as error:
        if not str(error).startswith("All search candidates failed to fit."):
            raise
        print(error)
    else:
        raise AssertionError("Expected every candidate to fail")
print(invalid_depths.cv_results_dataframe(
    columns=["param_max_depth", "all_fits_failed", "rank_test_score"],
    preserve_index=True,
).reset_index().to_string(index=False))
print("Attempted fold fits", invalid_depths.fit_counts_["search_fold_fits"])
```

```text
All search candidates failed to fit. Set error_score='raise' to debug.
 candidate_index  param_max_depth  all_fits_failed  rank_test_score
               0                0             True                1
               1               -1             True                1
Attempted fold fits 6
```

### Inspect an optimizer error

Set `error_score="raise"` to surface the original estimator error. Read
`search_time_` after catching it to see how long the optimizer ran.

```python
fail_fast = clone(depths).set_params(error_score="raise")
try:
    fail_fast.fit(X_train, y_train)
except ValueError as error:
    if "max_depth" not in str(error):
        raise
    print(type(error).__name__)
    print(f"Elapsed search time: {fail_fast.search_time_:.3g} seconds")
else:
    raise AssertionError("Expected the invalid depth to raise")
```

```text
InvalidParameterError
Elapsed search time: 0.00113 seconds
```

## Inspect finalist validation without a winner

This example makes the scorer return `NaN` on the smaller validation
partitions from four-fold finalist CV. The initial three-fold search
still produces finite scores. Catch the validation error to inspect
the recorded finalists and their split scores.

```python
def score_with_minimum_validation_size(estimator, X, y):
    if len(y) < len(y_train) // 3:
        return float("nan")
    return balanced_accuracy_score(y, estimator.predict(X))

failed_finalists = clone(search).set_params(
    scoring=score_with_minimum_validation_size, final_selection=True,
    final_selection_cv=4, final_selection_top_k=2,
    population_size=4, generations=0, error_score="raise",
)
try:
    failed_finalists.fit(X_train, y_train)
except ValueError as error:
    if str(error) != "All final-selection candidates produced non-finite scores":
        raise
    print(error)
else:
    raise AssertionError("Expected the finalist scores to be non-finite")

diagnostics = failed_finalists.final_selection_results_
print("Completed finalists", len(diagnostics["candidates"]))
print("Selected index", diagnostics["selected_index"])
print("Finalist fold fits", failed_finalists.fit_counts_["finalist_fold_fits"])
print("Per-fold scores", diagnostics["candidates"][0]["cv_scores"])
```

```text
All final-selection candidates produced non-finite scores
Completed finalists 2
Selected index None
Finalist fold fits 8
Per-fold scores [nan, nan, nan, nan]
```

```python
from sklearn_genetic.plots import plot_finalist_scores

plot_finalist_scores(failed_finalists)
```

![Search and finalist scores with two unavailable finalist evaluations](/images/optimizer_finalist_validation_without_winner.png)

*Search scores remain visible for both finalists.*

## Diagnose an undefined objective

Here the objective scorer returns `NaN` on every fold and a second scorer still
reports accuracy. The search warns about the objective once and keeps both sets
of CV results. Use the split scores to see where the objective is undefined.

```python
def unavailable(estimator, X, y):
    return float("nan")

report_only = clone(search).set_params(
    scoring={"unavailable": unavailable, "accuracy": "accuracy"},
    objective_metric="unavailable", refit=False,
    population_size=2, generations=0,
)
with warnings.catch_warnings(record=True) as recorded:
    warnings.simplefilter("always", UserWarning)
    report_only.fit(X_train, y_train)
for warning in recorded:
    if str(warning.message).startswith("No candidate has a finite mean CV score"):
        print(warning.message)
print("First candidate accuracy", round(report_only.cv_results_["mean_test_accuracy"][0], 3))
```

```text
No candidate has a finite mean CV score for the objective metric 'unavailable'. Inspect the split scores in cv_results_.
First candidate accuracy 0.911
```

## Configure a combined scorer

A standalone callable returns one number per fold. A callable returning
several metrics raises a configuration error. Supply the scorers by name
to record each metric and choose the objective through `refit`.

```python
def combined_score(estimator, X, y):
    prediction = estimator.predict(X)
    return {"accuracy": accuracy_score(y, prediction),
            "balanced_accuracy": balanced_accuracy_score(y, prediction)}

probe = clone(search).set_params(scoring=combined_score, population_size=4, generations=0)
try:
    probe.fit(X_train, y_train, callbacks=CVBudgetStopping(1))
except ValueError as error:
    print(error)
else:
    raise AssertionError("Expected the combined scorer to raise")

named = clone(probe).set_params(
    scoring={"accuracy": "accuracy", "balanced_accuracy": "balanced_accuracy"},
    refit="balanced_accuracy",
)
named.fit(X_train, y_train, callbacks=CVBudgetStopping(1))
print("Recorded metrics:", list(named.scorer_))
```

```text
A callable passed as scoring must return a single numeric score per fold. Pass a dictionary of named scorers to scoring for multiple metrics.
INFO: Stopping the algorithm
Recorded metrics: ['accuracy', 'balanced_accuracy']
```

## Inspect parameters with names shared by CV diagnostics

A custom estimator can have a constructor parameter named `fit_time`.
`param_fit_time` keeps that parameter, while `mean_fit_time` reports the
measured time across folds. The final refit uses the stored parameter value.
For this single-parameter grid, the search builds both the parameter column
and the candidate dictionaries from the same ordered values.

```python
class TreeWithCustomParameter(DecisionTreeClassifier):
    def __init__(self, fit_time=2):
        super().__init__(random_state=7)
        self.fit_time = fit_time

    def fit(self, X, y, sample_weight=None, check_input=True):
        self.max_depth = self.fit_time
        return super().fit(X, y, sample_weight=sample_weight, check_input=check_input)

custom = GASearchCV(
    TreeWithCustomParameter(), param_grid={"fit_time": Integer(2, 6)},
    cv=folds, population_size=4, generations=0, random_state=7, verbose=False,
).fit(X_train, y_train)
print("Candidate parameters", custom.cv_results_["param_fit_time"])
assert custom.cv_results_["param_fit_time"] == [
    row["fit_time"] for row in custom.cv_results_["params"]
]
print("Refitted parameter", custom.best_estimator_.fit_time)
print("Recorded timing for each candidate",
      len(custom.cv_results_["mean_fit_time"]) == len(custom.cv_results_["params"]))
```

```text
Candidate parameters [2, 3, 4, 6]
Refitted parameter 3
Recorded timing for each candidate True
```

Use the `param_*` columns when plotting a parameter that shares a name with
a diagnostic. Here one column shows the estimator parameter and the other
shows its mean validation score.

```python
from sklearn_genetic.plots import plot_search_space

plot_search_space(
    custom, features=["param_fit_time", "mean_test_score"], kind="heatmap"
)
```

![Correlation heatmap for param_fit_time and the mean test score](/images/optimizer_parameter_diagnostic_heatmap.png)

*The parameter column uses the values supplied to the estimator.*
