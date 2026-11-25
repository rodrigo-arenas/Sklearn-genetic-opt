---
title: "Resume a Scikit-learn Search from a Checkpoint"
description: "Resume a scikit-learn search with ModelCheckpoint. Recover saved CV scores, check partial generations and count new evaluations in GASearchCV or GAFeatureSelectionCV."
---

:::warning Development version
You are reading the **latest (development)** docs. For stable documentation, see [stable](/stable/).
:::

# Resume a Stopped Search from a Checkpoint

**Time:** 8 min | **Difficulty:** Intermediate

## What This Solves

Both `GASearchCV` and `GAFeatureSelectionCV` accept `ModelCheckpoint`. The
first example uses the default DEAP engine. For SciPy, pymoo, Nevergrad
and CatCMAwM, see [Resume an alternative engine](#resume-an-alternative-engine).

Use a generation checkpoint to resume a search with its saved candidate scores
and cumulative evaluation counts. Give each experiment its own checkpoint path.
The first run below assumes the file does not exist.

## Save a time-limited search

Run the following from a directory where you can write files.

```python
from pathlib import Path

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from sklearn_genetic import GASearchCV
from sklearn_genetic.callbacks import ModelCheckpoint, TimerStopping
from sklearn_genetic.space import Integer

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
path = Path("tree-search.pkl")

def make_search():
    return GASearchCV(
        DecisionTreeClassifier(random_state=42),
        param_grid={"max_depth": Integer(2, 20), "min_samples_leaf": Integer(1, 16)},
        scoring="roc_auc", cv=3, population_size=6, generations=2,
        random_state=42, n_jobs=1, verbose=False,
    )

first = make_search()
first.fit(X_train, y_train, callbacks=[ModelCheckpoint(path), TimerStopping(total_seconds=0.01)])
print("Last saved generation", first.history["gen"][-1])
print("Candidate CV calls", first.fit_stats_["cross_validate_calls"])
```

## Resume from the saved file

Recreate the same data, split and search definition if you start a new Python
session. Leave out the short timer to allow more evaluations. Pass a checkpoint callback pointing to the existing file on the next
`fit`. The generation log continues from its last saved entry. DEAP initializes
a new population, reuses scores saved in the cache, and applies the configured
`generations` to the new search phase.

When the saved fit used metadata routing, start the resumed fit inside
`config_context(enable_metadata_routing=True)` and pass the original fit
metadata again. New snapshots record the setting and reject a mismatch
before evaluating candidates. Older snapshots without the field rely on
you to restore the original setting. See the
[weighted resume example](../../examples/checkpointing#resume-a-search-with-metadata-routing).

Use the estimator's `save(path)` and `load(path)` methods to restore the
fitted results of a completed search. Loading into an existing instance
replaces its previous fitted results, including when the saved search used
`refit=False`. Use the `ModelCheckpoint` callback to resume optimization
from a generation boundary.

```python
checkpoint = ModelCheckpoint(path)
previous = checkpoint.load()
last_generation = previous["logbook"][-1]["gen"]

resumed = make_search()
resumed.fit(X_train, y_train, callbacks=[checkpoint])
print("Generation before and after resume", last_generation, resumed.history["gen"][-1])
print("Additional candidate CV calls", resumed.fit_counts_["search_cv_calls"])
```

With caching enabled, the resumed search reuses saved candidate scores. Keep
the scoring rule and the training inputs from the original search. New DEAP
checkpoints record a fingerprint of the ordered rows, targets, groups and fit
parameters when those inputs can be hashed. If they change, `fit` raises before
using the saved scores. Start a new checkpoint file to search on changed data.

New checkpoints with a fingerprint also store the folds produced by an
explicit splitter. Resume uses those folds for fresh candidates. It restores
the Python and NumPy proposal streams and archived elites before starting a
new population. Warm-start candidates or an exhausted space can still yield
cache hits. If a timer expires before DEAP evaluates any candidates on resume,
`search_space_exhausted_` reports whether the saved cache covers the finite
search space. A generation interrupted by a time or CV limit has `partial=True`
in `history` and in a new DEAP checkpoint. DEAP saves the candidate results
available at that boundary. If DEAP runs local refinement at the end of
search, the callback updates that snapshot so the next fit can recover those
candidate scores too. See the
[local refinement example](../../examples/checkpointing#resume-after-local-refinement)
for the saved and fresh CV counts.
New DEAP checkpoints also restore the saved `criteria`, `error_score`,
`return_train_score` and feature selection's `max_features`. This keeps the
evaluation and feature limit consistent across search phases.
For deterministic searches using `final_selection="reuse_folds"`, a DEAP
resume can also reuse matching finalist fold scores once the training
inputs are verified and the saved search used independent fold indices.
Only unmatched folds need fresh fits. Read `fit_counts_["finalist_reused_folds"]`
to see how many fold fits were avoided.
Older checkpoints load the available state. For files without a fingerprint,
verify the training inputs before resuming. Unverified checkpoints validate
all finalist folds afresh.

If a DEAP snapshot lacks the runtime CV counter, `fit` continues the
cumulative count from the last saved generation when that record contains a
valid count. Missing auxiliary counters start at zero for the resumed work.
When the cumulative count is absent from both locations, the resumed search
counts the CV calls it performs. `fit_counts_` reports fresh work in all of
these cases. See the
[counter recovery example](../../examples/checkpointing#recover-a-cumulative-cv-count).

Each successful checkpoint replaces the previous snapshot. If serialization
fails, the callback logs a warning and the previous file remains available.
This includes a `ValueError` raised by a custom serializer or a
`RecursionError` raised while pickling a deeply nested object. Read the
warning to identify the failed save.
If an existing file contains an empty payload or lacks the expected snapshot
structure, `fit` reports the invalid checkpoint before evaluating candidates.
Use a new path to start an independent search.
To keep a fitted estimator for prediction, use the search object's `save`
method after `fit`.

## Resume an alternative engine

To try the previous tree search with SciPy, use its own checkpoint file.
This example stops after the first completed generation report, then refits
an identically configured estimator from that file. The same workflow works
with `optimizer_backend="nevergrad"`, `"cmaes"` and `"pymoo"`.

```python
from sklearn.base import clone
from sklearn_genetic.callbacks import ThresholdStopping

search = make_search().set_params(optimizer_backend="scipy")
checkpoint = ModelCheckpoint("tree-scipy.pkl")
clone(search).fit(X_train, y_train, callbacks=[ThresholdStopping(0.0), checkpoint])
saved = checkpoint.load()
print("Last completed generation", saved["logbook"][-1]["gen"])
print("Ready to resume", saved["resumable"])

resumed = clone(search).fit(X_train, y_train, callbacks=[checkpoint])
print("Additional CV calls", resumed.fit_counts_["search_cv_calls"])
```

:::warning Finish an in-progress search before upgrading
SciPy, pymoo, Nevergrad and CatCMAwM require the same installed version of
`sklearn-genetic-opt` as the one that wrote the checkpoint, even if all other
package versions match. To finish a 0.16.0 run, resume it in its original
0.16.0 environment. To run it on 0.17.0 instead, start a new search with a
new checkpoint path and keep the original file. DEAP uses a different format;
do not assume its checkpoints are portable across arbitrary versions either.
:::

For these four engines, keep the Python environment and dependencies unchanged. Keep the training rows and targets in the same order,
along with the original splits, fit metadata, scorer, model, seed, population
size, generation budget and solver options. If you configured the adapter with `functools.partial`,
recreate it with the same adapter function and keyword arguments. A stopping
callback starts a fresh lifecycle on the next fit. Leave the previous timer
or stopping condition out if you want to finish the original budget.
New fingerprints handle explicit fold selectors supplied as plain lists or
tuples of Python integers and booleans directly. Preserve the selector types
and order when resuming. Checkpoints with older fingerprints can still use
their original fold selectors.
If the file declares a resumable snapshot but lacks required fields, the error
names the missing fields before the search evaluates any candidates. The
required fields include the counters in `fit_stats`, such as cache hits and
worker batches. Every fit counter must be a nonnegative integer. See the
[invalid counter example](../../tutorials/optimizer-backends#diagnose-an-invalid-checkpoint-counter).
When a checkpoint records a cumulative search CV count, `fit` checks that it
is a nonnegative integer. If the last saved generation also records the
count, the two values must agree. A validation error identifies the file
and counter before candidate evaluation.
If an older file overwrote estimator parameters with CV fields of the same
name, start with a new checkpoint path. Current snapshots retain both values.

Nevergrad and CatCMAwM restore their saved solver state. SciPy and pymoo replay
completed proposals, verify them against the saved record and return stored
evaluations to the solver. This adds solver work on restart while avoiding new
CV fits for the completed generations. When saving and replaying integer
feature masks, both engines compute signatures with the existing checkpoint
format directly.
A timer or interruption partway through a generation can require those
unfinished proposals to be evaluated again.
When a partial generation follows a completed one, the in-memory history
includes the partial row and the file keeps the completed boundary. A stop
in the first generation can leave an inspection-only snapshot.
Use [fitness evolution on the CV axis](../../tutorials/optimizer-backends#find-partial-generations-before-and-after-resume)
to see which plotted rows are partial before and after resume.
After each fit, read the callback's `resumable_cv_calls_` and
`unsaved_cv_calls_` to see how much fresh search work can be recovered from
its file. These counters exclude work loaded from a previous fit.
Compare `fit_counts_["search_cv_calls"]` with
`fit_stats_["cross_validate_calls"]` to see fresh calls and the cumulative
count after resume. Their difference is the number restored from the
checkpoint. Reusing a score from the cache uses zero fresh CV calls.
When the full generation budget was already saved, the search skips evolution.
The checkpoint covers the evolutionary CV evaluations through the last completed
generation. Optional validation of finalists and the final model refit happen
after that checkpoint and can run again even when you resume a completed search.

Earlier progress-only files with `resumable=False` remain readable with
`checkpoint.load()`. Start a new experiment with its own file. Only load
trusted pickle files. See the [feature selection example](../../tutorials/optimizer-backends#select-features-and-inspect-progress)
for a run with a binary mask, and the [plugin protocol](../../api/algorithms#callbacks-and-checkpoints)
for third-party optimizers.

## Reduce checkpoint writes

Specify `every_n_generations` when writing a full snapshot after each
generation takes a meaningful share of the fit. In this example, the search
reports generations 0 through 3 and saves generations 0 and 2. Use a fresh
checkpoint path and the SciPy optimizer extra.

```python
from sklearn_genetic.backends import scipy_optimizer

search = make_search().set_params(
    optimizer_backend=scipy_optimizer(stop_on_convergence=False),
    generations=3,
    use_cache=False,
)
checkpoint = ModelCheckpoint("tree-scipy-interval.pkl", every_n_generations=2)
search.fit(X_train, y_train, callbacks=checkpoint)
print("Last generation in memory", search.history["gen"][-1])
print("Last saved generation", checkpoint.load()["logbook"][-1]["gen"])
print("Fresh CV calls to repeat on resume", checkpoint.unsaved_cv_calls_)
```

The next fit resumes from generation 2. Choose the interval based on how
much work you can afford to lose between saves. You can change the interval
on resume. The schedule uses the same generation indices shown in `history`.
See the [budget and resume tutorial](../../tutorials/budgeted-optimizer-search)
for an example with two compute sessions and a partial generation.

## Diagnose a resume error

| Error names | What to check |
| --- | --- |
| `logbook` and `generation boundary` | The saved generation numbers must be consecutive, with valid candidate and population counts. |
| `candidate_records` and `generation boundary` | The number of saved candidate rows must match the sum of the evaluation counts in the logbook. |
| `solver_state` and `generation boundary` | The replay journal or native solver state must match the last completed generation. |
| `solver_state` and `unreadable data` | The native solver payload could not be decoded in the active environment. |
| `telemetry` or a `telemetry.*` field | The saved best score must be a real number or `None`, and the best generation and stagnation counts must be nonnegative integers. |
| `context` or `problem` | The checkpoint needs data and configuration fingerprints and the saved definition of a parameter or feature search. |
| `cv_splits`, `search_cv_splits`, or `outlier_cv_splits` | The file needs materialized train/test pairs. Row indices must be in bounds and Boolean masks must match the training row count. |

`fit` raises before starting fresh CV and preserves the existing file. The
telemetry and configuration checks also apply to resumable third-party optimizers. Resume from an
earlier checkpoint you trust, or start an independent search with a new path.

## See Also

- [Reproducibility and Checkpointing](../../guide/reproducibility) — full guide
- [Finalist validation and fold reuse](../../guide/advanced-optimizer-control#cheaper-search-with-full-training-finalist-validation)
- [Checkpointing & Resume Example](../../examples/checkpointing) — annotated example
- [Stop After a Time Budget](./time-budget) — stop cleanly at a time limit, then resume
