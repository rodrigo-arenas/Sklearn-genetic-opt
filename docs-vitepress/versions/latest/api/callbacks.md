---
title: Callbacks API
description: API reference for stopping, progress, checkpoint, and legacy TensorBoard callbacks.
---

:::warning Development version
You are reading the **latest (dev)** docs. For the stable version, see [stable](/stable/).
:::

# Callbacks API

```python
from sklearn_genetic.callbacks import (
    ThresholdStopping,
    ConsecutiveStopping,
    DeltaThreshold,
    CVBudgetStopping,
    TimerStopping,
    ProgressBar,
    LogbookSaver,
    ModelCheckpoint,
    TensorBoard,
    BaseCallback,
)
```

Callbacks are passed to `fit(X, y, callbacks=[...])`. After each generation, `on_step` is called on every callback. If any callback returns `True` from `on_step`, the search stops at the end of that generation.

With DEAP, this also skips local refinement. After `on_end`, configured
finalist validation and the full-data refit can still run. See the
[DEAP stopping example](../tutorials/optimizer-backends#stop-deap-with-local-refinement-enabled).

Score-based stopping callbacks accept `"fitness"`, `"fitness_std"`,
`"fitness_best"`, `"fitness_best_observed"`, `"fitness_max"`, and
`"fitness_min"`. Use `"fitness_best_observed"` to include all finite candidate
scores seen so far, including discarded offspring and repeated evaluations.
A missing value in older history delays the decision until the required
scores are available.

`ThresholdStopping`, `DeltaThreshold`, and `TimerStopping` raise `ValueError`
if you configure a threshold or time limit with `NaN`. Supply a defined value
when a configuration file or calculation produces a missing value.

`ConsecutiveStopping` and `DeltaThreshold` accept positive Python and NumPy
integers for the `generations` window, including unsigned NumPy scalars.

## BaseCallback

Base class for custom callbacks. Subclass and override the methods you need.

```python
class BaseCallback:
    def on_start(self, estimator=None): ...      # called before generation 0
    def on_step(self, record=None, logbook=None, estimator=None) -> bool: ...  # True = stop
    def on_end(self, logbook=None, estimator=None): ...  # called after the last generation
```

`on_step` returning `True` stops the search. `False` (or `None`) continues.

## ThresholdStopping

Stops when a finite score reaches the target in the search direction.
For `fitness_std`, the dispersion must reach or exceed the target.

```python
ThresholdStopping(threshold, metric="fitness")
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `threshold` | float | — | Stop when `metric ≥ threshold` for `criteria="max"` or `fitness_std`, and when `metric ≤ threshold` for other metrics with `criteria="min"` |
| `metric` | str | `"fitness"` | History column to watch |

To stop when the best accuracy in the population reaches 0.98, use
`fitness_max` in a maximizing search.

```python
ThresholdStopping(threshold=0.98, metric="fitness_max")
```

## ConsecutiveStopping

Compares the current value with each of the previous `N` values and stops if
it improves on none of them. Score metrics follow the search criterion. For
`fitness_std`, larger values count as improvement. The current value and
all `N` previous values must be finite.

```python
ConsecutiveStopping(generations, metric="fitness")
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `generations` | int | — | Number of non-improving consecutive generations before stopping |
| `metric` | str | `"fitness"` | History column to watch |

**Example:** stop if `fitness_best` hasn't improved in 8 generations:

```python
ConsecutiveStopping(generations=8, metric="fitness_best")
```

## DeltaThreshold

Stops when the spread (max − min) across the last `N` generations is at
most the threshold. Each value in the window must be finite.
If the spread exceeds the score dtype's range, it can still be compared
with the threshold.

```python
DeltaThreshold(threshold, generations=2, metric="fitness")
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `threshold` | float | — | Stop when `max(metric[-N:]) - min(metric[-N:]) ≤ threshold` |
| `generations` | int | `2` | Window size |
| `metric` | str | `"fitness"` | History column to watch |

## CVBudgetStopping

Sets a hard limit on fresh candidate cross-validation calls in one fit.
The search reserves calls before dispatching work in parallel and can stop
partway through a generation. Each candidate scored over its folds uses one
call. Cache hits and restored checkpoint results use no new calls.
With a CV call budget and no timer, the evaluator stops preparing cache
keys at the first proposal that would need an additional call. Cache
hits before that proposal are still recorded in order.
When only one call remains at the start of a population batch, evaluation uses the
calling thread for that candidate. With `parallel_backend="cv"`, the
candidate's folds can still run in parallel.

```python
CVBudgetStopping(max_cv_calls=24)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `max_cv_calls` | int | Positive maximum number of fresh search CV calls |

Read `fit_counts_["search_cv_calls"]` for the calls used during the current
fit. Finalist validation and full-data refitting have separate counts.
If the cap interrupts a generation, its history row has `partial=True`.
The in-memory `cv_results_` includes the candidates completed before the stop.
See [the optimizer tutorial](../tutorials/optimizer-backends#compare-feature-selection-at-the-same-cv-budget)
for a search whose initial population exceeds the number of available calls.
See the [search budget example](../guide/callbacks#cvbudgetstopping).
Before dispatching candidates, the search uses the smallest positive limit
configured by the budget callbacks after `on_start`. See
[how to choose a budget at startup](../tutorials/optimizer-backends#choose-a-cv-budget-at-startup).

## TimerStopping

Stops starting new candidate batches after the elapsed time limit. A monotonic
clock measures search time across system clock adjustments. An in-flight
cross-validation batch, final selection and refit can finish after the limit.
A search without usable checkpoint results completes at least one initial batch.
DEAP can return saved results when the deadline has already expired at resume.
When the search records an interrupted generation, it marks it `partial=True`
in `history`. DEAP leaves the previous history intact if the deadline prevents
every evaluation in the next batch. When the last evaluation batch finishes as
the deadline is reached, the search records the complete generation and its
survivors.

```python
TimerStopping(total_seconds)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `total_seconds` | float | Stop after this many elapsed seconds |

## ProgressBar

Displays a tqdm progress bar. No parameters required.

```python
ProgressBar()
```

## LogbookSaver

Writes the per-candidate evaluation records with joblib after each generation.
For regular file paths, it also saves new rows from DEAP local refinement.
The records cover candidate search. Finalist validation and refit happen
after the callback finishes.
Each record's `index` gives its zero-based position in `cv_results_`. Rows
that share a cached score have their own positions. Pass
`preserve_index=True` to `cv_results_dataframe()` to use these positions as
labels in a ranked table. When reading a file written by an older version,
use `enumerate(records)` to recover the positions.

For a regular file path, it replaces the previous file after the new write
succeeds. If the write or replacement fails, the callback logs the path and
error and keeps the previous file available. Use `ModelCheckpoint` for search
snapshots that can be resumed.

```python
LogbookSaver(checkpoint_path)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `checkpoint_path` | str, path-like object, or binary stream | Destination for the logbook. A regular file path needs a writable parent directory for atomic replacement. Writes to a stream follow joblib's stream behavior. |
| `**dump_options` | keyword arguments | Options passed to joblib, including compression and pickle protocol. Filename-based compression is preserved. |

Restore with:

```python
from joblib import load
logbook = load("./logbook.pkl")
```

## ModelCheckpoint

Saves a snapshot at a reported generation. DEAP retains candidate scores from
partial rows and starts a new population on resume. When DEAP finishes local
refinement, the callback updates the final snapshot with those scores and the
latest CV count. Resuming a checkpoint restores candidate row positions in
the logbook. SciPy, pymoo, Nevergrad
and CatCMAwM use the last completed generation for exact continuation when
you pass the same checkpoint path to `fit` again with the same data, settings
and environment. Older shared-runtime snapshots with `resumable=False` remain
readable for inspection. See the [resume recipe](../recipes/advanced/checkpointing)
and the [optional plugin protocol](../api/algorithms#callbacks-and-checkpoints).
Use a fresh path for an independent run and load only trusted snapshots.
If the first existing checkpoint contains an empty payload or an unrecognized
snapshot, `fit` raises an error before candidate evaluation and preserves the file.
For an incomplete optimizer snapshot that claims to support resume, the error
lists the missing fields and the checkpoint path. Its `fit_stats` counters
must all be present and contain nonnegative integers.
For a DEAP snapshot with a missing runtime CV counter, `fit` can recover the
cumulative count from the last saved generation. The callback's counters
still report fresh work from the resumed fit.
New snapshots save scikit-learn's `enable_metadata_routing` setting. Restore
that setting before calling `fit` to resume, so the saved and new evaluations
use the same metadata requests. Older snapshots without the setting skip this
check. `load()` can inspect either snapshot regardless of the active setting.

```python
ModelCheckpoint(checkpoint_path, *, every_n_generations=1)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `checkpoint_path` | str or Path | File path for the checkpoint |
| `every_n_generations` | positive int, default `1` | Schedule writes at generation indices `0, N, 2*N, …`, including after resume. |

Larger intervals reduce serialization and disk writes. A fit can finish
between scheduled writes, leaving the latest generations available only on
the fitted estimator. The existing snapshot rules for partial generations
apply at scheduled indices. DEAP also saves after completed local refinement.
See [how to reduce checkpoint writes](../recipes/advanced/checkpointing#reduce-checkpoint-writes).

After fitting, read these counters on the callback instance you passed to `fit`.

| Attribute | What it counts |
|-----------|---------------|
| `resumable_cv_calls_` | Fresh search CV calls from the latest fit retained in the last successfully written resumable snapshot |
| `unsaved_cv_calls_` | Fresh search CV calls from the latest fit outside that resumable snapshot |

Both counters reset at the start of each fit and sum to
`fit_counts_["search_cv_calls"]` on the fitted estimator. Earlier calls loaded
from a checkpoint stay outside these counts. A progress-only snapshot retains
no calls for resume. If a write fails, the counters use the last successful
write from that fit. Pickle errors, including `ValueError` from a custom
serializer and `RecursionError` for deeply nested data, produce a warning.
The search continues and keeps the previous file. See the
[example before and after resume](../tutorials/optimizer-backends#find-partial-generations-before-and-after-resume).

Inspect a saved snapshot from either format:

```python
from sklearn_genetic.callbacks import ModelCheckpoint

snapshot = ModelCheckpoint("./checkpoint.pkl").load()
print(len(snapshot["logbook"]))  # generation records
if "optimizer_checkpoint_version" in snapshot:
    print(snapshot["backend"], snapshot["resumable"], len(snapshot["candidate_records"]))
else:
    print(snapshot["estimator_state"]["generations"])  # historical DEAP format
```

## Legacy TensorBoard callback

`TensorBoard` remains available for existing integrations. It writes the five
generation fitness metrics as TensorFlow scalars and requires a separate
TensorFlow installation. Importing `sklearn_genetic.callbacks` does not import
TensorFlow.

```python
TensorBoard(log_dir="./logs", run_id=None)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `log_dir` | str | Parent directory for summary files |
| `run_id` | str or None | Subdirectory; defaults to a timestamp |

For current workflows, use [`LivePlot`](./plots) and
[`plot_search_overview`](./plots) with `sklearn-genetic-opt[plot]` instead.

## See Also

- [Callbacks Guide](../guide/callbacks) — usage examples and combining callbacks
- [Custom Callbacks](../guide/custom-callback) — writing your own callback
