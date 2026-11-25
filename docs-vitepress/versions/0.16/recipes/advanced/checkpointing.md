---
title: "Resume a Stopped Hyperparameter Search from a Checkpoint"
description: "Recipe to save GASearchCV state to disk and resume from a checkpoint, preserving all evaluated candidates."
---

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
`fit`. The generation log continues from its last saved entry.

```python
checkpoint = ModelCheckpoint(path)
previous = checkpoint.load()
last_generation = previous["logbook"][-1]["gen"]
previous_calls = previous["runtime_state"]["fit_stats_"]["cross_validate_calls"]

resumed = make_search()
resumed.fit(X_train, y_train, callbacks=[checkpoint])
print("Generation before and after resume", last_generation, resumed.history["gen"][-1])
print("Additional candidate CV calls", resumed.fit_stats_["cross_validate_calls"] - previous_calls)
```

With caching enabled, the resumed search reuses saved candidate scores. Use
the same training rows in the same order, CV splits and scoring rule so those
scores remain comparable. A current checkpoint restores the Python and NumPy
proposal streams and archived elites before evolution continues with a newly
initialized population. Its proposals can differ from a single uninterrupted
run. Warm-start candidates or an exhausted space can still yield cache hits.
Use the additional CV calls above to measure new evaluations. Older checkpoints
load the available state and initialize any missing search state on resume.

Each successful checkpoint replaces the previous snapshot. If serialization
fails, the callback logs a warning and the previous file remains available.
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
print("Additional CV calls",
      resumed.fit_stats_["cross_validate_calls"] - saved["fit_stats"]["cross_validate_calls"])
```

For these four engines, keep the training rows and targets in the same order,
along with the original splits, fit metadata, scorer, model, seed, population
size, generation budget and solver options. Use the same Python environment
and dependency versions. If you configured the adapter with `functools.partial`,
recreate it with the same adapter function and keyword arguments. A stopping
callback starts a fresh lifecycle on the next fit. Leave the previous timer
or stopping condition out if you want to finish the original budget.

Nevergrad and CatCMAwM restore their saved solver state. SciPy and pymoo replay
completed proposals, verify them against the saved record and return stored
evaluations to the solver. This adds solver work on restart while avoiding new
CV fits for the completed generations. A timer or interruption partway through
a generation can require those unfinished proposals to be evaluated again.
When the full generation budget was already saved, the search skips evolution.
The checkpoint covers the evolutionary CV evaluations through the last completed
generation. Optional validation of finalists and the final model refit happen
after that checkpoint and can run again even when you resume a completed search.

Earlier progress-only files with `resumable=False` remain readable with
`checkpoint.load()`. Start a new experiment with its own file. Only load
trusted pickle files. See the [feature selection example](../../tutorials/optimizer-backends#select-features-and-inspect-progress)
for a run with a binary mask, and the [plugin protocol](../../api/algorithms#callbacks-and-checkpoints)
for third-party optimizers.

## See Also

- [Reproducibility and Checkpointing](../../guide/reproducibility) — full guide
- [Checkpointing & Resume Example](../../examples/checkpointing) — annotated example
- [Stop After a Time Budget](./time-budget) — stop cleanly at a time limit, then resume
