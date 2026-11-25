---
title: Early Stopping and Callbacks for Hyperparameter Search
description: Stop scikit-learn hyperparameter search at a score or loss threshold, detect convergence, track progress and save checkpoints with callbacks.
---

:::warning Development version
You are reading the **latest (dev)** docs. For the stable version, see [stable](/stable/).
:::

**Estimated reading time:** 12 minutes  
**Difficulty:** Intermediate  
**Prerequisites:** [Getting Started with GASearchCV](./basic-usage)

# Early Stopping and Callbacks for Hyperparameter Search

Callbacks hook into the search loop and run after each generation. They can stop the search early, log progress, save checkpoints, or trigger any custom action.

Callbacks are evaluated in three phases:

1. **`on_start`** — once before generation 0
2. **`on_step`** — after every generation. Return `True` to stop the search
3. **`on_end`** — once after the final generation (or after a stop)

## Prerequisites

- Completed [Basic Usage](./basic-usage)

## Using Callbacks

Pass a single callback or a list to `fit`:

```python
from sklearn_genetic.callbacks import ConsecutiveStopping, ProgressBar

search.fit(X_train, y_train, callbacks=[
    ProgressBar(),
    ConsecutiveStopping(generations=5, metric="fitness_best"),
])
```

If any callback signals to stop, the search ends at the end of that generation.
DEAP skips local refinement when this happens. Configured finalist validation
and refit still run after `on_end`. See the
[DEAP stopping example](../tutorials/optimizer-backends#stop-deap-with-local-refinement-enabled).

## Built-in Callbacks

### ProgressBar

Displays a tqdm progress bar showing generation progress.

```python
from sklearn_genetic.callbacks import ProgressBar

callback = ProgressBar()
```

No parameters required. The bar length equals the configured number of generations.

---

### ConsecutiveStopping

Compares the current value with each of the previous `N` values and stops if
it improves on none of them. Score metrics follow the search criterion. For
`fitness_std`, larger values count as improvement. The callback waits for a
comparison window with finite scores, including the current generation.

```python
from sklearn_genetic.callbacks import ConsecutiveStopping

callback = ConsecutiveStopping(generations=5, metric="fitness")
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `generations` | int | — | Stop if the metric has not improved in this many consecutive generations |
| `metric` | str | `"fitness"` | History column to track: `"fitness"`, `"fitness_best"`, `"fitness_max"`, `"fitness_min"`, `"fitness_std"` |

---

### ThresholdStopping

Stops once a finite metric reaches the target in the search direction.
For `fitness_std`, it stops when dispersion reaches or exceeds the target.

```python
from sklearn_genetic.callbacks import ThresholdStopping

callback = ThresholdStopping(threshold=0.98, metric="fitness_max")
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `threshold` | float | — | Stop when the score is at least this value with `criteria="max"` or at most this value with `criteria="min"` |
| `metric` | str | `"fitness"` | History column to watch |

#### Stop when a loss reaches its target

Use a positive loss with `criteria="min"` and `metric="fitness_best_observed"`
to stop when any candidate's finite CV loss reaches or falls below your target.
This includes discarded offspring and repeated evaluations. With a negated loss such as
`scoring="neg_mean_absolute_error"`, use `criteria="max"` and negate the
target too. After stopping, the configured finalist validation and refit can
still run. The returned `best_score_` reflects the selected candidate and may
differ from the observed score that triggered stopping.

```python
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer, mean_absolute_error

from sklearn_genetic import GASearchCV
from sklearn_genetic.callbacks import ThresholdStopping
from sklearn_genetic.space import Continuous

X, y = load_diabetes(return_X_y=True)
search = GASearchCV(
    Ridge(),
    cv=3,
    param_grid={"alpha": Continuous(0.001, 10.0, distribution="log-uniform")},
    scoring=make_scorer(mean_absolute_error),
    criteria="min",
    population_size=6,
    generations=4,
    random_state=42,
    verbose=False,
)
search.fit(X, y, callbacks=ThresholdStopping(47.0, metric="fitness_best_observed"))
```

---

### DeltaThreshold

Stops when the spread (max − min) across the last `N` generations is at
most the threshold. A missing or non-finite value blocks the comparison
until it leaves the window.

```python
from sklearn_genetic.callbacks import DeltaThreshold

callback = DeltaThreshold(threshold=0.001, generations=5, metric="fitness_min")
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `threshold` | float | — | Stop when `max(metric[-N:]) - min(metric[-N:]) ≤ threshold` |
| `generations` | int | `2` | Window size for the comparison |
| `metric` | str | `"fitness"` | History column to watch |

---

### CVBudgetStopping

Use `CVBudgetStopping` to cap new candidate cross-validation calls across
optimizers. One call evaluates a candidate on all search folds. With caching
enabled, a repeated candidate can reuse its score without consuming a call.

```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

from sklearn_genetic import GASearchCV
from sklearn_genetic.callbacks import CVBudgetStopping
from sklearn_genetic.space import Integer

X, y = load_iris(return_X_y=True)
search = GASearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid={"max_depth": Integer(1, 8), "min_samples_leaf": Integer(1, 8)},
    scoring="accuracy",
    cv=3,
    population_size=6,
    generations=10,
    random_state=42,
    n_jobs=1,
    refit=False,
    verbose=False,
)
search.fit(X, y, callbacks=CVBudgetStopping(max_cv_calls=12))
print(search.fit_counts_["search_cv_calls"])
print(search.best_params_, search.best_score_)
```

The limit covers candidate evaluations during search, including DEAP local
refinement. Finalist validation and refit can run afterward when enabled.
When a population batch starts with one call left, candidate-level parallelism is skipped.
The [optimizer tutorial](../tutorials/optimizer-backends#check-worker-use-at-a-small-cv-budget)
shows how to check this in `fit_stats_`.
An optimizer can finish with fewer calls after covering its space or reaching
another stopping condition.

On resume, the limit applies to new CV calls in the current fit. The bundled
alternative engines resume from completed generations. A budget that cuts a
generation short leaves an earlier completed checkpoint available for resume.
Candidates from an interrupted generation may be evaluated again on the next
fit. A file containing only an incomplete first generation records progress.

---

### TimerStopping

Stops after a wall-clock time limit.

```python
from sklearn_genetic.callbacks import TimerStopping

callback = TimerStopping(total_seconds=300)  # stop after 5 minutes
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `total_seconds` | float | Maximum elapsed seconds before starting another candidate batch; an in-flight batch can finish after the limit |

Pass it to `fit` to cap a long search by wall-clock time — handy for notebook and local experiments where the search space may be larger than expected:

```python
from sklearn_genetic.callbacks import TimerStopping

callbacks = [TimerStopping(total_seconds=60)]

search.fit(X_train, y_train, callbacks=callbacks)
```

The search checks the budget between candidate batches, including batches
within a generation. Final selection and refit can extend the total fit time.

---

### LogbookSaver

Writes per-candidate evaluation records with joblib after each generation.
For a regular file, the callback also saves new rows from DEAP local
refinement, including rows that reuse cached scores. See the
[local refinement example](../examples/checkpointing#resume-after-local-refinement).
If writing a regular file fails, its last successful version stays readable
and the search continues. The error log identifies the affected path and
the cause. Use `ModelCheckpoint` for resumable search snapshots.

```python
from sklearn_genetic.callbacks import LogbookSaver

callback = LogbookSaver(checkpoint_path="./logbook.pkl")
```

Restore with joblib:

```python
from joblib import load

logbook = load("./logbook.pkl")
print(logbook)
```

---

### ModelCheckpoint

Saves a snapshot at a reported generation. DEAP stores its historical continuation state.
SciPy, pymoo, Nevergrad and CatCMAwM resume from a completed generation when
you pass the same checkpoint path to `fit` again with the same data, settings
and environment. Older shared-runtime snapshots with `resumable=False` remain
readable for inspection. See the [resume recipe](../recipes/advanced/checkpointing)
and the [optional plugin protocol](../api/algorithms#callbacks-and-checkpoints).
Use a fresh path for an independent run and load only trusted snapshots.

```python
from sklearn_genetic.callbacks import ModelCheckpoint

callback = ModelCheckpoint(checkpoint_path="./checkpoint.pkl")
```

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

---

### Live training plot

Use the plotting extra to watch best and average fitness as each generation finishes.
The same callback works for `GASearchCV` and `GAFeatureSelectionCV`.

```python
from sklearn_genetic.plots import LivePlot

callback = LivePlot(metrics=("fitness_best_observed", "fitness"), save_path="./training-progress.png")
search.fit(X, y, callbacks=callback)
```

Install with `pip install 'sklearn-genetic-opt[plot]'`. The initial population is generation 0. Each completed generation updates the same lines, including the last one before early stopping. A resumed search includes the recorded generations from its checkpoint. Local refinement can update the final point during `on_end`.

When all selected fields are fitness metrics, very large or very small
scores share a power-of-two divisor shown on the vertical axis. A later
generation can update that scale for the entire chart. The logbook and
fitted history retain the original scores.

The example uses `fitness_best_observed` for the best finite primary score among all candidates evaluated through the generation, including discarded offspring and repeated evaluations. The default `fitness_best` follows the elite archive, which can retain an older score for an identical configuration. Older checkpoints show a gap where the observed series was never recorded. On resume, the new series uses the candidate records available in the checkpoint and new evaluations. The dashed line (`fitness`) is the current population mean. Use `fitness_min` or `fitness_max` for the current population extremes. A negative sklearn scorer keeps its sign. A multi-metric search tracks its refit objective. Fitness may include search penalties.

- **Desktop**. Select a GUI backend such as QtAgg or TkAgg before importing pyplot. The callback processes window events between generations. A long CV evaluation can leave the window unchanged until that generation completes.
- **Notebook**. With ipympl installed, run `%matplotlib widget` and display the target canvas in a preceding cell, then pass its axis to `LivePlot(ax=ax)`. Updates depend on the kernel and frontend processing widget messages. With `%matplotlib inline`, the callback replaces a static snapshot through a Jupyter display handle at each update. Frontends may buffer these messages.
- **Agg / CI**. Pass `save_path="training-progress.png"` to overwrite a snapshot on each update. Rendering and file writes take time on the fitting thread. Close the figure with `plt.close(callback.ax.figure)` when finished.

To monitor diversity in its own units, pass a second callback with `LivePlot(metrics=("unique_individual_ratio", "genotype_diversity"))`. Both fields are ratios computed from the current population. A custom callback can call `plot_generation_metrics(logbook, metrics=..., ax=...)` for a single snapshot. Candidate rankings, fold distributions, parameter landscapes and the selected feature mask use completed fit results.

The legacy `TensorBoard` callback remains importable for existing users who
install TensorFlow separately. TensorFlow is no longer included in any extra.

## Full Example

This example uses `ThresholdStopping` and `DeltaThreshold` together — it stops when accuracy exceeds 0.98 OR when the improvement between consecutive generations drops below 0.001:

```python
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier

from sklearn_genetic import EvolutionConfig, GASearchCV, PopulationConfig, RuntimeConfig
from sklearn_genetic.callbacks import DeltaThreshold, ThresholdStopping
from sklearn_genetic.space import Categorical, Continuous, Integer

data = load_digits()
X, y = data["data"], data["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

param_grid = {
    "min_weight_fraction_leaf": Continuous(0, 0.5),
    "criterion": Categorical(["gini", "entropy"]),
    "max_depth": Integer(2, 20),
    "max_leaf_nodes": Integer(2, 30),
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

evolved_estimator = GASearchCV(
    estimator=DecisionTreeClassifier(),
    cv=cv,
    scoring="accuracy",
    param_grid=param_grid,
    evolution_config=EvolutionConfig(
        population_size=16,
        generations=30,
        tournament_size=3,
        elitism=True,
        crossover_probability=0.9,
        mutation_probability=0.05,
    ),
    population_config=PopulationConfig(initializer="smart"),
    runtime_config=RuntimeConfig(n_jobs=-1, verbose=True),
)

evolved_estimator.fit(X_train, y_train, callbacks=[
    ThresholdStopping(threshold=0.98, metric="fitness_max"),
    DeltaThreshold(threshold=0.001, metric="fitness"),
])

y_predict_ga = evolved_estimator.predict(X_test)
print(evolved_estimator.best_params_)
print("Accuracy:", accuracy_score(y_test, y_predict_ga))
```

## Tips & Gotchas

- Callbacks see the same metric names that appear in the generation log: `fitness`, `fitness_std`, `fitness_best`, `fitness_max`, `fitness_min`.
- `ConsecutiveStopping` watches `fitness` by default (mean population fitness). Use `metric="fitness_best"` if you want to stop when the overall best score stops improving.
- `TimerStopping` stops scheduling new batches after the limit; an in-flight CV batch and final selection/refit can still complete.
- When multiple callbacks are combined, **any single one** returning `True` stops the search.

## See Also

- [Writing Custom Callbacks](./custom-callback) — build your own stopping logic
- [Reproducibility & Checkpointing](./reproducibility) — use LogbookSaver and ModelCheckpoint
- [Cross-Validation in Hyperparameter Search](./understand-cv) — understand the metrics callbacks watch
- [Advanced Optimizer Control](./advanced-optimizer-control) — diversity control and stagnation recovery
- [API: Callbacks](../api/callbacks) — full callback reference
