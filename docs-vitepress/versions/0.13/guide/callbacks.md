---
title: Callbacks
description: Add early stopping, progress reporting, checkpoints, TensorBoard, and MLflow logging to your GASearchCV runs.
---

# Callbacks

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

Stops if the metric fails to improve for `N` consecutive generations.

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

Stops once the metric reaches or exceeds a target value.

```python
from sklearn_genetic.callbacks import ThresholdStopping

callback = ThresholdStopping(threshold=0.98, metric="fitness_max")
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `threshold` | float | — | Stop when the metric ≥ this value |
| `metric` | str | `"fitness"` | History column to watch |

---

### DeltaThreshold

Stops when the spread (max − min) across the last `N` generations falls below a threshold — useful for detecting convergence.

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

### TimerStopping

Stops after a wall-clock time limit.

```python
from sklearn_genetic.callbacks import TimerStopping

callback = TimerStopping(total_seconds=300)  # stop after 5 minutes
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `total_seconds` | float | Maximum elapsed seconds. Checked after each generation, so the current generation completes before stopping |

---

### LogbookSaver

Saves the generation logbook to a file after each generation. Useful for long runs — if the search is interrupted, you can inspect progress so far.

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

Saves the full estimator state (logbook + fitted model) to a pickle file after each generation.

```python
from sklearn_genetic.callbacks import ModelCheckpoint

callback = ModelCheckpoint(checkpoint_path="./checkpoint.pkl")
```

Restore with pickle:

```python
import pickle

with open("./checkpoint.pkl", "rb") as f:
    checkpoint = pickle.load(f)

estimator_state = checkpoint["estimator_state"]
logbook = checkpoint["logbook"]
```

---

### TensorBoard

Logs per-generation fitness metrics to TensorBoard.

```python
import tensorflow as tf
from sklearn_genetic.callbacks import TensorBoard

writer = tf.summary.create_file_writer("./logs")
callback = TensorBoard(writer=writer)
```

Requires TensorFlow: `pip install sklearn-genetic-opt[all]`.

View results with:

```bash
tensorboard --logdir ./logs
```

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
- `TimerStopping` stops after the **next** generation completes once the time limit is exceeded — it won't interrupt a running generation.
- When multiple callbacks are combined, **any single one** returning `True` stops the search.

## Next Steps

- [Custom Callbacks](./custom-callback) — write your own callback
- [MLflow Integration](./mlflow) — log experiments using `MLflowConfig`
- [Advanced Optimizer Control](./advanced-optimizer-control) — tune diversity and convergence behavior
