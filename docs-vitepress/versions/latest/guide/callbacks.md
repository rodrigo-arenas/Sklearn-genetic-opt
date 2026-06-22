---
title: Callbacks
description: Add early stopping, progress reporting, checkpoints, TensorBoard, and MLflow logging to your GASearchCV runs.
---

:::warning Development version
You are reading the **latest (dev)** docs. For the stable version, see [0.13](/versions/0.13/).
:::

# Callbacks

Callbacks let you hook into the evolutionary search loop after each generation. They can stop the search early, log progress, save checkpoints, and more.

## Prerequisites

- Completed [Basic Usage](./basic-usage)

## Using Callbacks

Pass a list of callbacks to `fit`:

```python
from sklearn_genetic.callbacks import ConsecutiveStopping, ProgressBar

search.fit(X_train, y_train, callbacks=[
    ConsecutiveStopping(generations=5, metric="fitness_best"),
    ProgressBar(),
])
```

## Built-in Callbacks

### ConsecutiveStopping

Stops the search if the chosen metric does not improve for `N` consecutive generations:

```python
from sklearn_genetic.callbacks import ConsecutiveStopping

# Stop if fitness_best hasn't improved for 8 generations
callback = ConsecutiveStopping(generations=8, metric="fitness_best")
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `generations` | int | — | Number of consecutive generations without improvement |
| `metric` | str | `"fitness_best"` | History column to watch |

### DeltaThreshold

Stops when the improvement between generations falls below a threshold:

```python
from sklearn_genetic.callbacks import DeltaThreshold

callback = DeltaThreshold(threshold=0.001, metric="fitness")
```

### TimerStopping

Stops after a wall-clock time limit:

```python
from sklearn_genetic.callbacks import TimerStopping

callback = TimerStopping(total_seconds=300)  # stop after 5 minutes
```

### ProgressBar

Shows a tqdm progress bar:

```python
from sklearn_genetic.callbacks import ProgressBar

callback = ProgressBar()
```

### LogbookSaver

Saves the generation logbook to a file after each generation:

```python
from sklearn_genetic.callbacks import LogbookSaver

callback = LogbookSaver(checkpoint_path="./logbook.pkl")
```

### TensorBoard

Logs fitness metrics to TensorBoard (requires `tensorflow` extra):

```python
from sklearn_genetic.callbacks import TensorBoard
import tensorflow as tf

writer = tf.summary.create_file_writer("./logs")
callback = TensorBoard(writer=writer)
```

### MLflowCallback

Logs parameters and metrics to MLflow (requires `mlflow` extra). See [MLflow Integration](./mlflow) for a complete example.

## Combining Callbacks

All callbacks in the list are called after every generation. If any callback signals to stop, the search ends at the end of that generation:

```python
from sklearn_genetic.callbacks import ConsecutiveStopping, LogbookSaver, ProgressBar

search.fit(X_train, y_train, callbacks=[
    ProgressBar(),
    ConsecutiveStopping(generations=10, metric="fitness_best"),
    LogbookSaver(checkpoint_path="./checkpoint.pkl"),
])
```

## Tips & Gotchas

- Callbacks receive the current generation's logbook entry. The available metric names match the columns in `pd.DataFrame(search.history)`.
- `ConsecutiveStopping` is the most common early-stopping callback. Set `generations` based on your expected convergence speed — 5–10 generations is a reasonable default.
- `LogbookSaver` is useful for long runs: if the search is interrupted, you can inspect the progress so far.

## Next Steps

- [Custom Callbacks](./custom-callback) — write your own callback.
- [MLflow Integration](./mlflow) — log experiments to MLflow.
- [Advanced Optimizer Control](./advanced-optimizer-control) — tune diversity and convergence behavior.
