---
title: "Stop a Search Early When Fitness Plateaus"
description: "Recipe to use ConsecutiveStopping callback to end GASearchCV automatically when the best score stops improving."
---

:::warning Development version
You are reading the **latest (development)** docs. For stable documentation, see [stable](/stable/).
:::

# Stop Early When Fitness Plateaus

**Time:** 5 min | **Difficulty:** Beginner

## What This Solves

Running all N generations wastes time when the search converges after generation 5. `ConsecutiveStopping` ends the search automatically when the best score doesn't improve for K consecutive generations.

## Recipe

```python
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split

from sklearn_genetic import GASearchCV, EvolutionConfig, RuntimeConfig
from sklearn_genetic.callbacks import ConsecutiveStopping
from sklearn_genetic.space import Continuous, Integer

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

ga = GASearchCV(
    estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid={
        "n_estimators": Integer(50, 300),
        "max_depth":    Integer(3, 20),
        "max_features": Continuous(0.1, 1.0),
    },
    scoring="roc_auc",
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    evolution_config=EvolutionConfig(
        population_size=20,
        generations=50,    # upper limit — early stopping will terminate sooner
        elitism=True,
    ),
    runtime_config=RuntimeConfig(n_jobs=-1, verbose=True),
    random_state=42,
)

callback = ConsecutiveStopping(
    generations=5,          # stop if best score doesn't improve for 5 generations
    metric="fitness_best",  # track the running best (not generation mean)
)

ga.fit(X_train, y_train, callbacks=[callback])

print(f"Stopped after {ga.history[-1]['gen']} generations (max was 50)")
print(f"Best CV ROC AUC: {ga.best_score_:.4f}")
```

## Available Metrics for `ConsecutiveStopping`

| `metric=` | What it tracks |
|-----------|---------------|
| `"fitness_best"` | Best score seen so far (monotonically non-decreasing) |
| `"fitness"` | Current generation mean score |
| `"fitness_std"` | Standard deviation of current generation scores |

## Combine with a Time Budget

```python
from sklearn_genetic.callbacks import ConsecutiveStopping, TimerStopping

ga.fit(
    X_train, y_train,
    callbacks=[
        ConsecutiveStopping(generations=5, metric="fitness_best"),
        TimerStopping(total_seconds=120),   # also stop if search takes > 2 min
    ]
)
```

## Key Points

- **`generations=50` as upper limit**: Set a generous max — early stopping will end it sooner in practice.
- **`"fitness_best"` vs `"fitness"`**: `"fitness_best"` only increases; the search stops when the *best-ever* score stalls. `"fitness"` tracks the current generation mean, which can oscillate.
- **Multiple callbacks**: Both are checked after each generation. The search stops when any callback signals completion.

## See Also

- [Stop After a Time Budget](./time-budget) — wall-clock limit
- [Early Stopping and Callbacks](../../guide/callbacks) — all available callbacks
- [Writing Custom Callbacks](../../guide/custom-callback) — define your own stopping criterion
