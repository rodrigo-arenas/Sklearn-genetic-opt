---
title: "Stop Hyperparameter Search After a Time Budget"
description: "Recipe to use TimerStopping callback to limit GASearchCV to a fixed wall-clock time budget."
---

# Stop After a Time Budget

**Time:** 5 min | **Difficulty:** Beginner

## What This Solves

When you have a deployment pipeline with a fixed training window (e.g. must finish within 10 minutes), `TimerStopping` ends the search after the wall-clock time limit — however many generations that produces.

## Recipe

```python
import time
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split

from sklearn_genetic import GASearchCV, EvolutionConfig, RuntimeConfig
from sklearn_genetic.callbacks import TimerStopping
from sklearn_genetic.space import Categorical, Continuous, Integer

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

BUDGET_SECONDS = 60   # 1 minute budget

ga = GASearchCV(
    estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid={
        "n_estimators": Integer(50, 300),
        "max_depth":    Integer(3, 20),
        "max_features": Continuous(0.1, 1.0),
        "class_weight": Categorical([None, "balanced"]),
    },
    scoring="roc_auc",
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    evolution_config=EvolutionConfig(
        population_size=20,
        generations=100,   # large upper limit — timer will stop it sooner
        elitism=True,
    ),
    runtime_config=RuntimeConfig(n_jobs=-1, verbose=True),
    random_state=42,
)

started = time.perf_counter()
ga.fit(X_train, y_train, callbacks=[TimerStopping(total_seconds=BUDGET_SECONDS)])
elapsed = time.perf_counter() - started

gens_completed = ga.history[-1]["gen"]
print(f"Ran {gens_completed} generations in {elapsed:.1f}s (budget: {BUDGET_SECONDS}s)")
print(f"Best CV ROC AUC: {ga.best_score_:.4f}")
```

## Combine Timer with Plateau Detection

```python
from sklearn_genetic.callbacks import ConsecutiveStopping, TimerStopping

# Stop when either condition is met
ga.fit(
    X_train, y_train,
    callbacks=[
        TimerStopping(total_seconds=120),
        ConsecutiveStopping(generations=5, metric="fitness_best"),
    ]
)
```

## Key Points

- **Timer checks at generation boundaries**: The timer is checked after each complete generation, not mid-generation. Actual elapsed time will be slightly over the budget by one generation.
- **`generations=100` as a fallback**: Set a large upper limit so the search runs as many generations as possible before the timer fires.
- **`ga.history[-1]["gen"]`**: Reports the last completed generation — useful for logging how many generations ran within the budget.
- **First generation always completes**: The timer can't interrupt mid-generation, so you always get at least one result.

## See Also

- [Stop When Fitness Plateaus](./early-stopping-consecutive) — stop on convergence instead
- [Early Stopping and Callbacks](../../guide/callbacks) — all available callbacks
- [Resume a Stopped Search](./checkpointing) — continue from a checkpoint after stopping
