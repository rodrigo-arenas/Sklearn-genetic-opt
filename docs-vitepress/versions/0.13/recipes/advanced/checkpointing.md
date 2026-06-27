---
title: "Resume a Stopped Hyperparameter Search from a Checkpoint"
description: "Recipe to save GASearchCV state to disk and resume from a checkpoint, preserving all evaluated candidates."
---

# Resume a Stopped Search from a Checkpoint

**Time:** 8 min | **Difficulty:** Intermediate

## What This Solves

Long searches can be interrupted (OOM, timeout, instance preemption). Checkpointing saves the search state after each generation so you can resume without re-evaluating already-visited candidates.

## Recipe: Save a Checkpoint

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

param_grid = {
    "n_estimators": Integer(50, 300),
    "max_depth":    Integer(3, 20),
    "max_features": Continuous(0.1, 1.0),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

ga = GASearchCV(
    estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid=param_grid,
    scoring="roc_auc",
    cv=cv,
    evolution_config=EvolutionConfig(population_size=20, generations=30, elitism=True),
    runtime_config=RuntimeConfig(n_jobs=-1, verbose=True),
    random_state=42,
)

# checkpoint_path: file to save state after each generation
ga.fit(
    X_train, y_train,
    checkpoint_path="./ga_search_checkpoint.pkl",
    callbacks=[ConsecutiveStopping(generations=5, metric="fitness_best")],
)

print(f"Best CV ROC AUC: {ga.best_score_:.4f}")
```

## Recipe: Resume from a Checkpoint

```python
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Load the saved search
ga = joblib.load("./ga_search_checkpoint.pkl")

print(f"Resumed from generation {ga.history[-1]['gen']}")
print(f"Best CV ROC AUC so far: {ga.best_score_:.4f}")

# Continue from where it stopped — provide the same data
ga.fit(
    X_train, y_train,
    checkpoint_path="./ga_search_checkpoint.pkl",
    # Increase generations or remove early stopping to run more
)

print(f"Final best CV ROC AUC: {ga.best_score_:.4f}")
```

## Key Points

- **`checkpoint_path`**: Saves the full `GASearchCV` object (including population, history, best params) after each generation using `joblib.dump`.
- **Same data required**: Resume with the same `X_train`, `y_train` and `cv` splits. Different data produces incorrect results.
- **`random_state` preserved**: The checkpoint restores the PRNG state, so resumed generations are deterministic.
- **Cache hits**: Evaluated candidates from the checkpoint are cached — they won't be re-evaluated on resume.
- **File size**: Checkpoints include the full population. For large estimators, this can be 10–100 MB.

## See Also

- [Reproducibility and Checkpointing](../../guide/reproducibility) — full guide
- [Checkpointing & Resume Example](../../examples/checkpointing) — annotated example
- [Stop After a Time Budget](./time-budget) — stop cleanly at a time limit, then resume
