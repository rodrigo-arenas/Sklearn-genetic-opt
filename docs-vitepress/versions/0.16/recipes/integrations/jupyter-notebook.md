---
title: "Run Hyperparameter Search in a Jupyter Notebook"
description: "Recipe to run GASearchCV in a Jupyter Notebook with tqdm progress bars and inline matplotlib plots."
---

# Run in a Jupyter Notebook

**Time:** 5 min | **Difficulty:** Beginner

## What This Solves

Running `GASearchCV` in a notebook needs a few adjustments: enabling tqdm notebook mode, configuring matplotlib inline, and handling warnings from parallel workers.

## Recipe

```python
import warnings
warnings.filterwarnings("ignore")

# tqdm notebook mode — must be called before GASearchCV
from tqdm.notebook import tqdm as tqdm_notebook

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split

from sklearn_genetic import GASearchCV, EvolutionConfig, RuntimeConfig
from sklearn_genetic.space import Categorical, Continuous, Integer
from sklearn_genetic.plots import plot_fitness_evolution, plot_search_space

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

param_grid = {
    "n_estimators": Integer(50, 200),
    "max_depth":    Integer(3, 15),
    "max_features": Continuous(0.1, 1.0),
    "class_weight": Categorical([None, "balanced"]),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

ga = GASearchCV(
    estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid=param_grid,
    scoring="roc_auc",
    cv=cv,
    evolution_config=EvolutionConfig(population_size=15, generations=10, elitism=True),
    runtime_config=RuntimeConfig(
        n_jobs=-1,
        verbose=True,     # shows tqdm progress bar per generation
    ),
    random_state=42,
)
ga.fit(X_train, y_train)

print(f"Best CV ROC AUC: {ga.best_score_:.4f}")
print(f"Test ROC AUC:    {ga.score(X_test, y_test):.4f}")
print("Best params:", ga.best_params_)
```

## Inline Plots

After fitting, plot fitness evolution and search space directly in the notebook:

```python
# Fitness over generations
plot_fitness_evolution(ga, metric="fitness_max")
plt.title("ROC AUC over generations")
plt.show()

# Search space density
plot_search_space(ga, features=["n_estimators", "max_depth", "max_features"])
plt.tight_layout()
plt.show()
```

## Key Points

- **`verbose=True`**: Shows a tqdm progress bar per generation. In Jupyter, this renders as an interactive widget with ETA.
- **`warnings.filterwarnings("ignore")`**: Parallel workers emit convergence/sklearn warnings to stdout — suppress them for clean notebook output.
- **`%matplotlib inline`**: Add this magic command at the top of the notebook cell (before imports) to render plots inline.
- **`n_jobs=-1` caution in notebooks**: On some systems (particularly macOS), joblib's `loky` backend can hang in notebooks. If this happens, set `n_jobs=1` to use the serial backend.

## See Also

- [Plotting Gallery](../../examples/plotting-gallery) — all available plots
- [Getting Started with GASearchCV](../../guide/basic-usage) — first run walkthrough
- [MLflow Integration](./mlflow-logging) — track experiments from notebooks
