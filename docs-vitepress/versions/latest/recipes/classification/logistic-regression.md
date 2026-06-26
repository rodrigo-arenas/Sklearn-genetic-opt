---
title: "Tune LogisticRegression with Genetic Algorithms"
description: "Copy-paste recipe to tune LogisticRegression hyperparameters including C, penalty, and solver compatibility with GASearchCV."
---

:::warning Development version
You are reading the **latest (development)** docs. For stable documentation, see [stable](/stable/).
:::

# Tune LogisticRegression

**Time:** 5 min | **Difficulty:** Beginner

## What This Solves

`LogisticRegression` has a solver/penalty compatibility constraint — not all solver + penalty combinations are valid. This recipe shows how to search `C`, `penalty`, and `solver` jointly while avoiding invalid combos.

## Solver/Penalty Compatibility

| Solver | `l1` | `l2` | `elasticnet` | `none` |
|--------|------|------|--------------|--------|
| `liblinear` | ✓ | ✓ | — | — |
| `lbfgs` | — | ✓ | — | ✓ |
| `saga` | ✓ | ✓ | ✓ | ✓ |
| `sag` | — | ✓ | — | ✓ |

**Recommended approach:** Fix the solver to `saga` (the only one that supports all penalties) and search over penalty and `C`.

## Recipe

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn_genetic import GASearchCV, EvolutionConfig, RuntimeConfig
from sklearn_genetic.space import Categorical, Continuous

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# StandardScaler is required — LR is not scale-invariant
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(solver="saga", max_iter=2000, random_state=42)),
])

param_grid = {
    "lr__C":       Continuous(1e-3, 100.0, distribution="log-uniform"),
    "lr__penalty": Categorical(["l1", "l2", "elasticnet"]),
    "lr__l1_ratio": Continuous(0.0, 1.0),  # only used when penalty="elasticnet"
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

ga = GASearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=cv,
    evolution_config=EvolutionConfig(population_size=15, generations=12, elitism=True),
    runtime_config=RuntimeConfig(n_jobs=-1, verbose=True),
    random_state=42,
)
ga.fit(X_train, y_train)

print("Best ROC AUC (CV):", round(ga.best_score_, 4))
print("Best params:", ga.best_params_)
```

## Key Points

- **Always scale**: `LogisticRegression` is sensitive to feature scales. Put it in a `Pipeline` with `StandardScaler`.
- **Fix solver to `saga`**: Only `saga` supports all three penalties (`l1`, `l2`, `elasticnet`). Including other solvers requires enumerating valid combos.
- **`l1_ratio` is inactive for l1/l2**: The genetic search learns to route around this. A waste of a parameter slot, but harmless and simpler than filtering.
- **`log-uniform` for C**: The default range spans 4 orders of magnitude — log-uniform gives equal probability to C=0.01 and C=10.

## See Also

- [Logistic Regression Hyperparameter Tuning](../../tutorials/tune-logistic-regression) — full tutorial
- [Tune a Preprocessing + Estimator Pipeline](../pipelines/preprocessing-pipeline) — Pipeline with step prefixes
- [Choosing the Right Search Space](../../guide/choosing-search-spaces) — when to use log-uniform
