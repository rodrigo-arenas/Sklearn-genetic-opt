---
title: "Warm-start a scikit-learn Hyperparameter Search"
description: "Use warm_start_configs to seed GASearchCV with partial categorical and numeric configurations or reuse top candidates from an earlier search."
---

:::warning Development version
You are reading the **latest (development)** docs. For stable documentation, see [stable](/stable/).
:::

# Seed a Search with Known-Good Params (Warm Start)

**Time:** 5 min | **Difficulty:** Intermediate

## What This Solves

Use `warm_start_configs` to put known configurations or specific parameter
combinations into the starting population of `GASearchCV`. Each entry can be
partial. The search samples omitted parameters from their declared dimensions.

## Recipe

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier

from sklearn_genetic import GASearchCV, EvolutionConfig, PopulationConfig, RuntimeConfig
from sklearn_genetic.space import Continuous, Integer

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

param_grid = {
    "n_estimators": Integer(50, 400),
    "max_depth":    Integer(2, 10),
    "learning_rate": Continuous(0.01, 0.3, distribution="log-uniform"),
    "subsample":    Continuous(0.5, 1.0),
    "colsample_bytree": Continuous(0.4, 1.0),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

ga = GASearchCV(
    estimator=XGBClassifier(
        tree_method="hist", eval_metric="logloss",
        random_state=42, n_jobs=1,
    ),
    param_grid=param_grid,
    scoring="roc_auc",
    cv=cv,
    evolution_config=EvolutionConfig(
        population_size=15,
        generations=10,
        elitism=True,
        keep_top_k=3,
    ),
    population_config=PopulationConfig(
        initializer="smart",
        warm_start_configs=[
            # Seed with XGBoost's well-known defaults
            {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.3,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
            },
            # Also seed with a conservative, well-regularized config
            {
                "n_estimators": 200,
                "max_depth": 4,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
            },
        ],
    ),
    runtime_config=RuntimeConfig(n_jobs=-1, parallel_backend="cv", verbose=True),
    random_state=42,
)
ga.fit(X_train, y_train)

print("Best CV ROC AUC:", round(ga.best_score_, 4))
print("Best params:", ga.best_params_)
```

## Key Points

- Keys you supply must appear in `param_grid`, and their values must belong to the corresponding dimensions.
- The default `initializer="smart"` fills omitted values, skips duplicate completed configurations, and reads seeds in order until the population is full.
- When slots remain, smart initialization tries the estimator's defaults if they fit the space. It then samples numeric and categorical dimensions separately and fills any gaps left by deduplication.

## Seed categorical and numeric combinations

Reserve starting candidates for both an RBF and a linear SVC at the same
regularization strength. The search samples `gamma` for each seed because
the warm starts specify only `kernel` and `C`.

```python
from sklearn.datasets import load_wine
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn_genetic import GASearchCV, PopulationConfig
from sklearn_genetic.space import Categorical, Continuous

X, y = load_wine(return_X_y=True)
seeds = [
    {"svc__kernel": "linear", "svc__C": 0.1},
    {"svc__kernel": "rbf", "svc__C": 0.1},
]
search = GASearchCV(
    make_pipeline(StandardScaler(), SVC()),
    param_grid={
        "svc__kernel": Categorical(["linear", "rbf"]),
        "svc__C": Continuous(0.01, 10.0, distribution="log-uniform"),
        "svc__gamma": Continuous(0.0001, 0.1, distribution="log-uniform"),
    },
    population_config=PopulationConfig(warm_start_configs=seeds),
    population_size=6,
    generations=0,
    cv=3,
    scoring="balanced_accuracy",
    random_state=42,
    n_jobs=1,
    verbose=False,
).fit(X, y)

for candidate in search.cv_results_["params"][:len(seeds)]:
    print(candidate)
```

`generations=0` evaluates the starting population only. Check the first two
rows of `cv_results_["params"]` for the seeded pairs and sampled `gamma` values.

### Compare seeded and unseeded starts at the same CV budget

Seeds can help when they put a good configuration early in the search. They
also take slots in the initial population. Before reusing a seed strategy,
compare it across several splits. Choose your seeds independently of the
holdout data and keep the search space and CV folds fixed within each pair.

The following example reuses the pipeline, search space, and fixed seeds
above. `clone` creates an unfitted search for each run.

```python
from sklearn.base import clone
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn_genetic.callbacks import CVBudgetStopping

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)
folds = list(StratifiedKFold(3, shuffle=True, random_state=42).split(X_train, y_train))

for label, warm_starts in (("unseeded", []), ("seeded", seeds)):
    comparison = clone(search).set_params(
        population_config=PopulationConfig(warm_start_configs=warm_starts),
        cv=folds,
        generations=4,
    )
    comparison.fit(X_train, y_train, callbacks=CVBudgetStopping(12))
    calls = comparison.fit_counts_["search_cv_calls"]
    holdout = balanced_accuracy_score(y_test, comparison.predict(X_test))
    print(f"{label}: {calls} CV calls, CV={comparison.best_score_:.3f}, holdout={holdout:.3f}")
```

Check that both runs reached the same number of fresh CV calls before
comparing their scores. In a larger comparison, report how many pairs
reached the requested budget and summarize CV and holdout scores separately.
See [CV checkpoints in the optimizer catalog](../../benchmarks/optimizer-catalog#compare-several-cv-budgets-in-one-run).

## Multiple Seeds from Prior Runs

Seed a fresh search with candidates from an earlier `GASearchCV` fit. The
DataFrame is already ordered by test rank for the selected metric. Update
the population configuration to replace any warm starts it already contains.

```python
from sklearn.base import clone
from sklearn_genetic import PopulationConfig

ranked = old_ga.cv_results_dataframe(columns="params")
warm_configs = ranked["params"].head(3).tolist()
ga = clone(old_ga).set_params(
    population_config=PopulationConfig(
        initializer=old_ga.population_initializer,
        warm_start_configs=warm_configs,
    )
)
```

## See Also

- [Adaptive Crossover & Mutation Schedules](../../guide/adapters) — anneal exploration after warm start
- [Resume a Stopped Search](./checkpointing) — full resume vs warm start
- [XGBoost Hyperparameter Tuning](../../tutorials/tune-xgboost) — warm start in a full tutorial
