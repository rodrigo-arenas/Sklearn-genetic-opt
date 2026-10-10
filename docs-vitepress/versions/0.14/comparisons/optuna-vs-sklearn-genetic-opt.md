---
title: "Optuna vs sklearn-genetic-opt: Bayesian Optimization vs Genetic Algorithms"
description: "An honest comparison of Optuna (Bayesian optimization with TPE) and sklearn-genetic-opt (genetic algorithms) for hyperparameter tuning — with benchmarks, code examples, and a decision guide."
---

# Optuna vs sklearn-genetic-opt: When to Use Each

Optuna and sklearn-genetic-opt are both popular choices for hyperparameter optimization beyond GridSearchCV, but they use fundamentally different algorithms. Optuna uses Tree-structured Parzen Estimators (Bayesian optimization), while sklearn-genetic-opt uses genetic/evolutionary algorithms. This page compares them honestly — including scenarios where each method wins, loses, and ties. The goal is to help you pick the right tool, not to sell you on one.

---

## How Each Method Works

### Optuna (TPE / Bayesian Optimization)

- **Maintains a probabilistic model** of `f(hyperparams) → score`, updated after every trial
- **Each new trial is chosen to maximize "expected improvement"** over the current best — so every evaluation is informed by all previous ones
- **Very data-efficient**: learns from every trial to guide the next, which matters most when evaluations are expensive
- **Handles conditional spaces natively**: Optuna's define-by-run API lets you suggest a parameter only when it's relevant (e.g., `gamma` only when `kernel='rbf'`)
- **Sequential by default**: one trial at a time (though parallel backends exist)

### sklearn-genetic-opt (Genetic / Evolutionary Algorithms)

- **Maintains a population** of complete hyperparameter configurations (individuals)
- **Each generation**: select the best individuals, apply crossover (combine two configs) and mutation (randomly perturb a config), then evaluate the new generation
- **Population-based**: evaluates multiple configurations per generation — a natural fit for parallel execution
- **Captures cross-parameter interactions through crossover**: combining the `max_depth` of one good config with the `n_estimators` of another is the genetic analog of "try this combination we haven't seen"
- **Evaluates against sklearn's scoring function with cross-validation**: `GASearchCV` fits directly into the scikit-learn `fit`/`predict` pattern with the same attributes as `GridSearchCV`

---

## Quick Decision Guide

| Situation | Use Optuna | Use sklearn-genetic-opt |
|-----------|:----------:|:----------------------:|
| Evaluations are very expensive (> 5 min each) | ✓ | |
| Want to minimize total evaluations | ✓ | |
| Conditional parameter spaces (e.g., kernel-specific params) | ✓ | |
| Combined algorithm selection + tuning (CASH) | ✓ | |
| Already using scikit-learn API (GridSearchCV pattern) | | ✓ |
| Want parallel evaluation of multiple configs per round | | ✓ |
| Need diagnostic plots (convergence, search space, diversity) | | ✓ |
| MLflow / TensorBoard integration out of the box | | ✓ |
| Feature selection alongside hyperparameter search | | ✓ |
| Callbacks for early stopping in sklearn style | | ✓ |
| Budget of 50–500 evaluations, mixed numeric parameter types | Tie | Tie |
| Standard hyperparameter tuning on a fixed estimator | Tie | Tie |

---

## Performance Benchmark

The numbers below come from the [Benchmarks](../benchmarks/) page, using the Bayesmark suite: standard scikit-learn datasets and estimators, equal evaluation budget across all methods (budget = 48, 3 seeds, 3-fold CV). Classification is scored by accuracy (higher is better); regression by MSE (lower is better).

| Dataset | Model | Metric | GASearchCV | Optuna | RandomizedSearchCV | Winner |
|---------|-------|--------|:----------:|:------:|:-----------------:|--------|
| wine | knn | accuracy ↑ | **0.9832** | 0.9832 | 0.9813 | tie |
| wine | svm | accuracy ↑ | **0.9869** | 0.9851 | 0.9850 | gasearch |
| wine | dt | accuracy ↑ | 0.8520 | **0.8651** | 0.8463 | optuna |
| wine | rf | accuracy ↑ | 0.9645 | **0.9719** | 0.9532 | optuna |
| breast | knn | accuracy ↑ | 0.9684 | **0.9695** | 0.9678 | optuna |
| breast | svm | accuracy ↑ | **0.9789** | 0.9783 | 0.9783 | gasearch |
| breast | dt | accuracy ↑ | 0.9133 | **0.9151** | 0.9133 | optuna |
| breast | rf | accuracy ↑ | 0.9256 | **0.9285** | 0.9215 | optuna |
| diabetes | knn | MSE ↓ | **3223.40** | 3229.59 | 3223.47 | gasearch |
| diabetes | svm | MSE ↓ | 3035.66 | **3034.17** | 3053.55 | optuna |
| diabetes | dt | MSE ↓ | 3994.28 | **3837.88** | 3852.95 | optuna |
| diabetes | rf | MSE ↓ | 3553.56 | **3320.70** | 3498.15 | optuna |

Optuna wins 7 of 12 tasks, GASearchCV wins 3, and there are 2 ties. Optuna has a consistent edge on tree-based models (`dt`, `rf`) at this small budget.

:::info Does more budget change the picture?
At **budget = 96**, the gap narrows. On wine/dt, GASearchCV overtakes Optuna (0.8690 vs 0.8670). On diabetes/rf, the gap collapses from ~233 to ~36 MSE. On the tree *classification* tasks (breast/rf), Optuna maintains its lead. Both methods improve with more budget — this is not a story about one outrunning the other, but about where each one benefits most from extra evaluations.

See the [Benchmarks page](../benchmarks/) for the full budget-scaling results.
:::

**The performance difference on standard hyperparameter search is usually small.** Both methods beat random search consistently, and the margin between them is often within one standard deviation across seeds. The bigger differences show up in workflow integration, feature set, and how each method handles the edges of the problem (very expensive evaluations, conditional spaces, feature selection).

### CASH: Where the Picture Changes

For **Combined Algorithm Selection and Hyperparameter optimization** (choosing the estimator family *and* its hyperparameters simultaneously), the benchmark tells a clear and honest story:

| Optimizer | Mean Accuracy ± Std | Mean Evals | Mean Wall Time |
|-----------|:-------------------:|:----------:|:--------------:|
| Optuna | **0.8636 ± 0.0050** | 184 | 59 s |
| RandomizedSearchCV | 0.8591 ± 0.0055 | 184 | 27 s |
| GASearchCV | 0.8582 ± 0.0087 | 184 | 205 s |

:::warning Honest result: CASH is Optuna's territory
For combined algorithm selection + tuning, **GASearchCV finished last** — and 3–8× slower in wall time. The reason is structural: Optuna's define-by-run API only models the active parameters of the selected family. GASearchCV carries every family's parameters as genes simultaneously, and crossover between individuals that picked different families mostly recombines inactive genes.

If your problem is CASH, use Optuna or another define-by-run Bayesian optimizer.
:::

---

## Code Comparison — Same Problem, Both Tools

The same `RandomForestClassifier` + `breast_cancer` problem solved with both tools, with a matched 60-evaluation budget.

### Optuna

```python
import optuna
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

# install: pip install optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        "ccp_alpha": trial.suggest_float("ccp_alpha", 0.0, 0.03),
    }
    rf = RandomForestClassifier(random_state=42, **params)
    return cross_val_score(rf, X_train, y_train, cv=cv, scoring="roc_auc").mean()

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=60, n_jobs=-1)
print(f"Optuna best CV ROC-AUC: {study.best_value:.4f}")
print(f"Optuna best params: {study.best_params}")
```

**Expected output:**
```
Optuna best CV ROC-AUC: 0.9956
Optuna best params: {'n_estimators': 247, 'max_depth': 11, 'min_samples_split': 2,
                      'min_samples_leaf': 1, 'max_features': 'sqrt', 'ccp_alpha': 0.001}
```

### sklearn-genetic-opt

```python
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn_genetic import GASearchCV, EvolutionConfig, PopulationConfig, RuntimeConfig
from sklearn_genetic.space import Integer, Continuous, Categorical

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

param_grid = {
    "n_estimators": Integer(50, 300),
    "max_depth": Integer(3, 20),
    "min_samples_split": Integer(2, 20),
    "min_samples_leaf": Integer(1, 10),
    "max_features": Categorical(["sqrt", "log2"]),
    "ccp_alpha": Continuous(0.0, 0.03),
}

search = GASearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=cv,
    scoring="roc_auc",
    # population_size=15, generations=10 → ~150 nominal evals,
    # but the GA caches and deduplicates — realized unique evals ≈ 60–80
    evolution_config=EvolutionConfig(population_size=15, generations=10),
    population_config=PopulationConfig(initializer="smart"),
    runtime_config=RuntimeConfig(
        n_jobs=-1, parallel_backend="population", use_cache=True
    ),
    random_state=42,
)
search.fit(X_train, y_train)
print(f"GA best CV ROC-AUC: {search.best_score_:.4f}")
print(f"GA best params: {search.best_params_}")

# GASearchCV exposes the same sklearn attributes as GridSearchCV
print(f"CV results shape: {len(search.cv_results_['mean_test_score'])} candidates evaluated")
```

**Expected output:**
```
GA best CV ROC-AUC: 0.9953
GA best params: {'n_estimators': 212, 'max_depth': 13, 'min_samples_split': 2,
                  'min_samples_leaf': 1, 'max_features': 'sqrt', 'ccp_alpha': 0.0009}
CV results shape: 68 candidates evaluated
```

:::tip What the output shows
Both methods land within a few hundredths of each other on ROC-AUC. The important difference is not the score — it's the API. Optuna requires a custom `objective` function; GASearchCV plugs into sklearn's existing `fit`/`cv_results_`/`best_params_` pattern directly. The right choice depends on which workflow fits your project better.
:::

---

## When Optuna Is the Better Choice

Be specific about when to reach for Optuna over sklearn-genetic-opt.

**1. Very expensive evaluations (deep learning, large datasets)**
Optuna's Bayesian approach minimizes total evaluations needed. If each cross-validation run takes 30 minutes, the difference between 60 and 80 evaluations is hours. Sequential TPE, which learns from each result before choosing the next trial, is better suited here than a population-based approach that commits to a batch of N individuals per generation.

**2. Conditional parameter spaces**
Optuna's define-by-run API handles "if `kernel='rbf'`, tune `gamma`; else skip `gamma`" natively and efficiently. The CASH benchmark above shows this concretely. GASearchCV uses a flat encoding where all parameters exist as genes regardless of whether they're active.

**3. Advanced pruning**
Optuna can terminate unpromising trials *mid-evaluation* using `MedianPruner`, `SuccessiveHalvingPruner`, or `HyperbandPruner`. This is useful when you're training iterative models (gradient boosting, neural networks) where you can measure intermediate performance. GASearchCV always evaluates the full cross-validation before acting.

**4. Real-time dashboard**
[optuna-dashboard](https://github.com/optuna/optuna-dashboard) gives a live web UI showing trial history, parameter importances, and contour plots without any additional setup. sklearn-genetic-opt has built-in plots (11+), but they render post-hoc, not live.

**5. Distributed optimization**
Optuna supports distributed trials with pluggable storage backends (PostgreSQL, Redis, etc.) so multiple machines can contribute to the same study. sklearn-genetic-opt parallelizes within a machine via `joblib` but does not natively distribute across machines.

---

## When sklearn-genetic-opt Is the Better Choice

**1. sklearn-first workflow**
`GASearchCV` is a drop-in replacement for `GridSearchCV`. It has the same `cv_results_`, `best_params_`, `best_estimator_`, and `score()` attributes — meaning it fits into any sklearn pipeline, model registry, or tooling that already expects a `BaseSearchCV` object. Optuna requires writing an `objective` function and accessing results via `study.best_params` instead.

**2. Parallel population evaluation**
Genetic search naturally evaluates all N individuals in a generation simultaneously. With `parallel_backend="population"` and `n_jobs=-1`, you fill your CPU budget in a single parallelized batch. Optuna's default sequential TPE makes one decision at a time; its `n_jobs` parameter runs multiple trials in parallel but without the generational coupling that makes population evaluation coherent.

**3. Feature selection alongside hyperparameter search**
`GAFeatureSelectionCV` is unique to this library. It jointly optimizes which features to include and what hyperparameter values to use — a combined search that no other sklearn-compatible tool handles in the same interface. Optuna has no equivalent out of the box.

**4. Diagnostic plots**
sklearn-genetic-opt includes 11+ built-in visualizations: convergence curves, search space scatter plots, parameter evolution across generations, population diversity, and fitness distributions. These are available via `plot_search_space(search)`, `plot_convergence(search)`, and friends — no extra code required. See the [Benchmarks page](../benchmarks/) for examples of how to read them.

**5. MLflow auto-logging**
Every candidate configuration is logged as a child run with zero configuration:

```python
import mlflow
from sklearn_genetic.callbacks import LogbookCallback

with mlflow.start_run():
    search.fit(X_train, y_train, callbacks=[LogbookCallback()])
    # Every candidate is a child run in MLflow with params and metrics
```

Optuna has its own [MLflow integration](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.integration.MLflowCallback.html), but it requires a separate callback setup and does not integrate with sklearn's `fit` lifecycle automatically.

**6. Callbacks for early stopping in sklearn style**
`ConsecutiveStopping`, `ThresholdStopping`, and `TimerStopping` plug directly into `search.fit(X, y, callbacks=[...])`. They are composable and fit naturally into a script that otherwise uses sklearn conventions.

**7. Mixed parameter types without conditional logic**
`Integer(low, high)`, `Continuous(low, high)`, and `Categorical([...])` work uniformly — you define the grid and the optimizer handles the rest. No `if`/`else` branching inside an objective function.

---

## Using Both Together

Optuna and sklearn-genetic-opt are not mutually exclusive. Some practitioners use a two-phase approach:

- **Phase 1 — Optuna for global exploration**: Run `n_trials=30` with a wide parameter space to identify which regions are productive. Optuna's TPE is efficient at eliminating dead zones quickly.
- **Phase 2 — GASearchCV for population-based refinement**: Take the top-N configs from Optuna as initial population seeds, narrow the bounds, and use `GASearchCV` with crossover to explore interactions within the productive region.

```python
import optuna
from sklearn_genetic import GASearchCV, EvolutionConfig, PopulationConfig, RuntimeConfig
from sklearn_genetic.space import Integer, Continuous, Categorical
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score

optuna.logging.set_verbosity(optuna.logging.WARNING)

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Phase 1: Optuna explores broadly
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 10, 500),
        "max_depth": trial.suggest_int("max_depth", 2, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 30),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
    }
    rf = RandomForestClassifier(random_state=42, **params)
    return cross_val_score(rf, X_train, y_train, cv=cv, scoring="roc_auc").mean()

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)

# Extract best params from Optuna to narrow bounds for Phase 2
best = study.best_params
n_est_center = best["n_estimators"]
depth_center = best["max_depth"]

# Phase 2: GASearchCV refines within a narrower space
narrow_grid = {
    "n_estimators": Integer(
        max(10, n_est_center - 50), min(500, n_est_center + 50)
    ),
    "max_depth": Integer(
        max(2, depth_center - 3), min(30, depth_center + 3)
    ),
    "min_samples_split": Integer(2, 20),
    "max_features": Categorical(["sqrt", "log2"]),
}

search = GASearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=narrow_grid,
    cv=cv,
    scoring="roc_auc",
    evolution_config=EvolutionConfig(population_size=12, generations=8),
    runtime_config=RuntimeConfig(n_jobs=-1, parallel_backend="population"),
    random_state=42,
)
search.fit(X_train, y_train)
print(f"Phase 2 (GA) best CV ROC-AUC: {search.best_score_:.4f}")
```

The reverse order (GA first for diversity, then Optuna to exploit) also works. The point is that the two tools address different parts of the exploration–exploitation tradeoff, and combining them is a legitimate strategy.

---

## The Honest Bottom Line

- For most hyperparameter tuning tasks with a fixed estimator, **both methods achieve similar quality given an equal budget**. The benchmarks show this: differences are typically within one standard deviation.
- **Optuna is the better tool** when evaluations are expensive, when you have conditional spaces, or when you need pruning or a distributed setup. For combined algorithm selection + tuning (CASH), Optuna's structural advantage is real and measurable.
- **sklearn-genetic-opt is the better fit** when you're already in scikit-learn land, when you want feature selection alongside hyperparameter search, or when you want diagnostic plots and MLflow logging with minimal setup.
- The choice often comes down to **workflow preferences and ecosystem fit** rather than raw benchmark scores.

:::tip
Both libraries are free and well-maintained. Try both on your actual problem — the winner depends more on your specific search space and workflow than on general benchmarks. If you're not sure, start with GASearchCV (it's a two-line change from GridSearchCV) and switch to Optuna if you hit a conditional-space or pruning requirement.
:::

---

## See Also

- [Grid Search vs Random Search vs Bayesian vs Genetic Algorithms](./grid-search-vs-genetic-algorithms) — broader comparison of all four methods with benchmarks
- [Benchmarks: GASearchCV vs Optuna vs RandomizedSearchCV](../benchmarks/) — detailed Bayesmark suite results including CASH and mixed-space results
- [When to Use Genetic Algorithm Search](../guide/when-to-use) — decision guide for choosing between search methods
- [Getting Started with GASearchCV](../guide/basic-usage) — try it yourself with a working example
- [Random Forest Hyperparameter Tuning](../tutorials/tune-random-forest) — complete tutorial using sklearn-genetic-opt
- [Feature Selection Guide](../guide/feature-selection-guide) — GAFeatureSelectionCV in detail
