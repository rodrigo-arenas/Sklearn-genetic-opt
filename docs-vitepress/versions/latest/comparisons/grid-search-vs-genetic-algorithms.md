---
title: "Grid Search vs Random Search vs Bayesian vs Genetic Algorithms: Which Should You Use?"
description: "A practical, benchmark-backed comparison of GridSearchCV, RandomizedSearchCV, Optuna (Bayesian), and GASearchCV (genetic) for hyperparameter optimization in Python."
---

:::warning Development version
You are reading the **latest (dev)** docs. For stable documentation, see [stable](/stable/).
:::

# Grid Search vs Random Search vs Bayesian vs Genetic Algorithms: Which Should You Use?

**Estimated reading time:** 15 min &nbsp;|&nbsp; **Difficulty:** Intermediate

:::info Prerequisites
- `scikit-learn` installed (`pip install scikit-learn`)
- `sklearn-genetic-opt` installed (`pip install sklearn-genetic-opt`)
- Basic familiarity with `fit` / `predict` and cross-validation
:::

There are four mainstream approaches to hyperparameter optimization in Python. Each has genuine strengths and genuine failure modes. This page gives you the numbers to choose wisely, including the cases where `GASearchCV` is the wrong tool.

---

## TL;DR — Quick Decision Guide

| Method | Best for | Weakness | Speed | Code complexity |
|--------|----------|----------|-------|-----------------|
| `GridSearchCV` | Small grids, ≤ 3 params, must check every combo | Candidate count multiplies exponentially | Fast (low `n_iter`) | Trivial |
| `RandomizedSearchCV` | Baseline search, fast experiments, 3–6 params | Treats every parameter independently — misses interactions | Fast | Trivial |
| Bayesian (Optuna) | Data-efficient search, very expensive evaluations | More setup, sequential by default | Moderate | Moderate |
| Genetic (`GASearchCV`) | 5+ interacting params, mixed types, large spaces | Overhead on tiny spaces; CASH via flat grids | Moderate | Low (sklearn API) |

If you are starting out: **run `RandomizedSearchCV` first**. It is fast, effective on smooth spaces, and takes 5 lines to set up. Switch to `GASearchCV` when you hit its limits (many parameters, known interactions, mixed types). Switch to Optuna when each individual evaluation costs more than a few minutes.

---

## Grid Search: Exhaustive but Limited

### How it works

Grid search defines a finite list of values for each parameter and evaluates every possible combination. With 3 parameters and 3 values each, that is 3³ = 27 evaluations. Add a fourth parameter with 3 values: 81. Add a fifth: 243.

### The combinatorial explosion problem

| Parameters | Values each | Candidates | At 5 s/eval |
|-----------|-------------|------------|-------------|
| 2 | 3 | 9 | 45 s |
| 3 | 3 | 27 | 2.25 min |
| 4 | 3 | 81 | 6.75 min |
| 5 | 3 | 243 | 20 min |
| 6 | 3 | 729 | 1 hr |
| 7 | 3 | 2,187 | 3 hrs |

At 3 values per parameter this is already impractical beyond 5–6 parameters. And 3 values is a very coarse grid — especially for continuous parameters like `learning_rate`, where the sweet spot might lie between any two of your chosen values.

### Code: GridSearchCV on breast_cancer

```python
import time
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 6, None],
    "min_samples_leaf": [1, 5, 10],
}
# 3 × 3 × 3 = 27 candidates

t0 = time.perf_counter()
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1,
)
grid_search.fit(X_train, y_train)
grid_time = time.perf_counter() - t0

test_auc = roc_auc_score(y_test, grid_search.predict_proba(X_test)[:, 1])
print(f"GridSearchCV")
print(f"  Candidates: {len(grid_search.cv_results_['params'])}")
print(f"  Best CV AUC:  {grid_search.best_score_:.4f}")
print(f"  Test AUC:     {test_auc:.4f}")
print(f"  Time:         {grid_time:.1f}s")
print(f"  Best params:  {grid_search.best_params_}")
```

```text
GridSearchCV
  Candidates: 27
  Best CV AUC:  0.9955
  Test AUC:     0.9972
  Time:         3.2s
  Best params:  {'max_depth': None, 'min_samples_leaf': 1, 'n_estimators': 200}
```

### When grid search wins

- You have 2–3 parameters and you must check every combination (e.g., regulatory requirement for exhaustive search)
- All parameters are discrete with a small, well-reasoned set of candidate values
- You want to reproduce an existing paper's exact hyperparameter grid

---

## Random Search: Surprisingly Effective

### The Bergstra & Bengio insight

In their 2012 paper, Bergstra & Bengio showed that random search almost always matches or beats grid search at the same budget — and often requires far fewer evaluations. The reason is subtle: most parameters have low importance. If only 2 of your 6 parameters actually move the needle, a grid wastes 4/6 of its resolution on dimensions that don't matter. Random search samples all dimensions independently, so it covers the important ones more densely per evaluation.

### Why it beats grid search on continuous spaces

A grid forces you to commit to specific values upfront. If the best `learning_rate` is 0.047 and your grid includes 0.01 and 0.1, you will never find it. Random search samples from a distribution, so the entire range gets coverage proportional to your budget.

### Code: RandomizedSearchCV on the same problem

```python
import numpy as np
from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV

param_dist = {
    "n_estimators": randint(50, 500),
    "max_depth": [3, 5, 8, 10, 15, None],
    "min_samples_leaf": randint(1, 20),
    "max_features": uniform(0.2, 0.8),   # continuous: 0.2–1.0
}

BUDGET = 50  # same budget used for all methods below

t0 = time.perf_counter()
random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_dist,
    n_iter=BUDGET,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1,
    random_state=42,
)
random_search.fit(X_train, y_train)
random_time = time.perf_counter() - t0

test_auc = roc_auc_score(y_test, random_search.predict_proba(X_test)[:, 1])
print(f"RandomizedSearchCV")
print(f"  Candidates: {len(random_search.cv_results_['params'])}")
print(f"  Best CV AUC:  {random_search.best_score_:.4f}")
print(f"  Test AUC:     {test_auc:.4f}")
print(f"  Time:         {random_time:.1f}s")
```

```text
RandomizedSearchCV
  Candidates: 50
  Best CV AUC:  0.9962
  Test AUC:     0.9972
  Time:         7.1s
```

### When random search wins

- Fast, cheap evaluations where you can afford a large budget
- 3–6 parameters on a smooth, continuous objective
- You need a strong baseline before committing to a more complex optimizer
- You want simplicity: no extra dependencies, no extra concepts

---

## Bayesian Optimization: Smart Sampling

### Surrogate model concept

Bayesian optimization fits a *surrogate model* — typically a Gaussian Process (GP) or Tree-structured Parzen Estimator (TPE) — to the observed (configuration, score) pairs. The surrogate is cheap to evaluate, so the optimizer can ask it: "where should I look next?" without running another full cross-validation.

### Acquisition function: exploration vs exploitation

The next configuration is chosen by maximizing an *acquisition function* — a formula that balances:

- **Exploitation:** try configurations where the surrogate predicts a high score
- **Exploration:** try configurations where the surrogate is uncertain (it might be wrong in a good way)

Expected Improvement (EI) is the most common choice: it estimates the probability and magnitude of improvement over the current best.

### Code: Optuna TPE

:::info Install Optuna separately
Optuna is not a dependency of `sklearn-genetic-opt`. Install it with `pip install optuna`.
:::

```python
# pip install optuna
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.model_selection import cross_val_score

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_categorical("max_depth", [3, 5, 8, 10, 15, None]),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "max_features": trial.suggest_float("max_features", 0.2, 1.0),
    }
    model = RandomForestClassifier(random_state=42, **params)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    return scores.mean()

t0 = time.perf_counter()
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=BUDGET, show_progress_bar=False)
optuna_time = time.perf_counter() - t0

best_params = study.best_params
best_model = RandomForestClassifier(random_state=42, **best_params)
best_model.fit(X_train, y_train)
test_auc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])
print(f"Optuna (TPE Bayesian)")
print(f"  Candidates: {len(study.trials)}")
print(f"  Best CV AUC:  {study.best_value:.4f}")
print(f"  Test AUC:     {test_auc:.4f}")
print(f"  Time:         {optuna_time:.1f}s")
```

```text
Optuna (TPE Bayesian)
  Candidates: 50
  Best CV AUC:  0.9964
  Test AUC:     0.9972
  Time:         9.3s
```

### When Bayesian optimization wins

- Each evaluation takes **more than a few minutes** (large neural networks, deep ensembles)
- You have a tight evaluation budget (20–100 total evaluations)
- You need conditional hyperparameters where some parameters only apply when others are set a certain way — Optuna's define-by-run API handles this natively
- You are running combined algorithm selection + tuning (CASH) — Optuna's conditional API is structurally better suited than a flat parameter grid

---

## Genetic Algorithms: Population-Based Search

### How crossover + mutation finds interactions

A genetic algorithm maintains a *population* of complete configurations. Each generation:

1. **Selection** — configurations with higher scores are more likely to become parents
2. **Crossover** — two parent configurations exchange parameter values to produce offspring
3. **Mutation** — random changes introduce diversity and prevent premature convergence
4. **Elitism** — the best configurations survive unchanged

The critical difference from random search: a genetic algorithm recombines **entire configurations that worked well**, not individual parameter values. If `(learning_rate=0.05, n_estimators=300)` performed well, its offspring inherits both values together. Random search treats each parameter independently and can never exploit that joint relationship.

### Key advantage: evaluates joint configurations

Suppose the optimal region of your search space is where `learning_rate` is low *and* `n_estimators` is high. Random search might find a high `n_estimators` candidate and a low `learning_rate` candidate but never combine them. A genetic algorithm, after a few generations, will recognize that good individuals share both traits and produce more offspring in that joint region.

### Code: GASearchCV on the same problem

```python
from sklearn_genetic import GASearchCV, EvolutionConfig, PopulationConfig, RuntimeConfig
from sklearn_genetic.space import Categorical, Continuous, Integer

param_grid_ga = {
    "n_estimators": Integer(50, 500),
    "max_depth": Categorical([3, 5, 8, 10, 15, None]),
    "min_samples_leaf": Integer(1, 20),
    "max_features": Continuous(0.2, 1.0),
}

t0 = time.perf_counter()
ga_search = GASearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid_ga,
    cv=cv,
    scoring="roc_auc",
    evolution_config=EvolutionConfig(
        population_size=10,
        generations=6,   # 10 × 6 ≈ 60 evaluations; cached duplicates skipped
        elitism=True,
        keep_top_k=3,
    ),
    population_config=PopulationConfig(initializer="smart"),
    runtime_config=RuntimeConfig(n_jobs=-1, use_cache=True, verbose=False),
    random_state=42,
)
ga_search.fit(X_train, y_train)
ga_time = time.perf_counter() - t0

test_auc = roc_auc_score(y_test, ga_search.predict_proba(X_test)[:, 1])
unique_evals = ga_search.fit_stats_["unique_candidates"]
print(f"GASearchCV (Genetic)")
print(f"  Candidates: {unique_evals} unique (cache hit rate: "
      f"{ga_search.fit_stats_['cache_hits'] / ga_search.fit_stats_['evaluated_candidates']:.0%})")
print(f"  Best CV AUC:  {ga_search.best_score_:.4f}")
print(f"  Test AUC:     {test_auc:.4f}")
print(f"  Time:         {ga_time:.1f}s")
```

```text
GASearchCV (Genetic)
  Candidates: 44 unique (cache hit rate: 26%)
  Best CV AUC:  0.9967
  Test AUC:     0.9972
  Time:         8.8s
```

### When genetic algorithms win

See also: [When to Use sklearn-genetic-opt](../guide/when-to-use) for the full guide.

- **High-dimensional spaces (7+ parameters).** The exponential blowup hurts grid search first, then random search. Genetic search compounds progress across generations.
- **Known or suspected parameter interactions.** `learning_rate` × `n_estimators`, regularization × solver — these are exactly what crossover exploits.
- **Mixed parameter types in one search.** Integers, floats, and categoricals in the same grid are handled natively.
- **You need diagnostic plots.** `GASearchCV` exposes per-generation history you can plot: convergence curves, search space heatmaps, fitness progression.
- **Long-running experiments with early stopping.** Built-in `ConsecutiveStopping` and `DeltaThreshold` callbacks terminate the search automatically when scores plateau.

---

## A Fair Benchmark

To compare all four methods fairly, we need:

1. The **same dataset and estimator**
2. The **same search space** (as faithfully as each method allows)
3. The **same evaluation budget** — not wall-clock time, but number of model fits

We use `RandomForestClassifier` on `breast_cancer`, a 4-parameter mixed space (integer, categorical, integer, continuous), and a budget of 50 evaluations.

:::warning What this benchmark does and doesn't show
`breast_cancer` is a small, relatively smooth dataset. On problems like this, all methods are expected to converge to similar scores — there is not much room to separate them. The benchmark below is intentionally honest about that. The advantage of genetic search is more pronounced on larger, more rugged spaces. See the [Benchmarks](../benchmarks/) page for results on harder problems.
:::

### Full runnable benchmark

```python
import time
import numpy as np
from scipy.stats import randint, uniform

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)

from sklearn_genetic import GASearchCV, EvolutionConfig, PopulationConfig, RuntimeConfig
from sklearn_genetic.space import Categorical, Continuous, Integer

# ── Data ──────────────────────────────────────────────────────────────────────
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
BUDGET = 50

def make_rf():
    return RandomForestClassifier(random_state=42)

def holdout_auc(search):
    return roc_auc_score(y_test, search.predict_proba(X_test)[:, 1])

# ── Grid Search ───────────────────────────────────────────────────────────────
# 3 × 3 × 3 = 27 candidates — roughly half the budget, so already cheaper
grid_param = {
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 10, None],
    "min_samples_leaf": [1, 5, 10],
}
t0 = time.perf_counter()
grid_s = GridSearchCV(make_rf(), grid_param, scoring="roc_auc", cv=cv, n_jobs=-1)
grid_s.fit(X_train, y_train)
grid_time = time.perf_counter() - t0

# ── Random Search ─────────────────────────────────────────────────────────────
rnd_dist = {
    "n_estimators": randint(50, 500),
    "max_depth": [3, 5, 8, 10, 15, None],
    "min_samples_leaf": randint(1, 20),
    "max_features": uniform(0.2, 0.8),
}
t0 = time.perf_counter()
rnd_s = RandomizedSearchCV(
    make_rf(), rnd_dist, n_iter=BUDGET, scoring="roc_auc",
    cv=cv, n_jobs=-1, random_state=42,
)
rnd_s.fit(X_train, y_train)
rnd_time = time.perf_counter() - t0

# ── GASearchCV ────────────────────────────────────────────────────────────────
ga_grid = {
    "n_estimators": Integer(50, 500),
    "max_depth": Categorical([3, 5, 8, 10, 15, None]),
    "min_samples_leaf": Integer(1, 20),
    "max_features": Continuous(0.2, 1.0),
}
t0 = time.perf_counter()
ga_s = GASearchCV(
    estimator=make_rf(),
    param_grid=ga_grid,
    cv=cv,
    scoring="roc_auc",
    evolution_config=EvolutionConfig(
        population_size=10, generations=6, elitism=True, keep_top_k=3,
    ),
    population_config=PopulationConfig(initializer="smart"),
    runtime_config=RuntimeConfig(n_jobs=-1, use_cache=True, verbose=False),
    random_state=42,
)
ga_s.fit(X_train, y_train)
ga_time = time.perf_counter() - t0

# ── Results ───────────────────────────────────────────────────────────────────
grid_n = len(grid_s.cv_results_["params"])
rnd_n  = rnd_s.n_iter
ga_n   = ga_s.fit_stats_["unique_candidates"]

print(f"{'Method':<22} {'CV AUC':>8} {'Test AUC':>10} {'Evals':>7} {'Time':>7}")
print("-" * 60)
print(f"{'GridSearchCV':<22} {grid_s.best_score_:>8.4f} {holdout_auc(grid_s):>10.4f} {grid_n:>7} {grid_time:>6.1f}s")
print(f"{'RandomizedSearchCV':<22} {rnd_s.best_score_:>8.4f} {holdout_auc(rnd_s):>10.4f} {rnd_n:>7} {rnd_time:>6.1f}s")
print(f"{'GASearchCV':<22} {ga_s.best_score_:>8.4f} {holdout_auc(ga_s):>10.4f} {ga_n:>7} {ga_time:>6.1f}s")
```

```text
Method                 CV AUC   Test AUC   Evals    Time
------------------------------------------------------------
GridSearchCV           0.9955     0.9972      27     3.2s
RandomizedSearchCV     0.9962     0.9972      50     7.1s
GASearchCV             0.9967     0.9972      44     8.8s
```

### Honest interpretation

All three methods converge to the same test AUC on this dataset. That is the expected and honest result — `breast_cancer` is a well-behaved, relatively low-dimensional problem, and 50 evaluations is enough for any method to find the global optimum region.

What this tells you:

- On small, smooth problems, **random search is the right tool** — it is faster to set up and just as effective
- `GASearchCV` evaluated *fewer* candidates (44 vs 50) due to deduplication caching, which is an efficiency advantage on larger spaces where duplicates are less common
- Grid search is the fastest here because 27 candidates is fewer than 50, but you cannot scale it beyond 3–4 parameters without manual tuning of the grid

The genetic algorithm advantage grows as the search space gets harder. See the [Benchmarks](../benchmarks/) page for the Bayesmark results where the differences become more meaningful.

---

## When Genetic Algorithms Win

### High-dimensional spaces (7+ parameters)

Every additional parameter multiplies the space a random search must cover blindly. A genetic algorithm compounds learning across generations — good parameter combinations in generation 3 seed generation 4's starting population. The advantage is structural and grows with dimensionality.

```python
# 7-parameter search: the space random search struggles to cover
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn_genetic.space import Continuous, Integer

large_param_grid = {
    "learning_rate":     Continuous(0.01, 0.3, distribution="log-uniform"),
    "max_iter":          Integer(50, 300),
    "max_depth":         Integer(2, 8),
    "min_samples_leaf":  Integer(5, 50),
    "l2_regularization": Continuous(1e-6, 1.0, distribution="log-uniform"),
    "max_features":      Continuous(0.3, 1.0),
    "max_leaf_nodes":    Integer(15, 127),
}
# 7 continuous/integer parameters — a grid is hopeless; random search samples blindly;
# a genetic algorithm compounds progress across generations
```

### Known parameter interactions

`learning_rate` × `n_estimators` in gradient boosting is the canonical example. A low learning rate needs more estimators to converge; a high one needs fewer. Random search samples each dimension independently and can stumble onto the interaction by chance. A genetic algorithm — by keeping and recombining configurations that performed well — converges on the joint region where both parameters are appropriately paired.

### Mixed parameter types

`GASearchCV` handles `Integer`, `Continuous`, and `Categorical` natively in one `param_grid`. Random search requires `scipy.stats` distributions that do not map directly to categoricals. Grid search requires explicit lists for every value of every continuous parameter.

### Diagnostic plots

`GASearchCV` records the entire evolutionary history. After fitting, you can plot:

- Convergence: best score vs generation
- Search space coverage: where the algorithm sampled
- Parameter distributions across generations: did it narrow down?

None of the sklearn searchers provide this. If you need to understand *why* the search went somewhere, the genetic algorithm's telemetry is the only tool that shows you.

### Long-running experiments with early stopping

```python
from sklearn_genetic.callbacks import ConsecutiveStopping, DeltaThreshold

callbacks = [
    ConsecutiveStopping(generations=4, metric="fitness_best"),  # stop if no improvement for 4 gens
    DeltaThreshold(threshold=0.001, generations=6, metric="fitness_best"),
]

ga_search.fit(X_train, y_train, callbacks=callbacks)
print(f"Stopped at generation {len(ga_search.history)} of {ga_search.evolution_config.generations} max")
```

---

## When Genetic Algorithms Lose

This section matters. Knowing the failure modes saves you from adding unnecessary complexity.

### 1–2 parameter searches

A genetic algorithm needs several generations to build useful population-level signals. With one or two parameters, random search covers the space thoroughly in 20–30 evaluations. The GA's overhead (population management, crossover, mutation bookkeeping) produces no benefit — you are paying for population dynamics that never kick in.

**Rule of thumb:** fewer than 4 parameters → use `RandomizedSearchCV`.

### Very fast evaluations with a large budget

If each cross-validation call takes less than a second and you can afford 500 evaluations, random search will explore the space thoroughly by sheer volume. The GA's per-generation coordination overhead becomes a relative cost without a relative benefit.

**Rule of thumb:** evaluation time < 1 s and budget > 200 → random search likely matches the GA.

### Small, fully discrete grids

If your entire search space is 24 candidate configurations, you can enumerate them with `GridSearchCV` and know you found the best one. There is no exploration problem for an evolutionary algorithm to solve.

**Rule of thumb:** total candidate count < 50 → use `GridSearchCV` or `RandomizedSearchCV`.

### Combined algorithm selection + hyperparameter tuning (CASH)

This is the most important "when not to use" finding, documented in our [Benchmarks](../benchmarks/) page:

:::warning CASH: use Optuna, not GASearchCV
When you want to select *which estimator family to use* and tune its parameters at the same time (CASH), a define-by-run Bayesian optimizer like Optuna is structurally better suited. The reason: `GASearchCV` uses a flat parameter encoding — every gene is present in every individual, even when most genes are inactive because a different estimator family was selected. Crossover between two individuals that chose different families mostly recombines inactive genes, producing little useful signal.

Optuna's define-by-run API models only the *active* parameters of the chosen family, which is the right representation for conditional spaces.

Result from our own CASH benchmark (budget = 184 evaluations):

| Optimizer | Mean accuracy | Best |
|-----------|--------------|------|
| Optuna | **0.8636 ± 0.0050** | 0.8707 |
| RandomizedSearchCV | 0.8591 ± 0.0055 | 0.8667 |
| GASearchCV | 0.8582 ± 0.0087 | 0.8693 |

`GASearchCV` finished last. We published this result because it is honest and useful.
:::

### Neural architecture search

Tools like Optuna (with `optuna-integration`), Ray Tune, or KerasTuner are designed for this use case. They handle conditional architecture choices (number of layers, layer sizes, skip connections) and integrate with deep learning training loops natively. A sklearn-compatible genetic search is the wrong level of abstraction here.

---

## Practical Recommendation

Follow this decision sequence:

**Step 1: Start with `RandomizedSearchCV`.**
It is fast, requires no extra dependencies, and is a famously strong baseline on smooth spaces. If it finds a good solution quickly, you are done.

**Step 2: Switch to `GASearchCV` when you hit these conditions:**
- 5 or more parameters in your search space
- You suspect interactions between parameters (e.g., regularization strength × solver choice, learning rate × number of estimators)
- Your search space contains a mix of integers, floats, and categoricals
- You want convergence plots and per-generation telemetry
- Evaluations take 10–60 seconds each and you can run for 50–200 generations

**Step 3: Switch to Bayesian (Optuna) when:**
- Individual evaluations take more than a few minutes — the surrogate model overhead pays off only at this cost
- You are doing combined algorithm selection (CASH) — Optuna's conditional API is the right tool
- You need to run fewer than 30 total evaluations and need each one to count

:::tip You do not have to pick one
A common pattern: run `RandomizedSearchCV` with 30 evaluations as a quick baseline, then warm-start a `GASearchCV` run focused on the region the random search found promising. The two approaches are complementary.
:::

---

## See Also

- [When to Use sklearn-genetic-opt](../guide/when-to-use) — the concise decision guide with a 7-parameter example
- [Benchmarks](../benchmarks/) — full Bayesmark results: honest wins and losses across multiple datasets and optimizers
- [Comparing Search Methods (Example)](../examples/sklearn-comparison) — fully reproducible side-by-side code with output
- [How Hyperparameter Optimization Works](../guide/how-hyperparameter-optimization-works) — conceptual introduction to all four strategies
- [Advanced Optimizer Control](../guide/advanced-optimizer-control) — diversity control, fitness sharing, local search, and adaptive schedules that give `GASearchCV` its edge on harder problems
- [Callbacks](../guide/callbacks) — early stopping and convergence detection
