---
title: "How Hyperparameter Optimization Works: A Complete Guide"
description: "Understand the four main hyperparameter optimization strategies — Grid Search, Random Search, Bayesian Optimization, and Evolutionary/Genetic Search — with Python examples and practical guidance on when to use each."
---

# How Hyperparameter Optimization Works: A Complete Guide

Every machine learning model has settings you choose before training begins — the number of trees in a forest, how fast a network learns, how deeply a decision tree can grow. Getting these settings right can be the difference between a mediocre model and a great one. This guide explains what hyperparameters are, why they matter, and how the four main strategies for tuning them work — from the simplest grid search to evolutionary algorithms that mimic natural selection.

**Estimated reading time:** 12 min &nbsp;|&nbsp; **Difficulty:** Beginner

:::info Prerequisites
- Python installed with `scikit-learn` (`pip install scikit-learn`)
- Basic familiarity with `fit` / `predict` in scikit-learn
- No prior knowledge of optimization theory required
:::

---

## What Are Hyperparameters?

Machine learning models have two kinds of values:

**Parameters** are learned automatically during training. A neural network's weights, a linear model's coefficients, a decision tree's split thresholds — the training algorithm adjusts these to fit your data. You never set them by hand.

**Hyperparameters** are set by *you* before training begins. They control *how* the algorithm learns, not *what* it learns. Common examples:

| Hyperparameter | Model | What it controls |
|---|---|---|
| `n_estimators` | Random Forest | How many trees to build |
| `max_depth` | Decision Tree / XGBoost | How deep each tree can grow |
| `learning_rate` | Gradient Boosting / Neural Networks | How large each update step is |
| `C` | SVM / Logistic Regression | Regularization strength |
| `alpha` | Ridge / Neural Networks | Penalty on large weights |
| `kernel` | SVM | The shape of the decision boundary |

### Why the right hyperparameters matter

Consider a Random Forest trained on the `breast_cancer` dataset. The default settings give reasonable results, but the right hyperparameters can push accuracy noticeably higher:

```python
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import numpy as np

X, y = load_breast_cancer(return_X_y=True)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Default hyperparameters
default_rf = RandomForestClassifier(random_state=42)
default_scores = cross_val_score(default_rf, X, y, cv=cv, scoring="roc_auc")
print(f"Default ROC-AUC:  {default_scores.mean():.4f} ± {default_scores.std():.4f}")

# Poorly chosen hyperparameters
poor_rf = RandomForestClassifier(
    n_estimators=5,      # too few trees — high variance
    max_depth=1,         # too shallow — underfits
    random_state=42
)
poor_scores = cross_val_score(poor_rf, X, y, cv=cv, scoring="roc_auc")
print(f"Poor params ROC-AUC: {poor_scores.mean():.4f} ± {poor_scores.std():.4f}")

# Well-chosen hyperparameters
good_rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_leaf=3,
    max_features=0.5,
    random_state=42
)
good_scores = cross_val_score(good_rf, X, y, cv=cv, scoring="roc_auc")
print(f"Good params ROC-AUC: {good_scores.mean():.4f} ± {good_scores.std():.4f}")
```

```text
Default ROC-AUC:      0.9934 ± 0.0035
Poor params ROC-AUC:  0.9197 ± 0.0195
Good params ROC-AUC:  0.9963 ± 0.0022
```

The gap between poor and good hyperparameters (~7.7 pp ROC-AUC) is larger than most architectural improvements you could make. On real-world problems with noisy data and complex interactions, this gap widens further.

:::tip The fundamental challenge
You cannot know the best hyperparameters in advance — they depend on your dataset, your model, and your objective. The only way to find them is to *try configurations and measure their performance*. Hyperparameter optimization is the systematic approach to doing that efficiently.
:::

---

## The Four Main Optimization Strategies

All four strategies share the same loop: propose a configuration, evaluate it with cross-validation, record the score, repeat. They differ in *how* they propose the next configuration.

### 1. Grid Search — Exhaustive but Explosive

Grid search defines a fixed set of values for each hyperparameter and evaluates every possible combination. With 3 values for each of 4 parameters, that's 3⁴ = 81 evaluations. Add a fifth parameter with 3 values: 243 evaluations. Add a sixth: 729.

```
┌─────────────────────────────────────────┐
│         Grid Search                     │
│                                         │
│  n_estimators: [50, 100, 200]           │
│  max_depth:    [3, 5, 8]               │
│                                         │
│  . . .   ← evaluates all 9 combos      │
│  . . .                                  │
│  . . .                                  │
└─────────────────────────────────────────┘
```

**Strengths:** Guaranteed to find the best configuration within your grid. Completely reproducible. Simple to reason about.

**Weaknesses:** The number of evaluations multiplies with each new parameter — this is the *curse of dimensionality*. With continuous parameters, any fixed grid leaves most of the space unsampled.

**Best for:** Small search spaces with 2–3 discrete parameters and a small number of values per parameter.

### 2. Random Search — Surprisingly Effective

Instead of every combination, random search samples hyperparameter values randomly. Bergstra & Bengio (2012) showed that for most problems — where only a few parameters really matter — random search finds equally good or better results than grid search with far fewer evaluations.

```
┌─────────────────────────────────────────┐
│         Random Search                   │
│                                         │
│  n_estimators: Uniform(10, 500)         │
│  max_depth:    Uniform(1, 20)           │
│                                         │
│    ·     ·      ← random samples        │
│  ·    ·    ·                            │
│     ·    ·                              │
└─────────────────────────────────────────┘
```

The insight is that **most parameters have little effect**. If only 2 of your 6 parameters matter, a grid wastes 4/6 of its resolution on irrelevant dimensions. Random search dedicates more of its budget to varying the parameters that actually count.

**Strengths:** Scales to many parameters. Works well with continuous distributions. Simple to implement.

**Weaknesses:** Each trial is independent — it learns nothing from previous trials. Misses interactions between parameters (more on this below).

**Best for:** Large search spaces with 3–6 parameters, especially when you don't know which parameters matter most.

### 3. Bayesian Optimization — Learning From History

Bayesian optimization is the idea that previous evaluations carry information about the fitness landscape. Instead of sampling blindly, it fits a *surrogate model* — typically a Gaussian Process or Tree Parzen Estimator — to the scores observed so far, then uses that model to predict which unexplored configurations are most promising.

```
┌─────────────────────────────────────────┐
│         Bayesian Optimization           │
│                                         │
│  Trial 1: random         score = 0.91   │
│  Trial 2: random         score = 0.94   │
│  Fit surrogate model                    │
│  Trial 3: model says try here → 0.96   │
│  Update model                           │
│  Trial 4: model says try here → 0.97   │
│  ...                                    │
└─────────────────────────────────────────┘
```

The surrogate model balances **exploitation** (trying configurations near known good points) and **exploration** (trying poorly-sampled regions that might be even better). This trade-off is controlled by an *acquisition function* such as Expected Improvement.

**Strengths:** Data-efficient — typically finds good solutions in fewer evaluations than random search. Principled uncertainty quantification.

**Weaknesses:** The surrogate model itself has overhead. Becomes expensive in very high dimensions. Typically sequential (each trial depends on results so far), though parallel variants exist.

**Best for:** Problems where each evaluation is expensive (minutes to hours), and you have a budget of 20–100 total evaluations.

### 4. Genetic / Evolutionary Search — Population-Based

Genetic algorithms are inspired by natural selection. Instead of evaluating one configuration at a time, they maintain a *population* of configurations and evolve it over multiple *generations*.

```
Generation 0 (random population):
  [lr=0.1, n=100, depth=5]  → 0.91
  [lr=0.3, n=50,  depth=3]  → 0.88
  [lr=0.01,n=200, depth=8]  → 0.93
  [lr=0.05,n=150, depth=6]  → 0.95   ← best

Generation 1 (select, crossover, mutate):
  [lr=0.05,n=150, depth=6]  → kept (elitism)
  [lr=0.1, n=150, depth=6]  → crossover of best + 1st
  [lr=0.05,n=200, depth=7]  → crossover + mutation
  [lr=0.04,n=160, depth=6]  → mutation of best
```

Each generation:
1. **Selection** — configurations with higher scores are more likely to be chosen as parents.
2. **Crossover** — two parent configurations exchange parts of their values to produce offspring.
3. **Mutation** — random small changes are applied to introduce diversity and avoid local optima.
4. **Elitism** — the best configurations survive unchanged to the next generation.

**Strengths:** Naturally captures interactions between parameters (because it recombines *complete configurations*, not individual values). Handles mixed parameter types (integers, floats, categories). Parallelizes naturally within a generation.

**Weaknesses:** Adds overhead on trivially small search spaces. Requires tuning the evolutionary parameters themselves (population size, mutation rate).

**Best for:** Large, mixed search spaces with 5+ parameters, especially when you expect parameter interactions.

---

## Why Parameter Interactions Matter

Grid search and random search share a hidden assumption: **parameters are independent**. They sample each parameter as if the others were fixed. In practice, this is almost never true.

Consider `learning_rate` and `n_estimators` in gradient boosting. A low learning rate needs many estimators to converge; a high learning rate needs fewer. The optimal `learning_rate` *depends on* `n_estimators`. If you search them independently, you will never find the sweet spot where both are jointly tuned.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import numpy as np

X, y = load_breast_cancer(return_X_y=True)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Mismatch: high learning rate + many estimators → overfits
high_lr_many_trees = GradientBoostingClassifier(
    learning_rate=0.3, n_estimators=300, max_depth=4, random_state=42
)
score_mismatch_a = cross_val_score(
    high_lr_many_trees, X, y, cv=cv, scoring="roc_auc"
).mean()

# Mismatch: low learning rate + few estimators → underfits
low_lr_few_trees = GradientBoostingClassifier(
    learning_rate=0.005, n_estimators=10, max_depth=4, random_state=42
)
score_mismatch_b = cross_val_score(
    low_lr_few_trees, X, y, cv=cv, scoring="roc_auc"
).mean()

# Interaction: low learning rate + many estimators → well-tuned
low_lr_many_trees = GradientBoostingClassifier(
    learning_rate=0.05, n_estimators=300, max_depth=3, random_state=42
)
score_interaction = cross_val_score(
    low_lr_many_trees, X, y, cv=cv, scoring="roc_auc"
).mean()

print(f"High LR + many trees (overfits):  {score_mismatch_a:.4f}")
print(f"Low LR  + few trees  (underfits): {score_mismatch_b:.4f}")
print(f"Low LR  + many trees (balanced):  {score_interaction:.4f}")
```

```text
High LR + many trees (overfits):  0.9926
Low LR  + few trees  (underfits): 0.9451
Low LR  + many trees (balanced):  0.9975
```

:::info Why genetic search handles this better
Grid search evaluates the same value of `learning_rate` paired with every value of `n_estimators` — it cannot prefer one pairing over another. Random search samples them independently. A genetic algorithm recombines *complete configurations* that performed well: if `(lr=0.05, n=300)` is good, its offspring will inherit both values together, so the algorithm naturally discovers and preserves beneficial interactions.
:::

---

## Choosing the Right Method

Use this table as a starting point. Overlap is normal — multiple methods may be appropriate, and the best choice often comes down to your evaluation budget.

| | Grid Search | Random Search | Bayesian Opt. | Genetic Search |
|---|:---:|:---:|:---:|:---:|
| **Parameters** | ≤ 3 | 3 – 6 | 3 – 10 | 5+ |
| **Evaluation budget** | Any | Any | 20 – 100 | 50 – 300 |
| **Captures interactions** | No | No | Partially | Yes |
| **Continuous parameters** | Poor | Good | Good | Good |
| **Mixed types** | OK | Good | Varies | Good |
| **Parallel evaluations** | Yes | Yes | Limited | Yes |
| **Implementation complexity** | Low | Low | Medium | Medium |

### Rules of thumb

:::tip Grid Search
Use when you have 2–3 discrete parameters and a small number of meaningful values for each. Perfect for final fine-grained search once other methods have narrowed the space.
:::

:::tip Random Search
Use when you have 3–6 parameters and do not know which ones matter. Good default choice for an initial broad search.
:::

:::tip Bayesian Optimization
Use when each model evaluation is expensive (minutes or more) and you have a budget under 100 evaluations. The surrogate model pays for itself only when evaluations are costly.
:::

:::tip Genetic / Evolutionary Search
Use when you have 5+ parameters, suspect interactions, or have mixed parameter types. Also a strong choice when you can run many evaluations in parallel.
:::

---

## Python Examples

All three examples tune the same `RandomForestClassifier` on the same `digits` dataset so you can compare them fairly.

### Setup (shared across all examples)

```python
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples:     {X_test.shape[0]}")
print(f"Classes:          {len(np.unique(y))}")
```

```text
Training samples: 1347
Test samples:     450
Classes:          10
```

### Grid Search

```python
import time
from sklearn.model_selection import GridSearchCV

param_grid_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 10, None],
    "min_samples_leaf": [1, 3],
}

# 3 × 3 × 2 = 18 combinations × 3 folds = 54 model fits
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid_grid,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1,
)

t0 = time.perf_counter()
grid_search.fit(X_train, y_train)
grid_time = time.perf_counter() - t0

y_pred_grid = grid_search.predict(X_test)
print(f"Grid Search")
print(f"  Candidates evaluated: {len(grid_search.cv_results_['mean_test_score'])}")
print(f"  Best CV accuracy:     {grid_search.best_score_:.4f}")
print(f"  Test accuracy:        {accuracy_score(y_test, y_pred_grid):.4f}")
print(f"  Time:                 {grid_time:.1f}s")
print(f"  Best params:          {grid_search.best_params_}")
```

```text
Grid Search
  Candidates evaluated: 18
  Best CV accuracy:     0.9718
  Test accuracy:        0.9778
  Time:                 4.3s
  Best params:          {'max_depth': None, 'min_samples_leaf': 1, 'n_estimators': 200}
```

### Random Search

Random search lets us sample continuous distributions and explore a much larger space in the same number of evaluations.

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

param_dist_random = {
    "n_estimators": randint(50, 500),
    "max_depth": [3, 5, 8, 10, 15, None],
    "min_samples_leaf": randint(1, 20),
    "max_features": uniform(0.2, 0.8),   # continuous: 0.2 to 1.0
}

random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_dist_random,
    n_iter=30,        # same budget ceiling as genetic example below
    cv=cv,
    scoring="accuracy",
    n_jobs=-1,
    random_state=42,
)

t0 = time.perf_counter()
random_search.fit(X_train, y_train)
random_time = time.perf_counter() - t0

y_pred_random = random_search.predict(X_test)
print(f"Random Search")
print(f"  Candidates evaluated: {len(random_search.cv_results_['mean_test_score'])}")
print(f"  Best CV accuracy:     {random_search.best_score_:.4f}")
print(f"  Test accuracy:        {accuracy_score(y_test, y_pred_random):.4f}")
print(f"  Time:                 {random_time:.1f}s")
print(f"  Best params:          {random_search.best_params_}")
```

```text
Random Search
  Candidates evaluated: 30
  Best CV accuracy:     0.9762
  Test accuracy:        0.9800
  Time:                 8.2s
  Best params:          {'max_depth': None, 'max_features': 0.6243, 'min_samples_leaf': 1, 'n_estimators': 387}
```

### Genetic Search with GASearchCV

`GASearchCV` from `sklearn-genetic-opt` uses evolutionary search. The same 4 parameters are searched over a richer space, and the algorithm recombines well-performing configurations across generations.

```python
from sklearn_genetic import GASearchCV, EvolutionConfig, PopulationConfig, RuntimeConfig
from sklearn_genetic.space import Categorical, Continuous, Integer

param_grid_ga = {
    "n_estimators": Integer(50, 500),
    "max_depth": Categorical([3, 5, 8, 10, 15, None]),
    "min_samples_leaf": Integer(1, 20),
    "max_features": Continuous(0.2, 1.0),
}

ga_search = GASearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid_ga,
    cv=cv,
    scoring="accuracy",
    evolution_config=EvolutionConfig(
        population_size=10,
        generations=5,    # 10 × 5 = ~50 evaluations, cached duplicates skipped
    ),
    population_config=PopulationConfig(initializer="smart"),
    runtime_config=RuntimeConfig(n_jobs=-1, verbose=False),
    random_state=42,
)

t0 = time.perf_counter()
ga_search.fit(X_train, y_train)
ga_time = time.perf_counter() - t0

y_pred_ga = ga_search.predict(X_test)
print(f"Genetic Search (GASearchCV)")
print(f"  Candidates evaluated: {ga_search.fit_stats_['evaluated_candidates']}")
print(f"  Best CV accuracy:     {ga_search.best_score_:.4f}")
print(f"  Test accuracy:        {accuracy_score(y_test, y_pred_ga):.4f}")
print(f"  Time:                 {ga_time:.1f}s")
print(f"  Best params:          {ga_search.best_params_}")
```

```text
Genetic Search (GASearchCV)
  Candidates evaluated: 48
  Best CV accuracy:     0.9792
  Test accuracy:        0.9822
  Time:                 9.7s
  Best params:          {'n_estimators': 423, 'max_depth': None, 'min_samples_leaf': 1, 'max_features': 0.5831}
```

### Results comparison

```python
print(f"\n{'Method':<20} {'CV Accuracy':>12} {'Test Accuracy':>14} {'Evaluations':>12}")
print("-" * 62)
print(f"{'Grid Search':<20} {grid_search.best_score_:>12.4f} "
      f"{accuracy_score(y_test, y_pred_grid):>14.4f} "
      f"{len(grid_search.cv_results_['mean_test_score']):>12}")
print(f"{'Random Search':<20} {random_search.best_score_:>12.4f} "
      f"{accuracy_score(y_test, y_pred_random):>14.4f} "
      f"{len(random_search.cv_results_['mean_test_score']):>12}")
print(f"{'Genetic Search':<20} {ga_search.best_score_:>12.4f} "
      f"{accuracy_score(y_test, y_pred_ga):>14.4f} "
      f"{ga_search.fit_stats_['evaluated_candidates']:>12}")
```

```text
Method               CV Accuracy  Test Accuracy  Evaluations
--------------------------------------------------------------
Grid Search               0.9718         0.9778           18
Random Search             0.9762         0.9800           30
Genetic Search            0.9792         0.9822           48
```

:::info Interpreting these results
No single method wins on every dataset. On this example the genetic search finds a slightly better configuration, but with more evaluations. The advantage of genetic search grows on datasets with stronger parameter interactions and larger search spaces — the digits dataset with 4 parameters is intentionally simple to keep these examples fast to run.
:::

---

## Search Budget vs Quality

Every evaluation costs time. The relationship between budget and quality follows a rough pattern: early evaluations produce big gains; later evaluations produce diminishing returns.

```python
import matplotlib.pyplot as plt
import pandas as pd

# Use the random search results to illustrate budget vs quality
results = pd.DataFrame(random_search.cv_results_)
results = results.sort_values("rank_test_score")

# Running best score as we add more evaluations (in evaluation order)
scores_in_order = results.sort_index()["mean_test_score"].values
running_best = np.maximum.accumulate(scores_in_order)

plt.figure(figsize=(8, 4))
plt.plot(range(1, len(running_best) + 1), running_best, marker="o", markersize=4)
plt.axhline(running_best[-1], color="gray", linestyle="--", alpha=0.5, label="Final best")
plt.xlabel("Evaluations completed")
plt.ylabel("Best CV accuracy so far")
plt.title("Diminishing returns: budget vs quality (Random Search, 30 trials)")
plt.legend()
plt.tight_layout()
plt.savefig("budget_vs_quality.png", dpi=100)
plt.show()

print(f"After  5 evaluations: {running_best[4]:.4f}")
print(f"After 10 evaluations: {running_best[9]:.4f}")
print(f"After 20 evaluations: {running_best[19]:.4f}")
print(f"After 30 evaluations: {running_best[29]:.4f}")
```

```text
After  5 evaluations: 0.9725
After 10 evaluations: 0.9740
After 20 evaluations: 0.9762
After 30 evaluations: 0.9762
```

The first 10 evaluations recovered most of the gain. The next 20 added very little. This pattern argues for **early stopping**: once scores plateau across several iterations, additional evaluations rarely help.

### Early stopping with GASearchCV

`sklearn-genetic-opt` has built-in convergence detection. You can stop the search automatically when the score stops improving:

```python
from sklearn_genetic.callbacks import ConsecutiveStopping

# Stop if the best score doesn't improve for 3 consecutive generations
early_stop = ConsecutiveStopping(generations=3, metric="fitness_best")

ga_search_with_stop = GASearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid_ga,
    cv=cv,
    scoring="accuracy",
    evolution_config=EvolutionConfig(population_size=10, generations=20),  # ceiling
    population_config=PopulationConfig(initializer="smart"),
    runtime_config=RuntimeConfig(n_jobs=-1, verbose=False),
    random_state=42,
)

ga_search_with_stop.fit(X_train, y_train, callbacks=early_stop)
print(f"Stopped at generation: {len(ga_search_with_stop.history)}")
print(f"Best CV accuracy:      {ga_search_with_stop.best_score_:.4f}")
```

```text
Stopped at generation: 6
Best CV accuracy:      0.9792
```

The search stopped 14 generations early once it confirmed the score was no longer improving.

---

## Common Pitfalls

### Overfitting to the validation set

Every time you evaluate a configuration, you use your validation data to measure it. If you evaluate enough configurations on the same validation set, you will eventually find one that looks great by chance — even though it won't generalize. This is called **hyperparameter overfitting** (or evaluation set overfitting).

**What to do:** Always hold out a separate test set and evaluate your final model on it *once*. Use cross-validation (multiple validation folds) rather than a single train/validation split. The more evaluations you run, the more conservative you should be about trusting your validation score.

### Data leakage in hyperparameter search

If you preprocess your data (scale, encode, impute) *before* creating train/validation splits, information from the validation folds leaks into training. This inflates your cross-validation score and produces a model that appears better than it is.

**What to do:** Always wrap preprocessing in a `Pipeline` and pass the pipeline as the estimator to your search object. That way, the preprocessing is fit only on the training fold and applied to the validation fold.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Wrong: scaling before CV leaks validation statistics into training
# X_scaled = StandardScaler().fit_transform(X)   # <-- do NOT do this
# RandomizedSearchCV(SVC(), ...).fit(X_scaled, y)

# Right: scaling inside a pipeline, refitted per fold
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC()),
])

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform

param_dist = {
    "svc__C": loguniform(1e-3, 1e3),
    "svc__gamma": loguniform(1e-4, 1e1),
    "svc__kernel": ["rbf", "poly"],
}

# Each fold: scaler fits on train, transforms both train and val
search = RandomizedSearchCV(pipeline, param_dist, n_iter=20, cv=3,
                            scoring="accuracy", n_jobs=-1, random_state=42)
```

### Ignoring parameter interactions

Tuning `learning_rate` with everything else fixed, then tuning `n_estimators` with everything else fixed, does not give you the jointly optimal combination. If you must tune one parameter at a time, restart the process with the newly found value and iterate — but a joint search is almost always better.

### Not setting `random_state` for reproducibility

Both `RandomizedSearchCV` and `GASearchCV` are stochastic — they sample randomly. Without a fixed seed, two runs with identical code can find different best parameters. Always set `random_state` on both the search object and the estimator.

```python
# Always do this — set random_state everywhere
rf = RandomForestClassifier(random_state=42)  # estimator seed
search = RandomizedSearchCV(rf, param_dist, random_state=42)  # search seed
```

:::details What if my results are still not reproducible?
Even with `random_state` set, results can vary if you use `n_jobs > 1` on some platforms. Parallel execution order is not deterministic on all operating systems. If exact reproducibility matters more than speed, set `n_jobs=1`. See [Reproducibility](./reproducibility) for a full discussion.
:::

---

## Next Steps

You now understand the landscape of hyperparameter optimization. Depending on what you want to do next:

- [When to Use Genetic Algorithm Search](./when-to-use) — a deeper look at when a genetic approach wins over random or grid search, with a 7-parameter real-world example
- [Getting Started with GASearchCV](./basic-usage) — your first genetic search in 10 minutes, with full runnable code
- [Grid Search vs Genetic Algorithms: Benchmark](../comparisons/grid-search-vs-genetic-algorithms) — head-to-head benchmark on larger search spaces
- [Random Forest Hyperparameter Tuning](../tutorials/tune-random-forest) — a comprehensive step-by-step tutorial for Random Forests specifically
- [XGBoost Hyperparameter Tuning](../tutorials/tune-xgboost) — tuning 9 interacting XGBoost parameters with `GASearchCV`
