---
title: "Common Hyperparameter Tuning Mistakes (and How to Avoid Them)"
description: "The 10 most common mistakes in hyperparameter optimization — overfitting the validation set, data leakage, wrong metrics, bad search spaces — with concrete fixes."
---

# Common Hyperparameter Tuning Mistakes (and How to Avoid Them)

Hyperparameter tuning has a lot of ways to go wrong quietly — searches that appear to be working, but produce models that don't generalize. This page walks through the ten most common mistakes, explains why each one hurts, and shows the fix with runnable Python code.

**Estimated reading time:** 10 min &nbsp;|&nbsp; **Difficulty:** Intermediate

:::info Prerequisites
- Basic familiarity with scikit-learn's `fit` / `predict` API
- Completed [Basic Usage](./basic-usage) or equivalent experience with `GridSearchCV` / `RandomizedSearchCV`
:::

---

## Mistake 1: Overfitting to the Validation Set

Every time you evaluate a hyperparameter configuration, you're implicitly using the validation set to select it. Evaluate enough configurations on the same validation set and you will eventually find one that looks great by chance — even if it won't generalize to new data.

The problem grows with the number of evaluations. A search that tries 200 configurations has 200 chances to get lucky on the same validation fold.

**The fix:** Hold out a separate test set that is never touched during the search. Use cross-validation (multiple validation folds) to reduce variance in each estimate.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score

from sklearn_genetic import EvolutionConfig, GASearchCV, PopulationConfig, RuntimeConfig
from sklearn_genetic.space import Categorical, Continuous, Integer

X, y = load_breast_cancer(return_X_y=True)

# Split BEFORE any tuning: train is used for search, test is held out
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

# Cross-validation runs entirely on X_train — X_test is never seen
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

search = GASearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid={
        "n_estimators": Integer(50, 300),
        "max_depth": Integer(2, 12),
        "max_features": Categorical(["sqrt", "log2"]),
    },
    cv=cv,
    scoring="roc_auc",
    evolution_config=EvolutionConfig(population_size=15, generations=8),
    population_config=PopulationConfig(initializer="smart"),
    runtime_config=RuntimeConfig(n_jobs=-1, verbose=False),
    random_state=42,
)

search.fit(X_train, y_train)

# Report both — expect a small gap
cv_score = search.best_score_
test_score = roc_auc_score(y_test, search.predict_proba(X_test)[:, 1])
print(f"Best CV ROC-AUC  : {cv_score:.4f}")
print(f"Holdout ROC-AUC  : {test_score:.4f}")
print(f"Optimism gap     : {cv_score - test_score:+.4f}")
```

```text
Best CV ROC-AUC  : 0.9963
Holdout ROC-AUC  : 0.9930
Optimism gap     : +0.0033
```

A small positive gap is normal and expected. A large gap (> 0.02–0.05 on typical problems) suggests overfitting to the validation set.

:::warning Never use the test set during hyperparameter search
If you look at test-set performance to decide which configurations to try next, you have turned the test set into a validation set. You will no longer have an unbiased estimate of generalization performance. Reserve the test set for a single final evaluation.
:::

---

## Mistake 2: Data Leakage Through the Pipeline

The most common form of data leakage in hyperparameter search happens when you preprocess your data — scale, impute, encode — before creating cross-validation splits. The result: preprocessing statistics (mean, variance, category frequencies) are computed using information from the validation fold. The model then benefits from that information during training, inflating the cross-validation score.

```python
# BAD — scaler sees the entire dataset including validation folds
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score, StratifiedKFold
import numpy as np

X, y = load_breast_cancer(return_X_y=True)

scaler = StandardScaler()
X_scaled_all = scaler.fit_transform(X)  # leaks validation statistics

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores_leaked = cross_val_score(SVC(), X_scaled_all, y, cv=cv, scoring="accuracy")
print(f"Leaked CV accuracy:  {scores_leaked.mean():.4f} ± {scores_leaked.std():.4f}")
```

```text
Leaked CV accuracy:  0.9790 ± 0.0093
```

```python
# GOOD — scaler is inside a Pipeline, fitted per fold on training data only
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC()),
])

scores_clean = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
print(f"Clean CV accuracy:   {scores_clean.mean():.4f} ± {scores_clean.std():.4f}")
```

```text
Clean CV accuracy:   0.9789 ± 0.0096
```

On this well-separated dataset the gap is small, but on high-dimensional or small-sample datasets it can be several percentage points.

:::danger Leakage inflates your CV score silently
Leakage inflates the CV score without you knowing — the model scores higher on the validation fold than it would on truly unseen data. The only reliable fix is to ensure all preprocessing lives inside a `Pipeline` so it is refit from scratch on each training fold.
:::

When using `GASearchCV`, pass the pipeline as the estimator and use `step__param` naming in `param_grid`:

```python
from sklearn_genetic import GASearchCV, EvolutionConfig, PopulationConfig, RuntimeConfig
from sklearn_genetic.space import Continuous

search = GASearchCV(
    estimator=pipeline,   # the pipeline, not the raw SVC
    param_grid={
        "svc__C": Continuous(0.01, 100.0, distribution="log-uniform"),
        "svc__gamma": Continuous(1e-4, 1.0, distribution="log-uniform"),
    },
    cv=cv,
    scoring="accuracy",
    evolution_config=EvolutionConfig(population_size=10, generations=6),
    population_config=PopulationConfig(initializer="smart"),
    runtime_config=RuntimeConfig(n_jobs=-1, verbose=False),
    random_state=42,
)
search.fit(X, y)
print(f"Best CV accuracy: {search.best_score_:.4f}")
print(f"Best params:      {search.best_params_}")
```

```text
Best CV accuracy: 0.9842
Best params:      {'svc__C': 12.47, 'svc__gamma': 0.0023}
```

See [Pipeline Tuning](./pipeline-tuning) for a full walkthrough.

---

## Mistake 3: Ignoring Class Imbalance in Scoring

Using `accuracy` as the metric on an imbalanced dataset gives a misleadingly high score. A classifier that predicts the majority class for every sample achieves (100 − minority_fraction)% accuracy while being completely useless.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
import numpy as np

X, y = load_breast_cancer(return_X_y=True)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Class distribution
unique, counts = np.unique(y, return_counts=True)
print("Class distribution:")
for cls, cnt in zip(unique, counts):
    print(f"  class {cls}: {cnt} samples ({cnt/len(y)*100:.1f}%)")

# A classifier that always predicts the majority class
majority_clf = DummyClassifier(strategy="most_frequent")
acc_dummy = cross_val_score(majority_clf, X, y, cv=cv, scoring="accuracy").mean()
bal_dummy = cross_val_score(majority_clf, X, y, cv=cv, scoring="balanced_accuracy").mean()
print(f"\nDummy classifier  — accuracy: {acc_dummy:.4f}  balanced_accuracy: {bal_dummy:.4f}")

# A real classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
acc_rf = cross_val_score(rf, X, y, cv=cv, scoring="accuracy").mean()
bal_rf = cross_val_score(rf, X, y, cv=cv, scoring="balanced_accuracy").mean()
roc_rf = cross_val_score(rf, X, y, cv=cv, scoring="roc_auc").mean()
print(f"Random Forest     — accuracy: {acc_rf:.4f}  balanced_accuracy: {bal_rf:.4f}  roc_auc: {roc_rf:.4f}")
```

```text
Class distribution:
  class 0: 212 samples (37.3%)
  class 1: 357 samples (62.7%)

Dummy classifier  — accuracy: 0.6274  balanced_accuracy: 0.5000
Random Forest     — accuracy: 0.9578  balanced_accuracy: 0.9477  roc_auc: 0.9934
```

When optimizing with `GASearchCV`, choose the scoring metric that reflects the actual business objective. On imbalanced datasets, prefer `balanced_accuracy`, `roc_auc`, or `f1`:

```python
from sklearn_genetic import GASearchCV, EvolutionConfig, PopulationConfig, RuntimeConfig
from sklearn_genetic.space import Integer, Categorical

# Use roc_auc instead of accuracy for imbalanced classification
search = GASearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid={
        "n_estimators": Integer(50, 300),
        "max_depth": Integer(2, 12),
        "class_weight": Categorical([None, "balanced"]),
    },
    cv=cv,
    scoring="roc_auc",   # not "accuracy"
    evolution_config=EvolutionConfig(population_size=12, generations=6),
    population_config=PopulationConfig(initializer="smart"),
    runtime_config=RuntimeConfig(n_jobs=-1, verbose=False),
    random_state=42,
)
search.fit(X, y)
print(f"Best ROC-AUC: {search.best_score_:.4f}")
print(f"Best params:  {search.best_params_}")
```

```text
Best ROC-AUC: 0.9961
Best params:  {'n_estimators': 218, 'max_depth': 10, 'class_weight': None}
```

:::tip Recommended metrics by task type
| Task | Preferred metrics |
|------|------------------|
| Balanced classification | `accuracy`, `roc_auc` |
| Imbalanced classification | `balanced_accuracy`, `roc_auc`, `f1` |
| Regression | `neg_root_mean_squared_error`, `r2`, `neg_mean_absolute_error` |
| Ranking / probability | `roc_auc`, `average_precision` |
:::

---

## Mistake 4: Search Spaces That Are Too Wide or Too Narrow

A search space that is too wide wastes most of your budget exploring regions where no good configuration exists. A space that is too narrow may not contain the true optimum at all.

```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

X, y = load_breast_cancer(return_X_y=True)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Demonstrate: bad C values vs good ones for LogisticRegression
for C_val in [0.00001, 0.001, 0.1, 1.0, 10.0, 1000.0]:
    score = cross_val_score(
        LogisticRegression(C=C_val, max_iter=1000, random_state=42),
        X, y, cv=cv, scoring="roc_auc"
    ).mean()
    print(f"  C={C_val:>10}  ROC-AUC={score:.4f}")
```

```text
  C=    1e-05  ROC-AUC=0.6329
  C=    0.001  ROC-AUC=0.9851
  C=      0.1  ROC-AUC=0.9970
  C=      1.0  ROC-AUC=0.9973
  C=     10.0  ROC-AUC=0.9961
  C=   1000.0  ROC-AUC=0.9925
```

The optimum is in the range 0.1–10. A search from `1e-6` to `1e3` covers a huge region that is mostly bad, while a search from `0.5` to `2.0` is too narrow to find the peak. The recommended approach: start wide, inspect `plot_search_space`, then refine.

**Use log-uniform for parameters that span orders of magnitude.** The `C` parameter above is a good example: the interesting range spans five orders of magnitude. With a uniform distribution, nearly all samples land near `1e3` (the upper bound), missing the interesting region around `0.1–10`.

```python
from sklearn_genetic import GASearchCV, EvolutionConfig, PopulationConfig, RuntimeConfig
from sklearn_genetic.space import Continuous

# Wrong: uniform distribution concentrates samples near the upper bound
search_uniform = GASearchCV(
    estimator=LogisticRegression(max_iter=1000, random_state=42),
    param_grid={"C": Continuous(1e-4, 1e3)},           # uniform by default
    cv=cv, scoring="roc_auc",
    evolution_config=EvolutionConfig(population_size=12, generations=5),
    population_config=PopulationConfig(initializer="smart"),
    runtime_config=RuntimeConfig(n_jobs=-1, verbose=False),
    random_state=42,
)
search_uniform.fit(X, y)

# Right: log-uniform samples evenly across orders of magnitude
search_log = GASearchCV(
    estimator=LogisticRegression(max_iter=1000, random_state=42),
    param_grid={"C": Continuous(1e-4, 1e3, distribution="log-uniform")},
    cv=cv, scoring="roc_auc",
    evolution_config=EvolutionConfig(population_size=12, generations=5),
    population_config=PopulationConfig(initializer="smart"),
    runtime_config=RuntimeConfig(n_jobs=-1, verbose=False),
    random_state=42,
)
search_log.fit(X, y)

print(f"Uniform  — best C: {search_uniform.best_params_['C']:.4f}  ROC-AUC: {search_uniform.best_score_:.4f}")
print(f"Log-unif — best C: {search_log.best_params_['C']:.4f}  ROC-AUC: {search_log.best_score_:.4f}")
```

```text
Uniform  — best C: 182.3341  ROC-AUC: 0.9938
Log-unif — best C: 0.7214    ROC-AUC: 0.9973
```

:::tip Use log-uniform for orders-of-magnitude parameters
Parameters like `learning_rate`, `alpha`, `C`, `gamma`, and `lambda` often span multiple orders of magnitude. Use `Continuous(lower, upper, distribution="log-uniform")` so the search samples each decade equally rather than concentrating near the upper bound.
:::

---

## Mistake 5: Using Accuracy as the Only Metric

Accuracy measures how often a model is correct, but it does not distinguish between types of errors, and it does not reflect relative costs. In many real problems, false negatives and false positives have very different consequences.

| Scenario | Why accuracy is misleading |
|----------|---------------------------|
| Fraud detection (1% fraud rate) | 99% accuracy by predicting "not fraud" for everything |
| Medical diagnosis (rare disease) | High accuracy by always predicting "healthy" |
| Churn prediction (20% churn rate) | 80% accuracy by predicting "stays" for everyone |
| Multi-label classification | Per-class imbalance is invisible in aggregate accuracy |

For classification tasks, think about what matters:

- **Does threshold matter?** Use `roc_auc` (threshold-independent).
- **Do precision and recall both matter?** Use `f1` or `average_precision`.
- **Is class imbalance the main concern?** Use `balanced_accuracy`.
- **Are false negatives much worse than false positives?** Use `recall` or a custom scorer.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

X, y = load_breast_cancer(return_X_y=True)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)

for metric in ["accuracy", "balanced_accuracy", "roc_auc", "f1", "average_precision"]:
    score = cross_val_score(rf, X, y, cv=cv, scoring=metric).mean()
    print(f"  {metric:<22} {score:.4f}")
```

```text
  accuracy               0.9578
  balanced_accuracy      0.9477
  roc_auc               0.9934
  f1                     0.9686
  average_precision      0.9960
```

Different metrics rank the same model differently. Choose the one that matches your deployment objective before you start the search — changing the metric after the search has run means you optimized for the wrong thing.

---

## Mistake 6: Not Accounting for Parameter Interactions

`learning_rate` and `n_estimators` are the classic interacting pair in gradient boosting: a low learning rate needs many estimators to converge; a high learning rate needs fewer. Tuning them independently — fixing one and optimizing the other, then vice versa — gives a suboptimal result because the optimal `learning_rate` changes depending on `n_estimators`.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

X, y = load_breast_cancer(return_X_y=True)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

configs = [
    dict(learning_rate=0.30, n_estimators=50,  label="high LR + few trees"),
    dict(learning_rate=0.01, n_estimators=50,  label="low LR  + few trees (underfits)"),
    dict(learning_rate=0.30, n_estimators=300, label="high LR + many trees (overfits)"),
    dict(learning_rate=0.05, n_estimators=300, label="low LR  + many trees (balanced)"),
]

for cfg in configs:
    label = cfg.pop("label")
    clf = GradientBoostingClassifier(**cfg, max_depth=3, random_state=42)
    score = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc").mean()
    print(f"  {label:<40}  ROC-AUC={score:.4f}")
```

```text
  high LR + few trees                       ROC-AUC=0.9930
  low LR  + few trees (underfits)           ROC-AUC=0.9689
  high LR + many trees (overfits)           ROC-AUC=0.9929
  low LR  + many trees (balanced)           ROC-AUC=0.9975
```

A coordinate-descent approach (fix `n_estimators`, optimize `learning_rate`, then swap) will find the high-LR or low-LR local optima, but miss the jointly optimal combination (low LR + many trees).

Genetic algorithms search over *complete configurations*, recombining parents that performed well. If `(lr=0.05, n=300)` is a good individual, its offspring inherits both values together, so the algorithm naturally discovers and preserves beneficial interactions. See [How Hyperparameter Optimization Works](./how-hyperparameter-optimization-works#why-parameter-interactions-matter) for a deeper explanation.

---

## Mistake 7: Running Search Without Cross-Validation

Using a single train/validation split to evaluate hyperparameters gives a noisy estimate of model quality. The split you happen to get might be unusually easy or unusually hard for certain configurations, and you will select the configuration that got lucky on *that* split.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
import numpy as np

X, y = load_breast_cancer(return_X_y=True)

# Single split: scores vary widely across seeds
print("Single split ROC-AUC (same model, different splits):")
for seed in [0, 1, 2, 3, 4]:
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_tr, y_tr)
    from sklearn.metrics import roc_auc_score
    score = roc_auc_score(y_val, rf.predict_proba(X_val)[:, 1])
    print(f"  seed={seed}  ROC-AUC={score:.4f}")

# Cross-validation: stable across seeds
print("\nCross-validation ROC-AUC (same model, different random seeds for CV):")
for seed in [0, 1, 2, 3, 4]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = cross_val_score(
        RandomForestClassifier(n_estimators=100, random_state=42),
        X, y, cv=cv, scoring="roc_auc"
    )
    print(f"  seed={seed}  ROC-AUC={scores.mean():.4f} ± {scores.std():.4f}")
```

```text
Single split ROC-AUC (same model, different splits):
  seed=0  ROC-AUC=0.9971
  seed=1  ROC-AUC=0.9929
  seed=2  ROC-AUC=0.9984
  seed=3  ROC-AUC=0.9948
  seed=4  ROC-AUC=0.9955

Cross-validation ROC-AUC (same model, different random seeds for CV):
  seed=0  ROC-AUC=0.9934 ± 0.0035
  seed=1  ROC-AUC=0.9934 ± 0.0040
  seed=2  ROC-AUC=0.9939 ± 0.0026
  seed=3  ROC-AUC=0.9934 ± 0.0030
  seed=4  ROC-AUC=0.9939 ± 0.0027
```

Single-split scores range from 0.9929 to 0.9984 (a 55 bp spread) while cross-validation means are stable within a few basis points. Hyperparameter search amplifies this: it will select whichever configuration happened to get lucky on the single split.

**Minimum recommendation:** 3-fold stratified CV for fast development. Use 5-fold or 10-fold when you can afford the compute — the additional folds reduce variance and give the evolutionary algorithm a more reliable fitness signal. Pass a `StratifiedKFold` to `GASearchCV`'s `cv` argument for classification tasks.

---

## Mistake 8: Setting `random_state` Inconsistently

For reproducible searches, you need to seed four independent sources of randomness. Missing any one of them means different runs may produce different results:

| Source | How to seed |
|--------|-------------|
| Train/test split | `train_test_split(..., random_state=42)` |
| Cross-validation splitter | `StratifiedKFold(..., random_state=42)` (requires `shuffle=True`) |
| Estimator | `RandomForestClassifier(random_state=42)` |
| Search algorithm | `GASearchCV(..., random_state=42)` |

```python
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split

from sklearn_genetic import EvolutionConfig, GASearchCV, PopulationConfig, RuntimeConfig
from sklearn_genetic.space import Categorical, Continuous, Integer

X, y = load_breast_cancer(return_X_y=True)

# All four random_state values are set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42   # (1) split
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # (2) CV

estimator = RandomForestClassifier(random_state=42)               # (3) estimator

search = GASearchCV(
    estimator=estimator,
    random_state=42,                                              # (4) search
    param_grid={
        "n_estimators": Integer(50, 300),
        "max_depth": Integer(2, 12),
        "max_features": Categorical(["sqrt", "log2"]),
    },
    cv=cv,
    scoring="roc_auc",
    evolution_config=EvolutionConfig(population_size=12, generations=6),
    population_config=PopulationConfig(initializer="smart"),
    runtime_config=RuntimeConfig(n_jobs=1, verbose=False),  # n_jobs=1 for full determinism
)

search.fit(X_train, y_train)
print(f"Best CV ROC-AUC: {search.best_score_:.4f}")
print(f"Best params:     {search.best_params_}")
```

```text
Best CV ROC-AUC: 0.9963
Best params:     {'n_estimators': 247, 'max_depth': 10, 'max_features': 'sqrt'}
```

See [Reproducibility & Checkpointing](./reproducibility) for a full discussion including why `n_jobs=1` is required for strict reproducibility.

---

## Mistake 9: Stopping Too Early (or Too Late)

**Stopping too early** means the genetic algorithm has not had enough generations to converge. The population may still be diverse and improving — you are leaving score on the table.

**Stopping too late** wastes compute time after the population has converged. Once `fitness_best` stops improving and diversity has collapsed, additional generations produce no benefit.

Detect both problems by inspecting `plot_fitness_evolution` after the search:

```python
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split

from sklearn_genetic import EvolutionConfig, GASearchCV, PopulationConfig, RuntimeConfig
from sklearn_genetic.callbacks import ConsecutiveStopping
from sklearn_genetic.space import Categorical, Integer
import pandas as pd

X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

search = GASearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid={
        "n_estimators": Integer(50, 300),
        "max_depth": Integer(2, 15),
        "min_samples_leaf": Integer(1, 10),
        "max_features": Categorical(["sqrt", "log2"]),
    },
    cv=cv,
    scoring="accuracy",
    evolution_config=EvolutionConfig(population_size=15, generations=30),  # generous ceiling
    population_config=PopulationConfig(initializer="smart"),
    runtime_config=RuntimeConfig(n_jobs=-1, verbose=False),
    random_state=42,
)

# Stop when fitness_best hasn't improved for 5 consecutive generations
early_stopper = ConsecutiveStopping(generations=5, metric="fitness_best")
search.fit(X_train, y_train, callbacks=[early_stopper])

history = pd.DataFrame(search.history)
print(f"Search stopped at generation : {len(history)}")
print(f"Best CV accuracy             : {search.best_score_:.4f}")
print(f"\nLast 5 generations:")
print(history[["gen", "fitness", "fitness_best", "genotype_diversity",
               "stagnation_generations"]].tail(5).to_string(index=False))
```

```text
Search stopped at generation : 11
Best CV accuracy             : 0.9784

Last 5 generations:
 gen  fitness  fitness_best  genotype_diversity  stagnation_generations
   6  0.97538       0.97843            0.233333                       1
   7  0.97453       0.97843            0.200000                       2
   8  0.97527       0.97843            0.166667                       3
   9  0.97492       0.97843            0.133333                       4
  10  0.97473       0.97843            0.133333                       5
```

:::tip Use ConsecutiveStopping to detect convergence automatically
`ConsecutiveStopping(generations=5, metric="fitness_best")` stops the search when the best score has not improved for 5 generations. Set the ceiling (total `generations` in `EvolutionConfig`) high enough that early stopping is what actually terminates the search — not the ceiling itself.
:::

---

## Mistake 10: Forgetting to Refit on Full Training Data

After cross-validation selects the best hyperparameters, you need a model trained on all the training data — not just one of the CV folds — to achieve the best generalization.

`GASearchCV` handles this automatically when `refit=True` (the default). After the search completes, it refits the best configuration on the entire `X_train` you passed to `.fit()`. The result is available as `best_estimator_`.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score

from sklearn_genetic import EvolutionConfig, GASearchCV, PopulationConfig, RuntimeConfig
from sklearn_genetic.space import Categorical, Integer

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

search = GASearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid={
        "n_estimators": Integer(50, 300),
        "max_depth": Integer(2, 12),
        "max_features": Categorical(["sqrt", "log2"]),
    },
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring="roc_auc",
    refit=True,   # default — refits best params on all of X_train
    evolution_config=EvolutionConfig(population_size=12, generations=6),
    population_config=PopulationConfig(initializer="smart"),
    runtime_config=RuntimeConfig(n_jobs=-1, verbose=False),
    random_state=42,
)

search.fit(X_train, y_train)

# Predict directly — uses best_estimator_ which was refit on all of X_train
y_proba = search.predict_proba(X_test)[:, 1]
test_roc = roc_auc_score(y_test, y_proba)

print(f"Best CV ROC-AUC   : {search.best_score_:.4f}")
print(f"Holdout ROC-AUC   : {test_roc:.4f}")
print(f"Refit completed   : {search.refit}")
print(f"Best estimator    : {search.best_estimator_}")
```

```text
Best CV ROC-AUC   : 0.9963
Holdout ROC-AUC   : 0.9930
Refit completed   : True
Best estimator    : RandomForestClassifier(max_depth=10, max_features='sqrt', n_estimators=247, random_state=42)
```

If you set `refit=False`, `best_estimator_` is not populated and calling `predict` will raise an error. You would then need to refit manually:

```python
# Only needed when refit=False
best_model = RandomForestClassifier(**search.best_params_, random_state=42)
best_model.fit(X_train, y_train)   # fit on the entire training set
```

Note that the holdout score may differ slightly from the CV score — this is normal and expected. A large gap (> 0.03–0.05) suggests overfitting to the validation set (see Mistake 1).

---

## Quick Checklist Before Running a Search

Before launching a long hyperparameter search, confirm each item:

1. The test set is split off and will not be touched until final evaluation.
2. All preprocessing is inside a `Pipeline` (not applied before the split).
3. The cross-validation strategy matches the task (`StratifiedKFold` for classification, `KFold` for regression).
4. The scoring metric reflects the actual business objective, not just convenience.
5. Search space bounds are sensible — not too wide, not too narrow.
6. Parameters that span orders of magnitude use `distribution="log-uniform"`.
7. `random_state` is set on the split, CV splitter, estimator, and search object.
8. `refit=True` (default) so `best_estimator_` is ready to use after the search.
9. A `ConsecutiveStopping` callback is wired up with a generous generation ceiling.
10. `fit_stats_["skipped_invalid_candidates"]` will be checked after the search to catch silent errors.

---

## See Also

- [Reproducibility & Checkpointing](./reproducibility) — seed every random source and save progress mid-search
- [Cross-Validation in Hyperparameter Search](./understand-cv) — how GASearchCV evaluates candidates and what the generation log columns mean
- [Troubleshooting](./troubleshooting) — symptoms, causes, and fixes for common search problems
- [How Hyperparameter Optimization Works](./how-hyperparameter-optimization-works) — the four main optimization strategies explained from first principles
- [Choosing the Right Search Space](./choosing-search-spaces) — recommended bounds for common sklearn estimators
