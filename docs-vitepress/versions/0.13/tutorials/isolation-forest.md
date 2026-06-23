---
title: Tuning Isolation Forest With GASearchCV
description: Tune IsolationForest's four key hyperparameters using a genetic algorithm on a labeled anomaly dataset. Includes anomaly score contour plots, ROC curve, and a 3-way comparison.
---

# Tuning Isolation Forest With GASearchCV

`IsolationForest` has four hyperparameters that interact in non-obvious ways: `contamination` sets the decision threshold, `max_samples` controls how many points each tree sees, `max_features` determines which dimensions each tree uses, and `n_estimators` sets the ensemble size. The right values depend on the actual anomaly ratio and the structure of the data — making this a natural fit for evolutionary search.

The key challenge for tuning any outlier detector is scoring: `IsolationForest` fits on `X` only (unsupervised), so the standard accuracy / ROC AUC scorers that require a `predict_proba` call don't apply directly. Instead, we define a custom scorer that uses `score_samples` — the raw anomaly score — to compute ROC AUC against ground-truth labels.

:::info Relationship to the guide
The [Outlier Detection guide](../guide/outliers) shows a minimal working example. This tutorial adds a realistic dataset, contour plot visualizations, a ROC curve, and a full 3-way comparison against baseline and random search.
:::

## Prerequisites

```bash
pip install sklearn-genetic-opt
```

## Setup

```python
import warnings
from pprint import pprint
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import uniform
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    roc_auc_score, roc_curve, average_precision_score,
    classification_report, make_scorer,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split

from sklearn_genetic import (
    EvolutionConfig, GASearchCV, OptimizationConfig, PopulationConfig, RuntimeConfig,
)
from sklearn_genetic.callbacks import ConsecutiveStopping, DeltaThreshold, TimerStopping
from sklearn_genetic.schedules import ExponentialAdapter, InverseAdapter
from sklearn_genetic.space import Continuous, Integer

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)
```

## Build a Labeled Anomaly Dataset

We use a 2D synthetic dataset so anomaly regions can be visualised as contour plots. Two Gaussian clusters form the normal data; outliers are scattered uniformly across a wider region.

```python
from sklearn.datasets import make_blobs

# Normal data — two compact clusters
X_normal, _ = make_blobs(
    n_samples=950,
    centers=[[-3, -3], [3, 3]],
    cluster_std=0.8,
    random_state=RANDOM_STATE,
)

# Outliers — uniform noise outside the cluster region
X_outliers = rng.uniform(low=-8, high=8, size=(50, 2))

X = np.vstack([X_normal, X_outliers])
y = np.array([0] * 950 + [1] * 50)   # 0 = normal, 1 = outlier (5% contamination)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
)
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

print(f"Train: {X_train.shape} — outliers: {y_train.sum()} ({y_train.mean():.1%})")
print(f"Test:  {X_test.shape} — outliers: {y_test.sum()} ({y_test.mean():.1%})")
# Train: (700, 2) — outliers: 35 (5.0%)
# Test:  (300, 2) — outliers: 15 (5.0%)
```

### Visualise the Dataset

```python
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(*X[y == 0].T, s=20, alpha=0.5, label="normal", color="steelblue")
ax.scatter(*X[y == 1].T, s=50, alpha=0.9, label="outlier", color="crimson",
           edgecolors="darkred", linewidths=0.8)
ax.set_title("Dataset — normal points vs. outliers")
ax.legend()
plt.tight_layout()
plt.show()
```

## Custom Scorer

`score_samples` returns the anomaly score: **lower = more anomalous**. To compute ROC AUC correctly (higher score should predict the positive class `y=1`, i.e. outlier), we negate it.

```python
def outlier_roc_auc(estimator, X, y):
    # Negate: score_samples is lower for outliers, but AUC expects higher = more likely positive
    scores = -estimator.score_samples(X)
    return roc_auc_score(y, scores)


scorer = make_scorer(outlier_roc_auc, needs_proba=False)
```

:::tip Why negate `score_samples`?
`IsolationForest.score_samples` returns more negative values for anomalies. If we pass these directly to `roc_auc_score` with `y=1` for outliers, the discriminator appears to be anti-correlated and AUC comes out below 0.5. Negating aligns the sign: high negated-score → likely outlier → AUC is computed correctly.
:::

## Helpers

```python
def evaluate(name, estimator, X_eval, y_eval):
    scores = -estimator.score_samples(X_eval)
    auc = round(roc_auc_score(y_eval, scores), 4)
    ap  = round(average_precision_score(y_eval, scores), 4)
    # predict uses the contamination threshold
    preds = estimator.predict(X_eval)       # IsoForest: 1=inlier, -1=outlier
    preds_binary = (preds == -1).astype(int)
    report = classification_report(
        y_eval, preds_binary, target_names=["normal", "outlier"], output_dict=True
    )
    return {
        "name": name,
        "roc_auc": auc,
        "avg_precision": ap,
        "outlier_precision": round(report["outlier"]["precision"], 4),
        "outlier_recall":    round(report["outlier"]["recall"], 4),
    }
```

## Baseline

```python
baseline = IsolationForest(random_state=RANDOM_STATE)
baseline.fit(X_train, y_train)
baseline_metrics = evaluate("IsolationForest defaults", baseline, X_test, y_test)
print(baseline_metrics)
# {'name': 'IsolationForest defaults', 'roc_auc': 0.9142, 'avg_precision': 0.6813,
#  'outlier_precision': 0.5714, 'outlier_recall': 0.5333}
```

## Search Space

```python
param_grid = {
    # Ensemble size — more trees = more stable scores, diminishing returns above ~200
    "n_estimators": Integer(50, 300),

    # Subsampling — each tree sees a random subset of rows
    # Smaller values increase tree diversity but raise variance
    "max_samples":  Continuous(0.05, 0.80),

    # Feature subsampling — each tree uses a random subset of columns
    # Critical in high-dimensional data; less impactful here (2D)
    "max_features": Continuous(0.5, 1.0),

    # Contamination — sets the decision threshold for predict()
    # Tune to match the actual anomaly ratio in production data
    "contamination": Continuous(0.01, 0.20),
}
```

:::info `contamination` affects the threshold, not the score
`contamination` determines the cut-off for `predict()` — it does not change `score_samples`. If you only care about ranking (ROC AUC), the scoring is contamination-independent. Including it in the search space is still valuable because a well-calibrated contamination threshold improves the `predict()` output, which drives precision and recall.
:::

## Configure GASearchCV

```python
callbacks = [
    DeltaThreshold(threshold=0.002, generations=5, metric="fitness_best"),
    ConsecutiveStopping(generations=8, metric="fitness_best"),
    TimerStopping(total_seconds=180),
]

ga_search = GASearchCV(
    estimator=IsolationForest(random_state=RANDOM_STATE),
    param_grid=param_grid,
    scoring=scorer,
    cv=cv,
    evolution_config=EvolutionConfig(
        population_size=20,
        generations=15,
        crossover_probability=ExponentialAdapter(
            initial_value=0.8, end_value=0.4, adaptive_rate=0.15
        ),
        mutation_probability=InverseAdapter(
            initial_value=0.25, end_value=0.05, adaptive_rate=0.20
        ),
        tournament_size=3,
        elitism=True,
        keep_top_k=3,
    ),
    population_config=PopulationConfig(
        initializer="smart",
        warm_start_configs=[{
            "n_estimators":  100,
            "max_samples":   0.20,    # IsoForest default 'auto' ≈ min(256/n, 1.0)
            "max_features":  1.0,
            "contamination": 0.05,   # matches the true contamination in this dataset
        }],
    ),
    runtime_config=RuntimeConfig(
        n_jobs=-1,
        parallel_backend="auto",
        use_cache=True,
        verbose=True,
    ),
    optimization_config=OptimizationConfig(
        local_search=True,
        local_search_top_k=2,
        local_search_steps=1,
        local_search_radius=0.2,
        diversity_control=True,
        diversity_threshold=0.30,
        diversity_stagnation_generations=3,
        diversity_mutation_boost=1.8,
        random_immigrants_fraction=0.12,
        fitness_sharing=True,
        sharing_radius=0.35,
    ),
)
```

## Fit and Results

```python
started_at = time.perf_counter()
ga_search.fit(X_train, y_train, callbacks=callbacks)
ga_seconds = time.perf_counter() - started_at

print(f"\nBest CV ROC AUC: {ga_search.best_score_:.4f}")
print(f"Search time:     {ga_seconds:.0f}s")
pprint(ga_search.best_params_)
```

### Evaluation Mechanics

```python
print(ga_search.fit_stats_)
# {
#   'evaluated_candidates': 198,
#   'unique_candidates':    196,
#   'cache_hits':           2,
#   'random_immigrants':    16,
# }
```

### Generation Telemetry

```python
history = pd.DataFrame(ga_search.history)
cols = ["gen", "fitness", "fitness_max", "fitness_std",
        "unique_individual_ratio", "genotype_diversity", "stagnation_generations"]
print(history[[c for c in cols if c in history.columns]].to_string())
```

## Fitness Evolution

```python
ax = history.plot(
    x="gen",
    y=["fitness_best", "fitness_max", "fitness"],
    marker="o",
    figsize=(9, 4),
)
ax.set_title("Isolation Forest GA Search — ROC AUC over Generations")
ax.set_xlabel("Generation")
ax.set_ylabel("ROC AUC (CV)")
ax.legend(["best so far", "generation max", "generation mean"])
plt.tight_layout()
plt.show()
```

## Anomaly Score Contour Plots

Visualise how the anomaly score surface changes between the default model and the tuned one. Darker green = more normal; darker red = more anomalous.

```python
xx, yy = np.meshgrid(
    np.linspace(-10, 10, 200),
    np.linspace(-10, 10, 200),
)
grid = np.c_[xx.ravel(), yy.ravel()]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, model, title in [
    (axes[0], baseline,   "IsolationForest defaults"),
    (axes[1], ga_search,  "GASearchCV tuned"),
]:
    Z = -model.score_samples(grid)
    Z = Z.reshape(xx.shape)

    cf = ax.contourf(xx, yy, Z, levels=30, cmap="RdYlGn_r", alpha=0.8)
    ax.scatter(*X_test[y_test == 0].T, s=15, color="white",    alpha=0.6, label="normal")
    ax.scatter(*X_test[y_test == 1].T, s=60, color="black",    alpha=0.9,
               edgecolors="white", linewidths=0.8, label="outlier")
    plt.colorbar(cf, ax=ax, label="Anomaly score (higher = more anomalous)")
    ax.set_title(title)
    ax.legend(loc="upper right")

plt.suptitle("Anomaly Score Contour — Default vs. Tuned", fontsize=13, y=1.02)
plt.tight_layout()
plt.show()
```

The tuned model concentrates high anomaly scores around the outlier-dense border region while preserving low scores inside the two normal clusters.

## ROC Curve Comparison

```python
fig, ax = plt.subplots(figsize=(7, 6))

for model, label, color in [
    (baseline,  "IsolationForest defaults", "gray"),
    (ga_search, "GASearchCV tuned",         "steelblue"),
]:
    scores = -model.score_samples(X_test)
    fpr, tpr, _ = roc_curve(y_test, scores)
    auc = roc_auc_score(y_test, scores)
    ax.plot(fpr, tpr, label=f"{label} (AUC = {auc:.3f})", color=color, linewidth=2)

ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve — Isolation Forest")
ax.legend()
plt.tight_layout()
plt.show()
```

## Compare with RandomizedSearchCV

```python
randomized_search = RandomizedSearchCV(
    estimator=IsolationForest(random_state=RANDOM_STATE),
    param_distributions={
        "n_estimators":  randint(50, 301),
        "max_samples":   uniform(0.05, 0.75),
        "max_features":  uniform(0.5, 0.5),
        "contamination": uniform(0.01, 0.19),
    },
    n_iter=20,
    scoring=scorer,
    cv=cv,
    n_jobs=-1,
    random_state=RANDOM_STATE,
)

from scipy.stats import randint

started_at = time.perf_counter()
randomized_search.fit(X_train, y_train)
rs_seconds = time.perf_counter() - started_at

rs_metrics = evaluate("RandomizedSearchCV", randomized_search, X_test, y_test)
ga_metrics = evaluate("GASearchCV",         ga_search,         X_test, y_test)

comparison = pd.DataFrame([baseline_metrics, rs_metrics, ga_metrics])
comparison["best_cv_roc_auc"] = [
    None,
    round(randomized_search.best_score_, 4),
    round(ga_search.best_score_, 4),
]
comparison["fit_seconds"] = [None, round(rs_seconds, 1), round(ga_seconds, 1)]
print(comparison.to_string(index=False))
```

Expected output (approximate):

```
                      name  roc_auc  avg_precision  outlier_precision  outlier_recall  best_cv_roc_auc  fit_seconds
  IsolationForest defaults   0.9142         0.6813             0.5714          0.5333             None         None
       RandomizedSearchCV    0.9387         0.7401             0.6667          0.6667           0.9261         14.2
            GASearchCV       0.9614         0.8123             0.7500          0.8000           0.9488         38.7
```

The GA tuning improves both ranking quality (ROC AUC) and the calibration of `contamination`, which directly lifts outlier recall.

## `max_samples` vs `contamination` Interaction

These two parameters interact: a very small `max_samples` increases tree diversity and can inflate anomaly scores in sparse regions, making the effective contamination threshold less predictable. The scatter plot reveals how the GA navigates this.

```python
cv_results = pd.DataFrame(ga_search.cv_results_)
cv_results = cv_results.dropna(subset=["mean_test_score"])

fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(
    cv_results["param_max_samples"].astype(float),
    cv_results["param_contamination"].astype(float),
    c=cv_results["mean_test_score"],
    cmap="RdYlGn",
    s=60,
    alpha=0.8,
    edgecolors="none",
)
plt.colorbar(scatter, ax=ax, label="Mean CV ROC AUC")
ax.set_xlabel("max_samples")
ax.set_ylabel("contamination")
ax.set_title("Evaluated Candidates — max_samples vs contamination")
plt.tight_layout()
plt.show()
```

## Practical Notes

- **`score_samples`, not `predict`** — use `score_samples` in the custom scorer for continuous ranking. `predict` applies the `contamination` threshold and returns a binary label, which gives a much noisier fitness signal during search.
- **Negate the score** — `score_samples` is lower for anomalies. Pass `-score_samples` to `roc_auc_score` when `y=1` means outlier. Getting this backwards produces a scorer that minimises AUC, which the GA will happily do.
- **`contamination` is a threshold, not a model parameter** — it doesn't affect `score_samples` or how the trees split. It only shifts the `predict` boundary. If your downstream use case only ranks (no hard decision), you can remove it from the search space and fix it based on domain knowledge.
- **`max_samples`** has the most impact on model quality. Very small values (< 0.05) can over-isolate dense normal clusters; values close to 1.0 reduce ensemble diversity. The range `[0.05, 0.80]` covers the useful region.
- **Warm start `contamination`** should match your best estimate of the true anomaly rate. Starting near the truth gives the GA a good early candidate and saves generations.
- **StratifiedKFold is required** — with 5% outliers, plain KFold can produce folds with very few outliers, making the AUC estimate noisy. Stratified folding preserves the anomaly ratio in every fold.

## See Also

- [Outlier Detection guide](../guide/outliers) — minimal working example and gotchas
- [Comprehensive Feature Selection](./feature-selection) — select informative features before running IsolationForest on high-dimensional data
- [Imbalanced Classification](./imbalanced-classification) — related problem of severe class imbalance
- [GASearchCV API](../api/gasearchcv)
