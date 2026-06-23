---
title: Comprehensive GA Feature Selection
description: Three-stage workflow — GA feature selection on 50 features, hyperparameter retune on the selected subset, cross-estimator robustness validation — on breast cancer data with added noise.
---

# Comprehensive GA Feature Selection

Feature selection and hyperparameter tuning are usually done separately, but they interact: the best hyperparameters for 50 features are different from the best for 15. This tutorial shows a three-stage workflow that treats both together:

1. **Baseline** — all 50 features, default hyperparameters
2. **GA feature selection** — find the best 15-feature subset from 50 (30 real + 20 noise)
3. **Hyperparameter retune** — tune a new model on the selected 15 features (faster evaluations, better budget)

Then a **robustness check** fits a completely different estimator on the selected features. If a second estimator also improves, the selection is genuinely informative and not an artifact of the scoring model.

:::info Complementary to the simpler example
The [Feature Selection (Noisy Data)](../examples/feature-selection) example in the Examples section uses Iris + 12 noise features in a single stage. This tutorial uses breast cancer + 20 noise features and adds hyperparameter retuning and cross-estimator validation.
:::

## Prerequisites

```bash
pip install sklearn-genetic-opt
```

## Setup

```python
import warnings
from pprint import pprint

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn_genetic import (
    EvolutionConfig, GAFeatureSelectionCV, GASearchCV,
    OptimizationConfig, PopulationConfig, RuntimeConfig,
)
from sklearn_genetic.callbacks import ConsecutiveStopping, DeltaThreshold, TimerStopping
from sklearn_genetic.schedules import ExponentialAdapter, InverseAdapter
from sklearn_genetic.space import Categorical, Continuous, Integer

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)
```

## Build the Dataset

Start with breast cancer's 30 real features, then add 20 independent Gaussian noise columns. A good selector should recover most real features and drop all noise ones.

```python
data = load_breast_cancer(as_frame=True)
X_real = data.data          # 30 real features
y = data.target

noise = pd.DataFrame(
    rng.normal(size=(X_real.shape[0], 20)),
    columns=[f"noise_{i:02d}" for i in range(20)],
)
X = pd.concat([X_real.reset_index(drop=True), noise], axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
)
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

real_feature_names = set(data.feature_names)

print(f"Real features:  {X_real.shape[1]}")
print(f"Noise features: {noise.shape[1]}")
print(f"Total features: {X.shape[1]}")
print(f"Train: {X_train.shape}, Test: {X_test.shape}")
```

## Helpers

```python
def evaluate(name, estimator, X_eval, y_eval):
    predictions = estimator.predict(X_eval)
    try:
        probabilities = estimator.predict_proba(X_eval)[:, 1]
        roc = round(roc_auc_score(y_eval, probabilities), 4)
    except AttributeError:
        roc = None
    return {
        "name": name,
        "n_features": X_eval.shape[1],
        "accuracy": round(accuracy_score(y_eval, predictions), 4),
        "balanced_accuracy": round(balanced_accuracy_score(y_eval, predictions), 4),
        "roc_auc": roc,
    }


def make_svc():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="rbf", C=2.0, gamma="scale",
                    probability=True, random_state=RANDOM_STATE)),
    ])
```

## Stage 1 — Baseline on All 50 Features

```python
rf_baseline = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
rf_baseline.fit(X_train, y_train)
baseline_rf = evaluate("RF baseline (50 features)", rf_baseline, X_test, y_test)

svc_baseline = make_svc()
svc_baseline.fit(X_train, y_train)
baseline_svc = evaluate("SVC baseline (50 features)", svc_baseline, X_test, y_test)

print(baseline_rf)
# {'name': 'RF baseline (50 features)', 'n_features': 50, 'accuracy': 0.9474, ...}
print(baseline_svc)
# {'name': 'SVC baseline (50 features)', 'n_features': 50, 'accuracy': 0.9298, ...}
```

The noise features dilute both models. The RF ignores them via feature importance; the SVC's kernel distance is degraded by irrelevant dimensions.

## Stage 2 — GA Feature Selection

`GAFeatureSelectionCV` optimizes a binary mask over the 50 columns. Setting `max_features=15` steers the search toward compact subsets.

```python
selector = GAFeatureSelectionCV(
    estimator=RandomForestClassifier(
        n_estimators=100, random_state=RANDOM_STATE, n_jobs=1
    ),
    cv=cv,
    scoring="roc_auc",
    max_features=15,
    evolution_config=EvolutionConfig(
        population_size=24,
        generations=20,
        crossover_probability=ExponentialAdapter(
            initial_value=0.8, end_value=0.4, adaptive_rate=0.15
        ),
        mutation_probability=InverseAdapter(
            initial_value=0.30, end_value=0.08, adaptive_rate=0.25
        ),
        tournament_size=3,
        elitism=True,
        keep_top_k=3,
    ),
    population_config=PopulationConfig(initializer="smart"),
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
        local_search_radius=0.15,
        diversity_control=True,
        diversity_threshold=0.30,
        diversity_stagnation_generations=3,
        diversity_mutation_boost=1.8,
        random_immigrants_fraction=0.12,
        fitness_sharing=True,
        sharing_radius=0.40,
    ),
)

selection_callbacks = [
    DeltaThreshold(threshold=0.001, generations=5, metric="fitness_best"),
    ConsecutiveStopping(generations=8, metric="fitness_best"),
    TimerStopping(total_seconds=120),
]

selector.fit(X_train, y_train, callbacks=selection_callbacks)

print(f"\nBest CV ROC AUC (selection): {selector.best_score_:.4f}")
print(f"Selected {selector.n_features_} of {X.shape[1]} features")
```

### Which Features Were Selected?

```python
selected_mask = selector.support_
selected_names = X_train.columns[selected_mask].tolist()

summary = pd.DataFrame({
    "feature": X_train.columns,
    "selected": selected_mask,
    "kind": ["real" if c in real_feature_names else "noise" for c in X_train.columns],
})

print("\nSelected features:")
print(summary[summary["selected"]].to_string(index=False))

n_real_selected = summary[summary["selected"] & (summary["kind"] == "real")].shape[0]
n_noise_selected = summary[summary["selected"] & (summary["kind"] == "noise")].shape[0]
print(f"\nReal features kept: {n_real_selected} / {X_real.shape[1]}")
print(f"Noise features kept: {n_noise_selected} / {noise.shape[1]}")
```

### Feature Selection Breakdown Chart

```python
color_map = {"real": "steelblue", "noise": "salmon"}
colors = [color_map[k] for k in summary[summary["selected"]]["kind"]]

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(range(len(selected_names)), [1] * len(selected_names), color=colors)
ax.set_xticks(range(len(selected_names)))
ax.set_xticklabels(selected_names, rotation=45, ha="right", fontsize=8)
ax.set_title("Selected Features — real (blue) vs noise (red)")
ax.set_ylabel("Selected")
ax.set_yticks([])
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color="steelblue", label="real"), Patch(color="salmon", label="noise")])
plt.tight_layout()
plt.show()
```

### Feature Selection Telemetry

```python
print(selector.fit_stats_)

history = pd.DataFrame(selector.history)
ax = history.plot(
    x="gen",
    y=["fitness_best", "fitness_max", "fitness"],
    marker="o",
    figsize=(9, 4),
)
ax.set_title("Feature Selection — Fitness over Generations")
ax.set_xlabel("Generation")
ax.set_ylabel("ROC AUC (CV)")
plt.tight_layout()
plt.show()
```

## Stage 3 — Hyperparameter Retune on Selected Features

With 15 features instead of 50, each cross-validation call is faster. The same search budget covers more candidates.

```python
X_train_sel = selector.transform(X_train)
X_test_sel = selector.transform(X_test)

print(f"Train shape after selection: {X_train_sel.shape}")

rf_param_grid = {
    "n_estimators":      Integer(40, 200),
    "max_depth":         Integer(2, 12),
    "min_samples_split": Integer(2, 12),
    "min_samples_leaf":  Integer(1, 8),
    "max_features":      Categorical(["sqrt", "log2", None]),
}

ga_rf = GASearchCV(
    estimator=RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
    param_grid=rf_param_grid,
    scoring="roc_auc",
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
        keep_top_k=2,
    ),
    population_config=PopulationConfig(
        initializer="smart",
        warm_start_configs=[{
            "n_estimators": 100,
            "max_depth": 6,
            "min_samples_split": 4,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
        }],
    ),
    runtime_config=RuntimeConfig(n_jobs=-1, parallel_backend="auto", use_cache=True, verbose=True),
    optimization_config=OptimizationConfig(
        local_search=True, diversity_control=True, fitness_sharing=True,
    ),
)

ga_rf.fit(X_train_sel, y_train, callbacks=[
    ConsecutiveStopping(generations=8, metric="fitness_best"),
    TimerStopping(total_seconds=90),
])

print(f"\nBest CV ROC AUC (retune): {ga_rf.best_score_:.4f}")
pprint(ga_rf.best_params_)
```

## Robustness Validation — Second Estimator

If the selected features are genuinely informative (not overfitted to the RF scorer), an independent SVC should also benefit.

```python
svc_on_selected = make_svc()
svc_on_selected.fit(X_train_sel, y_train)

svc_sel_metrics = evaluate(
    "SVC on selected features", svc_on_selected, X_test_sel, y_test
)

rf_sel_metrics = evaluate(
    "RF retuned (selected features)", ga_rf, X_test_sel, y_test
)

print("\nSVC baseline vs SVC on selected features:")
print(pd.DataFrame([baseline_svc, svc_sel_metrics]).to_string(index=False))
```

If the SVC improves on the selected subset, it confirms the features are model-agnostic signal, not RF-specific artefacts.

## Full Comparison Table

```python
comparison = pd.DataFrame([
    baseline_rf,
    baseline_svc,
    evaluate("RF retuned (selected features)", ga_rf, X_test_sel, y_test),
    svc_sel_metrics,
])
print(comparison.to_string(index=False))
```

Expected output (approximate):

```
                          name  n_features  accuracy  balanced_accuracy  roc_auc
      RF baseline (50 features)         50    0.9474             0.9432   0.9891
     SVC baseline (50 features)         50    0.9298             0.9252   0.9857
  RF retuned (selected features)        15    0.9649             0.9613   0.9948
   SVC on selected features            15    0.9532             0.9497   0.9921
```

Both estimators improve after selection, confirming the selected subset carries the majority of predictive signal.

## Feature Importance Before and After Selection

```python
# Before: RF fitted on all 50 features
imp_all = pd.Series(
    rf_baseline.feature_importances_, index=X_train.columns
).sort_values(ascending=False).head(20)

# After: RF fitted on selected features only
imp_sel = pd.Series(
    ga_rf.best_estimator_.feature_importances_, index=selected_names
).sort_values(ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 7))

colors_all = ["steelblue" if n in real_feature_names else "salmon" for n in imp_all.index]
imp_all.plot(kind="barh", ax=axes[0], color=colors_all[::-1])
axes[0].set_title("Top-20 Importances — All 50 Features")
axes[0].invert_yaxis()

colors_sel = ["steelblue" if n in real_feature_names else "salmon" for n in imp_sel.index]
imp_sel.plot(kind="barh", ax=axes[1], color=colors_sel[::-1])
axes[1].set_title("Importances — 15 Selected Features")
axes[1].invert_yaxis()

plt.tight_layout()
plt.show()
```

## Practical Notes

- **`max_features` in `GAFeatureSelectionCV`** is a soft upper bound — the GA prefers subsets below it, but can exceed it if fitness requires. Check `selector.n_features_` for the actual count.
- **Stage ordering matters** — run feature selection before hyperparameter tuning when evaluations are expensive. Feature selection narrows the input space, making every subsequent CV call cheaper.
- **Cross-estimator validation** is the most reliable check that selected features are signal and not model-specific noise. If only the scoring estimator improves, suspect scorer-feature circularity.
- **`use_cache=True`** in feature selection is particularly impactful — many binary masks differ by only one or two features, and cached evaluations avoid redundant CV calls.
- **Diversity metrics in history** — if `unique_individual_ratio` drops below 0.5 before generation 10, increase `random_immigrants_fraction` or widen `sharing_radius`.

## See Also

- [Feature Selection (Noisy Data)](../examples/feature-selection) — simpler single-stage example
- [GAFeatureSelectionCV API](../api/gafeatureselectioncv)
- [Advanced Optimizer Control](../guide/advanced-optimizer-control)
- [Adaptive Schedules](../guide/adapters)
