---
title: Basic Usage
description: Full workflow for hyperparameter tuning with GASearchCV and feature selection with GAFeatureSelectionCV.
---

:::warning Development version
You are reading the **latest (dev)** docs. For the stable version, see [0.13](/versions/0.13/).
:::

# Basic Usage

This tutorial covers the two most common workflows:

- Hyperparameter tuning with `GASearchCV`
- Feature selection with `GAFeatureSelectionCV`

## Prerequisites

- `sklearn-genetic-opt` installed (`pip install sklearn-genetic-opt`)
- Basic familiarity with scikit-learn's `fit`/`predict` API

## Hyperparameter Tuning

We will tune an `MLPClassifier` on the [digits dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html).

```python
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier

from sklearn_genetic import EvolutionConfig, GASearchCV, PopulationConfig, RuntimeConfig
from sklearn_genetic.space import Categorical, Continuous, Integer
```

Load and split the data:

```python
data = load_digits()
n_samples = len(data.images)
X = data.images.reshape((n_samples, -1))
y = data["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)
```

Define the hyperparameter search space. The keys in `param_grid` must match valid estimator parameters:

- `Integer` — samples integer values from a range
- `Continuous` — samples floating-point values from a range
- `Categorical` — samples from a fixed list of choices

```python
param_grid = {
    "tol": Continuous(1e-2, 1e10, distribution="log-uniform"),
    "alpha": Continuous(1e-5, 2e-5),
    "activation": Categorical(["logistic", "tanh"]),
    "batch_size": Integer(300, 350),
}
```

Create the estimator, CV strategy, and the search:

```python
clf = MLPClassifier(hidden_layer_sizes=(50, 30))
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

evolved_estimator = GASearchCV(
    estimator=clf,
    cv=cv,
    scoring="accuracy",
    param_grid=param_grid,
    evolution_config=EvolutionConfig(population_size=10, generations=20),
    population_config=PopulationConfig(initializer="smart"),
    runtime_config=RuntimeConfig(n_jobs=-1, verbose=True),
)
```

`RuntimeConfig.n_jobs` controls parallel execution — unique candidates in the same generation are evaluated in parallel. `EvolutionConfig.population_size` and `EvolutionConfig.generations` determine how many candidate solutions are explored. `PopulationConfig(initializer="smart")` builds a more diverse initial population using estimator defaults, Latin hypercube samples for numeric hyperparameters, and stratified categorical values.

Run the optimization:

```python
evolved_estimator.fit(X_train, y_train)
```

Rendered output from the documentation run:

<!-- docs-example:basic-usage-hyperparameter-output:start -->
```text
Best CV accuracy: 0.9327
Holdout accuracy: 0.9158
Best parameters: { activation: logistic, alpha: 0.0000, batch_size: 214, tol: 0.0046 }
```
<!-- docs-example:basic-usage-hyperparameter-output:end -->

During training you will see a generation-by-generation log. Each row summarizes one generation:

| Column | Meaning |
|--------|---------|
| `gen` | generation number |
| `nevals` | number of evaluated individuals |
| `fitness` | average CV score |
| `fitness_std` | standard deviation of CV scores |
| `fitness_best` | best score found so far |
| `div` | genotype diversity (1.0 = diverse, 0.0 = converged) |
| `unique` | fraction of population with distinct configurations |
| `stag` | consecutive generations without improvement |

Inspect the full history as a DataFrame:

```python
import pandas as pd

history = pd.DataFrame(evolved_estimator.history)
print(history[[
    "gen", "fitness_best", "genotype_diversity",
    "unique_individual_ratio", "stagnation_generations",
]])
```

The last generations from the rendered docs example:

<!-- docs-example:basic-usage-history-output:start -->
| Generation | Best CV accuracy | Diversity | Unique ratio | Stagnation |
| --- | --- | --- | --- | --- |
| 1 | 0.9244 | 0.4500 | 0.8333 | 0 |
| 2 | 0.9252 | 0.1500 | 0.5000 | 0 |
| 3 | 0.9252 | 0.2000 | 0.5000 | 1 |
| 4 | 0.9327 | 0.2500 | 0.5000 | 0 |
| 5 | 0.9327 | 0.5000 | 0.8333 | 1 |
<!-- docs-example:basic-usage-history-output:end -->

Check evaluation cost via `fit_stats_`:

```python
print(evolved_estimator.fit_stats_)
# evaluated_candidates: total individuals presented to the evaluator
# unique_candidates:    distinct configurations actually cross-validated
# cache_hits:           evaluations reused from the fitness cache
# random_immigrants:    individuals injected when diversity control triggered
# skipped_invalid_candidates: configs that raised exceptions during fit
```

The generated run records the concrete evaluation cost:

<!-- docs-example:basic-usage-fit-stats-output:start -->
```text
evaluated_candidates: 66
unique_candidates: 65
cache_hits: 1
random_immigrants: 4
skipped_invalid_candidates: 0
```
<!-- docs-example:basic-usage-fit-stats-output:end -->

After fitting, `GASearchCV` behaves like a fitted scikit-learn estimator:

```python
print(evolved_estimator.best_params_)

y_predict_ga = evolved_estimator.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_predict_ga))
```

Plot the fitness evolution over generations:

```python
from sklearn_genetic.plots import plot_fitness_evolution

plot_fitness_evolution(evolved_estimator)
plt.show()
```

Use `plot_search_overview` for a compact diagnostic view of convergence, diversity, optimizer decisions, and the strongest candidate solutions:

```python
from sklearn_genetic.plots import plot_search_overview

plot_search_overview(evolved_estimator, top_k=6)
plt.show()
```

![Search overview dashboard](/images/basic_usage_search_overview.png)

See which hyperparameter values were sampled:

```python
from sklearn_genetic.plots import plot_search_space

plot_search_space(evolved_estimator, features=["tol", "batch_size", "alpha"])
plt.show()
```

When you want to see how sampled values changed in evaluation order, use `plot_parameter_evolution`:

```python
from sklearn_genetic.plots import plot_parameter_evolution

plot_parameter_evolution(evolved_estimator, parameters=["tol", "batch_size", "alpha"])
plt.show()
```

![Parameter evolution over candidate evaluations](/images/basic_usage_parameter_evolution.png)

## Feature Selection

For this example we use the Iris dataset with added random noise features. The goal is to recover the informative features while ignoring noise.

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from sklearn_genetic import (
    EvolutionConfig,
    GAFeatureSelectionCV,
    PopulationConfig,
    RuntimeConfig,
)
from sklearn_genetic.plots import plot_feature_selection, plot_fitness_evolution

data = load_iris()
X, y = data["data"], data["target"]

rng = np.random.default_rng(42)
noise = rng.uniform(0, 10, size=(X.shape[0], 10))
X = np.hstack((X, noise))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=0
)
```

`GAFeatureSelectionCV` evaluates subsets of columns and tries to maximise the CV score while selecting a compact feature set. The estimator should already be configured with the hyperparameters you want to use:

```python
clf = SVC(gamma="auto")

evolved_estimator = GAFeatureSelectionCV(
    estimator=clf,
    cv=3,
    scoring="accuracy",
    evolution_config=EvolutionConfig(
        population_size=30,
        generations=20,
        keep_top_k=2,
        elitism=True,
    ),
    population_config=PopulationConfig(initializer="smart"),
    runtime_config=RuntimeConfig(n_jobs=-1, verbose=True),
)

evolved_estimator.fit(X_train, y_train)
```

Rendered output from the documentation run:

<!-- docs-example:basic-usage-feature-output:start -->
```text
Holdout accuracy: 0.9200
Selected features: 5 of 14
Selected noise features: 1
Selected feature names:
- sepal length (cm)
- sepal width (cm)
- petal length (cm)
- petal width (cm)
- noise_2
```
<!-- docs-example:basic-usage-feature-output:end -->

After fitting, `GAFeatureSelectionCV` behaves like a fitted scikit-learn estimator. Prediction methods use only the selected columns:

```python
features = evolved_estimator.support_  # boolean mask

y_predict_ga = evolved_estimator.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_predict_ga))
```

`support_` is a boolean mask: `True` means the feature was selected, `False` means it was discarded. In this example the optimizer selects the informative Iris features and ignores the random noise features.

Plot the fitness evolution for the feature-selection search:

```python
plot_fitness_evolution(evolved_estimator)
plt.show()
```

Visualize the selected feature mask directly:

```python
feature_names = list(data.feature_names) + [f"noise_{i}" for i in range(noise.shape[1])]
plot_feature_selection(evolved_estimator, feature_names=feature_names)
plt.show()
```

![Selected feature mask](/images/basic_usage_feature_selection.png)

## Tips & Gotchas

- Set `RuntimeConfig(verbose=True)` to see the per-generation log during fit.
- If `accuracy` is already near 1.0 on your dataset, try a more discriminative metric.
- `population_config=PopulationConfig(initializer="smart")` is strongly recommended — it produces a more diverse starting population and usually finds better solutions faster than `"random"`.
- Check `fit_stats_["skipped_invalid_candidates"]` after fit — a non-zero value means some parameter combinations caused the estimator to raise exceptions.

## Next Steps

- [Understanding Cross-Validation](./understand-cv) — learn what the generation log columns mean.
- [Pipeline Tuning](./pipeline-tuning) — tune a scikit-learn `Pipeline` with the `step__param` naming convention.
- [Callbacks](./callbacks) — add early stopping, progress bars, and checkpoints.
- [Troubleshooting](./troubleshooting) — common errors and slow-search diagnosis.
