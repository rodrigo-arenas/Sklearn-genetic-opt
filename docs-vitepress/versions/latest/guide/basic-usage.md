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

Check evaluation cost via `fit_stats_`:

```python
print(evolved_estimator.fit_stats_)
# evaluated_candidates: total individuals presented to the evaluator
# unique_candidates:    distinct configurations actually cross-validated
# cache_hits:           evaluations reused from the fitness cache
# random_immigrants:    individuals injected when diversity control triggered
# skipped_invalid_candidates: configs that raised exceptions during fit
```

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

See which hyperparameter values were sampled:

```python
from sklearn_genetic.plots import plot_search_space

plot_search_space(evolved_estimator, features=["tol", "batch_size", "alpha"])
plt.show()
```

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
from sklearn_genetic.plots import plot_fitness_evolution

data = load_iris()
X, y = data["data"], data["target"]

noise = np.random.uniform(0, 10, size=(X.shape[0], 10))
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
