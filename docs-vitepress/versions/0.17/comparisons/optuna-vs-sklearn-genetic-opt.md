---
title: "Optuna vs sklearn-genetic-opt for scikit-learn"
description: "Compare Optuna TPE and sklearn-genetic-opt for scikit-learn tuning. Run both at the same CV budget and review benchmark results and search-space tradeoffs."
---

# Optuna vs sklearn-genetic-opt: When to Use Each

Optuna and sklearn-genetic-opt both support hyperparameter search for scikit-learn
estimators. Optuna runs trials defined by an objective function. `GASearchCV`
accepts a scikit-learn estimator, search space, scorer and CV splitter. This page
compares Optuna's TPE sampler with the default DEAP backend in `GASearchCV`.

## How Each Method Works

### Optuna with TPE

Optuna's default sampler for a single objective is the Tree-structured Parzen
Estimator (TPE). After its startup trials, TPE models parameter values associated
with better objective scores and compares their density with that of the remaining
values. It uses the density ratio to select new suggestions. Optuna also provides
[other samplers](https://optuna.readthedocs.io/en/stable/reference/samplers/index.html)
and [conditional search spaces](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/002_configurations.html).
See the [TPE sampler reference](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html).

### sklearn-genetic-opt with DEAP

The default DEAP backend maintains a population of configurations and uses
selection, crossover and mutation to propose new ones. `GASearchCV` scores each
candidate with the requested CV splitter and exposes the scores in `cv_results_`.
The library also provides
[four other optimization backends](../tutorials/optimizer-backends).

## Quick Decision Guide

| You need | Start with |
| --- | --- |
| Parameters that depend on earlier choices, such as an SVM kernel | Optuna's conditional trial suggestions |
| A search object with scikit-learn's `fit`, `best_params_` and `cv_results_` | `GASearchCV` |
| Scoring feature subsets with CV for a chosen estimator | `GAFeatureSelectionCV` |
| Trial pruning based on intermediate training metrics | An Optuna objective that reports intermediate values |
| The best quality for a fixed resource budget | Compare methods on the same splits and count the work each performs |

## Performance Benchmark

This historical [Bayesmark comparison](../benchmarks/) used a budget of 48
candidate evaluations, three seeds and three CV folds. The table reports mean
best inner CV scores. Accuracy is higher-is-better. MSE is lower-is-better.

| Dataset | Model | Metric | GASearchCV | Optuna | RandomizedSearchCV | Best displayed mean |
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

At the displayed precision, Optuna has the best mean in eight rows and
`GASearchCV` in three, with one tie. At the larger budget of 96 evaluations,
the ordering reversed on Wine with a decision tree and the MSE gap narrowed
on Diabetes with a random forest. Optuna retained the higher mean accuracy
on both random-forest classification tasks.

The [optimizer catalog](../benchmarks/optimizer-catalog#compare-with-random-search-and-optuna-tpe)
compares five engines with random search and Optuna TPE using observed CV calls
and separate holdout scores.

### CASH: Algorithm Selection and Tuning

In the [combined algorithm selection and hyperparameter tuning experiment](../benchmarks/#cash-algorithm-selection--tuning),
each optimizer completed 184 candidate evaluations per run, using three seeds
and three CV folds. The comparison used a conditional Optuna objective and
a flat encoding for `GASearchCV`.

| Optimizer | Mean accuracy ± SD | Mean evaluations | Mean wall time |
| --- | ---: | ---: | ---: |
| Optuna | 0.8636 ± 0.0050 | 184 | 59 s |
| RandomizedSearchCV | 0.8591 ± 0.0055 | 184 | 27 s |
| GASearchCV | 0.8582 ± 0.0087 | 184 | 205 s |

## Code Comparison — Same Problem, Both Tools

Install the example dependencies with `pip install optuna sklearn-genetic-opt`.
Run the following blocks in order. Both searches tune a `RandomForestClassifier`
over the same space with one worker, three fixed folds and a budget of 24
candidate CV calls. The train/test split is made before either search. Evaluate
the selected models on the test set after both searches finish.

### Set up the benchmark

```python
import optuna
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from sklearn_genetic import GASearchCV
from sklearn_genetic.callbacks import CVBudgetStopping
from sklearn_genetic.space import Categorical, Integer

seed = 42
budget = 24
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=seed
)
splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
folds = list(splitter.split(X_train, y_train))
forest = RandomForestClassifier(random_state=seed, n_jobs=1)
```

### Optuna

```python
from sklearn.base import clone

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 20, 80),
        "max_depth": trial.suggest_int("max_depth", 2, 12),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 6),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
    }
    candidate = clone(forest).set_params(**params)
    return cross_val_score(
        candidate, X_train, y_train, cv=folds, scoring="roc_auc", n_jobs=1,
        error_score="raise",
    ).mean()

optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(
    direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
)
study.optimize(objective, n_trials=budget, n_jobs=1)
optuna_winner = clone(forest).set_params(**study.best_params).fit(X_train, y_train)

print("Optuna search CV calls", len(study.trials))
print("Optuna best CV ROC AUC", study.best_value)
print("Optuna holdout ROC AUC", roc_auc_score(y_test, optuna_winner.predict_proba(X_test)[:, 1]))
```

### sklearn-genetic-opt

```python
search = GASearchCV(
    forest,
    param_grid={
        "n_estimators": Integer(20, 80),
        "max_depth": Integer(2, 12),
        "min_samples_leaf": Integer(1, 6),
        "max_features": Categorical(["sqrt", "log2"]),
    },
    cv=folds,
    scoring="roc_auc",
    population_size=6,
    generations=8,
    n_jobs=1,
    use_cache=False,
    error_score="raise",
    random_state=seed,
    verbose=False,
)
search.fit(X_train, y_train, callbacks=CVBudgetStopping(budget))

print("GASearchCV search CV calls", search.fit_counts_["search_cv_calls"])
print("GASearchCV best CV ROC AUC", search.best_score_)
print("GASearchCV holdout ROC AUC", roc_auc_score(y_test, search.predict_proba(X_test)[:, 1]))
```

One CV call fits three folds here. Both methods also fit the selected
model on the full training set. Compare the printed search CV counts
before comparing scores, especially if you change the budget or enable
caching. The holdout scores describe one test partition.

## When Optuna Is the Better Choice

- Use conditional trials when parameters depend on model choices.
- Use a [pruner](https://optuna.readthedocs.io/en/stable/reference/pruners.html)
  with a training loop that can report intermediate metrics.
- Use [parallel trials with shared storage](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/004_distributed.html)
  when multiple processes or machines contribute to a study.

## When sklearn-genetic-opt Is the Better Choice

- Use `GASearchCV` when the search should fit into scikit-learn's estimator
  workflow and use its scorers and CV splitters.
- Use `GAFeatureSelectionCV` to search over feature masks. It can serve as
  a transformer in a pipeline. See [feature selection](../tutorials/feature-selection).
- Use the [parallel CV and candidate options](../api/config#runtimeconfig),
  [callbacks](../api/callbacks), [plots](../api/plots), and
  [optional MLflow configuration](../guide/mlflow) to observe
  and control the search.

## Using Both Together

To try a two-stage search, pass parameter dictionaries from completed Optuna
trials into `PopulationConfig(warm_start_configs=...)`. Reuse the example above
and keep the starting values within the same search space.

```python
from sklearn_genetic import PopulationConfig

trials = sorted(
    study.get_trials(states=(optuna.trial.TrialState.COMPLETE,)),
    key=lambda trial: trial.value,
    reverse=True,
)
warm_search = clone(search).set_params(
    population_config=PopulationConfig(warm_start_configs=[trial.params for trial in trials[:3]])
)
warm_search.fit(X_train, y_train, callbacks=CVBudgetStopping(budget))
print("Additional search CV calls", warm_search.fit_counts_["search_cv_calls"])
```

Include both stages' CV calls in the total if you compare this workflow with
a single search. See [warm-start the search](../recipes/advanced/warm-start)
for examples and how the initializer handles duplicates.

## See Also

- [Switch optimization backends](../tutorials/optimizer-backends)
- [Optimizer catalog and matched CV counts](../benchmarks/optimizer-catalog)
- [Grid Search vs Random Search vs Bayesian vs Genetic Algorithms](./grid-search-vs-genetic-algorithms)
