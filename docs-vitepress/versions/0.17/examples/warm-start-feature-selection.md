---
title: "Warm-Start Feature Selection"
description: "Reuse a Boolean feature mask from a previous GAFeatureSelectionCV fit and score the subset on fresh cross-validation folds."
---

# Warm-Start Feature Selection

When a new batch uses the same columns, give GAFeatureSelectionCV a subset that worked on an earlier batch. The new search evaluates that subset first and can try others within its own CV budget.

```python
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, train_test_split

from sklearn_genetic import GAFeatureSelectionCV, PopulationConfig
from sklearn_genetic.callbacks import CVBudgetStopping

X, y = load_diabetes(return_X_y=True)
X_prior, X_next, y_prior, y_next = train_test_split(
    X, y, train_size=0.5, random_state=17,
)
X_train, X_test, y_train, y_test = train_test_split(
    X_next, y_next, test_size=0.25, random_state=18,
)

def selector(population_config=None):
    return GAFeatureSelectionCV(
        Ridge(alpha=0.01),
        population_config=population_config,
        population_size=8,
        generations=2,
        max_features=5,
        cv=KFold(3, shuffle=True, random_state=19),
        scoring="neg_mean_absolute_error",
        random_state=20,
        verbose=False,
    )

prior = selector().fit(X_prior, y_prior, callbacks=CVBudgetStopping(12))
known = prior.get_support()
current = selector(PopulationConfig(warm_start_masks=[known]))
current.fit(X_train, y_train, callbacks=CVBudgetStopping(4))

print("First candidate matches prior mask:", (current.cv_results_["features"][0] == known).all())
print("Current CV calls:", current.fit_counts_["search_cv_calls"])
print("Current CV MAE:", round(-current.best_score_, 2))
print("Held-out MAE:", round(-current.score(X_test, y_test), 2))
```

```text
INFO: Stopping the algorithm
INFO: Stopping the algorithm
First candidate matches prior mask: True
Current CV calls: 4
Current CV MAE: 51.07
Held-out MAE: 43.98
```

The earlier fit uses disjoint rows. Only its mask is reused. The new
fit computes each candidate's CV score from the current training rows.
For an outer CV comparison, obtain the seed from independent data or
learn it inside each outer training split.

- [Feature selection API](../api/gafeatureselectioncv#start-from-known-feature-subsets)
- [CVBudgetStopping](../api/callbacks#cvbudgetstopping)
