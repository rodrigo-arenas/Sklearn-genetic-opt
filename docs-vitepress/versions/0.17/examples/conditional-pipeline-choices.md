---
title: "Conditional Hyperparameter Choices in a Pipeline"
description: "Tune a finite set of conditional scikit-learn SVM configurations with GASearchCV by passing preconfigured estimators as categorical choices."
---

# Conditional Hyperparameter Choices in a Pipeline

A linear SVM uses `C`. An RBF SVM uses both `C` and `gamma`. List each configured estimator once in `Categorical` so the search evaluates only the combinations that affect the model.

## Define the alternatives and fit

The pipeline scales features inside each fold. The grid has one parameter,
`model`, with six choices. The warm starts use copies of those
estimators to cover the list in the initial population.

```python
from sklearn.base import clone
from sklearn.datasets import load_wine
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn_genetic import GASearchCV
from sklearn_genetic.space import Categorical

X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.25, random_state=42
)
choices = [SVC(kernel="linear", C=c) for c in (0.1, 1.0)]
choices += [
    SVC(kernel="rbf", C=c, gamma=gamma)
    for c in (0.1, 1.0)
    for gamma in (0.01, 0.1)
]
pipeline = Pipeline([("scale", StandardScaler()), ("model", choices[0])])
search = GASearchCV(
    pipeline,
    param_grid={"model": Categorical(choices)},
    warm_start_configs=[{"model": clone(choice)} for choice in choices],
    cv=StratifiedKFold(3, shuffle=True, random_state=42),
    population_size=len(choices),
    generations=4,
    scoring="accuracy",
    use_cache=True,
    n_jobs=1,
    random_state=42,
    verbose=False,
).fit(X_train, y_train)

print("Search CV calls", search.fit_counts_["search_cv_calls"])
print("Finite space covered", search.search_space_exhausted_)
print("Best model", search.best_params_["model"])
print("Holdout accuracy", round(search.score(X_test, y_test), 3))
```

```text
Search CV calls 6
Finite space covered True
Best model SVC(gamma=0.1)
Holdout accuracy 0.978
```

The search covers two linear configurations and four RBF configurations.
A flat grid over `kernel`, `C`, and `gamma` would contain eight
configurations. Two extra linear configurations would each run
cross-validation with an inactive `gamma` value. Use separate grid
dimensions for parameters that apply to every alternative, such as a
shared preprocessing option.

A warm start can supply a clone of a declared estimator. The search
matches its constructor values and gives exact types priority. This
applies inside object arrays used as constructor parameters, including
cells that contain arrays or dictionaries.

- [Pipeline tuning guide](../guide/pipeline-tuning)
- [Finite coverage and fit counts](../api/optimizer-backends#fit-work-by-phase)
