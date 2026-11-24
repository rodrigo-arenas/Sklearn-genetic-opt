---
title: "Hyperparameter Search with a Precomputed Kernel"
description: "Tune a scikit-learn SVC with GASearchCV on a precomputed RBF kernel. Use separate training and test kernel matrices."
---

:::warning Development version
This is the **latest (dev)** documentation. It may contain unreleased features or breaking changes. For the stable release, use [stable](/stable/).
:::

# Hyperparameter Search with a Precomputed Kernel

GASearchCV can tune an estimator that consumes a precomputed kernel. Here the kernel uses a fixed RBF function and the search tunes the SVC regularization parameter C.

## Build the training and test kernels

Split the original data first. The square training kernel has one row
and column for each training sample. The rectangular test kernel has
one row per test sample and one column per training sample.

```python
from sklearn.datasets import load_iris
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.svm import SVC

from sklearn_genetic import GASearchCV
from sklearn_genetic.space import Categorical

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)
K_train = rbf_kernel(X_train, gamma=0.2)
K_test = rbf_kernel(X_test, X_train, gamma=0.2)
print("Training kernel shape", K_train.shape)
print("Test kernel shape", K_test.shape)

search = GASearchCV(
    SVC(kernel="precomputed"),
    param_grid={"C": Categorical([0.25, 0.5, 1.0, 2.0, 4.0, 8.0])},
    cv=StratifiedKFold(3, shuffle=True, random_state=42),
    scoring="accuracy",
    population_size=6,
    generations=2,
    n_jobs=1,
    random_state=42,
    verbose=False,
).fit(K_train, y_train)
print("Selected C", search.best_params_["C"])
print("Search CV accuracy", round(search.best_score_, 3))
print("Holdout accuracy", round(search.score(K_test, y_test), 3))
```

```text
Training kernel shape (112, 112)
Test kernel shape (38, 112)
Selected C 0.5
Search CV accuracy 0.964
Holdout accuracy 0.947
```

Within each split, scikit-learn takes a square training submatrix and
selects validation rows against only the training-fold columns.
For feature selection, pass the original feature matrix to
`GAFeatureSelectionCV` with an estimator such as
`SVC(kernel="rbf", gamma=0.2)`.

- [GASearchCV API](../api/gasearchcv)
- [Precomputed inputs in feature selection](../guide/troubleshooting#precomputed-kernels-and-distances)
