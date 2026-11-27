---
title: "Feature Selection for Categorical Columns"
description: "Select raw categorical columns with GAFeatureSelectionCV, using a pipeline that fits OneHotEncoder or a dtype-based ColumnTransformer in each CV fold."
---

:::warning Development version
This is the **latest (dev)** documentation. It may contain unreleased features or breaking changes. For the stable release, use [stable](/stable/).
:::

# Feature Selection for Categorical Columns

Select whole input columns by putting a categorical encoder inside the estimator passed to GAFeatureSelectionCV. For each candidate mask, cross-validation fits the encoder and classifier on the training fold.

## Fit on categorical inputs

This example uses fixed cutoffs to turn the Iris measurements into two
categories per column. `OneHotEncoder` receives the selected raw columns
by position.

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from sklearn_genetic import GAFeatureSelectionCV

iris = load_iris(as_frame=True)
values = np.where(iris.data.to_numpy() < [5.8, 3.0, 3.8, 1.2], "low", "high")
X = pd.DataFrame(values, columns=iris.feature_names)
X_train, X_test, y_train, y_test = train_test_split(
    X, iris.target, test_size=0.25, stratify=iris.target, random_state=42
)

model = make_pipeline(
    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
    DecisionTreeClassifier(max_depth=3, random_state=42),
)
selector = GAFeatureSelectionCV(
    model,
    cv=StratifiedKFold(3, shuffle=True, random_state=42),
    scoring="accuracy",
    max_features=2,
    population_size=6,
    generations=3,
    n_jobs=1,
    random_state=42,
    verbose=False,
).set_output(transform="pandas")
selector.fit(X_train, y_train)

print("Selected columns", selector.get_feature_names_out().tolist())
print("Search CV accuracy", round(selector.best_score_, 3))
print("Holdout accuracy", round(selector.score(X_test, y_test), 3))
```

```text
Selected columns ['sepal width (cm)', 'petal length (cm)']
Search CV accuracy 0.759
Holdout accuracy 0.684
```

`selector.transform(X_test)` returns the selected original columns with
their category values. `selector.predict(X_test)` applies the refitted
encoder and classifier to those columns. With `handle_unknown="ignore"`,
new category values can reach the encoder during validation or prediction.

## Route columns by dtype

A mixed DataFrame can use a `ColumnTransformer` to encode categories
and scale numeric columns. Set `preserve_dataframe=True` so the
pipeline sees the names and dtypes of the selected columns in each
fold. The callable selectors adapt to each candidate's subset.

```python
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.preprocessing import StandardScaler

mixed = iris.data.copy()
mixed["petal width (cm)"] = pd.Categorical(
    np.where(mixed["petal width (cm)"] < 1.2, "narrow", "wide")
)
X_train, X_test, y_train, y_test = train_test_split(
    mixed, iris.target, test_size=0.25, stratify=iris.target, random_state=42
)
preprocessing = make_column_transformer(
    (
        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
        make_column_selector(dtype_include="category"),
    ),
    (StandardScaler(), make_column_selector(dtype_include="number")),
)
pipeline = make_pipeline(
    preprocessing, DecisionTreeClassifier(max_depth=3, random_state=42)
)
selector = GAFeatureSelectionCV(
    pipeline,
    preserve_dataframe=True,
    cv=StratifiedKFold(3, shuffle=True, random_state=42),
    scoring="accuracy",
    max_features=3,
    population_size=6,
    generations=2,
    error_score="raise",
    random_state=42,
    verbose=False,
).fit(X_train, y_train)

fitted_preprocessing = selector.best_estimator_.named_steps["columntransformer"]
print("Columns seen by the refitted pipeline", fitted_preprocessing.feature_names_in_.tolist())
print("Holdout accuracy", round(selector.score(X_test, y_test), 3))
```

```text
Columns seen by the refitted pipeline ['sepal width (cm)', 'petal length (cm)']
Holdout accuracy 0.895
```

Pass the same input schema at prediction time. If you also want
`selector.transform(X_test)` to return a DataFrame, set
`selector.set_output(transform="pandas")`.

- [GAFeatureSelectionCV API](../api/gafeatureselectioncv)
- [Pipeline tuning guide](../guide/pipeline-tuning)
