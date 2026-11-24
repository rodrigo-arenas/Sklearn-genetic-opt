"""Generator for examples/unlabeled-outlier-scoring.md."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _nbgen import Notebook  # noqa: E402

nb = Notebook(
    path="examples/unlabeled-outlier-scoring.md",
    title="Custom Outlier Scoring Without Labels",
    description=(
        "Use a two-argument custom scorer with GASearchCV and GAFeatureSelectionCV "
        "to tune an IsolationForest on unlabeled data."
    ),
    intro=(
        "A custom scorer for an unlabeled outlier search receives the fitted "
        "detector and the validation rows. Use the same scorer to tune "
        "hyperparameters or select features."
    ),
)

nb.md("""
    ## Fit both searches on unlabeled samples

    The example uses the mean of `IsolationForest.score_samples`. Reserve
    test rows before comparing search candidates. `KFold` splits the
    training rows without labels.
    """)

nb.code("""
    from sklearn.datasets import make_blobs
    from sklearn.ensemble import IsolationForest
    from sklearn.model_selection import KFold, train_test_split

    from sklearn_genetic import GAFeatureSelectionCV, GASearchCV
    from sklearn_genetic.space import Categorical, Integer


    def mean_inlier_score(detector, X):
        return float(detector.score_samples(X).mean())


    X, _ = make_blobs(n_samples=96, n_features=4, centers=2, random_state=42)
    X_train, X_test = train_test_split(X, test_size=0.25, random_state=42)
    cv = KFold(n_splits=3, shuffle=True, random_state=42)
    detector = IsolationForest(n_estimators=8, n_jobs=1, random_state=42)
    options = dict(
        scoring=mean_inlier_score,
        cv=cv,
        population_size=6,
        generations=2,
        n_jobs=1,
        random_state=42,
        verbose=False,
    )
    search = GASearchCV(
        detector,
        param_grid={
            "n_estimators": Integer(6, 12),
            "max_samples": Categorical([0.5, 0.75, 1.0]),
        },
        **options,
    ).fit(X_train)
    selector = GAFeatureSelectionCV(detector, max_features=3, **options).fit(X_train)

    for name, result in (("Tuning", search), ("Feature selection", selector)):
        print(name, "CV mean", round(result.best_score_, 4))
        print(name, "holdout mean", round(result.score(X_test), 4))
    """)

nb.md("""
    `score(X_test)` calls the custom scorer on the refitted detector. The
    feature selector applies its chosen mask before scoring the test rows.
    For metrics that use known anomalies, supply the labels to `fit` and
    `score`.

    The two-argument scorer also works with other estimators that accept
    `fit(X)`. The search omits the target throughout validation and refit
    and when you call `score(X_test)`. Feature selection can take a list of
    rows, a NumPy array or a pandas DataFrame for `X`.

    - [Outlier detection guide](../guide/outliers)
    - [Choosing cross-validation folds](../guide/understand-cv)
    """)

nb.write()
print("ok unlabeled-outlier-scoring")
