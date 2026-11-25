"""Generator for examples/multi-output-feature-selection.md."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _nbgen import Notebook  # noqa: E402

nb = Notebook(
    path="examples/multi-output-feature-selection.md",
    title="Feature Selection for Multi-Output Regression",
    description=(
        "Use GAFeatureSelectionCV and scikit-learn Ridge to select one set of "
        "input features for several regression targets, with per-target test errors."
    ),
    intro=(
        "Pass a two-dimensional target to select one set of features for all outputs. "
        "Ridge fits both targets. The scorer averages their errors into one number "
        "per fold to rank candidates."
    ),
)

nb.md("""
    ## Fit and evaluate the selector

    Keep the test rows aside while the selector chooses features using only the
    training rows. Predictions retain one column per target after refit.
    """)

nb.code("""
    from sklearn.datasets import make_regression
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import KFold, train_test_split

    from sklearn_genetic import GAFeatureSelectionCV

    X, y = make_regression(
        n_samples=120, n_features=8, n_informative=4, n_targets=2,
        noise=3.0, random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42,
    )
    joint_selector = GAFeatureSelectionCV(
        Ridge(), scoring="neg_mean_squared_error",
        cv=KFold(n_splits=3, shuffle=True, random_state=42),
        max_features=5, population_size=6, generations=2,
        random_state=42, n_jobs=1, verbose=False,
    ).fit(X_train, y_train)

    predictions = joint_selector.predict(X_test)
    target_mse = mean_squared_error(y_test, predictions, multioutput="raw_values")
    print("Prediction shape", predictions.shape)
    print("Holdout MSE by target", target_mse.round(2).tolist())
    """)


nb.md("""
    The search uses the negative mean squared error averaged across outputs.
    The holdout evaluation uses positive mean squared error and retains each
    target's error separately.

    ## Read a ranked shortlist

    Request the three highest-ranked rows by the search objective. The
    `candidate_index` labels locate the rows in `cv_results_`.
    """)

nb.code("""
    shortlist = joint_selector.cv_results_dataframe(
        columns=["n_features", "mean_test_score"],
        top_k=3,
        preserve_index=True,
    )
    print(shortlist.reset_index().to_string(index=False))
    """)

nb.md("""
    - [GAFeatureSelectionCV API](../api/gafeatureselectioncv)
    - [Cross-validation guide](../guide/understand-cv)
    """)

nb.write()
print("ok multi-output-feature-selection")
