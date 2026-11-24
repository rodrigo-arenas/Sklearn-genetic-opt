"""Generate an executable sparse feature selection tutorial."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _nbgen import Notebook  # noqa: E402

nb = Notebook(
    path="tutorials/sparse-feature-selection.md",
    title="Feature Selection with SciPy Sparse Matrices",
    description=(
        "Use GAFeatureSelectionCV with SciPy CSR and CSC matrices. Select columns "
        "on training data and evaluate the refitted scikit-learn classifier on held-out data."
    ),
    intro=(
        "Pass a SciPy sparse matrix to `GAFeatureSelectionCV` when the wrapped estimator "
        "supports sparse data. This example selects pixel columns from a dataset of handwritten "
        "digits and evaluates the selected features on held-out rows."
    ),
)

nb.md("""
## Fit on sparse training data

Split the data before fitting. With `refit=True` (the default), the selector
fits the winning estimator on the selected columns from all training rows.
""")
nb.code("""
    from scipy.sparse import csc_matrix
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    from sklearn_genetic import GAFeatureSelectionCV

    X, y = load_digits(n_class=2, return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=9
    )
    selector = GAFeatureSelectionCV(
        DecisionTreeClassifier(random_state=9),
        scoring="balanced_accuracy",
        cv=2,
        population_size=6,
        generations=2,
        max_features=16,
        random_state=9,
        n_jobs=1,
        verbose=False,
    ).fit(csc_matrix(X_train), y_train)
    print("Selected columns", selector.get_support(indices=True).size)
""")

nb.md("""
## Predict and score on sparse test data

`predict` and `score` apply the mask before calling the fitted estimator.
The selected columns reach that estimator in the supplied CSR or CSC format.
Choose a format your estimator supports. This decision tree accepts both.
""")
nb.code("""
    X_test_csc = csc_matrix(X_test)
    labels = selector.predict(X_test_csc)
    print("Predictions", labels.size)
    for matrix in (X_test_csc, X_test_csc.tocsr()):
        print(matrix.format, "balanced accuracy", round(selector.score(matrix, y_test), 3))
""")

nb.md("""
When you only need the reduced data, `selector.transform(X_test_csc)` returns
a CSR result using scikit-learn's transformer convention. See the
[GAFeatureSelectionCV reference](../api/gafeatureselectioncv#methods) for supported
prediction methods and sparse inputs.
""")

nb.md("""
## Use the selector in a sparse pipeline

Set `refit=False` when the selector is followed by a separate model in the
pipeline. The selector still searches over feature masks and applies the
winning mask. Scikit-learn's estimator tags report sparse input support
when the wrapped estimator supports it. The pipeline's remaining steps also
need to accept sparse input. On scikit-learn versions that propagate sparse
tags across pipeline steps, this support is also visible on the pipeline.
""")
nb.code("""
    import pandas as pd
    from sklearn.base import clone
    from sklearn.pipeline import make_pipeline

    pipeline = make_pipeline(
        clone(selector).set_params(refit=False),
        DecisionTreeClassifier(random_state=9),
    ).fit(csc_matrix(X_train), y_train)

    reduced = pipeline[0].transform(X_test_csc)
    named = pd.DataFrame.sparse.from_spmatrix(
        reduced, columns=pipeline[0].get_feature_names_out()
    )
    print("Reduced format", reduced.format, "shape", reduced.shape)
    print("First selected column names", list(named.columns[:3]))
    print("Predictions", pipeline.predict(X_test_csc).size)
""")
nb.md("""
The SciPy matrix has no column labels, so `get_feature_names_out()` reports
positions such as `x12`. The pandas sparse DataFrame attaches those labels
without converting the reduced data to a dense array.

## Compare training layouts at the same CV budget

Feature selection slices columns from the training matrix for every fresh
candidate. On a wide sparse matrix, CSC can reduce the cost of slicing.
Include the time to convert your training matrix when deciding whether to
use CSC. The example below starts both runs with the same CSR data, search
seed and folds. It disables caching to compare fresh evaluations.
""")
nb.code("""
    from time import perf_counter
    from scipy.sparse import csr_matrix
    from sklearn_genetic.callbacks import CVBudgetStopping

    training_csr = csr_matrix(X_train)
    for layout in ("csr", "csc"):
        run = clone(selector).set_params(use_cache=False)
        started = perf_counter()
        run.fit(training_csr.asformat(layout), y_train, callbacks=CVBudgetStopping(12))
        elapsed = perf_counter() - started
        counts = run.fit_counts_
        print(layout, "search CV calls", counts["search_cv_calls"],
              "search fold fits", counts["search_fold_fits"],
              "seconds", round(elapsed, 3),
              "held-out balanced accuracy", round(run.score(X_test_csc, y_test), 3))
""")

nb.md("""
Repeat the comparison on representative inputs and alternate which layout
runs first. Check that both runs reach the same number of search CV calls
and fold fits. These wall-clock times include the final refit. The held-out
score is calculated separately after timing.
""")

nb.write()
