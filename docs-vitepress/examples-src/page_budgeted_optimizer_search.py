"""Build the budgeted cross-validation and checkpoint resume tutorial."""

from _nbgen import Notebook

nb = Notebook(
    path="tutorials/budgeted-optimizer-search.md",
    title="Resume Scikit-learn Tuning with a Cross-Validation Budget",
    description=(
        "Use GASearchCV, CVBudgetStopping, and ModelCheckpoint to cap new "
        "cross-validation evaluations, save progress, and resume tuning with pymoo."
    ),
    intro=(
        "Split a `GASearchCV` run across two compute sessions. Each session gets a "
        "limit on new cross-validation (CV) calls. The checkpoint controls how "
        "much of that work you can carry into the next session."
    ),
)

nb.md("""
    ## Set up the search and fixed CV splits

    Use the development checkout of `sklearn-genetic-opt` with the pymoo extra
    installed (`python -m pip install -e '.[pymoo]'`). The code uses a temporary
    path for the example. Use a persistent path that both sessions can access
    for a real run. Start with a fresh file and load only trusted checkpoints.

    The test split stays outside model selection. Save the training row order,
    CV selectors, scorer, search settings, and dependency versions so the next
    session can recreate them. Here the selectors are Python lists of row indices.
    """)

nb.code("""
    from pathlib import Path
    from tempfile import TemporaryDirectory

    from sklearn.base import clone
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import StratifiedKFold, train_test_split
    from sklearn.tree import DecisionTreeClassifier

    from sklearn_genetic import GASearchCV
    from sklearn_genetic.callbacks import CVBudgetStopping, ModelCheckpoint
    from sklearn_genetic.space import Continuous, Integer

    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=41
    )
    splitter = StratifiedKFold(3, shuffle=True, random_state=41)
    folds = [(train.tolist(), test.tolist()) for train, test in splitter.split(X_train, y_train)]

    template = GASearchCV(
        DecisionTreeClassifier(random_state=41),
        param_grid={"max_depth": Integer(2, 20), "ccp_alpha": Continuous(0.0, 0.04)},
        scoring="balanced_accuracy",
        cv=folds,
        optimizer_backend="pymoo",
        population_size=6,
        generations=4,
        final_selection="reuse_folds",
        final_selection_top_k=2,
        use_cache=False,
        random_state=41,
        n_jobs=1,
        verbose=False,
        error_score="raise",
    )
    """)

nb.md("""
    ## Stop after 20 new CV calls

    A CV call scores one candidate across the three folds. Set
    `every_n_generations=2` to save at generation indices 0, 2, and 4.
    Here the first fit completes generation 2 after 18 calls and then scores
    two candidates in generation 3. Its last history row describes the partial
    generation. The file stores the completed generation 2, and the callback
    reports the two fresh evaluations that were outside that checkpoint.
    Caching is disabled here to make the generation counts predictable.
    """)

nb.code("""
    artifacts = TemporaryDirectory()
    path = Path(artifacts.name) / "tree-search.pkl"
    checkpoint = ModelCheckpoint(path, every_n_generations=2)

    first = clone(template).fit(
        X_train, y_train, callbacks=[CVBudgetStopping(20), checkpoint]
    )
    print("Fresh CV calls:", first.fit_counts_["search_cv_calls"])
    print("Partial final row:", first.history["partial"][-1])
    print("Last saved generation:", checkpoint.load()["logbook"][-1]["gen"])
    print("Recoverable CV calls:", checkpoint.resumable_cv_calls_)
    print("CV calls outside the saved snapshot:", checkpoint.unsaved_cv_calls_)
    """)

nb.md("""
    ## Resume for 12 more calls

    Point a new callback at the existing file and pass it to `fit` with the same
    data and search definition. The callback budget counts new search calls in
    this session. `fit_stats_` includes work recovered from the checkpoint.
    In this run the resumed fit starts with 18 saved calls and makes 12 new
    ones. Across the two sessions, 32 CV calls were actually performed.

    Finalist validation and the full-data refit are reported separately.
    Matching finalist folds reuse their saved scores in this example. The
    held-out split is scored only after the resumed search finishes.
    """)

nb.code("""
    resumed = clone(template).fit(
        X_train,
        y_train,
        callbacks=[CVBudgetStopping(12), ModelCheckpoint(path, every_n_generations=2)],
    )
    print("Fresh CV calls:", resumed.fit_counts_["search_cv_calls"])
    print("Cumulative CV calls:", resumed.fit_stats_["cross_validate_calls"])
    print("Reused finalist folds:", resumed.fit_counts_["finalist_reused_folds"])
    print("Refits:", resumed.fit_counts_["refit_fits"])
    print(f"Held-out balanced accuracy: {resumed.score(X_test, y_test):.3f}")
    artifacts.cleanup()
    """)

nb.md("""
    ## Interpret the budget and checkpoint

    | What to check | Where to look |
    | --- | --- |
    | Search work performed in the current session | `fit_counts_["search_cv_calls"]` and `fit_counts_["search_fold_fits"]` |
    | Fresh search work recoverable from the file | `resumable_cv_calls_` on the `ModelCheckpoint` callback |
    | Fresh work outside the checkpoint after a partial generation or skipped write | `unsaved_cv_calls_` on the callback |
    | Time in search, finalist validation, and refit | [`plot_fit_work(search, unit="seconds")`](../api/plots#plot-fit-work) |

    For a resume error, see the [checkpoint recipe](../recipes/advanced/checkpointing#diagnose-a-resume-error).
    To choose a backend, use the [matched-budget optimizer catalog](../benchmarks/optimizer-catalog).
    """)

nb.write()
