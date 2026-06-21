.. _pipeline-tuning:

Pipeline Tuning with GASearchCV
================================

scikit-learn ``Pipeline`` objects let you chain preprocessing steps and an
estimator into a single object. ``GASearchCV`` tunes pipelines the same way it
tunes plain estimators — the only difference is the parameter naming
convention.

Parameter Naming Inside a Pipeline
-----------------------------------

Pipeline parameters follow the pattern ``stepname__paramname`` (two
underscores). The step name is the string you assigned when creating the
pipeline. For example, a pipeline built with:

.. code:: python3

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import GradientBoostingRegressor

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", GradientBoostingRegressor()),
    ])

exposes parameters like ``regressor__n_estimators``,
``regressor__learning_rate``, and ``scaler__with_mean``. These are the same
names used in ``param_grid`` for any sklearn search method.

Full Example: Gradient Boosting Regression Pipeline
-----------------------------------------------------

This example tunes a ``GradientBoostingRegressor`` inside a preprocessing
pipeline on the diabetes regression dataset. The search space has six
parameters with known interactions (``learning_rate`` × ``n_estimators``,
``max_depth`` × ``min_samples_leaf``).

Setup
^^^^^

.. code:: python3

    from sklearn.datasets import load_diabetes
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import r2_score, mean_squared_error
    from sklearn.model_selection import KFold, train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from sklearn_genetic import EvolutionConfig, GASearchCV, PopulationConfig, RuntimeConfig
    from sklearn_genetic.space import Categorical, Continuous, Integer

    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    cv = KFold(n_splits=4, shuffle=True, random_state=42)

Build and evaluate a default baseline to have a comparison point:

.. code:: python3

    def make_pipeline(**kwargs):
        return Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", GradientBoostingRegressor(random_state=42, **kwargs)),
        ])

    baseline = make_pipeline()
    baseline.fit(X_train, y_train)

    baseline_r2 = r2_score(y_test, baseline.predict(X_test))
    baseline_rmse = mean_squared_error(y_test, baseline.predict(X_test)) ** 0.5
    print(f"Baseline R²: {baseline_r2:.4f}  RMSE: {baseline_rmse:.2f}")

Define the Search Space
^^^^^^^^^^^^^^^^^^^^^^^^

Note the ``regressor__`` prefix on every key:

.. code:: python3

    param_grid = {
        "regressor__n_estimators": Integer(50, 200),
        "regressor__learning_rate": Continuous(0.01, 0.2, distribution="log-uniform"),
        "regressor__max_depth": Integer(1, 5),
        "regressor__min_samples_leaf": Integer(1, 12),
        "regressor__subsample": Continuous(0.6, 1.0),
        "regressor__loss": Categorical(["squared_error", "absolute_error", "huber"]),
    }

The ``log-uniform`` distribution for ``learning_rate`` samples small values
more often than large ones, which matches the prior that small learning rates
are generally more interesting.

Configure and Run
^^^^^^^^^^^^^^^^^^

.. code:: python3

    search = GASearchCV(
        estimator=make_pipeline(),
        param_grid=param_grid,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        evolution_config=EvolutionConfig(
            population_size=20,
            generations=15,
            elitism=True,
            keep_top_k=3,
        ),
        population_config=PopulationConfig(
            initializer="smart",
            warm_start_configs=[
                {
                    "regressor__n_estimators": 100,
                    "regressor__learning_rate": 0.1,
                    "regressor__max_depth": 3,
                    "regressor__min_samples_leaf": 4,
                    "regressor__subsample": 0.8,
                    "regressor__loss": "squared_error",
                }
            ],
        ),
        runtime_config=RuntimeConfig(n_jobs=-1, parallel_backend="auto", use_cache=True),
    )

    search.fit(X_train, y_train)

    print("Best CV negative RMSE:", round(search.best_score_, 4))
    print("Best parameters:", search.best_params_)

A ``warm_start_configs`` entry seeds the initial population with a known-good
configuration. The optimizer then explores variations around it alongside the
LHS-sampled candidates.

Evaluate on the Holdout Set
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After fitting, the search object behaves like a fitted pipeline. Call
``predict`` directly:

.. code:: python3

    ga_r2 = r2_score(y_test, search.predict(X_test))
    ga_rmse = mean_squared_error(y_test, search.predict(X_test)) ** 0.5

    print(f"Baseline → R²: {baseline_r2:.4f}  RMSE: {baseline_rmse:.2f}")
    print(f"GA tuned → R²: {ga_r2:.4f}  RMSE: {ga_rmse:.2f}")

Inspect Evaluation Cost
^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python3

    print(search.fit_stats_)
    # evaluated_candidates: total individuals presented to the evaluator
    # unique_candidates:    distinct configurations actually cross-validated
    # cache_hits:           re-used scores from the fitness cache
    # random_immigrants:    individuals injected by diversity control

Visualize the Search
^^^^^^^^^^^^^^^^^^^^^

.. code:: python3

    import matplotlib.pyplot as plt
    from sklearn_genetic.plots import plot_fitness_evolution, plot_search_space

    plot_fitness_evolution(search)
    plt.show()

    # Inspect which learning_rate / n_estimators pairs were explored
    plot_search_space(
        search,
        features=["regressor__learning_rate", "regressor__n_estimators"],
    )
    plt.show()

Common Pitfalls
---------------

**Wrong step name in ``param_grid``**
    The step name must exactly match what you passed to ``Pipeline([...])``. If
    your pipeline uses ``("clf", LogisticRegression())``, the parameter is
    ``clf__C``, not ``logistic__C`` or ``C``.

**Tuning preprocessor parameters**
    You can also tune ``scaler__with_std``, ``pca__n_components``, or any
    preprocessor parameter using the same ``stepname__param`` pattern. When
    the preprocessor parameters change, the transformation changes, so the GA
    effectively searches the combined (preprocessing + model) space.

**Negative scorers for regression**
    sklearn convention is to maximize scores, so regression losses must be
    negated: ``"neg_root_mean_squared_error"``, ``"neg_mean_absolute_error"``.
    ``GASearchCV`` uses ``criteria="max"`` by default, which is correct for
    negative scorers.

**Nested parallelism**
    By default ``RuntimeConfig(parallel_backend="auto")`` parallelizes across
    unique candidates in a generation. If your pipeline itself uses ``n_jobs``
    internally (e.g., ``RandomForestClassifier(n_jobs=-1)``), you may get
    oversubscription. Either set the estimator's ``n_jobs=1``, or switch to
    ``RuntimeConfig(parallel_backend="cv")`` to parallelize within each
    candidate's cross-validation instead.

Next Steps
----------

* :doc:`callbacks` — stop the search early when the score plateaus.
* :doc:`adapters` — schedule crossover and mutation probabilities over
  generations.
* :doc:`advanced_optimizer_control` — diversity control, local refinement, and
  fitness sharing for harder pipeline spaces.
