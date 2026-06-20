.. _advanced-optimizer-control:

Advanced Optimizer Control
==========================

Introduction
------------

This tutorial shows how to use the optimizer-control features designed for
harder search spaces: spaces with many local optima, noisy validation scores,
or early population collapse.

The tools covered here are:

* ``PopulationConfig(initializer="smart")`` to start with better coverage.
* Adaptive mutation and crossover schedules.
* ``OptimizationConfig(diversity_control=True)`` to react when the population
  becomes too similar.
* ``OptimizationConfig(fitness_sharing=True)`` to reduce selection pressure on
  crowded niches.
* ``OptimizationConfig(local_search=True)`` to refine the best candidates after
  the genetic search.

These features are optional. The default behavior remains conservative, and the
advanced controls are best introduced one at a time when telemetry shows that
the optimizer needs them.

When to Use These Controls
--------------------------

Use these controls when one of the following patterns appears in
``estimator.history``:

* ``unique_individual_ratio`` drops quickly toward zero.
* ``genotype_diversity`` is low while the score is still not improving.
* ``stagnation_generations`` grows for several generations.
* Many high-scoring candidates are small variations of the same solution.
* The final solution is good, but nearby configurations might be better.

The controls target different parts of the optimization process:

============================== ==============================================
Control                        Main purpose
============================== ==============================================
``population_initializer``     Better initial exploration
Adaptive schedules             Change exploration pressure over generations
``diversity_control``          Recover diversity after collapse or stagnation
``fitness_sharing``            Keep multiple promising niches alive
``local_search``               Improve exploitation near hall-of-fame solutions
============================== ==============================================

Hyperparameter Search Example
-----------------------------

The following example tunes a random forest on a classification problem. It
uses a broad search space where multiple different regions can perform well.

.. code:: python3

    import pandas as pd

    from sklearn.datasets import load_breast_cancer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.model_selection import StratifiedKFold, train_test_split

    from sklearn_genetic import (
        EvolutionConfig,
        GASearchCV,
        OptimizationConfig,
        PopulationConfig,
        RuntimeConfig,
    )
    from sklearn_genetic.schedules import ExponentialAdapter, InverseAdapter
    from sklearn_genetic.space import Categorical, Integer

    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

    param_grid = {
        "n_estimators": Integer(50, 250),
        "max_depth": Integer(2, 20),
        "min_samples_split": Integer(2, 20),
        "min_samples_leaf": Integer(1, 10),
        "max_features": Categorical(["sqrt", "log2", None]),
        "criterion": Categorical(["gini", "entropy", "log_loss"]),
    }

    mutation_schedule = ExponentialAdapter(
        initial_value=0.8,
        end_value=0.25,
        adaptive_rate=0.08,
    )
    crossover_schedule = InverseAdapter(
        initial_value=0.25,
        end_value=0.55,
        adaptive_rate=0.05,
    )

    search = GASearchCV(
        estimator=RandomForestClassifier(random_state=42, n_jobs=1),
        param_grid=param_grid,
        cv=cv,
        scoring="roc_auc",
        evolution_config=EvolutionConfig(
            population_size=24,
            generations=18,
            mutation_probability=mutation_schedule,
            crossover_probability=crossover_schedule,
            tournament_size=3,
            elitism=True,
            keep_top_k=4,
        ),
        population_config=PopulationConfig(initializer="smart"),
        runtime_config=RuntimeConfig(n_jobs=-1, parallel_backend="auto", verbose=True),
        optimization_config=OptimizationConfig(
            diversity_control=True,
            diversity_threshold=0.18,
            diversity_stagnation_generations=4,
            diversity_mutation_boost=1.8,
            random_immigrants_fraction=0.15,
            fitness_sharing=True,
            sharing_radius=0.25,
            sharing_alpha=1.0,
            local_search=True,
            local_search_top_k=2,
            local_search_steps=4,
            local_search_radius=0.08,
        ),
    )

    search.fit(X_train, y_train)

    y_pred = search.predict(X_test)
    y_proba = search.predict_proba(X_test)[:, 1]

    print(search.best_params_)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_proba))

Reading Optimizer Telemetry
---------------------------

After fitting, convert ``history`` to a dataframe to inspect the search:

.. code:: python3

    history = pd.DataFrame(search.history)

    columns = [
        "gen",
        "fitness_best",
        "fitness_max",
        "unique_individual_ratio",
        "genotype_diversity",
        "stagnation_generations",
        "mutation_probability",
        "diversity_control_triggered",
        "random_immigrants",
        "duplicate_replacements",
        "fitness_sharing_applied",
        "mean_niche_count",
        "max_niche_count",
        "local_refinements",
    ]

    print(history[columns])
    print(search.fit_stats_)

The most useful fields are:

``unique_individual_ratio``
    Fraction of candidates in the generation that are unique. Low values mean
    the population contains many duplicates.

``genotype_diversity``
    Average per-gene diversity. Low values mean the candidates are becoming
    structurally similar, even if they are not exact duplicates.

``stagnation_generations``
    Number of generations since the best score improved.

``diversity_control_triggered``
    Whether diversity control reacted in the generation.

``random_immigrants``
    Number of random candidates injected into the offspring.

``duplicate_replacements``
    Number of duplicate offspring replaced before evaluation.

``fitness_sharing_applied``
    Whether niche-aware selection pressure was active.

``mean_niche_count`` and ``max_niche_count``
    How crowded candidates were during selection. Higher values mean more
    candidates occupied similar regions.

``local_refinements``
    Number of local neighbor candidates evaluated after the main genetic
    search. This is usually non-zero only in the final logbook row.

Feature Selection Example
-------------------------

The same controls can be used with
:class:`~sklearn_genetic.GAFeatureSelectionCV`. In feature selection,
``local_search_radius`` controls the fraction of feature bits flipped when
creating local neighbors.

.. code:: python3

    from sklearn.datasets import load_breast_cancer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, train_test_split

    from sklearn_genetic import (
        EvolutionConfig,
        GAFeatureSelectionCV,
        OptimizationConfig,
        PopulationConfig,
        RuntimeConfig,
    )
    from sklearn_genetic.schedules import ExponentialAdapter

    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

    selector = GAFeatureSelectionCV(
        estimator=RandomForestClassifier(random_state=42, n_jobs=1),
        cv=cv,
        scoring="roc_auc",
        max_features=18,
        evolution_config=EvolutionConfig(
            population_size=30,
            generations=16,
            mutation_probability=ExponentialAdapter(0.7, 0.2, 0.08),
            crossover_probability=0.35,
            keep_top_k=4,
        ),
        population_config=PopulationConfig(initializer="smart"),
        runtime_config=RuntimeConfig(n_jobs=-1, parallel_backend="auto", verbose=True),
        optimization_config=OptimizationConfig(
            diversity_control=True,
            diversity_threshold=0.2,
            diversity_stagnation_generations=4,
            random_immigrants_fraction=0.15,
            fitness_sharing=True,
            sharing_radius=0.2,
            local_search=True,
            local_search_top_k=2,
            local_search_steps=5,
            local_search_radius=0.1,
        ),
    )

    selector.fit(X_train, y_train)

    print("Selected features:", selector.best_features_)
    print("Test score:", selector.score(X_test, y_test))

Recommended Workflow
--------------------

Start from the simplest useful setup and add controls based on telemetry:

1. Use ``PopulationConfig(initializer="smart")`` and a reasonable adaptive mutation
   schedule.
2. Inspect ``unique_individual_ratio``, ``genotype_diversity``, and
   ``stagnation_generations``.
3. If diversity collapses early, enable ``OptimizationConfig(diversity_control=True)``.
4. If one candidate family dominates too quickly, enable
   ``OptimizationConfig(fitness_sharing=True)``.
5. If the final region looks promising but not fully refined, enable
   ``OptimizationConfig(local_search=True)``.

Tuning Guidelines
-----------------

``diversity_threshold``
    Values between ``0.1`` and ``0.3`` are a practical starting point. Increase
    it if you want diversity control to react earlier.

``diversity_mutation_boost``
    Values between ``1.5`` and ``2.5`` usually provide a noticeable mutation
    increase without turning the generation into a fully random search.

``random_immigrants_fraction``
    Start between ``0.05`` and ``0.2``. Larger values can help rugged spaces,
    but may slow convergence.

``sharing_radius``
    Values between ``0.15`` and ``0.35`` are often useful. Smaller values only
    penalize very similar candidates; larger values preserve broader niches.

``local_search_radius``
    Small values such as ``0.05`` to ``0.15`` keep refinement local. Larger
    values behave more like another mutation phase.

``local_search_steps``
    Keep this small at first. Each step creates extra candidates that must be
    evaluated with cross-validation.

Practical Notes
---------------

These controls improve search behavior, but they do not remove the need for a
good validation design. Use reproducible cross-validation splits, keep a final
holdout set for model assessment, and compare multiple random seeds when the
metric is noisy.

Because ``fitness_sharing`` only changes temporary selection pressure, it does
not alter raw cross-validation scores, ``best_score_``, or ``cv_results_``.
Because ``local_search`` evaluates extra neighbor candidates after the genetic
search, it can improve final quality but may increase total runtime.
