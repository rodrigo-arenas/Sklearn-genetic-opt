.. _troubleshooting:

Troubleshooting
===============

This page covers the most common problems encountered when using
sklearn-genetic-opt, organized by symptom.

.. contents::
   :local:
   :depth: 1

----

Parameter errors
----------------

**"ValueError: parameter X is not a valid parameter for estimator Y"**

The keys in ``param_grid`` must exactly match sklearn's parameter names for
the estimator. For plain estimators, check
``estimator.get_params().keys()``. For pipelines, the pattern is
``stepname__paramname``:

.. code:: python3

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier

    pipe = Pipeline([("scaler", StandardScaler()), ("clf", RandomForestClassifier())])
    print(list(pipe.get_params().keys()))
    # -> ['scaler', 'clf', 'scaler__copy', ..., 'clf__n_estimators', ...]

If the pipeline step is named ``"clf"`` then the key must be
``"clf__n_estimators"``, not ``"n_estimators"`` or ``"rf__n_estimators"``.

----

**"KeyError" or "unexpected keyword argument" during fit**

This usually means a parameter value from ``param_grid`` is invalid for the
estimator in that configuration. Check:

* Integer ranges do not include values the estimator rejects (e.g.,
  ``max_depth=0`` is invalid for most tree models; use ``Integer(1, ...)``)
* Categorical choices are all valid strings or ``None``, not mixed types
  that the estimator cannot handle

----

All candidates score the same
------------------------------

**Every generation shows the same ``fitness`` and ``fitness_best``**

Possible causes:

1. **The scoring metric is saturated.** On a small dataset a simple model
   can already achieve 100% accuracy. Try a harder dataset or a more
   discriminative metric.

2. **The search space is too narrow.** If all parameter combinations produce
   similar models, expand the ranges or add more parameters.

3. **The cross-validation strategy is deterministic across runs.** Set
   ``shuffle=True`` and a ``random_state`` on your CV splitter.

4. **``error_score`` is masking failures.** If the estimator raises
   exceptions for some configurations, ``error_score=nan`` (default)
   replaces scores with ``nan``, which can appear as a flat line. Check
   ``fit_stats_["skipped_invalid_candidates"]``:

   .. code:: python3

       print(search.fit_stats_)
       # skipped_invalid_candidates > 0 means some configs raised exceptions

   Switch to ``RuntimeConfig(error_score="raise")`` temporarily to see the
   actual exception.

----

Search is slow
--------------

**Fit takes much longer than expected**

1. **Nested parallelism.** The default ``parallel_backend="auto"``
   parallelizes across unique candidates in a generation. If the estimator
   itself uses parallelism (e.g., ``RandomForestClassifier(n_jobs=-1)``),
   you will oversubscribe the CPU. Either set the estimator's ``n_jobs=1``,
   or switch parallelism to the CV level:

   .. code:: python3

       runtime_config=RuntimeConfig(parallel_backend="cv", n_jobs=-1)

   This keeps candidate evaluation serial but parallelizes the cross-
   validation splits for each candidate.

2. **Too many unique candidates.** Check
   ``fit_stats_["unique_candidates"]``. If it equals
   ``fit_stats_["evaluated_candidates"]`` and ``cache_hits`` is zero, the
   cache is not helping. This is normal on the first run, but if you
   re-run with the same estimator, warm-starting or reducing the search
   space might help.

3. **Population is too large relative to the space.** On a small space with
   many duplicate candidates, reduce ``population_size`` and add more
   generations instead.

----

Population converges too fast
-------------------------------

**``genotype_diversity`` drops to zero within a few generations**

The population has converged prematurely — all individuals are nearly
identical and further generations find nothing new. Remedies:

* Check that ``diversity_control=True`` is active (it is by default as of
  0.13.0). Verify via ``search.diversity_control``.
* Lower ``diversity_threshold`` if it is higher than the observed
  ``genotype_diversity`` floor, or raise it if diversity control is not
  triggering early enough.
* Increase ``random_immigrants_fraction`` to inject more fresh individuals
  when triggered.
* Reduce ``tournament_size`` to lower selection pressure and keep weaker
  individuals in the gene pool longer.
* Increase ``population_size``. A larger population maintains diversity
  naturally.

Inspect the history to diagnose:

.. code:: python3

    import pandas as pd

    history = pd.DataFrame(search.history)
    print(history[["gen", "genotype_diversity", "unique_individual_ratio",
                    "stagnation_generations", "diversity_control_triggered"]])

----

**``stagnation_generations`` keeps growing**

The best score is not improving. This may mean:

* The search has found a genuine optimum. Check if the score is already
  close to what you expect from domain knowledge.
* The population has converged around a local optimum. See the diversity
  section above.
* The scoring metric is too noisy for the CV fold count. Increase
  ``cv`` splits to reduce variance in the fitness signal.

Use a callback to stop early and avoid wasting evaluations:

.. code:: python3

    from sklearn_genetic.callbacks import ConsecutiveStopping

    search.fit(X_train, y_train, callbacks=[
        ConsecutiveStopping(generations=8, metric="fitness_best"),
    ])

----

Understanding ``fit_stats_``
-----------------------------

After fitting, ``search.fit_stats_`` is a dictionary with evaluation
counters:

.. code:: python3

    {
        "evaluated_candidates":        420,  # total individuals presented
        "unique_candidates":           310,  # distinct configs cross-validated
        "cross_validate_calls":        310,  # actual CV calls made
        "cache_hits":                  110,  # scores reused from cache
        "duplicate_candidates":          0,  # within-generation duplicates
        "skipped_invalid_candidates":    0,  # configs that raised exceptions
        "population_parallel_batches":  21,  # batches run in parallel
        "population_serial_batches":     0,  # batches run serially
        "random_immigrants":            12,  # individuals injected by diversity control
        "local_refinement_candidates":   2,  # hall-of-fame neighbors evaluated
    }

Key ratios to check:

* ``cache_hits / evaluated_candidates`` — cache efficiency; above 20% is
  good for a search with many generations.
* ``skipped_invalid_candidates > 0`` — some parameter combinations caused
  the estimator to raise exceptions.
* ``random_immigrants > 0`` — diversity control was triggered at least once.

----

Reproducibility
---------------

**Results differ between runs with the same code**

sklearn-genetic-opt uses Python's ``random`` module and NumPy's random
generator. Set both before fitting:

.. code:: python3

    import random
    import numpy as np

    random.seed(42)
    np.random.seed(42)

    search.fit(X_train, y_train)

Also seed the CV splitter and any estimator that accepts ``random_state``.
See :doc:`reproducibility` for a complete example.

----

Warm-start configs are ignored
-------------------------------

**Warm-start seeds do not appear in the first generation**

Warm-start configs in ``PopulationConfig(warm_start_configs=[...])`` are
validated against the search space before being used. A config is silently
skipped if:

* A key is not in ``param_grid``.
* A value is out of range for an ``Integer`` or ``Continuous`` dimension.
* A value is not in the ``choices`` list of a ``Categorical`` dimension.

Check your config keys against ``list(search.param_grid.keys())`` and
the bounds of each dimension.

----

Multi-metric search: ``best_params_`` is not what I expected
-------------------------------------------------------------

With a multi-metric scoring dict, ``GASearchCV`` optimizes the ``refit``
metric during the evolutionary search. ``best_params_`` and ``best_score_``
always refer to the ``refit`` metric, not every metric at once.

To inspect what the best configuration would be for a different metric,
query ``cv_results_`` after fitting:

.. code:: python3

    import pandas as pd

    results = pd.DataFrame(search.cv_results_)
    best_by_f1 = results.sort_values("rank_test_f1").iloc[0]
    print(best_by_f1["params"])

See :doc:`multi_metric` for a full example.

----

Plots show nothing or raise errors
-----------------------------------

``plot_fitness_evolution`` and ``plot_search_space`` require the
``seaborn`` extra:

.. code:: bash

    pip install sklearn-genetic-opt[all]
    # or just seaborn:
    pip install seaborn

The plotting helpers operate on the fitted estimator's ``logbook``
attribute. If ``verbose=False`` was set during fit, the logbook is still
populated — the flag only controls printed output.

----

Getting more help
-----------------

* Check the :doc:`understand_cv` tutorial for a detailed explanation of the
  generation log and evaluation process.
* Inspect :doc:`advanced_optimizer_control` for guidance on diversity
  control, fitness sharing, and local search telemetry.
* Open an issue at
  `github.com/rodrigo-arenas/Sklearn-genetic-opt/issues
  <https://github.com/rodrigo-arenas/Sklearn-genetic-opt/issues>`__
  and include the output of ``search.fit_stats_``, the relevant section of
  ``pd.DataFrame(search.history)``, and a minimal reproducible example.
