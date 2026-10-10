# Changelog

Full release notes with code examples are in the [documentation](https://sklearngeneticopt.rodrigo-arenas.com/).

## Unreleased

- `GASearchCV` and `GAFeatureSelectionCV` now rank non-finite cross-validation scores last during evolution and final candidate selection, for both maximization and minimization. A candidate with one failed fold no longer receives a misleading average from its remaining folds. Raw fold scores remain visible in `cv_results_`; fitness sharing preserves valid ordering, and resumed checkpoints normalize older cached fitness values.

## 0.14.0

- Documentation now separates released pages from development pages in the sitemap, publishes `robots.txt`, and points the homepage at the current stable version. The release freeze updates homepage and component links for each new minor version. The homepage drops redundant sections, fixes broken routes, and describes trade-offs without claiming a universal speed or accuracy advantage.

### New Features

- Fast Random Forest and Extra Trees regression presets search only `squared_error`; the default `balanced` and the `wide` profiles retain `absolute_error`. A 104-fit comparison measures faster searches with the same candidate budget, with a documented MAE tradeoff on contaminated targets.

- Fresh feature searches with smart initialization and `TimerStopping` evaluate their largest initial feature set first, so a short deadline does not leave only small masks evaluated. The same candidate masks and `max_features` bound are retained; random initialization and checkpoint resumes preserve their ordering.

- Added `subsample_cv` to reduce training rows while retaining full validation folds. Both searches can re-evaluate finalists on full-training CV; `GAFeatureSelectionCV` now shares `final_selection`, `final_selection_top_k`, and `final_selection_cv` with `GASearchCV`. The default full-data search is unchanged.

- Local refinement in both searches now chains accepted elite improvements across `local_search_steps`, retaining per-step population batching and the configured candidate count.

- Added optional `algorithm="eaDifferentialEvolution"` for both searches, with domain-aware numeric mutation and bounded categorical and feature-mask handling. Benchmarks show mixed outcomes, so the default algorithm is unchanged.

- Cached finite search spaces stop after all unique configurations have been evaluated. This avoids redundant generations in small hyperparameter and feature spaces without changing their winners.

- Fitness sharing uses vectorized distances with bounded auxiliary memory for feature masks and mixed hyperparameter spaces. The isolated sharing benchmark is 7.9–14.8× faster; complete-fit timings remain dominated by model evaluation.

- Added a paired holdout benchmark against Optuna TPE, RandomizedSearchCV and optional FLAML-CFO, with per-run CV budgets, dependency versions and before/after evidence in `benchmarks/PERFORMANCE.md`.
- Extended the 0.14.0 benchmark record with sparse/imbalanced/regression feature-selection transfer tasks and corrected-default HPO comparisons against `RandomizedSearchCV` and Optuna TPE across four model families. The documented wins and losses use held-out data and retain per-run reproduction artifacts; they do not change evolutionary defaults.

- `GASearchCV` and `GAFeatureSelectionCV` now reject unsupported `fit` metadata up front with a clear error naming the parameter and the wrapped estimator, instead of failing deep inside cross-validation. Estimators whose `fit` takes `**kwargs` (e.g. `Pipeline`) are left to validate at fit time, so routed keys like `clf__sample_weight` keep working (#364).
- Added `GAFeatureSelectionCV.get_feature_names_out()` support for DataFrame column names and generated NumPy feature names.

### Bug Fixes

- `GASearchCV` now honors its documented `return_train_score=False` default when initializing scikit-learn's search base. Previously scikit-learn reset it to `True`, scoring every training fold and adding unwanted train-score fields to `cv_results_`. Explicit `return_train_score=True` still includes those fields. A paired four-family benchmark measured 1.23–1.84× lower elapsed time for cheap tree, KNN and ElasticNet searches with identical CV winners; random-forest timings showed no benefit.

- Unsupervised `GAFeatureSelectionCV.fit(X)` now accepts pandas DataFrames with outlier detectors. Input validation converts `X` to a sliceable array while retaining the original column names for `get_feature_names_out()`; previously mask evaluation raised `InvalidIndexError`. Unsupervised `GASearchCV.fit(X)` also accepts list input instead of assuming `X.shape` exists.

- `GAFeatureSelectionCV.score(X, y)` now uses the configured scorer on the selected columns, including a callable scorer or the chosen `refit` metric in multi-metric searches. Previously it always returned `best_estimator_.score(X_selected, y)`, which could report a different metric from the one optimized.

- The tagged-release workflow now refreshes the conda recipe from either a local checkout or a previous release archive, and fails clearly if it cannot recognize the complete source mapping. It uses the checksum of the exact sdist uploaded to PyPI instead of independently rebuilding one. The unreleased 0.14.0 recipe builds from the current checkout instead of the stale 0.13.2 archive and declares the missing core `pandas` dependency.

- `GASearchCV.fit` now rebuilds its search space after `set_params(param_grid=...)`, including changed parameter names and bounds. Previously a reused estimator could silently evaluate the old grid or ignore an invalid replacement.

- `GASearchCV.fit` now honors warm starts changed with `set_params(warm_start_configs=...)` or a replacement `PopulationConfig`, including removal of old seeds. Previously it kept using the configuration captured at construction, which could silently ignore new seeds or validate an obsolete one.

- Random Forest and Extra Trees regression presets omit the redundant `friedman_mse` criterion, which fails CV cloning after parameter assignment on scikit-learn 1.9. The default `balanced` and the `wide` profiles retain `squared_error` and `absolute_error`.

- Feature finalist selection preserves sparse and dense masks when preliminary scores tie at the cutoff, instead of spending every tied finalist slot on small masks. Better preliminary scores and the configured CV budget retain priority; final-score ties still favor fewer features.

- Numeric smart initialization works with SciPy versions before 1.15 by using their `seed` argument, while preserving the existing `rng` path and seeded samples on newer versions.

- `TimerStopping` stops new candidate batches within generations in both searches, including local refinement. Completed candidates remain eligible to win; in-flight CV and final selection/refit finish, and at least one initial batch is evaluated.

- Estimator-valued categorical choices preserve their configuration through crossover and cloned warm starts, reuse cached evaluations, and occupy distinct hall-of-fame slots. Applying nested parameters during search or finalist evaluation no longer modifies the caller's model choices.

- Finalist selection excludes NaN scores, raises when every finalist score is NaN, and validates an integer evaluation budget. Checkpoints retain finalist settings. An explicit non-None `final_selection_cv` takes precedence over its grouped configuration value, preserving sklearn cloning when that value contains explicit fold arrays.

- Both searches clone the winning estimator for final refit instead of fitting or changing the caller's estimator. Use `best_estimator_` (or prediction methods on the search) for the fitted model; `estimator` remains the constructor template. Numeric dimension sampling also retains global-seed reproducibility after cloning a search.

- Mu/lambda algorithms now preserve hall-of-fame elites in the next population when `elitism=True`. Best-so-far telemetry includes evaluated candidates even if stochastic survivor selection drops them.

- Diversity telemetry uses pairwise allele disagreement, avoiding permanent diversity recovery for balanced binary feature masks in larger populations.
- Integer local refinement samples neighbors in constant memory instead of allocating the full integer interval.

- Fresh fits clear cached scores while explicit checkpoint resumes retain their cache.
- Both searches use fitness-aware tournament selection with `elitism=False`, supporting negative scores and minimization without losing population members.
- Candidate ranks respect `criteria="min"`; final selection compares the requested number of distinct configurations rather than duplicate log rows.
- Feature selection rejects non-integer, zero, negative, or oversized `max_features` at fit time.

- `Continuous` now rejects a non-positive `lower` bound when `distribution="log-uniform"` at construction time, instead of failing later during `fit` with an opaque SciPy "Domain error in arguments".
- `Categorical` now rejects negative `priors`. Previously they passed validation because they summed to one, then either raised a NumPy error (seeded) or silently sampled with distorted weights (unseeded).
- Fixed pickling of fitted `GASearchCV` and `GAFeatureSelectionCV` instances. `pickle.dumps`/`pickle.loads` and `joblib.dump`/`joblib.load` previously failed on the unpicklable DEAP toolbox captured in `__dict__` (`Can't pickle <function ...>`) and now exclude the unpicklable DEAP internals (`toolbox`, `_stats`, `_pop`, `_hof`) via new `__getstate__`/`__setstate__` hooks on `GeneticEstimatorMixin`. The user-facing `hof` result is still preserved. `save()`/`load()` share the same protocol, and `load()` also accepts files written directly with `pickle.dump(self, f)` (#358).

## 0.13.4

### New Features

- Added cache-behavior benchmark coverage for candidate collisions, including checks for cache hits, duplicate candidates, and repeated `cross_validate` calls under controlled collision-heavy searches (#334).

### Bug Fixes

- Fixed `ModelCheckpoint` resume dropping prior-run history: `logbook`, `cv_results_`, and `history` were silently reset to only the post-resume generations, and `fit_stats_` counters were zeroed on every resume. `ModelCheckpoint` now persists the real per-candidate logbook and `fit_stats_`, and `GASearchCV`/`GAFeatureSelectionCV.fit` restore them after `_register()` rebuilds the (unpicklable) DEAP toolbox/population/hof (#299).
- Fixed checkpoint resume generation numbering so resumed runs continue from the saved generation instead of restarting the generation counter (#299, #331).
- Fixed checkpoint resume reproducibility by preserving and restoring optimizer random-state context across save/load cycles (#332).
- Fixed `Categorical.sample()` returning NumPy scalar values when seeded, preserving the original Python value types for sampled categorical choices (#333).

## 0.13.3

### New Features

- **Search-space conversion helper**: `from_sklearn_space` converts common `RandomizedSearchCV`-style spaces into native `Integer`, `Continuous`, and `Categorical` dimensions. It supports list-like categorical choices plus `scipy.stats.randint`, `uniform`, `loguniform`, and `reciprocal` frozen distributions.
- **Estimator presets**: new starter spaces for `RandomForestClassifier`, `RandomForestRegressor`, `HistGradientBoostingClassifier`, `HistGradientBoostingRegressor`, `LogisticRegression`, `SVC`, `XGBClassifier`, and `XGBRegressor`. Presets support `profile="fast"`, `"balanced"`, or `"wide"` and a `prefix` argument for sklearn pipelines.

### Bug Fixes

- Fixed `ModelCheckpoint` for `GAFeatureSelectionCV` by building constructor-compatible checkpoint state from a shared helper and omitting `param_grid` for estimators that do not define it.
- Fixed `random_state=0` handling for `Integer` and `Continuous` search-space dimensions so seed `0` creates a NumPy generator and produces deterministic samples.

---

## 0.13.1

### New Features

- **`random_state`**: `GASearchCV` and `GAFeatureSelectionCV` now accept a `random_state` argument that seeds the entire search at `fit` time (population init, mutation, crossover, random immigrants). Pass `random_state=None` (default) for non-deterministic behaviour.
- **Expanded plotting API**: eleven new functions — `plot_parameter_evolution`, `plot_search_decisions`, `plot_candidate_scores`, `plot_feature_selection`, `plot_convergence`, `plot_diversity`, `plot_optimizer_events`, `plot_score_landscape`, `plot_cv_scores`, `plot_candidate_rankings`, `plot_search_overview`.
- **Benchmarks**: new comparisons of `GASearchCV` vs Optuna and random search on tabular and CASH tasks.

### Bug Fixes

- Fixed Latin hypercube sampler reproducibility in the `smart` population initialiser.

---

## 0.13.0

### Breaking Changes

- `crossover_probability` default changed from `0.2` → `0.8`; `mutation_probability` default changed from `0.8` → `0.1` for both `GASearchCV` and `GAFeatureSelectionCV`.
- `diversity_control` now defaults to `True` and `diversity_threshold` now defaults to `0.25`. Previously `diversity_control` defaulted to `False` and `diversity_threshold` defaulted to `0.1`. Set `diversity_control=False` to restore the previous behavior.
- The fitness function for `GASearchCV` is now **single-objective** (CV score only). Previously a `novelty_score` based on Hamming distance was included as a second objective. `GAFeatureSelectionCV` retains its two-objective fitness (CV score + feature count) unchanged.

### New Features

- **Parallel candidate evaluation**: candidates within a generation are evaluated in parallel via `n_jobs`. New `parallel_backend` parameter (`"auto"`, `"population"`, `"cv"`) controls the parallelism strategy.
- **`fit_stats_`**: new attribute with evaluation counters — `evaluated_candidates`, `unique_candidates`, `cross_validate_calls`, `cache_hits`, `duplicate_candidates`, and others.
- **Optimizer telemetry in `history`**: new per-generation fields — `genotype_diversity`, `unique_individual_ratio`, `fitness_best`, `stagnation_generations`, `diversity_control_triggered`, and others.
- **Smart initialization**: `PopulationConfig(initializer="smart")` uses Latin hypercube sampling for numeric parameters, estimator defaults, warm-start seeds, and stratified categorical values.
- **Grouped config objects**: `EvolutionConfig`, `PopulationConfig`, `RuntimeConfig`, and `OptimizationConfig` provide a cleaner API. Flat keyword parameters remain supported for backward compatibility.
- **Local search**: `OptimizationConfig(local_search=True)` runs a short neighborhood search around hall-of-fame candidates after the genetic search.
- **Fitness sharing**: `OptimizationConfig(fitness_sharing=True)` reduces fitness of individuals in crowded niches to promote niche exploration.
- **Adaptive tournament selection**: `adaptive_selection=True` adjusts selection pressure based on population diversity and stagnation.
- **Final selection**: `final_selection=True` re-evaluates top-K candidates after the GA and selects the best before refitting.
- **Uniform crossover**: `GASearchCV` now uses `cxUniform` (50% per-gene swap probability) instead of two-point crossover for mixed-type hyperparameter spaces.
- **Expanded plots**: `plot_fitness_evolution` supports multiple metrics and smoothing; `plot_history` can visualize arbitrary telemetry fields; `plot_search_space` adds pair-plot mode and correlation heatmap.
- **Bayesmark-style benchmark**: new `benchmarks/benchmark_bayesmark.py` script and a Benchmarks documentation section compare `GASearchCV` head-to-head with Optuna (TPE) and `RandomizedSearchCV` on the standard Bayesmark datasets and search spaces under an equal evaluation budget. Optuna and SciPy are available as an optional `benchmark` extra (`pip install sklearn-genetic-opt[benchmark]`).

### Bug Fixes

- Fixed fitted estimator persistence by excluding volatile DEAP runtime objects from the saved state.
- Fixed type preservation for hyperparameter candidates across all population operations.
- Fixed smart feature-selection initialization to respect `max_features` and always select at least one feature.
- Fixed convergence telemetry so local refinement updates the final generation history row.

---

## 0.12.0

- Added compatibility for outlier detection algorithms.

## 0.11.1

- Fixed `AttributeError: 'GASearchCV' object has no attribute 'creator'`.

## 0.11.0

- Added `use_cache` parameter (default `True`) to skip re-evaluating already-seen configurations.
- Added `warm_start_configs` to `GAFeatureSelectionCV`.
- Introduced novelty search strategy to `GASearchCV`.

## 0.10.0

- `GAFeatureSelectionCV` now mimics the scikit-learn FeatureSelection API.
- Improved candidate generation when `max_features` is set.
- Dropped Python 3.7 support; added Python 3.10+ support.

## 0.9.0

- Introduced adaptive schedulers: `ConstantAdapter`, `ExponentialAdapter`, `InverseAdapter`, `PotentialAdapter`.
- Added `random_state` parameter to `Continuous`, `Categorical`, and `Integer`.

## 0.8.0

- Added `plot_search_space` for visualizing the explored search space.

## 0.7.0

- Added `GAFeatureSelectionCV` for wrapper-based feature selection.

## 0.6.0

- Added `MLflow` callback for experiment tracking.

## 0.5.0

- Added `ModelCheckpoint` callback.
- Added `plot_fitness_evolution` helper.

## 0.4.0

- Added `TensorBoard` callback for training visualization.

## 0.3.0

- Added `LogbookSaver` callback.
- Added `ProgressBar` callback.

## 0.2.0

- Added early-stopping callbacks: `ThresholdStopping`, `DeltaThreshold`, `ConsecutiveStopping`, `TimerStopping`.

## 0.1.0

- Initial release with `GASearchCV`, search spaces (`Continuous`, `Integer`, `Categorical`), and DEAP-powered genetic algorithms.
