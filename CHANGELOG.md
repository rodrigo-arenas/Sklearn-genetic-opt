# Changelog

Full release notes with code examples are in the [documentation](https://sklearngeneticopt.rodrigo-arenas.com/).

## Unreleased

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
