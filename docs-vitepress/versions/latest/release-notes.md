---
title: Release Notes
description: Changelog for sklearn-genetic-opt.
---

:::warning Development version
You are reading the **latest (dev)** docs. For the stable version, see [stable](/stable/).
:::

# Release Notes

## 0.13.1

### New Features

- **`random_state` parameter**: `GASearchCV` and `GAFeatureSelectionCV` now accept a `random_state` argument that seeds the entire search at `fit` time — population initialisation (including Latin hypercube sampling), mutation, crossover, and random immigrants. Runs are fully reproducible without manually seeding the global `random` / `numpy` RNGs. Pass `random_state=None` (default) to keep the previous non-deterministic behaviour.

- **Expanded plotting API**: eleven new functions in `sklearn_genetic.plots` — `plot_parameter_evolution`, `plot_search_decisions`, `plot_candidate_scores`, `plot_feature_selection`, `plot_convergence`, `plot_diversity`, `plot_optimizer_events`, `plot_score_landscape`, `plot_cv_scores`, `plot_candidate_rankings`, and `plot_search_overview`. See the [Plotting Gallery](../examples/plotting-gallery) for examples.

- **Benchmarks page**: new [Benchmarks](./benchmarks/) section in the docs with Bayesmark-style comparisons of `GASearchCV` against Optuna and random search on tabular regression/classification tasks and CASH (combined algorithm selection and hyperparameter optimisation) scenarios.

### Bug Fixes

- Fixed Latin hypercube sampler reproducibility: the `smart` initialiser now seeds `qmc.LatinHypercube` from the global RNG so numeric-parameter searches are reproducible when `random_state` is set.

---

## 0.13.0

### Breaking Changes

- `crossover_probability` default changed from `0.2` → `0.8`; `mutation_probability` default changed from `0.8` → `0.1` for both `GASearchCV` and `GAFeatureSelectionCV`.

- `diversity_control` now defaults to `True` and `diversity_threshold` now defaults to `0.25`. Previously `diversity_control` defaulted to `False` and `diversity_threshold` defaulted to `0.1`. Set `diversity_control=False` to restore the previous behavior.

- The fitness function for `GASearchCV` is now **single-objective** (CV score only). Previously a `novelty_score` based on Hamming distance was included as a second objective. This caused Pareto-dominance comparisons to favor diverse-but-lower-scoring candidates over better candidates, reducing search quality. Fitness sharing and diversity control already maintain population diversity without corrupting the primary fitness signal. `GAFeatureSelectionCV` retains its two-objective fitness (CV score + feature count) unchanged.

### New Features

- **Parallel candidate evaluation**: candidates within a generation are de-duplicated and unique candidates are evaluated in parallel via `n_jobs`. Added `parallel_backend` (`"auto"`, `"population"`, `"cv"`) to control the parallelism strategy.

- **`fit_stats_`**: new attribute with evaluation counters — `evaluated_candidates`, `unique_candidates`, `cross_validate_calls`, `cache_hits`, `duplicate_candidates`, `skipped_invalid_candidates`, `random_immigrants`, `local_refinement_candidates`.

- **Optimizer telemetry in `history`**: new per-generation fields — `genotype_diversity`, `unique_individual_ratio`, `fitness_best`, `stagnation_generations`, `diversity_control_triggered`, and others.

- **Smart initialization**: `PopulationConfig(initializer="smart")` uses Latin hypercube sampling for numeric parameters, estimator defaults, warm-start seeds, and stratified categorical values. Set `initializer="random"` to use the previous behavior.

- **Grouped config objects**: `EvolutionConfig`, `PopulationConfig`, `RuntimeConfig`, and `OptimizationConfig` provide a cleaner API for advanced settings. The previous flat keyword parameters remain supported for backward compatibility.

- **Local search**: `OptimizationConfig(local_search=True)` runs a short neighborhood search around hall-of-fame candidates after the genetic search.

- **Fitness sharing**: `OptimizationConfig(fitness_sharing=True)` reduces the fitness of individuals in crowded niches to promote niche exploration.

- **Adaptive tournament selection**: `adaptive_selection=True` adjusts selection pressure based on population diversity and stagnation.

- **Final selection**: `final_selection=True` re-evaluates the top-K candidates after the GA and selects the best before refitting.

- **Uniform crossover**: `GASearchCV` now uses `cxUniform` (50% per-gene swap probability) instead of two-point crossover for mixed-type hyperparameter spaces.

- **Compact verbose log**: the generation log now shows `div`, `unique`, `stag`, and `events` columns.

- **Expanded plots**: `plot_fitness_evolution` supports multiple metrics and smoothing; `plot_history` can visualize arbitrary telemetry fields; `plot_search_space` adds pair-plot mode and correlation heatmap.

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
- Introduced novelty search strategy to `GASearchCV` (later revised in 0.13.0).

## 0.10.0

- `GAFeatureSelectionCV` now mimics the scikit-learn FeatureSelection API.
- Improved candidate generation when `max_features` is set.
- Dropped Python 3.7 support; added Python 3.10+ support.

## 0.9.0

- Introduced adaptive schedulers: `ConstantAdapter`, `ExponentialAdapter`, `InverseAdapter`, `PotentialAdapter`.
- Added `random_state` parameter to `Continuous`, `Categorical`, and `Integer`.

## 0.8.0

- Added `max_features` to `GAFeatureSelectionCV`.
- Added multi-metric evaluation support.
- Training now gracefully handles `KeyboardInterrupt`, `SystemExit`, `StopIteration`.

## 0.7.0

- Added `GAFeatureSelectionCV` for wrapper-based feature selection.

## 0.6.0

- Added `ProgressBar`, `TensorBoard`, `TimerStopping` callbacks.
- Added `on_start` / `on_end` lifecycle hooks to `BaseCallback`.
- Seaborn and MLflow are now optional extras.

## 0.5.0

- Built-in MLflow integration via `MLflowConfig`.
- Added `LogbookSaver` callback.

## 0.4.0

- Added `ConsecutiveStopping`, `ThresholdStopping`, `DeltaThreshold` callbacks.
- Added `plot_search_space` function.
- Sphinx documentation on Read the Docs.

## 0.1–0.3

- Initial release of `GASearchCV`.
- Added `param_grid`, `plot_fitness_evolution`, DEAP integration.
- Added `Space`, `Integer`, `Continuous`, `Categorical` classes.
