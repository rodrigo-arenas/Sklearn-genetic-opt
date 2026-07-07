---
title: Release Notes
description: Changelog for sklearn-genetic-opt.
---

# Release Notes

## 0.13.3

### New Features and Behavior

- **Search-space conversion helper**: `from_sklearn_space` converts common `RandomizedSearchCV`-style spaces into native `Integer`, `Continuous`, and `Categorical` dimensions. It supports list-like categorical choices plus `scipy.stats.randint`, `uniform`, `loguniform`, and `reciprocal` frozen distributions.
- **Estimator presets**: new starter spaces for `RandomForestClassifier`, `RandomForestRegressor`, `HistGradientBoostingClassifier`, `HistGradientBoostingRegressor`, `LogisticRegression`, `SVC`, `XGBClassifier`, and `XGBRegressor`. Presets support `profile="fast"`, `"balanced"`, or `"wide"` and a `prefix` argument for sklearn pipelines.
- **Benchmark cache toggle**: `benchmarks/benchmark_fit.py` now exposes `--use-cache` / `--no-use-cache`, and threads the setting through both `GASearchCV` and `GAFeatureSelectionCV` benchmark builders. This makes it possible to measure the real impact of the fitness cache on repeated candidate evaluation.
- **Safer constructor validation**: `error_score` is now validated during estimator construction, so invalid values fail earlier with a clearer message.
- **More robust ranking utilities**: internal score ranking now handles `NaN` values consistently when ranking candidates.

### Bug Fixes and Error Messages

- **ModelCheckpoint cleanup**: removed a duplicate `param_grid` entry from `ModelCheckpoint` estimator state and added regression coverage for the core checkpoint keys.
- **Clearer checkpoint output**: `ModelCheckpoint` now reports successful saves with the clearer message `Checkpoint saved to ...`.
- **Better plotting errors**:
  - candidate plots now validate invalid `top_k` values before plotting
  - metric-column errors now list available metric names
  - `plot_history` errors now show available fields
  - plotting helpers now raise more actionable errors when called before fitting
- **Search-space validation improvements**:
  - `warm_start_configs` validation now reports clearer errors
  - feature-name count mismatch errors include the expected and received counts
  - preset `prefix` now validates its type
  - unsupported scipy distributions in `from_sklearn_space` now suggest list or `Categorical` alternatives
- **Optional dependency errors**: missing MLflow now raises a clearer error explaining how to install the optional extra.
- **Adapter validation**: scheduler adapter errors now include the received type.
- **Type annotations**: completed type annotations for the search-space parameter classes and fixed the `Categorical.random_state` annotation.

### Documentation

- **Clean canonical URLs**: docs canonical and Open Graph URLs now use clean VitePress-style URLs instead of raw `.html` targets.
- **Preset pipeline guidance**: added a guide section showing how to use preset search spaces with sklearn `Pipeline` objects via the preset `prefix` argument.
- **Community articles**:
  - added a Community Articles page for external posts, videos, tutorials, and articles
  - updated README and contributor-facing docs to point to the current community articles source file
  - added the first community article entry comparing `GASearchCV` and `RandomizedSearchCV`
- **Contributor guidance**: docs now remind contributors to check issue and PR status before starting work to avoid duplicated effort.
- Added [Estimator Presets](./api/presets) and expanded [Search Space](./api/space) with conversion examples and rules.
- **README improvements**:
  - added citation guidance
  - added a gentle GitHub star prompt and stars badge
  - updated community article contribution links
- **Docs navigation cleanup**: replaced the generic Links dropdown with direct navbar icons for GitHub, Star, Fork, Issues, Discussions, and PyPI; also removed the legacy Read the Docs link now that the RTD page is no longer available.
- **TimerStopping example**: callbacks docs now include a runnable `TimerStopping` example.
- **Troubleshooting expansion**: added error-reference and `parallel_backend` decision tables, plus refreshed stale `parallel_backend` examples.

### Tests and Maintenance

- Added direct tests for `_as_list` and extra edge cases including NumPy arrays, sets, empty strings, and falsy scalar values.
- Added `_candidate_label` tests covering hidden-parameter counts, `label_params`, truncation, and compact float formatting.
- Added `Categorical.random_state` determinism coverage and priors sampling validation.
- Added an internal Markdown link checker for versioned docs pages.
- Added `CITATION.cff` for GitHub citation support.

---

## 0.13.2

Documentation release. No changes to the Python package.

### New Documentation

**New Guides**

- **[How Hyperparameter Optimization Works](./guide/how-hyperparameter-optimization-works)** — complete conceptual guide comparing grid search, random search, Bayesian optimization, and genetic algorithms with worked Python examples and a method-selection flowchart.
- **[Common Hyperparameter Tuning Mistakes](./guide/common-mistakes)** — ten common pitfalls (data leakage, class imbalance, bad search spaces, missing seeds, premature stopping, and more) with diagnosis and fixes.
- **[Choosing the Right Search Space](./guide/choosing-search-spaces)** — when to use `Integer`, `Continuous`, `Categorical`; when to use log-uniform; per-estimator recommended parameter ranges.
- **[Feature Selection Methods Compared](./guide/feature-selection-guide)** — side-by-side comparison of filter, embedded, and wrapper methods with guidance on which to use and when `GAFeatureSelectionCV` is the right choice.

**New Tutorials**

- **[Random Forest Hyperparameter Tuning](./tutorials/tune-random-forest)** — 7-parameter joint search, which parameters matter most, classification and regression variants, baseline comparison.
- **[Gradient Boosting Hyperparameter Tuning](./tutorials/tune-gradient-boosting)** — `HistGradientBoostingClassifier` vs classic `GradientBoostingClassifier`, `max_leaf_nodes` vs `max_depth`, speed comparison.
- **[Logistic Regression Hyperparameter Tuning](./tutorials/tune-logistic-regression)** — solver/penalty compatibility table, multi-penalty search with SAGA, mandatory scaling in a Pipeline.
- **[SVM Hyperparameter Tuning (C, kernel, gamma)](./tutorials/tune-svm)** — C–gamma interaction visualization, mandatory `Pipeline` + `StandardScaler`, RBF vs linear kernel, O(n²) scaling note.

**New Comparisons Section**

- **[Comparisons overview](./comparisons/)** — new section hub for tool comparisons.
- **[Grid Search vs Random Search vs Bayesian vs Genetic Algorithms](./comparisons/grid-search-vs-genetic-algorithms)** — honest equal-budget benchmark across all four methods with code and result tables.
- **[Optuna vs sklearn-genetic-opt](./comparisons/optuna-vs-sklearn-genetic-opt)** — head-to-head on tabular benchmarks using the Bayesmark experimental design; honest about where each approach wins.

**New Recipes Section**

A new [Recipes](./recipes/) section provides 30 copy-paste ready solutions (5–10 min each) organized into seven categories:

- **[Classification](./recipes/classification/)** (8 recipes) — `RandomForestClassifier`, `LogisticRegression`, `SVC`, `XGBClassifier`, `LGBMClassifier`, `CatBoostClassifier`, `HistGradientBoostingClassifier`, `ExtraTreesClassifier`.
- **[Regression](./recipes/regression/)** (5 recipes) — `RandomForestRegressor`, `XGBRegressor`, `LGBMRegressor`, `CatBoostRegressor`, `ElasticNet`.
- **[Feature Selection](./recipes/feature-selection/)** (4 recipes) — high-dimensional datasets, two-stage select-then-tune, custom feature-count penalty scorer, leakage-free CV selection.
- **[Pipelines](./recipes/pipelines/)** (4 recipes) — preprocessing + estimator pipeline, `ColumnTransformer` with mixed types, imputer strategy as a hyperparameter, polynomial features degree.
- **[Scoring Metrics](./recipes/metrics/)** (5 recipes) — F1 (binary), ROC-AUC, balanced accuracy, MAE, RMSE.
- **[Integrations](./recipes/integrations/)** (3 recipes) — MLflow child-run logging, Joblib parallelism modes, Jupyter notebook setup.
- **[Advanced](./recipes/advanced/)** (5 recipes) — warm-start with known-good configs, `ConsecutiveStopping` setup, `TimerStopping` for wall-clock budgets, checkpointing and resume, custom scoring functions.

### Documentation Improvements

- **SEO titles and descriptions** — titles on 15+ existing pages rewritten to answer the search query directly (e.g. "LightGBM Hyperparameter Tuning with Genetic Algorithms" instead of "Tuning LightGBM").
- **Cross-linking** — "See Also" sections added to all tutorial and guide pages linking to related tutorials, recipes, and comparison pages.
- **Difficulty and reading-time metadata** — all tutorial pages now show difficulty level (Beginner / Intermediate / Advanced) and an estimated reading time.
- **Tutorials index** — updated with difficulty column, recommended reading order, and links to all new tutorials.
- **Sidebar** — Comparisons and Recipes sections added to the `latest` sidebar as collapsible trees. The `0.13` stable sidebar is unchanged.
- **README** — complete rewrite of `README.rst` as a high-converting GitHub landing page with value proposition, when-to-use / when-not-to-use guidance, a six-tool comparison table, condensed Quick Start, visual demo section, common use cases, and learning paths.
- **Canonical URLs and Open Graph tags** — `transformPageData` in `config.ts` now injects `<link rel="canonical">`, `og:title`, `og:description`, `og:url`, `og:image`, and `twitter:*` tags on every page.

---

## 0.13.1

### New Features

- **`random_state` parameter**: `GASearchCV` and `GAFeatureSelectionCV` now accept a `random_state` argument that seeds the entire search at `fit` time — population initialisation (including Latin hypercube sampling), mutation, crossover, and random immigrants. Runs are fully reproducible without manually seeding the global `random` / `numpy` RNGs. Pass `random_state=None` (default) to keep the previous non-deterministic behaviour.

- **Expanded plotting API**: eleven new functions in `sklearn_genetic.plots` — `plot_parameter_evolution`, `plot_search_decisions`, `plot_candidate_scores`, `plot_feature_selection`, `plot_convergence`, `plot_diversity`, `plot_optimizer_events`, `plot_score_landscape`, `plot_cv_scores`, `plot_candidate_rankings`, and `plot_search_overview`. See the [Plotting Gallery](./examples/plotting-gallery) for examples.

- **Benchmarks page**: new Benchmarks section in the docs with Bayesmark-style comparisons of `GASearchCV` against Optuna and random search on tabular regression/classification tasks and CASH (combined algorithm selection and hyperparameter optimisation) scenarios.

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
