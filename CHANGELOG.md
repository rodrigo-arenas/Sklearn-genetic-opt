# Changelog

Read the [development release notes](docs-vitepress/versions/latest/release-notes.md)
for upgrade guidance and links to the API examples.

## Unreleased

### Feature selection

- `GAFeatureSelectionCV(preserve_dataframe=True)` gives the wrapped estimator
  pandas columns with their names and dtypes in CV, finalist validation,
  refit and prediction. This supports fold-local preprocessing such as a
  `ColumnTransformer` with dtype-based selectors.
- `GAFeatureSelectionCV` lets a pipeline's first active step handle
  missing feature values, including sparse data passed to `SimpleImputer`.
- For large CSC inputs that retain at most half their columns,
  `GAFeatureSelectionCV.transform` converts only the selected columns to CSR.
- Sparse `inverse_transform` reconstructs the CSC column pointers directly
  for SciPy CSR and CSC inputs.
- `PopulationConfig(warm_start_masks=...)` lets `GAFeatureSelectionCV` try
  known feature subsets first on all five built-in backends. Each fresh search
  evaluates those masks on its own CV splits.

### Cross-validation

- Explicit CV folds report the setting, fold, partition, and dtype when
  a NumPy selector uses a dtype other than integer or Boolean.
- Failed finalist validation records its status and elapsed time, including
  failures while preparing folds or when every finalist score is non-finite.
  The elapsed fit-work plot labels the failed phase.
- `subsample_cv` identifies the fold if its training partition is empty
  or the requested fraction would leave it with zero training rows.
- Search ranks and finalist comparisons retain extended floating-point
  precision from custom scorers, including in reported scores.
- Finalist validation can reuse a subset of matching folds with very large
  finite scores when a retained score establishes the floating-point dtype.
- Finalist validation reuses small scores saved as `float64` when new folds
  already produce a `float64` array with large scores, avoiding repeat fits.
- Finalist validation can combine small saved scores with large new
  `longdouble` scores. A discarded search fold alone cannot establish
  extended precision for the finalist's score array.
- Finalist validation preserves integer scores when a discarded fold caused
  promotion to floating point. Small saved integers can also be reused when
  newly evaluated folds establish the floating-point dtype.
- When recovering original score types, finalist validation can keep new
  integer or Boolean scores and fit only the remaining folds.
- Finalist validation can also keep small new `float64` scores while
  recovering the types of matching folds stored in extended precision.
- Generation history and score-based stopping callbacks recognize finite
  NumPy extended-precision scores beyond the range of a Python float.
- Categorical matching compares object-array contents recursively when
  resolving copied estimator configurations. Nested arrays can match and
  choices with different scalar types retain their strict matching priority.
- Searches with an optional target keep `y` omitted during CV, refit and
  scoring. Feature selection validates lists and pandas DataFrames in this
  mode before slicing candidate columns.
- New unlabeled outlier searches call detectors and scorers without
  synthetic targets. Compatible older checkpoints keep the target policy
  used to produce their saved scores.
- `GASearchCV` publishes the refitted estimator only after a successful
  full-data fit. A failed refit leaves the recorded CV results available.
- Both estimators expose the elapsed time of a failed full-data refit as
  `refit_failure_time_`. The fit-work plot labels failed refits and shows
  their elapsed time in seconds mode.
- Feature selectors in pipelines can receive groups automatically when only
  the active finalist splitter requests them.
- Finalist validation preserves each recorded fold score when the two
  ordered fold sequences match, including repeated train/test pairs.
- Finalist validation can reuse matching Boolean train/test masks with
  `final_selection="reuse_folds"`, including when the other stage uses the
  same ordered row positions. Valid negative positions also match their
  equivalent row positions.
- When comparing separately supplied folds, finalist validation evaluates
  the fold afresh if either the search or finalist selector mixes integer
  positions and Boolean values in a list or tuple.
- Finalist validation scores folds with custom selector containers afresh.
  Score reuse compares plain NumPy arrays, lists, and tuples.
- Both estimators accept a custom `split(X, y)` method alongside an active
  search or finalist splitter that accepts `groups`.
- A `GASearchCV` group alias supplies its search and finalist splitters in
  nested validation, including when the outer splitter requests plain `groups`.
- `GAFeatureSelectionCV` uses the group request from a splitter such as
  `GroupKFold` when it runs inside a pipeline with metadata routing enabled.
  Explicit group aliases on the selector are supported.
- Empty search or finalist CV reports the setting that needs a train/test
  split. An empty finalist list or tuple is rejected before candidate search.
- Explicit `cv` and `final_selection_cv` lists and tuples report malformed
  train/test pairs, out-of-range positions and wrong-length Boolean masks
  before candidate search. Other iterables and splitters report the same
  errors before scoring the folds they yield.
- Both estimators report a configuration error when a callable scorer
  returns multiple scores. Use a dictionary of named scorers to track
  multiple metrics.
- Both search estimators reject `n_jobs=0` before candidate evaluation,
  including when the CV budget allows only one candidate.
- `GASearchCV` honors `set_fit_request(groups=...)` in pipelines and outer
  cross-validation, including when only the finalist splitter needs groups.
- `GASearchCV` retains estimator parameters named like CV diagnostics, such
  as `fit_time` or `test_accuracy`, when reporting and refitting candidates.
  New checkpoints retain both values. Resume rejects older records where
  the CV fields overwrote the original parameters.
- Candidate workers in both search estimators inherit the active scikit-learn
  configuration, including metadata routing for separate training and scoring
  weights. This applies to reused local threads and configured joblib backends.
- `GAFeatureSelectionCV` exposes estimator and scorer metadata requests
  to pipelines and external validation. It routes them during both
  `fit` and `fit_transform`.
- Both search estimators accept separate metadata requests for training
  and scoring during CV, including aliases. The final refit receives
  only metadata requested by the fitted estimator's `fit` method.
- `GAFeatureSelectionCV.score` accepts sample weights for its selected
  scorer. Pipelines and external cross-validation can route score weights
  through `set_score_request`.
- With refit enabled, `GAFeatureSelectionCV` exposes its wrapped estimator's
  role to scikit-learn. External classifier CV can stratify and use ROC AUC,
  and fitted classifiers expose `classes_`. Detectors with `score_samples`
  expose per-sample scores on the selected features.
- `GAFeatureSelectionCV` preserves `NaN` in input features during fit,
  transformation, inverse transformation and inference when the estimator
  advertises `allow_nan`.
- `GAFeatureSelectionCV` reports its wrapped estimator's sparse input
  support through scikit-learn's estimator tags, including inside pipelines.
- `GAFeatureSelectionCV` preserves categorical input for compatible
  estimators and pipelines whose first active step supports categories.
- `GAFeatureSelectionCV` detects estimators that expect precomputed kernels
  or distances and reports the input mismatch before evaluating masks.
- `GAFeatureSelectionCV` accepts multi-output regression and multilabel
  targets when the wrapped estimator supports them. Estimators such as
  `MultiOutputRegressor` receive a two-dimensional target even with one column.
- CatCMAwM rescales very large losses on continuous and integer axes,
  including finite values outside the range of a Python float. Earlier
  boundary penalties and later losses use the same scale. Checkpoints
  retain the scale, and `cv_results_` reports the original scores.
- SciPy matches survivors using the solver's double-precision loss and
  retains original scores for CV results and the candidate archive. Finite
  losses outside the solver's range use its nearest finite boundary.
  The stopping check handles very large plateaus and distinguishes tiny
  unequal losses. Replay verifies saved generations before applying the check.
- Both search estimators preserve the mean and dispersion of extreme finite
  fold scores in `cv_results_`. Search and finalist selection use the same
  overflow-safe fold mean.
- Nevergrad compresses extreme finite solver losses and assigns failed fits
  a higher loss. Cross-validation results retain the original scores.
- CatCMAwM keeps sampling when failed candidate fits leave its continuous
  boundary penalty without a usable lower loss quartile.
- Feature selection uses scikit-learn's per-fold clones directly during
  candidate evaluation. Candidate logging retains its own estimator clone.
- SciPy keeps failed candidates at infinite loss, avoiding overflow in
  convergence checks. Solver requests for an all-invalid population's
  existing losses reuse those losses without additional CV.
- With numeric `error_score`, a candidate that fails to fit every CV fold
  emits `FitFailedWarning` and ranks behind candidates that fitted. Searches
  with no successful candidate fits raise a configuration error.
- The default outlier scorer uses the fitted detector to evaluate validation
  rows. Its label fallback calls `predict`, and detectors that require fitting
  anew for labels get a configuration error.
- Both search estimators accept `objective_metric` to select the optimization
  metric in a multi-metric search with `refit=False` or callable `refit`.
- A callable `refit` selects a candidate row from `cv_results_`. In a
  multi-metric search, the refitted estimator's `score` uses the named objective.
- Both search estimators honor explicit CV splitters and folds for outlier
  detectors, including grouped and stratified validation. Checkpoint resumes
  retain the folds used to score saved candidates.
- `GAFeatureSelectionCV.score` can apply configured and default anomaly
  scorers after refit, including when the detector has no `score` method.
- Search and finalist validation preserve each fold when a custom splitter or
  iterable reuses its index arrays.

### Fit diagnostics

- `plot_finalist_scores` shows finite scores near or beyond the range of
  Matplotlib's coordinates using a shared, labeled power-of-two scale.
- `plot_fitness_evolution` applies the same scaling to extreme fitness
  histories before smoothing, preserving cost gaps and partial markers.
- `LivePlot` and `plot_generation_metrics` scale extreme fitness scores.
  Live updates rescale earlier points on the same lines when needed.
- `plot_cv_fold_scores` uses a labeled color scale for extreme finite
  test scores and retains the markers for missing scores and unavailable rows.
- `plot_cv_scores` preserves extreme finite scores in candidate rankings and
  uses a labeled scale for box, violin and strip plots.
- `plot_candidate_scores` and `plot_candidate_rankings` display extreme
  finite CV means on a labeled scale. Ranking error bars use the same units
  as the plotted means.
- Both searches retain `cv_results_` and `history` when every candidate
  fails to fit. The search raises before finalist validation or refit.
  Feature selection raises the same error when every attempted mask fails
  and the remaining masks were skipped.
- Automatic finalist selection keeps completed evaluations and elapsed time
  when validation raises. The result has no selected finalist in this case,
  and `plot_finalist_scores` can show the recorded evaluations.
- Candidate logs use the row position in `cv_results_` as `index`, including
  rows that reuse cached scores. Resuming an older checkpoint also restores
  the row positions.
- `cv_results_dataframe(top_k=...)` returns a ranked shortlist. The feature
  selector converts masks to parameter lists only for the returned rows.
- `GASearchCV` accepts a one-parameter grid without a generic warning.
  Backend-specific validation still applies.
- `GASearchCV.fit` and `GAFeatureSelectionCV.fit` raise a clear error before
  candidate evaluation when `y` is missing and the estimator or a standard
  scikit-learn pipeline step requires it.
- `plot_finalist_scores` and `SearchPlotter.finalist_scores()` compare search
  and finalist CV scores and identify the candidate selected for refit.
- Both estimators expose `search_time_` when the optimizer returns or raises.
  It measures elapsed search seconds from the current `fit`, including callbacks
  and any optimizer checkpoint replay. Search, finalist validation, and refit
  durations use the same high-resolution elapsed clock as the optimizer catalog.
- `plot_fit_work(unit="seconds")` and `SearchPlotter.fit_work(unit="seconds")`
  chart elapsed search, finalist validation, and refit time. Phases without a
  saved duration are marked `No timing`.
- The optimizer catalog exports elapsed search time. Its capped pair reports
  give search and full-data refit durations for both sides. Capped reports
  also identify which runs reached the CV target or confirmed finite-space
  coverage.
- The optimizer catalog accepts `--cv-checkpoints` to report CV scores,
  holdout scores, elapsed time, and pair completion at several call counts
  from one run per optimizer, case, seed, and cache setting.
- Each entry in `final_selection_results_["candidates"]` reports actual
  CV calls and fold fits, including any retry when scores cannot be reused.
- `plot_cv_scores` can display selected `cv_results_` rows with incomplete
  fold scores. Candidate labels give the number of finite folds or mark
  candidates flagged as `all_fits_failed`.
- Fold score heatmaps use a separate marker for candidates whose estimator
  failed to fit every fold, based on `cv_results_["all_fits_failed"]`.
- Hyperparameter search overviews display recorded fold scores when the
  requested metric has no finite candidate mean.
- The search overview keeps available plots when saved searches lack
  convergence, diversity or optimizer-event metrics.
- `LivePlot` and `plot_generation_metrics` retain generation positions and
  leave gaps when older checkpoint rows lack a plotted metric.
- Fitness evolution leaves a gap between points on the CV axis when an
  intervening generation has no recorded call count.
- `plot_search_space` accepts `cv_results_` columns and uses the optimization
  score in multi-metric searches. Parameter labels remain distinct from CV
  diagnostics when names overlap.
- `plot_cv_fold_scores` and `SearchPlotter.cv_fold_scores()` show the recorded
  score for each candidate and fold, including non-finite values. Fold plots
  keep similarly named scorers separate.
- Both search estimators warn when no candidate has a finite mean CV score
  for the optimization metric.
- Both search estimators expose `cv_results_["all_fits_failed"]` to identify
  candidates whose estimator failed to fit every search fold.
- Both search estimators accept `columns` in `cv_results_dataframe()` to
  return a narrower rank-sorted table. Feature selection only converts
  masks to candidate parameter dictionaries when the table includes `params`.
- `ModelCheckpoint` reports fresh search CV calls retained in a resumable
  snapshot and calls outside it for the latest fit.
- `plot_fitness_evolution` marks partial-generation scores with hollow diamonds,
  including on the cumulative CV axis and on grouped bars.
- `plot_fit_work` and `SearchPlotter.fit_work()` show fresh fold fits, the
  final refit and reused finalist scores for the latest call to `fit`.
- Callable refit in `GAFeatureSelectionCV` compares its chosen row with the
  archived winner that standard refit would use, including ties in CV score.
- DEAP fitness sharing computes symmetric distances for NumPy integer genes
  and keeps temporary selection scores finite when shifting extreme CV scores.
- `Continuous` rejects nonfinite bounds, intervals containing no representable
  floats and uniform intervals with overflowing spans. Sampling and DEAP
  operators keep valid log-uniform intervals usable when their logarithms
  round together.
- Replacing `OptimizationConfig` through `set_params` updates both search
  estimators and preserves cloning. An explicit finalist CV splitter takes
  precedence until reset to `None`. Fits validate updated controls before CV.
- Replacing `EvolutionConfig` through `set_params` updates both estimators'
  evolutionary controls and preserves cloning with DEAP schedules. Fresh
  DEAP fits apply updated probabilities, and checkpoint resumes rebuild the
  saved schedules before restoring their progress.
- Replacing `RuntimeConfig` through `set_params` updates both estimators'
  execution settings and preserves cloning. Fits validate the resulting
  parallel strategy and error policy before running CV.
- Replacing `PopulationConfig` through `set_params` updates the initializer for
  both estimators. The updated estimators can be cloned before fitting.
- A new call to `fit` clears the prior search's counters, coverage flag,
  scoring metadata and candidate archives before validation. Feature selection
  also drops the prior column names before checking the new inputs.
- `GAFeatureSelectionCV` validates input names and shape for prediction and
  scoring, then passes the selected columns to the refitted estimator in
  array or sparse form, independent of `set_output`.
- `GAFeatureSelectionCV` preserves CSR and CSC input formats when predicting
  and scoring. CSC inputs no longer require conversion before selecting columns.
- External engines compare finite observed scores directly, even when the
  gap between them exceeds the floating-point range. An unrepresentable
  `fitness_improvement` is recorded as signed infinity. `DeltaThreshold`
  compares overflowing finite ranges without losing the stopping decision.
- Score-based stopping callbacks accept `fitness_best_observed`, including
  evaluations outside the elite archive. Missing legacy history values delay
  the decision until its window is available.
- `ConsecutiveStopping` and `DeltaThreshold` use the same history window
  for Python and NumPy integer counts, including unsigned NumPy scalars.
- All engines validate population size, generation count and archive size
  before candidate CV. NumPy unsigned integer counts work across engines.
- External optimizers reuse candidate keys when maintaining a top-k
  hyperparameter archive with multiple entries.
- The shared optimizer archive uses binary search for insertion, preserving
  candidate order when ranks tie.
- `TimerStopping`, `ThresholdStopping`, and `DeltaThreshold` raise a
  configuration error for `NaN` limits or thresholds.
- `TimerStopping` measures elapsed search time with a monotonic clock so
  system clock adjustments leave the deadline intact.
- DEAP keeps the last recorded generation if the time limit prevents every
  evaluation in the next generation, including during checkpoint resume.
- An interruption handled during DEAP evolution ends the search at once and
  skips local refinement. The estimator uses candidates recorded so far.
- DEAP stops candidate evaluation when an `on_step` callback requests a stop,
  including when local refinement is enabled. Configured finalist validation
  and refit still run.
- DEAP reuses usable checkpoint results when the time limit has already
  expired at resume, without starting a new candidate evaluation.
- If the time limit expires before a resumed DEAP search evaluates any
  candidates, `search_space_exhausted_` reports coverage from the restored cache.
- DEAP reports a complete generation when the time limit expires after its
  last evaluation. The time limit also stops further generations when a
  custom timer leaves its step callback running.
- Fitness bar charts show generations with the same CV cost side by side in
  generation order.
- `ThresholdStopping` follows the search direction for score metrics.
  Score-based stopping callbacks wait for finite values in their window.
- Generation summaries use finite population fitness values and report
  `finite_fitness_count`. Fitness plots leave gaps where scores are unavailable,
  including when applying a rolling window.
- Generation summaries preserve the scale of extreme finite population scores
  when computing their mean and standard deviation for history and callbacks.
- DEAP reports `history["partial"]` when a timer or CV budget interrupts a
  generation, matching the other bundled engines.
- Both search estimators expose generation records through indexing and
  iteration with `refit=False`. Their length counts the recorded generations.
- `plot_fitness_evolution(x="cv_calls")` plots against the cumulative number
  of candidate CV evaluations.
- `GAFeatureSelectionCV` can transform data and report selected feature names
  with `refit=False`, including as a transformer in a scikit-learn pipeline.
- With `refit=False`, both search estimators report the selected candidate's
  search row and objective score. `GASearchCV` also reports its parameters.
- `GAFeatureSelectionCV.n_features_` reports the selected feature count after
  search, including when `refit=False`.
- `cv_results_dataframe(preserve_index=True)` keeps the original candidate
  row labels for lookup by `best_index_` in both search estimators.
- `GAFeatureSelectionCV` exposes the search row and CV score for the refitted
  mask when that row is available.
- Both search estimators expose `fit_counts_` after fitting. It reports fresh
  search, finalist and refit work for that call, including on checkpoint
  resume, and counts reused finalist fold scores separately.
- Finalist validation narrows potential reusable folds by their validation
  indices before comparing every index. When search and finalist validation
  share the same NumPy array, list, or tuple, the matcher converts the
  selector once and skips the repeated comparison.

### Checkpoint compatibility

- Optimizer resumes validate the saved best score and generation counters before
  starting fresh CV. Errors name the checkpoint file and invalid telemetry field.
- Optimizer resumes report malformed fingerprints or an invalid saved search
  problem before hashing the current training data.
- Resuming from saved CV splits checks fold pairs, row bounds and mask lengths
  before fitting new candidates, including with older DEAP snapshots.
- Bundled optimizer resumes identify malformed generation records, mismatched
  candidate counts, and invalid solver state before starting new CV.
- `ModelCheckpoint(every_n_generations=N)` can space out snapshot writes. Its
  counters report how many fresh CV calls the latest file can recover.
- `LogbookSaver` includes DEAP local refinement rows in its final file,
  including candidates whose scores came from the cache.
- `LogbookSaver` replaces a regular file only after joblib finishes writing
  the snapshot. A failed write keeps the previous log and reports the path
  and error.
- DEAP saves completed local refinement in the latest checkpoint, including
  the candidate scores and cumulative CV count. A failed write keeps the
  preceding generation snapshot.
- DEAP skips repeat candidate evaluation on resume when a verified
  checkpoint has complete cached coverage and uses standard refitting or
  `refit=False`, with finalist validation inactive.
- Resumable optimizer checkpoints compare explicit Boolean fold masks by
  value so equivalent copies can resume from the same saved search.
- DEAP resume recovers the cumulative CV count from the saved generation log
  when the runtime count is missing. Missing auxiliary counters start at
  zero for the resumed work.
- New checkpoints record scikit-learn's metadata routing setting. Resume
  rejects a changed setting before using saved CV scores. Older snapshots
  remain readable and resumable under the existing compatibility rules.
- Incomplete optimizer checkpoints report the missing resume fields and file
  path before candidate evaluation.
- Resumable optimizer checkpoints also validate the auxiliary fit counters
  before starting CV. Errors name missing or invalid counters.
- Checkpoint resume rejects invalid saved search CV counts and detects
  disagreement with the last recorded generation before evaluation.
- Searches reject empty or unrecognized checkpoint payloads before candidate
  evaluation and preserve the existing file.
- `ModelCheckpoint` keeps the previous file and lets the search continue
  when serialization raises `ValueError` or `RecursionError`.
- Loading a saved search replaces the receiving instance's fitted results
  with those in the file. A missing logbook entry in a legacy save raises
  before changing the receiving instance.
- Direct categorical choices containing Python float or NumPy float16, float32
  and float64 NaNs keep their type and bit pattern when resuming. Restored
  searches reuse cached CV scores for these choices.
- DEAP serializes checkpoint logs directly and can save custom report
  metadata that supports pickle without supporting deep copies.
- DEAP restores matching finalist fold scores with `final_selection="reuse_folds"`
  when the checkpoint verifies the inputs and preserves the original fold indices.
- DEAP checkpoints restore the saved search direction, CV error handling,
  train-score setting and feature limit.
- New DEAP checkpoints verify the training inputs before reuse of saved scores
  and preserve the original folds from explicit cross-validation splitters.
- Bundled optimizer checkpoints accept explicit search and finalist CV folds
  with the same ordered NumPy indices when estimator cloning copies a shared fold.
- Shared-runtime checkpoints fingerprint CV selectors made of Python integers
  and booleans directly. Explicit selector order and types are verified on
  resume, including for files with older configuration fingerprints.

### Search performance

- Feature selection skips repeated comparisons of shared cached masks
  when locating the archived winner in a long search history.
- Both searches use the original score array and return its ranks directly
  when every CV score is finite.
- Both searches reuse the computed mean when summarizing the standard
  deviation of ordinary `float64` fold scores and finite CV timings.
- Both searches compute the mean of short, ordinary CV score vectors
  without allocating a separate absolute-value array.
- Both searches keep the independent fold arrays produced by standard
  `KFold`, `StratifiedKFold`, and `GroupKFold` splitters.
- `TimerStopping` prepares cache keys in dispatch-sized batches, including
  when `CVBudgetStopping` also limits the number of fresh evaluations.
  Cache hits before the first unfinished proposal are recorded in order.
- Wide feature masks backed by integer lists or tuples use a faster byte
  conversion for binary validation, diversity and distinct-mask counts.
- pymoo passes repaired feature masks to the evaluator as immutable integer
  tuples, avoiding an intermediate list for each proposal.
- Optimizer adapters can pass feature masks as tuples of Python integers.
  The runtime reuses the tuple and validates repeated references once per batch.
- On sparse inputs with at least 4,096 columns, native feature evaluation
  uses a `bytearray` to build a writable Boolean mask from the cache
  lookup tuple.
- `GASearchCV` can use prepared candidate rows directly in batches with
  native integer dimensions and the standard repair hooks.
- CV fold heatmaps reuse the stable rank cutoff for short lists from long
  histories. Candidate score and fold distribution plots read the columns
  they display.
- Short `cv_results_dataframe(top_k=...)` queries can use a rank cutoff
  to sort only the leading candidates. Ties retain evaluation order.
- `GASearchCV` reads parameter names once when building `cv_results_` and
  reuses the recorded values for single-parameter grids.
- Both searches prepare candidate copies as the evaluator consumes the batch,
  reducing temporary memory for large populations.
- pymoo, Nevergrad and CatCMAwM reuse immutable feature masks from the
  initial population until those masks are dispatched for evaluation.
- DEAP evaluates the built-in initial feature masks directly with no
  callbacks or with the standard `CVBudgetStopping` callback.
- The built-in feature initializers reuse masks that already satisfy the
  feature bounds, saving a second normalization pass during setup.
- With a CV call budget and no timer, both searches prepare cache keys only
  through the first proposal that needs another call. Earlier proposals
  with cached scores keep their original row order.
- Matching a copied estimator to an estimator-valued categorical choice
  reuses its constructor parameters within each lookup.
- Search uses the smallest CV budget set by the callbacks after `on_start`
  and checks the limit before dispatching candidates.
- Feature finalist validation uses scikit-learn's per-fold estimator clones
  directly, matching the feature search phase.
- Finite-space stopping uses the cache setting after the `on_start` callback
  runs, including on resume.
- Finite categorical coverage counts floating-point `NaN` values with the
  same identity as the candidate cache, so a fully evaluated space can stop.
- Both searches track pending results by a shared candidate key within each
  evaluation batch, reducing repeated hashing and temporary key storage.
- Feature selection reuses validated integer masks during external optimizer
  evaluation, avoiding a second mask normalization pass.
- DEAP feature selection reuses the random draw function across columns in
  uniform crossover, preserving the sequence of masks from each seed.
- SciPy, Nevergrad, CMA-ES and pymoo reuse continuous coordinate bounds and
  logarithmic spans across proposals. DEAP's differential evolution reuses
  them within each generation.
- Smart initialization reuses continuous coordinate bounds across candidates,
  including the logarithmic span for `log-uniform` parameters.
- Smart initialization decodes Latin hypercube samples for integer parameters
  with scalar arithmetic and reads fixed integers directly from their bounds.
- `GASearchCV` skips copying candidate parameters made entirely of immutable
  Python scalars. Mutable choices and estimator-valued parameters still
  use scikit-learn's cloning rules.
- Both search estimators summarize finite fold timings directly when building
  `cv_results_`. Recorded missing timing values retain their existing handling.
- Both search estimators skip candidate worker dispatch when only one search
  CV call remains. The batch is counted as serial and fold parallelism still
  follows the configured strategy.
- Feature selection counts Boolean masks directly when formatting CV results
  and retains the existing sums for legacy mask formats.
- Cached feature selection batches share the lookup key for repeated masks,
  reducing temporary allocations while retaining all completed proposal rows.
- Uncached batches prepare only the candidates dispatched within the search
  limits, reducing bookkeeping allocations for oversized batches.
- Nevergrad decodes binary feature proposals by generation, retaining the
  seeded repair rules and the checkpoint format.
- pymoo builds the initial binary population directly and repairs each new batch
  of masks together, preserving seeded proposals and checkpoint replay.
- CatCMAwM computes binary starting probabilities and decodes feature masks in
  NumPy batches, preserving the elite counts and mask repair rules.
- SciPy, pymoo, Nevergrad and CatCMAwM reuse the feature selector's immutable
  initial masks, avoiding a second validation pass over those masks.
- Both search estimators copy candidates into private evaluation lists only
  when dispatching their batch, reducing allocation for proposals beyond an
  evaluation limit.
- Smart hyperparameter initialization fills available population slots with
  distinct warm-start candidates in input order, including candidates supplied
  after a prefix with duplicates.
- Smart hyperparameter initialization compares warm starts and estimator
  defaults using the values cross-validation receives. Equivalent estimator
  clones and continuous values that round together share one initial slot.
- SciPy and pymoo compute binary feature proposal signatures faster when writing
  and replaying checkpoints. The saved signatures and evaluations remain compatible.
- Finalist validation can combine cached small integers with large integers
  from new folds without repeating the full validation.
- Finalist validation can combine saved fractional scores with large integer
  scores from new folds when the combined scores use floating-point precision.
- Finalist validation can reuse a subset of large integer scores when every
  requested fold matches and the search retained an integer score array.
- When the search retained an integer score array, finalist validation can
  combine its large integer scores with scores from new folds.
- Interrupted finalist validation records the attempted CV call and candidate
  row. The fit-work plot marks known finalist fold fits as a lower bound.
- If the optimizer raises, `fit_counts_` retains newly recorded search CV
  work and the fit-work plot marks its fold count as a lower bound.
  The count includes results returned by an interrupted batch, including
  evaluations that finish out of order in the built-in worker pool.
  It also includes completed CV calls when preparing a later dispatch or
  recording the batch results raises.
- The finalist score plot identifies the interrupted candidate and shows
  its search score even when validation stops on the first finalist.
- Finalist validation uses each original score array when its folds match
  the complete search sequence or a permutation, including repeated folds.
  All finalists share the same ordering and retain the saved numeric dtype.
- When many feature proposals reuse evaluations, finalist selection packs
  each shared cached mask at most once per shortlist or tied cutoff scan.
- At a large tied cutoff, feature selection can pack only the largest eligible
  Boolean masks when looking for a denser replacement.
- `plot_fitness_evolution(x="fold_fits")` shows cumulative search fold fits
  for comparing runs with different numbers of CV splits.
- The optimizer catalog times candidate CV without replacing the built-in
  evaluators, so integer and sparse-mask evaluation keep their optimized paths.
- The optimizer catalog can export pairs of built-in engines at their shared
  CV budgets, with holdout scores and completion counts for each pair.
- The optimizer catalog counts CV calls with non-finite scores toward the
  budget and reports quality from finite candidates only.
- Skipped feature masks have missing CV scores and cannot become finalists
  or be refitted, even when valid scores have large magnitude.
- Fold score plots distinguish masks skipped before CV from estimator failures.
  The search overview displays recorded scores when feature selection ends
  without a selected mask.
- Feature selection computes size ranks from a bounded count of mask sizes
  in long histories, preserving tied ranks and candidate order.
- Feature-selection DataFrame exports reuse shared mask conversions while
  keeping an independently editable mask list in every `params` row.
- Finalist validation can reuse a matching fold with an undefined `NaN`
  score and keeps finalists with undefined mean scores ineligible.
- DEAP fitness sharing packs wide binary feature masks and computes each
  pairwise distance once while preserving the order of the crowding sums.
- Large tied finalist histories reuse the original row order for hyperparameter
  search and compare denser feature masks in bounded batches.
- Large finalist searches can sort a leading group of ranks while keeping
  complete ties and falling back when more distinct candidates are needed.
- Cached CV score ranks use the multiplicity of each shared summary,
  preserving tied ranks while sorting fewer values.
- Long CV result histories with a single finite score value compute ranks
  without sorting. Candidate multiplicity still determines the non-finite rank.
- When many candidates reuse cached scores, `cv_results_` summarizes the
  shared fold arrays once and restores the original row order for both searches.
  Grouping the shared arrays now takes one pass and uses less temporary memory.
  Feature selection also reuses the counts of shared Boolean masks.
- `scipy_optimizer(stop_on_convergence=False)` lets either search continue
  through equal population losses, subject to generation, coverage and
  callback limits. Checkpoints replay with the configured option.
- Hyperparameter repair traverses dimensions directly and clips native Python
  integers without converting them. Candidate results keep the grid's order.
- DEAP searches use internal individual types and preserve entries in the
  caller's DEAP creator registry, including after initialization fails.
- Categorical searches accept missing-value markers such as `pandas.NA`
  without treating ambiguous equality as a match or aborting coverage checks.
- Hyperparameter repair and optimizer decoding skip scalar NumPy clipping
  when a continuous proposal already lies strictly inside its bounds.
- Nevergrad reads binary choice indices directly during feature selection,
  reducing mask conversion work on wide inputs.
- `GAFeatureSelectionCV` accepts `population_initializer="spread"` to try a
  wider range of subset sizes early when the CV budget is small.
- `CVBudgetStopping` caps fresh search CV calls, including within a generation
  and during DEAP local refinement. It accounts for cache hits and resume.
- With `refit=False`, the bundled engines stop proposing cached candidates
  after finite-space coverage even when finalist validation is configured.
- Bundled optimizer adapters validate integer feature masks in bulk before
  evaluation, reducing overhead on wide feature sets.
- Categorical values that compare equal but have different declared types
  retain separate cached scores, coverage entries and finalists.
- DEAP counts local refinement toward finite-space coverage and skips remaining
  cached neighbor proposals once coverage is complete in search-only runs.
- Feature selection normalizes list-backed feature masks in bulk and counts
  selected features once during repair.
- Both search estimators reuse each proposal's cache key within its batch and
  reduce the score statistics in `cv_results_` in bounded NumPy batches.
- Repeated proposals share one persistent cache lookup when the batch records
  their candidate rows.
- The backend runtime reuses validation of repeated immutable feature masks
  to skip rehashing when they share the same cache key object.
- Both search estimators reuse those batches to populate per-fold score columns
  in `cv_results_`, reducing formatting time for long or many-fold searches.
- Both search estimators reuse worker threads across parallel candidate batches
  when joblib selects its standard local thread backend.
- SciPy, Nevergrad, CatCMAwM and pymoo stop at a completed generation after
  covering a finite search space when caching is enabled and finalist
  selection is off. Exact resume restores coverage from the checkpoint.

### Finalist selection

- Both estimators validate the finalist count at fit time, including after
  `set_params`. When finalists will run, invalid CV types and integer fold
  counts outside the sample limits raise before candidate cross-validation.
- Repeated feature masks with the same search score no longer change which
  distinct tied masks are re-evaluated during finalist selection.
- A new fit clears the previous fitted winner and finalist results. After
  `fit` with `refit=False`, both searches expose results from the current CV
  run without exposing a fitted estimator from an earlier run.

### Documentation and package metadata

- With `--public-refit`, the optimizer catalog reports the estimator's
  `best_score_` for the selected feature mask, including repeated evaluations.
- The optimizer catalog accepts `--cases`, `--engines` and `--seeds` for
  reproducing selected comparisons and trying new data splits. A subset
  needs only the optional engines and references it runs.
- The optimizer catalog can stop each fit at a fixed CV budget and record
  the candidate returned by the estimator and its total fit time.
- The optimizer catalog can export engine/reference pairs at their shared CV
  budget, including search time and test scores for the winners at that budget.
- The search comparison guides link to the matched-budget optimizer catalog.
  Package metadata adds search terms and links to the benchmarks and changelog.
- The troubleshooting guide explains cache reuse, batch duplicates and fresh
  fit counts after a checkpoint resume.

## 0.16.0

### Optimization backends

- Both estimators accept the keyword-only `optimizer_backend` parameter. DEAP
  remains the default. Install the `scipy`, `nevergrad`, `cmaes` or `pymoo` extra
  to use differential evolution, OnePlusOne, CatCMA with Margin or a mixed-variable
  or binary genetic algorithm respectively.
- Custom optimizers can be passed as callables or registered under
  `sklearn_genetic.optimizers`. The estimators own cross-validation, caching,
  candidate results and refitting across engines.
- The backend tutorial and benchmarks compare five engines at matched counts of
  actual cross-validation calls on classification and regression tasks.

### Callbacks, checkpoints and results

- LivePlot, progress logging and early stopping work across the five engines.
  Both estimators resume from a completed generation. Nevergrad and CatCMAwM
  restore native solver state. SciPy and pymoo verify and replay saved proposals
  without repeating completed CV evaluations. Earlier progress-only snapshots
  remain readable. External optimizers can supply state for continuation.
- Fitted history iteration stops at the last recorded generation. Multi-metric
  results can be pickled with the ordered list of scorer names.

### Compatibility

- DEAP now requires version 1.4.4 or newer. CI covers CPython 3.12–3.15 on Linux,
  macOS and Windows. On 3.15, pip selects versions of scikit-learn, NumPy, pandas
  and SciPy with upstream wheels. Earlier Python versions retain their existing
  minimums for those four packages.

## 0.15.0

### Search, performance and compatibility

- `GAFeatureSelectionCV` compares packed feature masks when selecting distinct finalists, reducing ranking time on wide inputs.

- Smart feature selection reuses initial mask draws for padding when the population exceeds the entire cached search space. Seeded populations in this case can differ from earlier versions.

- Smart initialization in `GASearchCV` varies fractional categorical allocations across search seeds while retaining exact stratified counts when the configured priors fit the available slots.
- Equal positive categorical priors now produce the same seeded smart-initialization draws as uniform sampling over the eligible choices, including when some choices have zero weight.
- `GAFeatureSelectionCV` can now select columns from COO sparse inputs during supervised and target-free outlier searches.
- Convergence, diversity and optimizer-event plots retain the recorded generation numbers when a history starts above zero or contains gaps.
- Both estimators expose cumulative candidate CV evaluations per generation in `history["cross_validate_calls"]`, including the cost of local refinement in the final generation.
- Fixed integer dimensions now return their exact value directly, including Python integers beyond the signed 64-bit range.

- Feature selection computes generation diversity for wide binary masks one row at a time and stores packed masks to count unique candidates.

- Seeded checkpoint resumes in both estimators continue the Python and NumPy proposal streams and retain the archived elites. Smart numeric hyperparameter initialization draws new strata on resume.

- `ModelCheckpoint` writes one snapshot per generation and replaces the previous checkpoint only after serialization succeeds. A failed update preserves the last readable snapshot.

- Resuming either search after updating `criteria` with `set_params` penalizes cached non-finite scores in the active optimization direction.

- Fixed premature stopping with `ConsecutiveStopping` when lower fitness scores improve a minimizing search.

- Smart feature-selection initialization samples the smaller of the selected and excluded column sets. This reduces temporary memory for wide inputs. Seeded runs remain reproducible, but the masks for a given seed can differ from earlier versions.

- Local refinement in `GASearchCV` now chooses variable dimensions when proposing neighbors in mixed fixed/variable grids.

- Differential evolution in `GASearchCV` directs its mandatory donor coordinate to a variable hyperparameter when a grid also contains fixed dimensions.

- Standard `GASearchCV` mutation skips numeric dimensions with equal bounds and single-choice categories, focusing proposals on variable dimensions in mixed grids.

- Both estimators support `final_selection="reuse_folds"` for deterministic searches with `error_score="raise"`. Finalists reuse scores from identical ordered splits of the current fit and report the number of reused folds.

- Fixed integer candidate repair and local refinement in `GASearchCV` so adjacent large integers retain distinct parameter values and fitness-cache entries.
- Fixed sampling of log-uniform continuous dimensions with equal bounds. Smart initialization now recognizes the same fixed value in warm-start and generated candidates, preserving room for distinct configurations.
- Candidate score, ranking and fold plots now honor the search's minimization or maximization criterion and omit non-finite means. Horizontal charts show the best candidate at the top, and identical display labels retain separate bars.
- Fixed population growth in `eaSimple` when the initial population contains duplicates and the elite archive grows during evolution. Both search estimators now keep archived elites within the configured population budget.

- Removed TensorFlow and TensorBoard from optional extras. Added Matplotlib-based live generation plots in `sklearn_genetic.plots`; the legacy `TensorBoard` callback remains available with a separate TensorFlow installation and no longer imports it at library startup.
- `GASearchCV` and `GAFeatureSelectionCV` now rank non-finite cross-validation scores last during evolution and final candidate selection, for both maximization and minimization. A candidate with one failed fold no longer receives a misleading average from its remaining folds. Raw fold scores remain visible in `cv_results_`; fitness sharing preserves valid ordering, and resumed checkpoints normalize older cached fitness values.

- Variable integer dimensions use NumPy sampling directly. Both searches build
  CV result columns with less repeated work and skip cache-key construction when
  caching is disabled. Feature selection avoids redundant mask scans, speeds up
  copying and validation of native masks, and limits the scan for tied finalists.

### Documentation, benchmarks and maintenance

- Added guidance and runnable examples for counting candidate evaluations,
  finalist fold fits, checkpoint resumes, live plotting and search budgets.
  Benchmark coverage now includes held-out comparisons with random search,
  exhaustive and greedy feature selection, search-space coverage, crossover,
  initialization, filter proposals, finalist budgets, fold pruning, and
  end-to-end time and memory attribution. Experimental policies are reported
  in the [benchmark record](benchmarks/PERFORMANCE_V2.md).
- Documentation pages use stable canonical URLs where a released counterpart
  exists. Search titles and descriptions were revised. The Pages workflow
  submits changed URLs to Bing through IndexNow after deployment. Added shared
  contributor instructions and updated optional dependency and CI guidance.

### Community contributions

- Both fitted searches expose `cv_results_dataframe()`, sorted by the selected
  metric's test rank with stable ties. Feature-selection rows include their
  masks in `params` (#363, #393).
- Both searches refresh optimization direction after `set_params(criteria=...)`
  and validate replacement values. Dictionary-valued categorical candidates
  work across search operations, including duplicate detection and caching
  (#391, #388).
- Added community resources on search spaces, callbacks and `GASearchCV`;
  clarified CV versus held-out MAE, corrected warm-start advice and a
  troubleshooting link, repaired the example-generation npm commands, and
  documented focused test runs (#389, #400, #401, #402, #404).
- Updated Vite, Vue, its server renderer and source-map-js for the docs build
  (#385, #386, #392). The 0.14 docs freeze and conda recipe refresh completed
  after publication; archive links, stable-page metadata and performance
  documentation were also updated.

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
- Added `warm_start_configs` to `GASearchCV`.
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
