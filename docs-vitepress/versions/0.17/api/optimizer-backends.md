---
title: Optimizer backend API
description: Engine names, optional extras, supported controls, generation history and checkpoint capabilities for both search estimators.
---

# Optimizer backend API

Set the keyword-only `optimizer_backend` on `GASearchCV` or `GAFeatureSelectionCV`.
It accepts an engine name or a synchronous optimizer callable and defaults to
`"deap"`. It is a constructor parameter, so `clone`, `get_params` and `set_params`
work as with other scikit-learn estimators. Start with the
[executable tutorial](../tutorials/optimizer-backends) or the
[installation instructions](../guide/installation#optimization-backends).

| Value | Solver and useful settings | Population summarized in `history` | Checkpoint |
| --- | --- | --- | --- |
| `"deap"` | Defaults to `eaMuPlusLambda`. Choose an [algorithm](./algorithms), schedules, elitism or local search. Supports categorical priors. | Algorithm population | DEAP continuation |
| `"scipy"` | Deferred `best1bin` differential evolution. Use at least 5 individuals. A fixed `crossover_probability` sets recombination. You can [continue through equal losses](./algorithms#continue-through-equal-population-losses). | Survivors | Resume by verified replay |
| `"nevergrad"` | OnePlusOne for both tasks. Set `strategy="TwoPointsDE"` on the adapter callable to try differential evolution. Leave estimator crossover at its default. | Evaluated batch | Restore native solver state |
| `"cmaes"` | CatCMA with Margin. Use at least 4 individuals. Set `sigma` on the adapter callable in `(0, 1]` (default 0.25) for the initial normalized numeric step. Leave estimator crossover at its default. | Evaluated batch | Restore native solver state |
| `"pymoo"` | Mixed-variable GA for parameters and binary GA for masks. Tune fixed `crossover_probability` in `[0, 1]` (default 0.8). | Survivors | Resume by verified replay |

## Choose a starting point

| Workload | Evidence from the catalog benchmark |
| --- | --- |
| Feature selection for linear regression | pymoo had the lowest mean held-out RMSE on both synthetic mask tasks. On the harder case it improved on the training-mean predictor by 9.1%. |
| Feature selection for noisy classification | CatCMAwM reached mean held-out balanced accuracy 0.704, followed by Nevergrad at 0.701. Use your own splits to resolve small differences. |
| Tuning a decision tree | SciPy led on breast-cancer classification. DEAP had the lowest mean RMSE on the Friedman and Diabetes tree tasks. |

These results use five held-out splits and the default DEAP algorithm. At the
larger budget, 38 of 45 groups reached 96 real CV calls. In seven groups every
engine was compared at SciPy's earlier convergence point. The
[full comparison](../benchmarks/optimizer-catalog#results-at-equal-cv-cost)
includes every engine, time outside CV, repeated proposals and the raw CSVs.

## Domain and budget

Both estimators supply the initial population, including warm starts. The four
alternative adapters accept finite continuous ranges (linear or log scale),
unweighted categories and variable integers within int64 with a span below
`2**52`. Fixed integers retain their value. Feature proposals reach CV as
nonempty masks respecting `max_features`. The [algorithm details](./algorithms#scipy-backend)
describe the encodings, including SciPy's ordered category bins and CatCMAwM's
rounded representation for wide integer ranges.

pymoo, Nevergrad and CatCMAwM can reuse the standard selector's immutable
initial masks in a fresh fit. SciPy decodes its solver coordinates into masks
for evaluation. The shared evaluator checks newly proposed masks and makes
private copies of candidates as it dispatches them. Reusing the initial
masks reduces temporary allocations when a budget cuts a batch short.
The [budgeted feature selection example](../tutorials/optimizer-backends#compare-feature-selection-at-the-same-cv-budget)
shows how to measure fresh evaluations and fold fits.

For grids made entirely of built-in integer dimensions with the standard
repair hooks, `GASearchCV` can pass repaired rows directly to the CV workers.
The proposed parameters still appear in evaluation order in `cv_results_`.

`population_size * (generations + 1)` bounds proposals on the shared optimizer
runtime. The initial population is generation zero. Check
`fit_stats_["cross_validate_calls"]` for actual CV calls used by the search.
Cache hits and convergence can reduce that count. Each search CV call fits one
model per fold. Finalist validation and the full-data refit run afterward.
DEAP's proposal count depends on the selected algorithm.
See the [measured cost and quality comparison](../benchmarks/optimizer-catalog).
Use [CVBudgetStopping](./callbacks#cvbudgetstopping) to give different engines
the same upper limit on fresh search CV calls. Check the observed count for
each fit because convergence, finite coverage and cached proposals can leave
part of the budget unused.

With caching enabled, `fit_stats_["duplicate_candidates"]` counts repeat
proposals that share a fresh evaluation in their batch. `cache_hits` counts
proposals whose score was already cached before the batch, including every
repeat of those proposals in that batch. Each completed proposal keeps its
own row in `cv_results_`. Equivalent proposals share one key during batch
evaluation and one cached result lookup when the batch records their rows.
The persistent cache retains value-based keys for resuming a search.
For a feature mask supplied more than once as the same immutable tuple, the
engine can also skip hashing the key again within the batch.
Hyperparameter keys sort pairs by name after resolving categorical values.
Candidate result columns follow the declared grid order.
In long histories with frequent cache reuse, result assembly can group
shared fold-score arrays in one pass. Each row keeps its original position
and contributes to tied ranks.
When every finite score in a long result column is the same, the formatter
assigns rank 1 directly. Non-finite scores take the rank after all finite
candidate rows.
With caching disabled, `unique_candidates` counts
fresh evaluations, including repeated configurations. See the
[worked cache example](../tutorials/optimizer-backends#read-both-cache-counters-in-one-batch).

With caching enabled, the bundled engines can stop at a generation boundary
after covering a finite search space. `search_space_exhausted_=True` confirms
that the engine detected full coverage. `False` can also mean that coverage
was untracked or that a limit interrupted a batch before the coverage check.
Read the actual search CV calls in `fit_counts_` and the last row's
`partial` flag in `history` alongside this attribute. See the
[budget and coverage example](../tutorials/optimizer-backends#compare-a-cv-limit-with-complete-coverage).
This stopping rule uses the cache setting after
`on_start`, including when resuming a checkpoint. SciPy, Nevergrad, CatCMAwM and pymoo keep collecting
their usual candidate history when the search will validate finalists.
With `refit=False`, coverage can stop the search even if `final_selection` is
configured. Third-party optimizers control their own stopping rules.
DEAP can resume from a verified checkpoint with complete cached coverage
without appending a generation with standard refitting or `refit=False`,
when finalist validation is inactive. See the
[exhausted checkpoint example](../tutorials/optimizer-backends#resume-an-exhausted-deap-search).
Floating-point `NaN` categories use the same identity for coverage and
score caching. See the
[finite categorical example](../tutorials/optimizer-backends#stop-when-a-categorical-space-is-covered).

DEAP also counts candidates scored during local refinement toward finite-space
coverage. Once the cached space is covered, DEAP ends refinement in search-only
runs with `refit=False`. Use `fit_counts_` for real fold work and
`fit_stats_["local_refinement_candidates"]` for neighbor proposals, including
ones resolved by the cache.

## Common estimator behavior

- The engine proposes candidates. The estimator owns fold splitting, scoring,
  caching, finalist selection, refit and prediction. `parallel_backend` controls
  where CV work is parallelized, independently of `optimizer_backend`.
- DEAP uses its own individual types, independent of classes registered in
  `deap.creator`. When fitting separate search objects in threads, their Python
  and NumPy random streams still share process-wide state. Running each search
  in a separate process isolates those streams for reproducible parallel fits.
- A fit with `refit=False` leaves the current CV results available and clears
  the fitted winner from any previous fit. Call `fit` with `refit=True` to
  use prediction again.
- Use `cv_results_` for per-candidate scores, `history` for per-generation
  metrics, and `fitness_best_observed` in [LivePlot](./plots) to follow the best
  finite primary score seen so far. The archive considers discarded candidates too.
- The shared optimizer archive keeps up to `keep_top_k` distinct candidates
  in rank order. A new candidate admitted to the archive goes ahead of
  entries with the same rank. At capacity, a candidate must outrank the
  last entry to enter. Finding the insertion point uses binary search.
- To resume any catalog engine, pass the same checkpoint path to a new `fit`
  with the same data, settings and environment. SciPy, pymoo, Nevergrad and
  CatCMAwM check the exact `sklearn-genetic-opt` version too: [finish the old
  run before upgrading or start a new checkpoint](../recipes/advanced/checkpointing#resume-an-alternative-engine). For SciPy and pymoo, replay
  uses recorded results for completed generations. Native solver state serves
  the same purpose for Nevergrad and CatCMAwM. A partially evaluated generation
  may repeat CV. A completed finite search resumes without evaluating new
  candidates. Additional finalist validation and final refit run after the
  checkpoint and can repeat on resume, including after the last generation.
  Newly written optimizer checkpoints compare NumPy index arrays and Boolean
  masks in explicit fold lists or tuples by value. Clones can resume when they
  hold copies of the same selectors in the same order. See the
  [Boolean fold checkpoint example](../tutorials/optimizer-backends#resume-with-shared-boolean-folds).
  Older progress-only files (`resumable=False`) can be inspected
  with `ModelCheckpoint.load()`. See the [checkpoint recipe](../recipes/advanced/checkpointing).
- Leave DEAP-specific controls at their constructor defaults when switching engines.
  These include `algorithm`, mutation probability, tournament size, elitism,
  adaptive schedules, immigrants, local search and fitness sharing. The shared
  runtime accepts `diversity_control` but does not adapt variation with it.
  Finalist selection remains available.

## Fit work by phase

After fitting either estimator, inspect `fit_counts_` for the most recent
call to `fit`:

| Key | Work measured |
| --- | --- |
| `search_cv_calls` | Fresh candidate cross-validation calls confirmed during search |
| `search_fold_fits` | Estimator fits across the confirmed search CV calls |
| `finalist_cv_calls` | Cross-validation calls started during finalist selection |
| `finalist_fold_fits` | Known fresh estimator fits during finalist validation, including any repeated folds |
| `finalist_reused_folds` | Finalist fold scores taken from search results |
| `refit_fits` | One completed full-data refit, zero otherwise |

For a completed fit, add the two `*_fold_fits` values and `refit_fits` to
obtain the total fit work. Reused fold scores and cache hits cost zero
fresh fits. On resume,
`fit_counts_` measures work done by that call. Use
`fit_stats_["cross_validate_calls"]` for cumulative search CV calls.
When `history` contains `partial`, check the flag on each row to identify
generations interrupted by a search budget.
Use [`plot_fit_work`](./plots#plot-fit-work) to see fresh fits and reused
finalist scores from the latest call to `fit` on separate bars.
For one finalist, read `cv_calls`, `fold_fits`, and `reused_folds` from its entry
in `final_selection_results_["candidates"]`. A retry can make `fold_fits`
larger than the number of returned fold scores. See the
[per-finalist example](../tutorials/optimizer-backends#inspect-actual-work-for-each-finalist).
If an exception interrupts a finalist CV call, the call counts toward
`finalist_cv_calls`. The partial fold count is unavailable, so
`finalist_fold_fits` reports a lower bound. Inspect
`final_selection_results_["failed_candidate"]` for the row index, parameters,
started calls, and known fold fits for the interrupted finalist. See the
[interrupted validation example](../tutorials/optimizer-backends#inspect-an-interrupted-finalist).

Failed automatic validation sets `final_selection_results_["failed"]` to
`True`, including when fold preparation raises or all finalist scores are
non-finite. Completed finalist rows and their scores remain available.

### Fit time by phase

These elapsed times are in seconds. Both search estimators and the optimizer
catalog measure durations with the same high-resolution clock, `time.perf_counter()`.
Use [`plot_fit_work(unit="seconds")`](./plots#plot-fit-work) to chart the recorded
phases.

| Field | Phase measured |
| --- | --- |
| `search_time_` | Optimizer run, including population setup, candidate CV, and callbacks |
| `final_selection_results_["time_seconds"]` | Automatic finalist validation, including the time to prepare its folds and reuse scores |
| `refit_time_` | Successful full-data estimator fit |
| `refit_failure_time_` | Time until the full-data refit phase raised |

After a refit error, `plot_fit_work(search, unit="seconds")` labels the failed
phase and shows its elapsed time. See the
[failed-refit example](../tutorials/optimizer-backends#inspect-a-failed-full-data-refit).

`search_time_` is recorded when the optimizer returns or raises, including with
`refit=False`. If an error ends the optimizer run, `search_interrupted_` is
true and `fit_counts_` retains confirmed search calls and fold fits from
that invocation, including completed candidate results from an interrupted
batch. If preparing a later dispatch or recording its results raises,
`fit_counts_` includes each completed CV call once. When the built-in
population thread pool handles a batch, the counts also include
later proposals whose evaluations return before or during worker shutdown.
Counts are lower bounds because an interrupted evaluation can perform work
before returning a result. `fit_stats_` tracks the optimizer's published
counters. Checkpoints retain their saved search state. An ordinary budget
stop returns successfully with
`search_interrupted_=False`.
On resume, the timer covers the current optimizer run, including any replay it
performs. Run `fit` to populate `search_time_` on an estimator saved with an
older version. Time the entire `fit` call to include input validation, search
split preparation, and construction of `cv_results_`.

## Extend the catalog

Pass a callable directly, or publish a Python entry point in the
`sklearn_genetic.optimizers` group. Both paths use
`optimize(problem, evaluate, report)`. The evaluator returns a loss to minimize
and an opaque handle for each decoded candidate. Report the selected handles
after initialization and after each generation. Plugins can return picklable
solver state at each boundary and implement `resume` for exact continuation.
See the [working plugin example and protocol](./algorithms#optimizer-plugins).
