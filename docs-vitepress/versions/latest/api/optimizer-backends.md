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
[development installation instructions](../guide/installation#optimization-backends).

| Value | Solver and useful settings | Population summarized in `history` | Checkpoint |
| --- | --- | --- | --- |
| `"deap"` | Defaults to `eaMuPlusLambda`. Choose an [algorithm](./algorithms), schedules, elitism or local search. Supports categorical priors. | Algorithm population | DEAP continuation |
| `"scipy"` | Deferred `best1bin` differential evolution. Use at least 5 individuals. A fixed `crossover_probability` sets recombination. Equal energies can end the run early. | Survivors | Progress |
| `"nevergrad"` | OnePlusOne for both tasks. Set `strategy="TwoPointsDE"` on the adapter callable to try differential evolution. Leave estimator crossover at its default. | Evaluated batch | Progress |
| `"cmaes"` | CatCMA with Margin. Use at least 4 individuals. Set `sigma` on the adapter callable in `(0, 1]` (default 0.25) for the initial normalized numeric step. Leave estimator crossover at its default. | Evaluated batch | Progress |
| `"pymoo"` | Mixed-variable GA for parameters and binary GA for masks. Tune fixed `crossover_probability` in `[0, 1]` (default 0.8). | Survivors | Progress |

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

`population_size * (generations + 1)` bounds proposals on the shared optimizer
runtime. The initial population is generation zero. Check
`fit_stats_["cross_validate_calls"]` for actual CV work. Cache hits and convergence
can reduce it, and optional finalist evaluation can add work. A real CV call fits
one model per fold. DEAP's proposal count depends on the selected algorithm.
See the [measured cost and quality comparison](../benchmarks/optimizer-catalog).

## Common estimator behavior

- The engine proposes candidates. The estimator owns fold splitting, scoring,
  caching, finalist selection, refit and prediction. `parallel_backend` controls
  where CV work is parallelized, independently of `optimizer_backend`.
- Use `cv_results_` for per-candidate scores, `history` for per-generation
  metrics, and `fitness_best_observed` in [LivePlot](./plots) to follow the best
  finite primary score seen so far. The archive considers discarded candidates too.
- Stopping and progress callbacks receive generation records. A shared-runtime
  checkpoint with `resumable=False` can be inspected with `ModelCheckpoint.load()`.
  Starting another fit with that file raises an error. Use a new path for a new
  search. The [checkpoint recipe](../recipes/advanced/checkpointing) covers DEAP.
- Leave DEAP-specific controls at their constructor defaults when switching engines.
  These include `algorithm`, mutation probability, tournament size, elitism,
  adaptive schedules, immigrants, local search and fitness sharing. The shared
  runtime accepts `diversity_control` but does not adapt variation with it.
  Finalist selection remains available.

## Extend the catalog

Pass a callable directly, or publish a Python entry point in the
`sklearn_genetic.optimizers` group. Both paths use
`optimize(problem, evaluate, report)`. The evaluator returns a loss to minimize
and an opaque handle for each decoded candidate. Report the selected handles
after initialization and after each generation. Plugins can return picklable
solver state at each boundary and implement `resume` for exact continuation.
See the [working plugin example and protocol](./algorithms#optimizer-plugins).
