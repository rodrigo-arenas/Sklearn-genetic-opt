# Performance and correctness evaluation — 2026-10-10

This run fixes demonstrated correctness and resource problems while keeping the evolutionary search API. It does **not** establish SOTA, a general training speedup, or a general accuracy improvement. The small holdout experiment contains regressions as well as ties and wins.

## Implemented changes

- Fresh fits discard stale cached scores; explicit checkpoint resumes retain their cache.
- Non-elitist selection supports negative scores and minimization. Candidate rankings honor the optimization direction, including train-score ranks.
- Final selection evaluates distinct top configurations instead of allowing cached duplicate rows to consume its shortlist.
- Feature bounds are validated after checkpoint restoration. Existing mask repair was verified across all three algorithms; this run did not invent that already-present fix.
- Diversity now measures pairwise allele disagreement. Balanced binary populations no longer look collapsed merely because their population is large.
- Local integer refinement samples uniformly in constant memory, excluding the parent when alternatives exist.
- Mu/lambda algorithms preserve archived elites as independent survivors. Best-so-far telemetry includes candidates that stochastic selection drops.
- Comparison scripts now account for `lambda = 2 * population`, disable unbudgeted local refinement in bounded runs, seed the GA, and report actual CV calls. The new paired harness retains per-seed raw data and untouched holdout scores.

Library commits: `253ee623b78ad9ca1deb65f32b915633d5ef15f1`, `0bbff86c5e0130c48c7c6ff476a3ca99907adc10`, `954127a305b2f3d8ea548fb134d6d66735b597f4`. Baseline: `8476a7f0c660017cc55573e88bf911bade9a4d7d`.

## Protocol

- Wine and Breast Cancer, SVM and decision tree, seeds 0–2. Spaces and preprocessing reused from `benchmark_bayesmark.py` without tuning them to holdout results.
- Stratified 75/25 train/test split per seed, shared three-fold CV on training only. Selection uses CV; holdout is scored once after refitting.
- Hyperparameter cap: 40 CV evaluations (120 fold fits), plus one final refit for every method. GA population 8, two mu/lambda generations, default library controls. Caching may leave some budget unused.
- Sequential execution, `n_jobs=1`, native thread pools limited to one. Timings include search and final refit, exclude dataset loading and holdout scoring. Single local timing pass; no claim of statistically reliable timing gains.
- Separate feature experiment: 240 synthetic samples, 20 features, five informative/five redundant, 75/25 split, tree depth 4, max_features 10, population 16, six generations, cap 208 CV evaluations. It is a before/after comparison, not a 40-evaluation comparison against the HPO optimizers.
- FLAML uses its CFO tuning API on the same estimator/search space, not FLAML AutoML. TPOT, H2O and AutoGluon were not benchmarked. No conclusion about parity with their full AutoML systems is supported.
- The final harness was run against both the exported baseline package and the improved package. Dependencies were identical for these paired runs. Optimizer order was fixed; timing order and operating-system load are potential confounders.

## Holdout accuracy and elapsed time

Each cell is mean holdout accuracy / mean seconds across three seeds. Accuracy differences on these small test sets should not be treated as significance tests.

| Dataset / model | GA before | GA after | Optuna TPE | RandomizedSearchCV | FLAML CFO |
|---|---:|---:|---:|---:|---:|
| wine / svm | 97.04% / 1.047s | 97.04% / 0.776s | 100.00% / 0.625s | 100.00% / 0.538s | 97.04% / 0.534s |
| wine / dt | 86.67% / 0.383s | 86.67% / 0.422s | 85.19% / 0.427s | 85.19% / 0.286s | 52.59% / 0.294s |
| breast / svm | 96.50% / 1.145s | 96.50% / 1.146s | 96.74% / 0.796s | 96.74% / 0.798s | 96.50% / 0.723s |
| breast / dt | 90.21% / 0.445s | 90.21% / 0.464s | 90.21% / 0.440s | 90.21% / 0.319s | 81.82% / 0.366s |

GA holdout accuracy and mean CV quality are unchanged versus the baseline on these four HPO scenarios. It does not consistently beat Optuna or random search, and remains slower in several of these small workloads. FLAML-CFO results here do not represent the capabilities of FLAML AutoML.

## Feature selection: improvement is not generalization

| Measurement | Before | After |
|---|---:|---:|
| Mean best CV accuracy | 0.83148 | 0.84074 |
| Mean holdout accuracy | 0.83889 | 0.81111 |
| Mean elapsed seconds | 1.47978 | 1.75193 |
| Mean CV evaluations | 207.00000 | 207.33333 |
| Mean recovery-triggered generations | 6.00000 | 0.00000 |

The improved CV objective did not improve the held-out metric: average holdout accuracy fell by 2.78 percentage points. Three seeds and 60 held-out samples per seed are insufficient to establish a general regression or improvement, but the observed decrease must not be hidden. No further algorithm tuning was performed against these holdout results. The recovery correction removes a false signal; it is not evidence that fewer immigrants always improve model quality.

## Constant-memory local integer sampling

Ten local-neighbor draws in `Integer(0, 1_000_000)`, parent 500,000, radius 0.1, measured with `tracemalloc`: peak Python allocations **8,028,406 → 4,848 bytes**, elapsed **1.753885 → 0.000695 seconds**. This is a microbenchmark with tracing overhead, not a training-speed claim. The implementation removes interval-size-dependent allocation entirely.

## Compatibility and validation

No public constructor arguments were removed or added. Search trajectories change due to corrected selection/diversity. `max_features=0`, booleans, non-integers and values above input width now raise `ValueError`; custom diversity thresholds may require recalibration.

- Initial focused baseline: 42 passing tests.
- Integrity checks: 212 passing; 43 repair checks after checkpoint/ranking findings.
- Efficiency checks: 84 passing, including bounded allocation and uniform-neighbor sampling.
- Integrated algorithms/search/features/benchmarks/callbacks: 194 passing, four pre-existing TensorBoard cases skipped because TensorFlow is absent.
- Final review repairs: 37 passing checks; the final benchmark-only check result is retained in the verification evidence.
- These counts overlap and must not be summed. Regression red runs and explicit surviving-mutant fixes were recorded. Independent reviews closed with no unresolved material findings. Black and test_guard passed.
- The full suite, optional MLflow/TensorFlow integrations, multiworker performance, release gates and deployment were not run. No push was performed. No tree-wide naming/date/copy guard was found; existing test files were extended.

## Reproduce (PowerShell)

Use the dependency versions recorded in the JSON. Core runtime dependencies were not added to the library; Optuna and FLAML are benchmark tools.

```powershell
.venv\Scripts\python.exe benchmarks\benchmark_evolution.py --methods gasearch optuna randomized flaml --features --output benchmarks\results\rerun.json
git archive --format=zip --output=.codex_tmp\performance-baseline.zip 8476a7f0c660017cc55573e88bf911bade9a4d7d sklearn_genetic
Expand-Archive -LiteralPath .codex_tmp\performance-baseline.zip -DestinationPath .codex_tmp\performance-baseline
.venv\Scripts\python.exe benchmarks\benchmark_evolution.py --source-root .codex_tmp\performance-baseline --methods gasearch optuna randomized flaml --features --output benchmarks\results\rerun-baseline.json
```

Python: 3.12.0. Packages: numpy 2.5.3, scipy 1.18.1, scikit-learn 1.9.1, deap 1.4.4, optuna 5.0.0, flaml 2.7.0.

Raw evidence: [baseline](results/performance-2026-10-10/baseline.json), [improved](results/performance-2026-10-10/improved.json), [verification](results/performance-2026-10-10/verification.txt).

Method references: [Optuna samplers](https://optuna.readthedocs.io/en/stable/reference/samplers/index.html), [FLAML tuning](https://microsoft.github.io/FLAML/docs/Use-Cases/Tune-User-Defined-Function/).

## Remaining evidence needed for SOTA

A larger predeclared suite with classification and regression, multiple compute budgets, more seeds and paired uncertainty estimates is still needed. Full AutoML comparisons also require an explicit common model-family/pipeline budget. The current evidence does not justify changing the evolutionary philosophy or adding ensembles, pruning or new dependencies speculatively.
