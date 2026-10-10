# Continued performance work

Baseline: `d6b2f3aab9cee50322934ec8c00c585e2fcbac2f`. This is an ongoing investigation of both `GASearchCV` and `GAFeatureSelectionCV`; a general 4× training or solution-quality improvement has not been established.

## Profile

A cProfile run of the existing paired benchmark (four HPO tasks and one feature task, one seed) attributed 9.27 of 9.36 seconds in the evolutionary loop to candidate evaluation, including 8.75 seconds in scikit-learn cross-validation. Profile overhead affects absolute times. Most end-to-end gains therefore need fewer expensive evaluations at comparable quality, rather than only faster Python evolution operators.

## Fitness-sharing execution

The optional sharing operation now vectorizes distances one population row at a time and uses O(population × dimensions) auxiliary memory rather than constructing a three-dimensional pairwise tensor. Its O(population² × dimensions) arithmetic is unchanged. Ordered contribution accumulation preserves selection ties. Numeric subtraction retains Python integer precision, including integer-valued continuous dimensions; categorical equality retains its original semantics.

Median of three isolated calls, with the original pairwise implementation as reference:

| Population | Binary features (100 dimensions) | Mixed HPO (3 dimensions) |
|---:|---:|---:|
| 50 | 9.24× | 7.90× |
| 150 | 14.44× | 10.04× |
| 300 | 13.51× | 14.76× |

Four complete seeded searches with sharing enabled produced the same best CV scores and evaluation counts. Complete-fit speed ratios ranged from **1.04× to 1.26×** in the final timing pass: these results do not demonstrate a general 4× training speedup. Timings use fixed before/after order on a shared Windows machine; they are descriptive, not significance estimates. Sharing is disabled by default, so this change does not accelerate the default evaluation path.

Reproduce in PowerShell:

```powershell
.venv\Scripts\python.exe benchmarks\benchmark_sharing.py
```

The benchmark includes reference equality checks, feature/HPO complete fits, two fit seeds, one native thread, and the raw per-case timings. It does not measure holdout quality because the operation is intended to preserve the existing search behavior, not introduce a new selection policy.

Verification: 102 focused algorithm/helper tests pass; `test_guard` is clean. New equality regressions were observed failing before their repairs; an unordered-accumulation mutant fails the selection regression. Independent review closed all material findings. Full suite not run. [Raw results](results/performance-v2/sharing.json), [verification evidence](results/performance-v2/sharing-verification.txt).

## Sequential local refinement

Previously, all `local_search_steps` neighbors were generated from the same initial elite. Each step now uses the updated elite archive: an improvement can lead to another improvement. The maximum candidate count is unchanged, and separate elite neighbors are still evaluated as a batch within each step. This applies to both searches and all three existing algorithms. Local search remains opt-in; one-step behavior is unchanged. Steps are dependent, so they are sequential rather than parallelized together; independent elites within a step retain parallel evaluation.

Twenty-four controlled algorithm cases (numeric/features, minimization/maximization, one/two elites, three algorithms) distinguish chained progress from one-shot sampling; all fail with the previous function. The focused algorithm/helper suite passes 126 tests. Real fits on Wine confirm identical serial/two-worker scores and chosen solutions, valid feature bounds, candidate limits, and final best-score telemetry. These are behavior checks, not claims of general holdout improvement or a 4× speedup. [Verification](results/performance-v2/refinement-verification.txt).

## Optional differential evolution

Both searches now accept `algorithm="eaDifferentialEvolution"`: deferred rand/1/bin, scale 0.8, one trial per parent, and greedy weighted replacement. Numeric differences use the declared linear/logarithmic domain; integers, categories and binary feature masks retain their domain constraints. Population size must be at least four. The standard mutation operator, cache, parallel evaluator, callbacks and optional local refinement are reused. Selection-specific tournament/sharing controls do not apply. The default algorithm is unchanged.

The new option is **not a general quality or speed improvement**. It is retained as an alternative evolutionary strategy; measured regressions below rule out replacing the default based on this experiment.

HPO validation used 240 searches: Iris, Wine, Breast Cancer and Diabetes × tree/SVM/KNN × five seeds (301–305) × default GA, differential evolution, Optuna TPE and FLAML CFO. Each run has a maximum of 40 candidate CV evaluations, identical folds, spaces and train/test split, and includes final refit in elapsed time. Logit-labelled Bayesmark dimensions use the same natural-range uniform distribution for all methods. Seeds 301–305 were declared after exploratory runs on 0–2 and before this validation. These are fixed-model HPO comparisons, not comparisons of full AutoML pipelines.

| Dataset/model | GA CV | DE CV | GA test | DE test | GA/DE time |
|---|---:|---:|---:|---:|---:|
| iris/dt | 0.9180 | 0.9319 | 0.9368 | 0.8947 | 1.03× |
| iris/svm | 0.9696 | 0.9697 | 0.9684 | 0.9579 | 1.07× |
| iris/knn | 0.9661 | 0.9643 | 0.9421 | 0.9368 | 1.18× |
| wine/dt | 0.8121 | 0.8180 | 0.7822 | 0.8133 | 1.01× |
| wine/svm | 0.9851 | 0.9851 | 0.9778 | 0.9778 | 1.05× |
| wine/knn | 0.9745 | 0.9760 | 0.9422 | 0.9333 | 1.36× |
| breast/dt | 0.9085 | 0.9075 | 0.9063 | 0.9035 | 1.03× |
| breast/svm | 0.9746 | 0.9742 | 0.9748 | 0.9734 | 1.06× |
| breast/knn | 0.9653 | 0.9667 | 0.9650 | 0.9650 | 1.15× |
| diabetes/dt | -4047.6632 | -4051.8790 | -4065.6570 | -4041.7889 | 1.00× |
| diabetes/svm | -3099.1697 | -3101.1773 | -3077.7374 | -3070.4122 | 0.99× |
| diabetes/knn | -3365.9392 | -3339.7368 | -3294.2080 | -3338.2356 | 1.12× |

Classification values are accuracy; Diabetes values are negative MSE (higher is better). Each cell averages five seeds. A ratio above 1 means DE was faster. Methods run in fixed order on a shared Windows machine; small timing differences should not be treated as robust speedups.

| Method | Mean classification CV | Mean classification test | Mean seconds/run |
|---|---:|---:|---:|
| differential | 0.9437 | 0.9284 | 0.925 |
| flaml | 0.9274 | 0.9216 | 0.639 |
| gasearch | 0.9415 | 0.9329 | 1.041 |
| optuna | 0.9510 | 0.9402 | 0.654 |

The classification aggregate contains 45 runs per method. It does not imply statistical significance or superiority on unseen tasks; the full per-seed data also records regression results. Optuna/FLAML remain stronger on several scenarios. TPOT, H2O and AutoGluon have not been run in this comparison.

Feature selection used 60 additional searches: 20/60/120 synthetic variables × tree/logistic regression × five new dataset/split seeds × two algorithms. Both methods propose 72 candidates (GA: 8 initial + 4 × 16; DE: 8 initial + 8 × 8), use three CV folds, and select between one and half the available features.

| Variables/model | GA CV | DE CV | GA test | DE test | GA/DE time |
|---|---:|---:|---:|---:|---:|
| 20/dt | 0.8400 | 0.8320 | 0.8220 | 0.8220 | 1.04× |
| 20/logistic | 0.8500 | 0.8493 | 0.8340 | 0.8420 | 1.03× |
| 60/dt | 0.8220 | 0.8187 | 0.7780 | 0.7620 | 1.00× |
| 60/logistic | 0.8573 | 0.8460 | 0.8160 | 0.7740 | 0.96× |
| 120/dt | 0.7927 | 0.7813 | 0.7780 | 0.7760 | 0.99× |
| 120/logistic | 0.7960 | 0.7960 | 0.8140 | 0.8440 | 0.95× |

The high-dimensional exploratory gains did not consistently repeat. For example, 120-variable logistic test accuracy improved by 3 percentage points, while 60-variable logistic test accuracy fell by 4.2 points. Overall these measurements do not meet the requested general 4× objective. No holdout-based choice of winner is made by the library.

Reproduce in PowerShell:

```powershell
.venv\Scripts\python.exe benchmarks\benchmark_evolution.py --output hpo.json --datasets iris wine breast diabetes --models dt svm knn --seeds 5 --seed-start 301 --budget 40 --methods gasearch differential optuna flaml
.venv\Scripts\python.exe benchmarks\benchmark_features.py --output features.json
```

Verification: 136 focused algorithm/helper tests and eight public search/validation tests pass. Separate real fits verify identical serial/two-worker solutions, six local refinement candidates, checkpoint cache reuse, and fitted save/load for both search classes. Missing algorithm behavior was recorded red before implementation; NaN callback recovery was recorded red before repair. Independent review rejected tie/category/NaN mutants and closed all material findings. `test_guard` and Black on tracked Python plus the new benchmark pass. Repository-root Black also scans ignored experiment scripts and reports formatting there; no tracked source failed. Full suite not run.

[HPO data](results/performance-v2/differential-hpo.json), [feature data](results/performance-v2/differential-features.json), [verification](results/performance-v2/differential-verification.txt).

## Complete finite-space coverage

Baseline: `f49c19ad1118bc415aade86dc6ff9eccb5128f03`. Cached search now stops after all possible configurations have been observed in the current run and preserved through the best-solution archive. This applies to both search classes and every evolutionary algorithm. Unknown/custom or non-fixed continuous dimensions, uncached runs, and runs without an archive keep the generation budget. Old cache contents alone never prove current coverage. Feature counts include every nonempty mask up to `max_features`.

Paired benchmark: 18 baseline/current pairs, three seeds per workload, 20 individuals, at most 40 generations, identical three-fold splits, one worker/native thread, final refit included. All pairs have exactly equal CV score, held-out score, predictions, selected configuration/mask, and CV evaluation counts.

| Workload | Baseline seconds (mean) | Current seconds (mean) | Paired speedup range | Current history records |
|---|---:|---:|---:|---|
| hpo-small | 0.636 | 0.076 | 7.82–9.01× | 1, 1, 1 |
| hpo-medium | 1.057 | 0.456 | 2.17–2.40× | 5, 4, 6 |
| hpo-continuous | 13.954 | 13.303 | 1.03–1.06× | 41, 41, 41 |
| features-small | 0.375 | 0.064 | 5.82–5.94× | 1, 1, 1 |
| features-medium | 0.870 | 0.652 | 1.23–1.43× | 11, 21, 11 |
| features-wide | 10.491 | 10.109 | 1.01–1.08× | 41, 41, 41 |

Baseline history has 41 records for every run (generation zero included). Small HPO and feature spaces exceed 4× with identical solutions; medium spaces improve less. Continuous and large-feature controls do not exhaust, so their small timing changes should be treated as noise. Runs use a fixed baseline-then-current order on a shared machine; these are workload-specific measurements, not a general 4× or SOTA claim.

Reproduce with `benchmarks/benchmark_exhaustion.py --output current.json`; export the baseline package with `git archive` and add `--source-root <export-directory>` for the baseline run.

[Baseline data](results/performance-v2/exhaustion-baseline.json), [current data](results/performance-v2/exhaustion-current.json), [verification](results/performance-v2/exhaustion-verification.txt).

## Feature masks compared with TPE and CFO

This comparison uses Optuna TPE and FLAML CFO as optimizers of a binary feature mask, not their complete AutoML workflows. All four methods use the same model, training/holdout split, three CV folds, 72 proposed candidates, one native thread, duplicate-score caching and timed final refit. Empty masks are infeasible; the GA repairs them and external optimizers receive negative infinity without a fit. No empty proposals occurred in the recorded runs. There is no feature-count cap, unlike the previous half-feature benchmark.

Data: synthetic classification with 400 samples, eight informative and four redundant variables, widths 20/60/120, seeds 501–503, depth-five decision tree and scaled logistic regression. All 72 runs completed. The holdout is used only for reporting.

| Method | Mean CV accuracy | Mean holdout accuracy | Mean seconds | Mean selected features |
|---|---:|---:|---:|---:|
| genetic | 0.7469 | 0.7178 | 1.1659 | 31.7222 |
| differential | 0.7496 | 0.7283 | 1.0845 | 31.7222 |
| optuna | 0.7648 | 0.7267 | 1.9438 | 35.1111 |
| flaml | 0.7448 | 0.7122 | 1.2497 | 32.4444 |

Means above cover 18 runs per method. Per-workload mean held-out accuracy follows; all timing and CV details remain in the raw results.

| Width/model | GA | Differential | TPE | CFO |
|---|---:|---:|---:|---:|
| 20/dt | 0.7033 | 0.7533 | 0.7367 | 0.7133 |
| 20/logistic | 0.7633 | 0.7767 | 0.7767 | 0.7700 |
| 60/dt | 0.7300 | 0.7067 | 0.7267 | 0.7200 |
| 60/logistic | 0.7367 | 0.7333 | 0.7000 | 0.7167 |
| 120/dt | 0.6667 | 0.7133 | 0.7033 | 0.6633 |
| 120/logistic | 0.7067 | 0.6867 | 0.7167 | 0.6900 |

Differential evolution takes about 1.79× less time than TPE on the aggregate mean, with similar average held-out accuracy; TPE has the better average CV score. This small synthetic suite does not establish statistical superiority or equivalence to native TPOT/H2O/AutoGluon feature handling. Method order is fixed and machine load is shared, so timing is descriptive.

The original bounded benchmark remains reproducible: all twelve seed-301 rows match the prior committed data exactly for CV/holdout scores, masks counts and evaluation counts. Regression tests exercise empty masks, duplicates, selection of an earlier CV winner, final refit, and CLI domain restrictions.

```powershell
.venv\Scripts\python.exe benchmarks\benchmark_features.py --output feature-optimizers.json --seed-start 501 --seeds 3 --methods genetic differential optuna flaml --unconstrained
```

[Raw data](results/performance-v2/feature-optimizers.json), [verification](results/performance-v2/feature-optimizers-verification.txt).

Additional exploratory ideas were rejected: partial default seeding and boundary seeding worsened several HPO tasks; heavy-tailed feature mutation did not retain its exploratory gains on five new seeds. These experiments did not change library behavior.
