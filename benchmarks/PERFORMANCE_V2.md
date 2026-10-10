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
