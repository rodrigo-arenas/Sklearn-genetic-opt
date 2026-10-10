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

## Independent categorical population coverage

Baseline: `8171f981dfa233df67f2d6577919c7bc8a5bc310`. Smart HPO initialization now independently shuffles each categorical dimension. Uniform dimensions sample the entire choice domain when the population is smaller than the domain; larger batches retain balanced category counts. Weighted dimensions retain prior-based strata, including exclusion of zero-prior choices. Warm starts and valid estimator defaults keep their existing priority. Numeric-only and feature-mask initialization are unchanged.

Previously, categorical columns shared one repeating index, so unrelated genes started in correlated combinations; small populations only visited the category prefix. Public regression cases now find a deliberately crossed category optimum and demonstrate coverage of all 20 choices across fixed seeds. These are coverage guarantees, not a claim that every task improves.

The paired baseline/current GA comparison covers Iris, Wine, Breast Cancer and Digits, decision trees/SVC/KNN, seeds 601–603 and a budget of 40 candidate evaluations. The new categorical profile uses identical domains, folds, splits and budgets for GA, Optuna TPE and FLAML CFO. Every method refits the CV winner before held-out reporting. All 144 fits completed (36 baseline plus 108 current/comparator fits).

| Method | Mean CV accuracy | Mean holdout accuracy | Mean seconds |
|---|---:|---:|---:|
| GA baseline | 0.95003 | 0.95821 | 1.92407 |
| GA current | 0.95287 | 0.95466 | 1.85349 |
| Optuna TPE | 0.95450 | 0.95595 | 1.14361 |
| FLAML CFO | 0.95135 | 0.95554 | 0.93759 |

GA improves mean CV accuracy by 0.00284, while held-out accuracy decreases by 0.00355. Wine/tree contributes the largest change: CV +0.01751 and holdout −0.03704. This prevents interpreting improved training-fold exploration as a generalization win. Current GA remains slower than TPE/CFO on this suite; its approximately 1.04× baseline timing ratio is small and potentially noise. Fixed execution order and shared machine load limit timing conclusions. No general 4× or SOTA claim follows.

The comparison harness converts FLAML NumPy scalar parameters into native Python values both during evaluation and final refit: Digits/KNN otherwise raises a real Cython type error for a NumPy string metric. A targeted regression demonstrates that failure and protects both conversion points.

```powershell
.venv\Scripts\python.exe benchmarks\benchmark_evolution.py --output categories-current.json --datasets iris wine breast digits --models dt svm knn --seeds 3 --seed-start 601 --budget 40 --methods gasearch optuna flaml --categorical
```

For the baseline, export the commit above, pass its directory with `--source-root`, and select only `--methods gasearch`. Four public initializer cases and 20 benchmark tests pass; independent review killed biased-subset, weighted-correlation, ignored-profile and missing-refit-normalization mutants. Scoped Black, test guard and diff checks pass. Full suite not run.

[Baseline data](results/performance-v2/categories-baseline.json), [current/comparator data](results/performance-v2/categories-current.json), [verification](results/performance-v2/categories-verification.txt).

## Stop impossible uniqueness retries during initialization

Baseline: `7c072d2331f54246e4cdd8e1dd3d52b44c42cf88`. Both smart initializers now reuse the bounded finite-domain count to stop retrying uniqueness after observing every possible configuration. They still fill the requested population size. Incomplete, unknown, continuous and uncached spaces retain the existing attempt bound. No new library option is introduced.

Twelve paired fits isolate initialization (zero generations, three seeds per workload/population size). All pairs have exactly identical CV/holdout scores, predictions, selected configuration/mask and CV evaluation counts. Three-fold CV and final refit are timed.

| Workload | Population | Baseline mean seconds | Current mean seconds | Paired speedup range |
|---|---:|---:|---:|---:|
| HPO, 6 configurations | 20 | 0.0848 | 0.0591 | 1.38–1.53× |
| HPO, 6 configurations | 100 | 0.1897 | 0.0930 | 1.95–2.13× |
| Features, 10 masks | 20 | 0.0679 | 0.0680 | 0.93–1.07× |
| Features, 10 masks | 100 | 0.0822 | 0.0790 | 0.99–1.08× |

HPO benefits because the discarded draws repeatedly call parameter distributions. Feature draws are cheap: their removal reduces work but shows no convincing total-fit timing gain here. These small, fixed-order runs are descriptive and do not support a general speedup. Six additional paired controls (continuous HPO and 20-feature selection, two generations) preserve scores, predictions, configurations, generation counts and CV counts exactly. Mean control time rises from 0.8665 to 0.9035 seconds for HPO and from 0.6606 to 0.7725 seconds for features. These controls show no speed gain; the small shared-machine sample cannot distinguish timing noise from overhead.

```powershell
.venv\Scripts\python.exe benchmarks\benchmark_exhaustion.py --output saturation.json --population-size 100 --generations 0 --cases hpo-small features-small
```

Repeat with population 20; export the baseline package and use `--source-root` for paired runs. Controls use `--population-size 20 --generations 2 --cases hpo-continuous features-wide`. Four public fit regressions fail without the optimization and pass with it. Two further cases detect premature stopping before all six configurations/masks are sampled. The focused initializer/finite-space consumer set passes 48 tests; four additional public-fit cases preserve retries for uncached HPO/features and unknown custom/continuous dimensions, and reject the premature-stop mutant. Full suite not run.

[20-individual baseline](results/performance-v2/saturation-baseline-20.json), [current](results/performance-v2/saturation-current-20.json); [100-individual baseline](results/performance-v2/saturation-baseline-100.json), [current](results/performance-v2/saturation-current-100.json); [control baseline](results/performance-v2/saturation-baseline-controls.json), [current](results/performance-v2/saturation-current-controls.json); [verification](results/performance-v2/saturation-verification.txt).

## Isolate final refit and preserve cloned-search reproducibility

Both searches now fit a cloned winner, preserving the supplied estimator's parameters and fitted state. The `estimator` attribute is the constructor template; callers that previously used it for prediction must use `best_estimator_`, `estimator_`, or the search's prediction methods. This compatibility change follows the existing documented fitted-winner interface and is recorded in the changelog. Estimator-valued hyperparameters are cloned too, preventing a winning pipeline component from being fitted in place.

A public-fit regression also exposed numeric sampling drift after `clone(search)`: copying a dimension copied its bound SciPy distribution's internal random state. Unseeded numeric draws now explicitly use the global RNG, so the search's seed controls the cloned search as intended. Explicit dimension generators keep their independent streams. The test compares candidate parameters and scores after cloning a fitted search; checking only the final winner would miss this drift.

Evidence includes original red ownership cases, red Integer/Continuous sampling mutants, a red nested-estimator mutation case, and negative prediction-method cases. The fitted paths exercise prediction, probabilities, decision function, score, feature transform, history and save/load. This is an API/reproducibility improvement, with no claimed training speedup. Verification includes 77 focused space/seed cases, 14 final ownership/prediction-guard cases and six checkpoint-resume cases. One pre-existing checkpoint test exhausted its three-value space before advancing any schedules; its space now exceeds the per-run proposal budget, preserving every adapter assertion. Full suite not run.

[Verification](results/performance-v2/refit-isolation-verification.txt).

A further exploratory current-to-pbest differential variant was rejected after 108 paired development fits: gains on some narrow feature tasks did not extend to wider masks or HPO regression, and there was no general timing gain. It is not part of the library.

## Native CPU AutoML comparison

The native comparison uses a shared 25% outer holdout, seeds 901–903, Breast Cancer, Digits, Diabetes and a fixed synthetic classification dataset (2,000 samples, 80 columns). Each method receives a requested 15-second budget and one CPU thread. Reported time is actual fitting time including final refitting, so generation-level stopping and native soft-limit overruns remain visible. Prediction time and optional-package imports are excluded. Classification uses holdout accuracy; regression uses negative mean squared error and is reported separately.

GASearchCV tunes five LightGBM hyperparameters with a population of eight and three-fold CV. GAFeatureSelectionCV selects a nonempty feature mask for fixed-parameter LightGBM with the same population and CV. Both use their default evolutionary algorithm. FLAML AutoML searches LightGBM, random forest and extra trees with native three-fold CV. AutoGluon trains those three CPU families, three bag folds and its weighted ensemble, with stacking disabled and sequential fitting. Native workflows receive a 1 GB memory setting; the genetic searches do not expose an equivalent cap. AutoGluon model persistence is included in its fitting time.

This is a comparison of complete configured workflows: the native validation procedures, model choices and ensembles differ from the genetic searches. It does not isolate optimizer quality, and this restricted CPU configuration does not represent every AutoGluon preset or establish overall SOTA. No TPOT or H2O measurement is included. Fixed method order, three splits and shared-machine load limit timing and generalization conclusions. The isolated environment records AutoGluon 1.6.3, FLAML 2.7.0, LightGBM 4.7.0 and the numerical dependency versions in the JSON artifact. See the [AutoGluon fit API](https://auto.gluon.ai/stable/api/autogluon.tabular.TabularPredictor.fit.html) for the native configuration semantics.

An earlier ExtraTrees surrogate-screening experiment was rejected after 108 development fits: fitting the surrogate and replacing cached proposals increased cost, while held-out gains were inconsistent across HPO and feature tasks. No surrogate is added to the library.

All 48 runs completed. Cells below show mean holdout score / mean actual seconds across three seeds. Higher scores are better, including negative MSE.

| Dataset / metric | GASearchCV | GAFeatureSelectionCV | FLAML AutoML | AutoGluon |
|---|---:|---:|---:|---:|
| breast / accuracy | 0.96737 / 15.77 | 0.95804 / 15.59 | 0.96037 / 15.00 | 0.96737 / 2.27 |
| digits / accuracy | 0.97185 / 23.45 | 0.96741 / 28.92 | 0.97852 / 15.26 | 0.97556 / 6.74 |
| diabetes / negative MSE | -2820.31296 / 15.34 | -3708.33279 / 15.30 | -2969.95390 / 15.01 | -2796.32627 / 1.58 |
| synthetic / accuracy | 0.89667 / 27.52 | 0.89467 / 17.64 | 0.87667 / 15.26 | 0.89000 / 8.78 |

GASearchCV matches AutoGluon's mean accuracy on Breast Cancer but takes about 6.94 times as long. On Digits it trails both native workflows; its generation boundary overshoots the requested budget. On the synthetic problem it exceeds FLAML's accuracy by 0.02000, but takes about 1.80 times as long. AutoGluon remains faster across all four tasks with broadly competitive quality in this configuration. These results do not establish parity with native AutoML.

Feature selection retains 11/21/16 of 30 columns on Breast Cancer, 58/49/64 of 64 on Digits, 6/7/2 of 10 on Diabetes, and 80/68/72 of 80 on the synthetic problem. Diabetes is a clear weakness: its mean MSE is 3708.33 versus 2820.31 for HPO, despite selecting fewer features. Because the two workflows also use different model parameters, that difference cannot be attributed solely to the evolutionary operators. No empty masks occurred. The longest measured fit was 31.53 seconds against a requested 15 seconds; reducing generation-level overshoot and model-evaluation cost remains necessary.

Reproduce with an isolated Python environment containing the recorded package versions and this repository:

```powershell
.agent-runs\performance-v2\automl-env\Scripts\python.exe benchmarks\benchmark_native_automl.py --output native-automl.json --model-dir native-models --label replication --seeds 3 --seed-start 901 --seconds 15
```

The harness writes results atomically after each completed run. Verification includes four smoke runs, 16 pilot runs, the 48-run matrix, 25 original benchmark tests and six final CLI boundary tests. The additional legal-boundary test rejects an over-restrictive seed-count mutant. Independent review found no remaining material issue; scoped Black, test guard and diff checks pass. Full suite not run.

[Raw results](results/performance-v2/native-automl.json), [verification](results/performance-v2/native-automl-verification.txt).

## Reduced-training evolution with full-training finalist selection

`subsample_cv` builds explicit CV splits with fewer training rows and unchanged validation rows. Both public searches can then re-evaluate distinct finalists on the original, full-training folds before choosing and refitting the winner. `GAFeatureSelectionCV` now shares the existing HPO final-selection engine, including grouped configuration, cloning, sample metadata and checkpoint support. Equal final feature scores prefer fewer columns. Search history and `cv_results_` remain proxy results; full-fold scores are recorded separately. This workflow is opt-in and adds no dependency.

The paired benchmark compares ordinary full-training search with training fractions 0.25 and 0.125. It uses population eight, four generations, three CV folds and four unique promoted finalists, with one CPU thread. Every fit has the same seeded 25% outer holdout, which never participates in search or finalist selection. Timing includes split preparation, search construction, evolutionary CV, full-training finalist CV and the final refit; imports and holdout prediction are excluded. Training subsampling uses independent seeded state and preserves all validation indices. Classification uses accuracy; Diabetes uses negative MSE, where higher is better.

SVM uses a fitted-within-fold `StandardScaler` pipeline: HPO searches logarithmic C/gamma, while feature selection uses fixed C=1 for classification and C=100 for regression. Random forest uses 80 trees: HPO searches depth, minimum leaf size and maximum feature fraction; feature selection uses fixed model parameters. The synthetic problem has 2,000 rows and 40 columns, including ten informative and eight redundant columns, with fixed dataset seed 113. SVM covers this problem plus Breast Cancer, Digits and Diabetes with seeds 1201–1205; random forest covers synthetic classification and Diabetes with seeds 1201–1203. These are configured workflow comparisons, not native AutoML comparisons or evidence of a universal optimizer improvement.

The fixed method order and shared Windows machine limit timing precision. Small focused checks occasionally ran alongside the measurements; no second heavy training workload ran concurrently. Lower-fidelity rankings can discard candidates that would have performed well with full training, and finalist promotion cannot recover them. The library therefore does not enable subsampling automatically or select its fraction using the outer holdout.

The RF process used the repaired one-sided sampler and started before a final lifecycle correction that clears three conditional result attributes on feature-fit entry. Every benchmark search is a new instance, so those attributes were absent and the cleanup would do nothing. Independent review confirmed this distinction does not require repeating RF. The final SVM replay uses the lifecycle correction too; [source hashes](results/performance-v2/fidelity-source-sha256.json) preserve both measured implementations.

Random-forest results: mean held-out score / total fitting seconds across three seeds.

| Dataset / search | Full training | 25% training + finalists | 12.5% training + finalists |
|---|---:|---:|---:|
| synthetic / hpo | 0.89800 / 102.04 | 0.90933 / 33.55 | 0.90667 / 28.27 |
| synthetic / features | 0.91400 / 44.12 | 0.90200 / 25.34 | 0.90067 / 22.79 |
| diabetes / hpo | -3008.37951 / 16.48 | -3051.39933 / 15.44 | -3186.97729 / 15.46 |
| diabetes / features | -3204.83070 / 18.09 | -3343.80777 / 14.91 | -3311.78857 / 14.22 |

On synthetic classification, RF HPO is 3.04x faster with a quarter of training rows and 3.61x faster with an eighth; held-out accuracy improves by 0.01133 and 0.00867 respectively. Feature selection is only 1.74x/1.94x faster and loses 0.01200/0.01333 accuracy. On Diabetes, HPO gains only about 1.07x and feature selection 1.21x/1.27x, while every reduced-training mean has worse held-out MSE. These RF results do not establish a 4x improvement and do not justify automatically subsampling every search.

[RF raw results](results/performance-v2/fidelity-rf.json).

SVM results: mean held-out score / total fitting seconds across five seeds.

| Dataset / search | Full training | 25% training + finalists | 12.5% training + finalists |
|---|---:|---:|---:|
| synthetic / hpo | 0.92520 / 26.15 | 0.92360 / 4.69 | 0.92480 / 2.86 |
| synthetic / features | 0.92040 / 9.93 | 0.92200 / 3.22 | 0.91920 / 2.22 |
| breast / hpo | 0.97063 / 2.29 | 0.96923 / 1.21 | 0.96923 / 1.09 |
| breast / features | 0.96783 / 1.09 | 0.97483 / 0.86 | 0.96084 / 0.82 |
| digits / hpo | 0.98133 / 26.23 | 0.98044 / 4.76 | 0.98044 / 2.81 |
| digits / features | 0.97867 / 10.23 | 0.98311 / 3.84 | 0.98267 / 2.61 |
| diabetes / hpo | -2809.24243 / 2.17 | -2783.42300 / 0.94 | -2834.50426 / 0.85 |
| diabetes / features | -3369.60862 / 1.33 | -3409.49632 / 0.73 | -3324.93156 / 0.65 |

SVM HPO reaches 9.16x speedup on synthetic classification and 9.35x on Digits at one-eighth training size, with held-out accuracy differences of -0.00040 and -0.00089. At a quarter, the speedups are 5.57x and 5.51x. Feature selection reaches 4.47x on synthetic classification with a -0.00120 accuracy difference and 3.92x on Digits with a +0.00400 difference. These are measured, workload-specific gains; they do not prove quality equivalence or a universal 4x improvement.

On Breast Cancer, quarter-training feature selection improves mean held-out accuracy by 0.00699 but is only 1.26x faster; using an eighth loses 0.00699 instead. Diabetes is mixed: quarter-training HPO lowers held-out MSE from 2809.24 to 2783.42 at 2.31x speed, while quarter-training feature selection increases MSE from 3369.61 to 3409.50. All 156 final fits completed; every feature-selection result retained at least one column. The raw rows include every seed, selected configuration/mask, number of search and finalist CV calls, fold sizes and dependency versions. No fraction is chosen automatically from these holdout scores.

Reproduce the final matrices with the public API:

```powershell
.venv\Scripts\python.exe benchmarks\benchmark_fidelity.py --output fidelity-svm.json --models svm --seeds 5 --seed-start 1201 --label final-sampler
.venv\Scripts\python.exe benchmarks\benchmark_fidelity.py --output fidelity-rf.json --models rf --datasets synthetic diabetes --seeds 3 --seed-start 1201
```

Verification includes recorded red finalist/NaN/budget, grouped-clone, checkpoint, sampling-boundary and repeated-fit regressions; 34 final focused cases; 57 adjacent HPO/feature/configuration/metadata/checkpoint cases; four initial smoke fits; both documented example fits; and result-integrity validation of all 156 final fits. Removing sampling uniqueness, class preservation or conditional-result cleanup is caught by the corresponding tests. Independent implementation review closed all material findings. Full suite not run.

[Final SVM data](results/performance-v2/fidelity-svm.json), [RF data](results/performance-v2/fidelity-rf.json), [verification](results/performance-v2/fidelity-verification.txt).

## Accelerated workflows versus native CPU AutoML on fresh splits

The follow-up uses the native benchmark dataset loader: Breast Cancer, Digits, Diabetes and a synthetic classification problem with 2,000 rows, 80 columns, 12 informative and eight redundant columns. The synthetic problem therefore differs from the 40-column fidelity experiment above. Five fresh seeds (1301-1305) define the same 25% outer holdout for every workflow. No holdout row enters evolutionary CV, finalist selection or refit.

This matrix contains 160 fits: full-, quarter- and eighth-training SVM workflows for both public searches, plus native FLAML and AutoGluon. Evolution uses population eight, at most four generations and a requested 15-second soft limit, then up to four distinct full-training finalists when subsampling is enabled. The full-training workflow can hit the timer before completing four generations, while a faster workflow can evaluate more candidates. These speedups consequently compare configured workflows under the same upper bounds, not identical candidate sequences. The native methods keep their preceding three-family, three-fold CPU settings; their validation and ensembles differ from the evolutionary pipelines.

Actual fitting time includes model construction, subsampling, evolutionary evaluations, full-training promotion and refit. The timer applies to evolution and is checked after generations; it does not cap finalist promotion or refit. Imports and outer prediction are excluded. The four profiles ran in fixed order: full, quarter, eighth, native. One heavy process ran at a time on the shared Windows machine; small diagnostic probes also ran, so subsecond timing differences should not be overinterpreted. Neither fraction nor model family is selected automatically using these holdout outcomes.

Mean held-out score / total fitting seconds across five seeds (higher score is better, including negative MSE):

| Dataset / search | Full training | 25% training + finalists | 12.5% training + finalists |
|---|---:|---:|---:|
| breast / gasearch | 0.97343 / 2.29 | 0.97483 / 1.15 | 0.97203 / 1.11 |
| breast / features | 0.97063 / 1.06 | 0.97203 / 0.83 | 0.97203 / 0.85 |
| digits / gasearch | 0.98311 / 16.35 | 0.98178 / 4.78 | 0.98400 / 2.79 |
| digits / features | 0.98044 / 10.14 | 0.98089 / 3.75 | 0.97867 / 2.57 |
| diabetes / gasearch | -3145.06237 / 2.08 | -3152.26357 / 0.95 | -3158.19614 / 0.86 |
| diabetes / features | -4236.90037 / 1.34 | -4000.68342 / 0.75 | -3815.68739 / 0.66 |
| synthetic / gasearch | 0.85280 / 17.92 | 0.85480 / 5.23 | 0.84920 / 3.27 |
| synthetic / features | 0.86360 / 12.70 | 0.85640 / 3.77 | 0.86240 / 2.65 |

| Dataset | FLAML native | AutoGluon native |
|---|---:|---:|
| breast | 0.96224 / 15.00 | 0.95245 / 2.03 |
| digits | 0.97333 / 15.30 | 0.97733 / 6.55 |
| diabetes | -3462.58553 / 15.01 | -3356.23817 / 1.55 |
| synthetic | 0.88240 / 15.26 | 0.88480 / 8.44 |

On Digits, eighth-training HPO is 5.87x faster than this full-training workflow, with accuracy 0.98400 versus 0.98311. Feature selection is 3.94x faster with accuracy 0.97867 versus 0.98044. Against native AutoGluon, these eighth-training HPO/feature workflows take 2.79/2.57 seconds versus 6.55, with mean accuracies 0.98400/0.97867 versus 0.97733. They also have higher mean accuracy and lower time than the configured FLAML workflow. This establishes competitive measured workflows on this dataset, not statistical superiority or an optimizer-only comparison.

Breast Cancer also favors the accelerated SVM workflows on mean accuracy and elapsed time against both configured native methods. Diabetes HPO at one-eighth training takes 0.86 seconds with MSE 3158.20, versus AutoGluon's 1.55 seconds and MSE 3356.24. Feature selection remains weaker on Diabetes: even though eighth-training improves MSE from 4236.90 to 3815.69 and halves its own fitting time, it is worse than both native AutoML means. It retains 7.4 of ten columns on average, compared with 5.8 for full-training selection; faster fitting here is not a larger dimensionality reduction.

The 80-column synthetic task does not reach native AutoML quality. Eighth-training HPO/feature searches take 3.27/2.65 seconds and score 0.84920/0.86240, versus AutoGluon's 8.44 seconds and 0.88480. Relative to their own full-training baselines, the speedups are 5.49x/4.79x, with accuracy changes -0.00360/-0.00120. The quarter-training feature workflow loses 0.00720. A single faster SVM workflow therefore cannot replace model-family selection, and these results do not justify enabling subsampling by default.

All 160 fits completed in 894.85 seconds of measured fitting time, with no empty selected mask. Each promoted workflow scored four distinct finalists on full training folds. Large-task full-training HPO evaluated 42.8 (Digits) and 39.8 (synthetic) candidates on average before the timer, versus about 71 under reduced training. Every raw seed, candidate count, mask/configuration, fold size, dependency version and source hash is retained. Sample standard deviations are in the verification summary; five overlapping dataset splits are not five independent datasets, and no quality-equivalence claim is made.

Reproduce with the recorded isolated environment; use a fresh native model label/directory:

```powershell
$benchmarkPython = '.agent-runs\performance-v2\automl-env\Scripts\python.exe'
foreach ($fraction in @(1, 0.25, 0.125)) {
    & $benchmarkPython benchmarks\benchmark_native_automl.py --output "svm-$fraction.json" --model-dir native-models-replication --label "svm-$fraction" --methods gasearch features --evolutionary-model svm --train-fraction $fraction --generations 4 --finalists 4 --seeds 5 --seed-start 1301 --seconds 15
}
& $benchmarkPython benchmarks\benchmark_native_automl.py --output native-fresh.json --model-dir native-models-replication --label native-fresh --methods flaml autogluon --evolutionary-model svm --generations 4 --seeds 5 --seed-start 1301 --seconds 15
```

Verification: seven new boundary/runtime cases failed before implementation; 14 final native benchmark cases pass, including real classification/regression fits, full-training and reduced-training paths, and exact outer-training inputs. Tests reject mutants that ignore positive generation counts or fit on the outer holdout. Scoped Black, test guard and diff checks pass. The complete 160-row matrix was checked for missing/duplicate combinations, finite metrics, fold sizes, promotion counts, valid masks, parameter domains and unchanged measured sources. Full suite not run.

[Raw matrix and source hashes](results/performance-v2/accelerated-native.json), [verification and per-workflow score dispersion](results/performance-v2/accelerated-native-verification.txt).


## Estimator-valued categorical inheritance

A public `GASearchCV.fit` reproduction exposed an evolutionary correctness issue: four warm-start parents all used a depth-one tree, yet crossover evaluated a depth-three tree with mutation and diversity replacement disabled. DEAP copies estimator objects; identity-based categorical membership rejected the copies and randomly sampled a replacement. Applying nested model parameters also modified the caller's categorical options before CV cloned the candidate.

Categorical membership now compares estimator constructor configurations and resolves valid copies to the original choice. Candidate and finalist evaluation clone parameter dictionaries before applying nested settings. The hall of fame compares canonical configurations, and its public winner-first ordering preserves distinct entries. This applies to all four HPO evolutionary algorithms, including cloned searches and notebook-local estimator classes; it introduces no new search option. Binary feature masks do not contain estimator categories and are unaffected by this correction.

Sixteen regression cases fail on the preceding implementation and pass after the change. In the crossover-only fixture, each of the three crossover GAs performs exactly four real CV evaluations for the four known configurations; repeated offspring use the cache. Differential evolution can create new numeric values and is checked for faithful model inheritance without imposing that four-call limit. Nested estimator, array and RNG configuration validation is covered through public warm-start sampling. Fifty-nine focused/adjacent tests pass, covering categorical spaces, cache, warm starts, refit and final selection. These are correctness and evaluation-count observations, not a general wall-time or held-out quality benchmark. Evidence: `results/performance-v2/categorical-inheritance-verification.txt`.


## Candidate-level time budgets in both searches

`TimerStopping` now stops scheduling candidate batches inside a generation, including local refinement. Serial evaluation and the CV backend check between candidates; population parallelism checks between batches of at most the effective worker count. At least one initial batch completes even when setup consumes the deadline, so a model remains available. In-flight cross-validation and final selection/refit finish: this is a soft search budget, not a hard bound on `fit`.

Partial generations retain the best completed candidate in the hall of fame. Unevaluated candidates receive no score, cache entry or evaluation count. Reusing a callback starts a new deadline; subsequent fits without it evaluate normally. The public API and full-generation behavior without a timer remain unchanged.

The paired benchmark uses the existing native-workflow harness with full-training SVM pipelines, four datasets and three fresh outer-split seeds (1701-1703). Each of both searches has population eight, up to 1,000 generations, three CV folds and a requested five-second evolutionary budget. Final refit is included; finalist promotion is disabled because training is full-size. The 48 runs alternate baseline/current order by seed and use immutable package snapshots. The baseline is `d821a1edbd367b22850ef513c0a8e7e5cacd55bb`. One heavy process ran at a time; small review probes ran concurrently on the shared Windows machine.

Mean held-out score / total fitting seconds across three seeds (higher score is better):

| Dataset / search | Generation boundary | Candidate boundary | Speedup |
|---|---:|---:|---:|
| breast / GASearchCV | 0.97436 / 5.35 | 0.97436 / 5.07 | 1.06x |
| breast / features | 0.96970 / 5.22 | 0.96737 / 5.06 | 1.03x |
| digits / GASearchCV | 0.98593 / 9.74 | 0.98667 / 5.37 | 1.81x |
| digits / features | 0.98222 / 5.71 | 0.98222 / 5.15 | 1.11x |
| diabetes / GASearchCV | -2724.17 / 5.07 | -2724.17 / 5.05 | 1.00x |
| diabetes / features | -3144.04 / 5.21 | -3144.04 / 5.07 | 1.03x |
| synthetic / GASearchCV | 0.89000 / 11.23 | 0.88467 / 5.44 | 2.07x |
| synthetic / features | 0.89600 / 7.31 | 0.89400 / 5.12 | 1.43x |

The maximum measured fit falls from 11.54 to 5.57 seconds; across all rows the mean falls from 6.85 to 5.16 seconds. This is reduced deadline overshoot, not increased evaluation throughput or a universal quality improvement. Synthetic HPO evaluates 11.7 candidates on average instead of 24 and loses 0.00533 held-out accuracy; synthetic feature selection evaluates 28 instead of 40 and loses 0.00200. Digits HPO is 1.81x faster with a +0.00074 accuracy difference; feature accuracy is unchanged. The three splits overlap and do not establish statistical equivalence. No 4x or SOTA claim follows from this timer experiment.

Reproduce either source snapshot with the existing harness and recorded dependency versions:

```powershell
.venv\Scripts\python.exe benchmarks\benchmark_native_automl.py --output timer-replication.json --model-dir timer-models --label timer-replication --methods gasearch features --evolutionary-model svm --train-fraction 1 --generations 1000 --finalists 4 --seeds 3 --seed-start 1701 --seconds 5
```

The raw artifact retains every seed, full mask/configuration, actual CV count, score, elapsed time, dependency version and source hash. [Raw paired matrix](results/performance-v2/timer-benchmark.json).

Verification: all 52 timer regression cases fail on the baseline. Final focused HPO/feature checks pass 109 cases; algorithm, internal-helper and callback files pass 200, with four pre-existing TensorFlow-unavailable skips. Coverage includes all four algorithms, serial/population/CV parallelism, cached and uncached evaluations, initial/later/refinement/setup deadlines, winners, counters and callback reuse. A counter-overcount mutant is rejected. Independent review closed its one material test gap. Black, test guard and diff checks pass; full suite not run. [Verification](results/performance-v2/timer-verification.txt).


## Compatibility of numeric smart initialization

The default numeric initializer failed before fitting under a supported older dependency combination: Python 3.12, NumPy 1.26.4, SciPy 1.11.4 and scikit-learn 1.5.0. SciPy renamed the QMC `seed` keyword to `rng` in 1.15 ([official migration note](https://docs.scipy.org/doc/scipy-1.15.0/reference/generated/scipy.stats.qmc.LatinHypercube.html)). The initializer now selects the keyword exposed by the constructor; it does not catch and mask constructor errors or add a dependency. Newer releases retain the existing `rng` path because changing the keyword can change seeded samples.

The existing public reproducibility test fails with the original code in that real older environment. Twenty-nine focused cases pass on each dependency stack after the fix. Additional public fits of both searches preserve exactly the current-stack candidate sequence, CV scores, predictions and counters, and both APIs fit successfully in the older environment. No new test duplicates the existing regression. This restores compatibility and makes no speed or quality claim. [Verification](results/performance-v2/qmc-compatibility-verification.txt).


## LightGBM fidelity and native AutoML comparison

A separate 96-fit matrix tests whether the earlier SVM acceleration transfers to tree boosting. It uses four datasets, three fresh outer-split seeds (1801-1803), a 25% holdout, one CPU, and 15-second soft search budgets. Both evolutionary searches use population eight, at most four generations, three CV folds and either full, quarter or eighth training folds. Reduced workflows reevaluate four distinct finalists on full training folds. Profile order rotates by seed; all measured sources remain at `f0105308780853f9c026fe6c4f17cfece888cf64`. FLAML and AutoGluon use their native CPU LightGBM/random-forest/extra-trees workflows, so model families and validation differ; this compares complete workflows, not identical optimizers. Every reported total includes final selection and refit.

Mean held-out score / fitting seconds over three seeds; classification uses accuracy and diabetes uses negative MSE (higher is better):

| Dataset / search | Full training | Quarter training | Eighth training |
|---|---:|---:|---:|
| breast / GASearchCV | .96037 / 6.12 | .96270 / 2.73 | .95571 / 2.46 |
| breast / features | .95338 / 5.85 | .95804 / 2.24 | .85781 / 1.15 |
| digits / GASearchCV | .97778 / 16.07 | .97778 / 19.40 | .97778 / 19.05 |
| digits / features | .98370 / 16.16 | .98222 / 21.31 | .97630 / 20.97 |
| diabetes / GASearchCV | -3325.75 / 3.64 | -3306.57 / 1.91 | -3535.46 / 1.63 |
| diabetes / features | -3532.41 / 2.34 | -3964.21 / 1.46 | -5593.70 / .82 |
| synthetic / GASearchCV | .89000 / 17.31 | .90467 / 24.68 | .89467 / 16.65 |
| synthetic / features | .90200 / 15.84 | .90600 / 14.86 | .90200 / 9.28 |

| Dataset | FLAML score / seconds | AutoGluon score / seconds |
|---|---:|---:|
| breast | .97436 / 15.00 | .94872 / 2.31 |
| digits | .97481 / 15.01 | .98370 / 6.59 |
| diabetes | -3278.48 / 15.00 | -3130.26 / 1.64 |
| synthetic | .89800 / 15.46 | .89800 / 8.76 |

The quarter workflow is 2.25x faster for breast HPO and 2.62x for breast features with small positive accuracy differences. Eighth-training synthetic features preserve mean accuracy at 1.71x speed and approach AutoGluon's elapsed time. However, eighth-training breast features lose 0.09557 accuracy despite a 5.07x time reduction, and diabetes feature error rises substantially. This is not a useful quality-preserving 4x result. Reduced workflows are slower on digits because they search more candidates before the deadline and still pay for full-training finalist reevaluation. Promotion/refit remain outside the soft evolutionary budget. Training fraction therefore stays explicit; these results do not justify automatically reducing it or selecting a fraction from holdout performance.

The small-training feature regressions motivate a separate investigation: a flat preliminary score can fill all finalist slots with small masks before full-training evaluation. That hypothesis is not a fix or a demonstrated improvement in this baseline matrix. Three overlapping splits do not establish statistical equivalence or general SOTA performance. TPOT and H2O were not measured.

Reproduce each evolutionary profile with the existing harness (replace the fraction with 1, .25 or .125):

```powershell
.agent-runs\performance-v2\automl-env\Scripts\python.exe benchmarks\benchmark_native_automl.py --output lightgbm-replication.json --model-dir lightgbm-models --label lightgbm-replication --methods gasearch features --evolutionary-model lightgbm --train-fraction .125 --generations 4 --finalists 4 --seeds 3 --seed-start 1801 --seconds 15
```

For native controls, use `--methods flaml autogluon --train-fraction 1`. Dependency versions, exact settings, all scores, masks, configurations, counters, timings and source hashes are retained in the [raw matrix](results/performance-v2/lightgbm-native.json). [Validation and dispersion](results/performance-v2/lightgbm-native-verification.txt) cover all 96 combinations. Total measured fit time is 971.16 seconds. This evidence-only slice changes no production code or tests; the full suite was not run.


## Feature finalist diversity when preliminary scores are flat

Reduced-training CV can leave no useful ranking between feature masks. In the LightGBM discovery matrix, eighth-training folds contain 35 rows for breast cancer and 27 for diabetes; the model uses `min_child_samples=20`. LightGBM documents this as an approximate minimum leaf size ([parameter reference](https://lightgbm.readthedocs.io/en/stable/Parameters.html#min_data_in_leaf)). The paired reproduction records the actual number of distinct preliminary scores rather than assuming that small folds must always produce a flat score.

Previously, final selection sorted tied masks by increasing feature count and spent every finalist slot on small masks. A public reproduction evaluates masks with 1, 2, 4 and 5 features, all with the same preliminary score, then promotes only the 1- and 2-feature masks despite the 5-feature mask winning under full-training evaluation.

When at least two finalist slots share the cutoff score, selection now keeps the sparse representative and replaces the last slot with a denser unselected mask at that same score when available. Higher-ranked candidates retain priority, masks remain distinct, and `final_selection_top_k` still bounds the number of extra CV calls. One-finalist searches remain unchanged; final-score ties still prefer fewer features. Masks respect `max_features` because selection draws only from evaluated candidates. This changes neither `GASearchCV` finalist ranking nor the evolutionary candidate sequence.


The paired validation uses five new seeds (1901-1905), four datasets and quarter/eighth training fractions: 80 feature-search fits before/after, plus eight HPO control fits at seed 1906. Both snapshots use exactly four generations, population eight, three CV folds, four full-training finalists and one CPU. A nonbinding 3,600-second timer preserves the harness path without truncating searches. Baseline/current and fraction order alternate by seed. Every one of the 44 pairs has identical evolutionary candidate/score hashes and actual CV counts; only 12 finalist lists change. Finalist score-rank multisets and the first finalist remain identical. The baseline production source is `f0105308780853f9c026fe6c4f17cfece888cf64`; the intervening benchmark-only commit adds no implementation change.

Mean held-out score / total fitting seconds across five seeds:

| Dataset / fraction | Previous finalist selection | Diverse tied finalists |
|---|---:|---:|
| breast / .25 | .967832 / 2.353 | .967832 / 2.278 |
| breast / .125 | .897902 / 1.183 | .969231 / 1.296 |
| digits / .25 | .973778 / 41.646 | .973778 / 41.505 |
| digits / .125 | .976889 / 24.715 | .976889 / 24.599 |
| diabetes / .25 | -3732.95 / 1.455 | -3732.95 / 1.376 |
| diabetes / .125 | -4581.02 / .830 | -3544.74 / .814 |
| synthetic / .25 | .892800 / 14.384 | .892800 / 14.068 |
| synthetic / .125 | .886800 / 8.870 | .886800 / 8.756 |

Eighth-training breast cancer has **3.32x lower mean classification error**, with accuracy improving by 0.07133 and elapsed time increasing by 9.5%. Diabetes mean squared error falls **22.6%**. All five seeds improve in each of these two cases; each records exactly one distinct preliminary score. The other six feature scenarios have unchanged held-out scores. The HPO controls retain the same candidates, finalists, selected parameters and scores. Timing differences in unchanged scenarios are not attributed to this code change.

The improvement preserves predictive quality by retaining more features: breast averages 30 instead of two, and diabetes ten instead of 1.8. It is not evidence of better dimensionality reduction, a universal 4x improvement or SOTA performance. Replacing one tied finalist can affect which configurations receive full evaluation; these five overlapping splits do not establish universal non-regression. Total measured fitting time is 1,055.99 seconds.

Reproduce each source with the native harness using `--methods features --evolutionary-model lightgbm --train-fraction .125 --generations 4 --finalists 4 --seeds 5 --seed-start 1901 --seconds 3600`; repeat at fraction `.25`. HPO controls use `--methods gasearch --train-fraction .25 --seeds 1 --seed-start 1906`. Supply the usual `--output`, `--model-dir` and `--label` arguments. The [paired raw artifact](results/performance-v2/feature-plateau.json) preserves per-seed masks, finalists, preliminary-score counts, exact candidate/score hashes, timing, settings and source hashes.

Verification: ten public regression cases fail without the change; 49 focused feature/HPO/final-selection/subsample checks pass. Coverage includes score direction, full-training refit and sample weights, score priority, fixed finalist counts, one finalist, bounded masks, final-score parsimony and uncached duplicate masks with differing observed scores. A mutant that removes duplicate protection fails both dedicated cases; independent review also rejects a mutant that ignores score rank and closes without material findings. Black, test guard and diff checks pass. No test file was added or renamed; the full suite was not run. [Verification and dispersion](results/performance-v2/feature-plateau-verification.txt).


## Rejected mutation-only change for fixed dimensions

An experiment changed `GASearchCV.mutate` to skip singleton categories and equal numeric bounds. The current operator can spend a mutation on an unchanged value; default diversity control can then replace that duplicate with a random candidate. The proposal reduced these replacements and preserved custom sampling, weighted priors and ordinary all-mutable seeded trajectories. However, it showed no consistent measured speed or held-out quality benefit, so **the production change was not adopted**.

The matrix contains 104 fits: two source snapshots x SVM/LightGBM x four datasets x three fresh seeds (2101-2103) x two budget regimes, plus eight ordinary SVM controls without fixed dimensions at seed 2104. One regime uses exactly four generations with a nonbinding 3,600-second timer; the other uses a five-second soft evolutionary budget and at most 1,000 generations. Population eight, full-training three-fold CV, a 25% held-out split, one CPU and no finalist promotion are shared within every pair. Execution order rotates across seeds. Final refit is included in elapsed time and can exceed the soft budget.

SVM adds fixed kernel=rbf, shrinking=True, max_iter=-1, tol=.001 and cache_size=200 to its C/gamma search. LightGBM adds n_jobs=1, verbosity=-1, random_state=seed, boosting_type=gbdt, objective=None, subsample=1 and colsample_bytree=1 to its five-variable search. These match estimator settings and introduce no new degrees of freedom. Each before/after pair has identical initialization and model domain.

Mean held-out score / fitting seconds over three seeds; classification uses accuracy and diabetes negative MSE (higher is better):

| Model / budget | Dataset | Existing mutation | Skip fixed dimensions |
|---|---|---:|---:|
| svm / 4 generations | breast | 0.972028 / 2.289 | 0.969697 / 2.259 |
| svm / 4 generations | digits | 0.982222 / 25.138 | 0.977778 / 25.791 |
| svm / 4 generations | diabetes | -3169.560603 / 2.135 | -3169.221639 / 2.129 |
| svm / 4 generations | synthetic | 0.881333 / 32.004 | 0.880667 / 31.526 |
| svm / 5 seconds | breast | 0.969697 / 5.057 | 0.969697 / 5.049 |
| svm / 5 seconds | digits | 0.979259 / 5.297 | 0.979259 / 5.211 |
| svm / 5 seconds | diabetes | -3155.955765 / 5.052 | -3156.900599 / 5.058 |
| svm / 5 seconds | synthetic | 0.880000 / 5.408 | 0.880000 / 5.482 |
| lightgbm / 4 generations | breast | 0.951049 / 5.581 | 0.941725 / 5.395 |
| lightgbm / 4 generations | digits | 0.972593 / 67.280 | 0.973333 / 69.129 |
| lightgbm / 4 generations | diabetes | -3496.545699 / 3.233 | -3532.346414 / 3.320 |
| lightgbm / 4 generations | synthetic | 0.904667 / 90.244 | 0.904667 / 101.569 |
| lightgbm / 5 seconds | breast | 0.951049 / 5.100 | 0.944056 / 5.082 |
| lightgbm / 5 seconds | digits | 0.973333 / 5.901 | 0.973333 / 5.930 |
| lightgbm / 5 seconds | diabetes | -3524.866826 / 5.048 | -3554.999184 / 5.051 |
| lightgbm / 5 seconds | synthetic | 0.899333 / 6.241 | 0.899333 / 6.352 |

At four generations, duplicate replacements fall from 520 to 443 for SVM and 302 to 262 for LightGBM across all datasets/seeds. Yet total SVM fitting time remains 184.69 versus 185.11 seconds; LightGBM rises from 499.01 to 538.24 seconds. Mean classification accuracy falls by 0.00248 and 0.00286 respectively. Under five seconds, SVM classification accuracy is unchanged; LightGBM loses 0.00233 on average. On expensive LightGBM tasks the deadline can end during initial population evaluation, before mutation runs, so identical scores there are not evidence that the mutation strategy is effective. Changes in sampled model complexity also affect fixed-generation timings.

These three overlapping train/test splits do not establish universal superiority of either strategy. They provide no basis for shipping this proposal as a performance improvement. The earlier native benchmark spaces contain no fixed dimensions and would not benefit from it. Total measured fitting time is 1,789.82 seconds. The eight ordinary controls preserve exact candidate/score hashes, selected parameters, scores, CV counts, cache hits and duplicate-replacement counters. All source snapshots remain unchanged.

The [raw experiment](results/performance-v2/fixed-mutation-experiment.json) preserves the rejected mutation method, execution/aggregation scripts, dependency versions, settings, per-seed scores, counters, histories and source hashes; baseline is `aa61cd20b9aa4914bace5ab0f4b023a5b745b3f4`. Scripts reuse `benchmark_native_automl.py` and add the constants listed above. The prototype had five discriminating public regressions fail against the original operator and 28 focused checks pass with the proposal. Independent mutation checks covered interleaved fixed coordinates, custom and weighted sampling, and exact Python/NumPy RNG preservation on mutable spaces. Prototype tests and implementation are retained with the local experiment, not shipped as a library change. [Validation and dispersion](results/performance-v2/fixed-mutation-verification.txt).


## A dense initial feature candidate under a short deadline

Smart feature initialization constructs masks in increasing feature-count order. Candidate-level stopping can therefore spend an entire short budget evaluating only the sparsest masks, even though the initial population already contains a dense candidate. Fresh `GAFeatureSelectionCV` searches with smart initialization and `TimerStopping` now move the densest existing initial mask first. The candidate set, `max_features` bound and soft deadline semantics are retained. Searches without a timer, random initialization, checkpoint resumes and HPO keep their ordering. This is a change in the timed feature-search trajectory, not a new evolutionary algorithm or a preprocessing filter.

The paired experiment has 136 fits: 120 feature searches across two sources, SVM/LightGBM, five datasets, three fresh seeds (2201-2203) and one-/five-second budgets, plus 16 no-timer initial-generation controls for features/HPO on the four earlier datasets at seed 2204. Each fit uses population eight, full-training three-fold CV, one CPU, a 25% held-out split and no finalist promotion. Timed runs allow at most 1,000 generations; execution order rotates by seed. In-flight CV and final refit finish beyond the soft deadline, and actual fitting times include this work.

This adds [UCI Dry Bean](https://archive.ics.uci.edu/dataset/602/dry%2Bbean%2Bdataset): 13,611 samples, 16 measured features and seven classes, licensed CC BY 4.0, DOI [10.24432/C50S4B](https://doi.org/10.24432/C50S4B). The source ZIP/ARFF and prepared-array hashes are recorded with its provenance. Each split contains 10,208 training and 3,403 held-out rows. No external test labels enter the search.

Mean held-out score / total fitting seconds across three seeds; classification uses accuracy and diabetes negative MSE:

| Model / requested seconds | Dataset | Ascending initial masks | Dense candidate first |
|---|---|---:|---:|
| svm / 1 | breast | 0.969697 / 1.022 | 0.965035 / 1.026 |
| svm / 1 | digits | 0.979259 / 1.115 | 0.977037 / 1.163 |
| svm / 1 | diabetes | -3272.372678 / 1.033 | -3214.995440 / 1.029 |
| svm / 1 | synthetic | 0.838000 / 1.150 | 0.864667 / 1.150 |
| svm / 1 | drybean | 0.595945 / 7.486 | 0.925850 / 2.203 |
| svm / 5 | breast | 0.958042 / 5.063 | 0.962704 / 5.062 |
| svm / 5 | digits | 0.979259 / 5.135 | 0.978519 / 5.099 |
| svm / 5 | diabetes | -3303.909203 / 5.067 | -3303.909203 / 5.066 |
| svm / 5 | synthetic | 0.873333 / 5.129 | 0.867333 / 5.193 |
| svm / 5 | drybean | 0.698012 / 7.922 | 0.925850 / 8.044 |
| lightgbm / 1 | breast | 0.972028 / 1.063 | 0.967366 / 1.082 |
| lightgbm / 1 | digits | 0.691852 / 1.800 | 0.973333 / 2.023 |
| lightgbm / 1 | diabetes | -3350.847136 / 1.032 | -3339.863577 / 1.048 |
| lightgbm / 1 | synthetic | 0.810000 / 1.722 | 0.898667 / 2.039 |
| lightgbm / 1 | drybean | 0.588794 / 1.852 | 0.922421 / 4.332 |
| lightgbm / 5 | breast | 0.962704 / 5.089 | 0.974359 / 5.090 |
| lightgbm / 5 | digits | 0.966667 / 6.627 | 0.973333 / 5.615 |
| lightgbm / 5 | diabetes | -3660.065813 / 5.061 | -3727.289848 / 5.056 |
| lightgbm / 5 | synthetic | 0.904000 / 6.768 | 0.903333 / 6.485 |
| lightgbm / 5 | drybean | 0.899500 / 6.337 | 0.922421 / 7.483 |

With a one-second request, Dry Bean SVM improves from .59595 to .92585 accuracy: **5.45x lower mean classification error**, while elapsed time falls from 7.486 to 2.203 seconds (**3.40x faster**). All three seeds improve. The original one-feature SVM fits are slower than fitting all features on this data. At five seconds the mean error is 4.07x lower, but elapsed time is slightly higher (8.044 versus 7.922 seconds). These results retain all 16 features rather than one to two; they improve prediction under a short budget, not dimensionality reduction.

Digits LightGBM at one second improves from .69185 to .97333 accuracy (**11.56x lower mean error**), at a 12.4% time increase. Dry Bean LightGBM at one second has 5.30x lower mean error but takes 4.332 instead of 1.852 seconds (2.34x as long). Synthetic accuracy also improves at one second with either model. Large gains occur where the old budget omitted the dense candidate; they do not demonstrate faster individual CV evaluations or universal SOTA performance.

Other scenarios are mixed: SVM accuracy decreases by .00466 on breast and .00222 on digits at one second, and by .006 on synthetic at five seconds. LightGBM breast accuracy decreases by .00466 at one second, while five-second diabetes MSE rises about 1.84%. Full per-seed scores, masks, evaluation counts and dispersion are retained. Three overlapping splits are descriptive evidence, not statistical equivalence or universal non-regression. The strongest quality gains often retain every feature. Total measured fitting time is 478.99 seconds.

Every timed current run starts with the dense mask, every baseline with one feature, and every returned mask is nonempty and within the feature bounds. The no-timer controls preserve exact candidate/score hashes, full histories, scores, chosen masks/parameters and counters in both APIs. Frozen source hashes remain unchanged. The [raw artifact](results/performance-v2/feature-timer.json) includes settings, dependencies, dataset provenance and driver/runner/reducer scripts that reuse `benchmark_native_automl.py`; baseline is `dac3879d15a26764f2107a133a2f11065c2a0192`. [Validation and dispersion](results/performance-v2/feature-timer-verification.txt).

Verification: 66 focused checks pass across both search APIs, including timed candidate batching, smart initialization, unchanged complete initial candidate sets and real checkpoint resume. The new behavior failed eight cases against the baseline; random-initialization controls already passed. An independent mutant probe exposed a population-backfill coverage gap; extending the existing parameterization now rejects taking the last mask instead of the densest. Black and the test guard pass. The full suite was not run.

## Larger real-data comparison on Dry Bean

A further 42 fits compare both evolutionary search APIs against native FLAML and AutoGluon on the 13,611-row, 16-feature, seven-class Dry Bean dataset above. Three new stratified 75/25 train/test splits use seeds 2401-2403. Every workflow receives one CPU. Evolutionary search uses SVM or LightGBM, population eight, up to 1,000 generations and three-fold CV. Full-training search requests 15 seconds; reduced workflows request 3.75 seconds using a quarter or an eighth of each training fold, then re-evaluate up to four finalists using full training folds. Validation rows are retained, and holdout labels are excluded from fitting. Actual times include final validation and refit.

FLAML and AutoGluon each request 15 seconds using their native validation/search and LightGBM, random-forest and extra-trees families. AutoGluon uses three bagging folds, no stacking and sequential fitting. Their model families, folds, ensembling and resource allocation differ from the fixed-model evolutionary workflows: this compares complete CPU workflows, not isolated optimizer quality or every configuration of those libraries.

Mean held-out accuracy / elapsed fitting seconds / selected features over three splits:

| Model / method | Full training | Quarter training + promotion | Eighth training + promotion |
|---|---:|---:|---:|
| svm / gasearch | 0.930845 / 17.683 / 16.00 | 0.931433 / 16.137 / 16.00 | 0.930649 / 12.096 / 16.00 |
| svm / features | 0.929376 / 16.885 / 13.00 | 0.929376 / 11.600 / 13.00 | 0.928984 / 11.355 / 12.33 |
| lightgbm / gasearch | 0.928299 / 17.906 / 16.00 | 0.927025 / 16.303 / 16.00 | 0.927025 / 16.651 / 16.00 |
| lightgbm / features | 0.925850 / 16.590 / 16.00 | 0.925850 / 12.230 / 16.00 | 0.925850 / 11.900 / 16.00 |

| Native workflow | Accuracy | Fitting seconds | Features |
|---|---:|---:|---:|
| flaml | 0.926437 | 15.524 | 16 |
| autogluon | 0.928886 | 13.168 | 16 |

SVM HPO with one-eighth training is **1.46x faster** than full search (12.096 versus 17.683 seconds), with a mean accuracy change of -0.000196. Quarter training improves accuracy by 0.000588 but provides only a 1.10x speedup. Feature selection with SVM is 1.46x faster at quarter training with unchanged mean accuracy, or 1.49x faster at eighth training with a -0.000392 change and 12.33 versus 13 selected features on average. LightGBM feature search keeps identical accuracy and all 16 features, at 1.36-1.39x speedups; LightGBM HPO gains only 1.08-1.10x and loses 0.001273 accuracy.

The reduced SVM HPO workflow averages .930649 accuracy in 12.096 seconds, versus .928886 in 13.168 seconds for AutoGluon and .926437 in 15.524 seconds for FLAML. This supports comparable measured time and quality on this dataset and CPU setting. It does not establish statistical superiority from three overlapping splits or general SOTA performance. AutoGluon finishes one split in 9.230 seconds, faster than that split's reduced SVM workflow. All per-seed values and standard deviations are preserved.

A fourfold smaller requested evolutionary budget does **not** produce fourfold lower total time here. Full-training finalist validation is a substantial remaining cost. The matrix totals 618.08 measured fitting seconds. Raw records verify finite scores, nonempty/bounded feature masks, explicit training-fold sizes and promotion counts; frozen sources remain unchanged.

[Raw results and reproduction scripts](results/performance-v2/drybean-native.json), [validation and dispersion](results/performance-v2/drybean-native-verification.txt). Source commit: `78a9f2d46cf20c11f47194fd1bc13ce0fcfb7329`. The embedded data-preparation script rebuilds the exact compressed NumPy dataset from the official UCI ZIP, checking ZIP, ARFF and array-file hashes. The drivers reuse `benchmark_native_automl.py`; no library behavior changes in this slice.

## Forest preset prerequisite: a deprecated alias breaks CV cloning

Real `GASearchCV.fit` checks of the regression forest presets found a current compatibility failure before the planned fast-profile performance experiment could run. In scikit-learn 1.9, assigning `criterion="friedman_mse"` with `set_params` leaves that value on the estimator, while its constructor normalizes it to `"squared_error"`. The subsequent CV clone rejects the changed constructor parameter with `RuntimeError`. Both Random Forest and Extra Trees are affected, including pipelines and every preset profile.

The two regression presets now offer `squared_error` and `absolute_error`, removing the redundant alias. The [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html) marks `friedman_mse` as deprecated; its installed constructor states it maps to `squared_error`. No model family, numeric range, profile name or prefix behavior changes. This prerequisite repairs usable searches; it is not a claimed training-speed improvement. Caller-supplied search spaces containing the upstream problematic alias are outside this preset-only change.

Twelve public preset/CV cases failed before the fix. All 76 focused preset checks pass on scikit-learn 1.9.1 and again on 1.5.0. Two duplicated factory-only criterion tests were replaced by one parametrized real-fit test covering both forests, all three profiles and direct/pipeline estimators; the absence-of-class-weight assertion is retained. The test guard reports a textual net reduction of two assertions for this consolidation, with both former contracts exercised by the replacement. Black and diff checks pass; no full suite was run. [Verification](results/performance-v2/forest-alias-verification.txt).

## Faster regression-forest presets, with an explicit robust-loss tradeoff

The `fast` profiles of `random_forest_regressor_space` and `extra_trees_regressor_space` now search only `squared_error`. The default `balanced` profile and `wide` retain both supported criteria. Names, prefixes, dimension keys and numeric ranges are unchanged. This reduces expensive candidate fits and frees a fixed time budget for more evaluations; it changes the search space and trajectory, not the implementation of scikit-learn trees. It applies to preset-driven `GASearchCV`, with no claimed feature-selection speedup from this particular change.

The baseline is the functional alias-repair commit `5ba9fe30dba5b5ce4ce3d9e6c0d2bce6b4d88094`. The matrix contains 96 searches: two versions × two forests × four datasets × three seeds (2501–2503) × two regimes. Each uses one CPU, a 25% held-out split, full-training three-fold CV and population six. Fixed searches run one generation and complete 18 CV evaluations each; timed searches use a five-second soft limit, at most 1,000 generations, and finish in-flight CV and final refit. Timings include that refit. Before/after execution order alternates by seed. Eight additional initial-generation fits test unchanged `balanced`/`wide` profiles with two-tree estimators.

Datasets are the real diabetes regression dataset, Friedman1 with 800 or 2,400 rows and 15 features (noise 1), and an 800-row Friedman1 case where independently sampled 10% target entries receive symmetric ±30 offsets before the outer split. The contaminated case optimizes MAE; others optimize MSE. Both held-out metrics are recorded for every run. A minus sign below means less held-out error. Ratios compare mean total fit times over three seeds.

| Forest | Budget | Dataset | Before → after seconds | Speedup | Held-out error change | CV calls before → after |
|---|---|---|---:|---:|---:|---:|
| rf | fixed | diabetes | 10.002 → 6.802 | 1.47× | -0.63% | 18.0 → 18.0 |
| rf | fixed | friedman800 | 18.816 → 9.617 | 1.96× | -1.02% | 18.0 → 18.0 |
| rf | fixed | friedman2400 | 56.034 → 21.050 | 2.66× | -1.49% | 18.0 → 18.0 |
| rf | fixed | outliers | 21.699 → 9.950 | 2.18× | +5.73% | 18.0 → 18.0 |
| rf | timer | diabetes | 5.432 → 5.419 | 1.00× | -0.78% | 9.0 → 14.0 |
| rf | timer | friedman800 | 7.179 → 5.698 | 1.26× | -4.27% | 5.7 → 9.0 |
| rf | timer | friedman2400 | 14.270 → 7.769 | 1.84× | -2.70% | 2.3 → 5.0 |
| rf | timer | outliers | 7.093 → 5.672 | 1.25× | +5.17% | 5.0 → 9.3 |
| et | fixed | diabetes | 13.855 → 4.727 | 2.93× | +1.10% | 18.0 → 18.0 |
| et | fixed | friedman800 | 32.937 → 9.536 | 3.45× | -1.05% | 18.0 → 18.0 |
| et | fixed | friedman2400 | 67.933 → 13.009 | 5.22× | -2.94% | 18.0 → 18.0 |
| et | fixed | outliers | 21.723 → 6.349 | 3.42× | +12.96% | 18.0 → 18.0 |
| et | timer | diabetes | 5.770 → 5.304 | 1.09× | +1.10% | 9.3 → 20.3 |
| et | timer | friedman800 | 8.191 → 5.434 | 1.51× | -3.87% | 5.7 → 16.3 |
| et | timer | friedman2400 | 18.128 → 5.811 | 3.12× | -6.08% | 2.3 → 9.3 |
| et | timer | outliers | 7.950 → 5.459 | 1.46× | +12.43% | 5.7 → 16.7 |

The strongest fixed-budget result is Extra Trees on 2,400-row Friedman1: 5.22× faster with 2.94% lower MSE. The same workload under the soft deadline completes four times as many CV evaluations on average, finishes 3.12× faster and lowers MSE by 6.08%. This is a workload-specific result, not a universal speedup. On diabetes, Extra Trees increases MSE by 1.10% in both regimes.

The contaminated-target experiment exposes the cost of removing the robust criterion: fixed-budget MAE rises 5.73% for Random Forest and 12.96% for Extra Trees; timed MAE rises 5.17% and 12.43%. [Scikit-learn describes the absolute-error criterion as using terminal-node medians](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html), unlike squared error's means. The change is accepted only for the explicitly fast profile; the ordinary default remains `balanced`. Three seeds, one small real dataset and synthetic workloads do not establish broad statistical superiority, and these numbers do not compare against native AutoML.

All four nonfast before/after control pairs preserve exact candidate/score hashes, predictions, best parameters, held-out/CV scores and evaluation counters. All 104 successful fits use matching before/after data and splits, finite scores, and valid full CV partitions; frozen sources are unchanged. Total measured fitting time is 1,334.48 seconds. The [raw artifact](results/performance-v2/fast-forest.json) contains per-seed results, dispersion, split hashes and sizes, raw-record hashes, dependency versions, all source hashes, the two changed source files and the execution/aggregation scripts. To reconstruct the experiment, populate `fast-forest-before` from the recorded baseline, copy it to `fast-forest-after` and replace the files in `changed_source_files`; extract the scripts and source manifests into `.agent-runs/performance-v2` and run the runner from the repository root using the recorded environment.

Four public fast-profile cases fail on the baseline while eight nonfast cases pass; the same 76 focused preset checks pass after the change on scikit-learn 1.9.1 and on 1.5.0. The cases exercise actual CV for both forests, every profile and direct/pipeline prefixes. Independent review exposed a rare-choice mutant that retained `absolute_error` with a 0.001 prior; all four fast cases now additionally reject that criterion through a fresh public warm-start request and kill the mutant. The artifact separates this later verification-test source from the original benchmark snapshots, and includes additional dependency versions captured at aggregation without an environment change. Black and test guard pass. No full suite or documentation build was run. [Verification](results/performance-v2/fast-forest-verification.txt).

## Rejected fast Random Forest classifier search-space reduction

After the regression-forest result, a paired classification experiment tested removing `max_features=None` from only the `fast` Random Forest classifier preset. The alternative kept `sqrt` and `log2`; the library preset was **not changed**. Pilot fits showed that all-feature tree splits can be expensive, but the full search exposed a material quality risk.

The matrix contains 48 real `GASearchCV` fits: baseline versus proposed space, breast cancer, digits, and 800/2,400-row synthetic classification, three seeds (2601–2603), one CPU, three-fold CV, 25% stratified holdout and population six. Fixed searches run one generation and 18 CV candidate evaluations each. Timed searches use a five-second soft limit, at most 1,000 generations, and include in-flight CV and final refit. Before/after order alternates by seed. Results below are means over three paired splits; the speed ratio divides baseline by proposed mean fitting time.

| Budget | Dataset | Baseline → proposed seconds | Speed ratio | Held-out accuracy baseline → proposed | CV calls baseline → proposed |
|---|---|---:|---:|---:|---:|
| fixed | breast | 12.088 → 9.716 | 1.24× | .94639 → .95105 | 18 → 18 |
| fixed | digits | 22.163 → 13.991 | 1.58× | .96963 → .97333 | 18 → 18 |
| fixed | synthetic 800 | 14.149 → 11.627 | 1.22× | .86833 → .86833 | 18 → 18 |
| fixed | synthetic 2,400 | 31.440 → 20.227 | 1.55× | .90444 → .89278 | 18 → 18 |
| timer | breast | 5.474 → 5.408 | 1.01× | .94406 → .94406 | 8.7 → 10.0 |
| timer | digits | 5.941 → 5.916 | 1.00× | .96889 → .96963 | 4.3 → 7.3 |
| timer | synthetic 800 | 6.471 → 5.718 | 1.13× | .86333 → .87000 | 7.0 → 9.0 |
| timer | synthetic 2,400 | 7.358 → 5.977 | 1.23× | .90000 → .89500 | 3.7 → 5.0 |

Across the 12 fixed pairs, total fit time falls from 239.52 to 166.68 seconds (1.44×), but the 2,400-row case at seed 2603 drops from .91833 to .88333 held-out accuracy with the proposed space. The baseline's winning candidate uses `max_features=None`, which the alternative excludes. Under the timer, the proposed space evaluates 94 instead of 71 candidates overall, but the same seed drops from .905 to .890 accuracy. This is enough to reject the preset change despite its useful speed gain. There is no claimed broad statistical equivalence from three overlapping splits, and the unchanged `balanced` and `wide` profiles were not re-benchmarked in this rejected experiment.

The [raw artifact](results/performance-v2/rf-fast-classifier-rejected.json) preserves all 48 candidate histories, split hashes, source hashes, dispersion, environment versions and reproduction scripts. Every pair used the same split, all scores are finite, each fixed run made 18 CV calls, and the frozen library source remained unchanged. Total measured fitting time was 550.99 seconds. This negative result guides further work toward cheaper evaluations that retain the useful candidate.

## Rejected smaller fast classification forests

A second classifier-preset experiment kept the full candidate space, including `max_features=None`, but reduced `n_estimators` from 50–180 to 25–90 in the `fast` Random Forest and Extra Trees spaces. The library preset was **not changed**. On breast cancer, digits and 800/2,400-row synthetic classification, six stratified train/test seeds (2701–2706) produced 96 fixed-budget fits: both spaces received 18 candidate CV calls, three folds, one CPU and one evolutionary generation. The first three seeds were a pilot; the other three were held apart for validation. A further 48 fits on validation seeds used a five-second soft timer and at most 1,000 generations. All fits include final refit in their elapsed time.

Means over six fixed-budget seeds (baseline → fewer trees):

| Dataset | Forest | Seconds | Held-out accuracy |
|---|---|---:|---:|
| breast | RF | 11.046 → 5.464 | .95921 → .95571 |
| breast | ET | 6.550 → 3.304 | .96970 → .96853 |
| digits | RF | 21.085 → 10.567 | .97593 → .97259 |
| digits | ET | 13.167 → 6.674 | .98370 → .98222 |
| synthetic 800 | RF | 13.980 → 7.139 | .88167 → .87750 |
| synthetic 800 | ET | 7.371 → 3.692 | .91000 → .90417 |
| synthetic 2,400 | RF | 30.297 → 14.803 | .89528 → .89389 |
| synthetic 2,400 | ET | 10.678 → 5.313 | .90917 → .91000 |

Across fixed runs, mean time fell from 19.102 to 9.493 seconds for RF (**2.01×**) and 9.442 to 4.746 seconds for ET (**1.99×**). Held-out accuracy fell by .00310 for RF and .00191 for ET. The proposed range lost on 25 of 48 pairs, won on 15 and tied on eight. The independent validation included a 2.8-point RF loss on breast and a 2.5-point RF loss on synthetic 800.

Under the five-second timer, RF evaluated 5.83 → 11.92 candidates on average, but mean accuracy fell .92620 → .92408 and elapsed time improved only 6.215 → 5.619 seconds. ET evaluated 11.08 → 22.33 candidates, while accuracy fell .94759 → .94282 and elapsed time improved 5.665 → 5.301 seconds. The fixed-budget speed gain therefore does not translate into a reliable short-deadline quality gain. Reducing the tree range is rejected as a default `fast` preset change; users who explicitly value speed over this measured quality loss can still define an `Integer(25, 90)` dimension themselves.

The [raw artifact](results/performance-v2/fast-classifier-tree-range-rejected.json) preserves all 144 fits, per-seed scores, times, winning parameters, CV counts, source hashes, dependency versions and scripts. The experiment used source commit `065300e`; the source files were unchanged throughout these runs. The data are descriptive: six overlapping splits per fixed workload and three timed splits do not establish statistical equivalence or broad superiority.

## One-fold evolution with full-CV finalist promotion

Both `GASearchCV` and `GAFeatureSelectionCV` already allow different search and finalist CV folds. To measure the workflow, 108 fits on breast cancer, digits and 800-row synthetic classification used a 25% stratified holdout, three original CV folds, one CPU, RBF SVC models, population eight and four generations (at most 72 proposed candidates). Full searches scored candidates on all three folds. Reduced searches used the first fold for evolution and re-evaluated either four or eight finalists on all three folds; a two-fold/four-finalist variant was also tested. Timings include finalist CV and final refit. Pilot seeds 901–903 tested one fold with four finalists; independent validation seeds 904–906 compared all four methods. This keeps the evolutionary search and its candidate budget, changing only the fidelity of candidate evaluation.

Validation means across three new seeds, shown as held-out accuracy / total fitting seconds:

| Dataset | API | Full three-fold search | One fold + eight finalists | Two folds + four finalists |
|---|---|---:|---:|---:|
| breast | `GASearchCV` | .96970 / 2.353 | .96970 / .975 | .96970 / 1.608 |
| breast | `GAFeatureSelectionCV` | .96737 / 1.009 | .96503 / .481 | .96970 / .742 |
| digits | `GASearchCV` | .98444 / 27.236 | .98370 / 9.932 | .98444 / 18.351 |
| digits | `GAFeatureSelectionCV` | .96889 / 9.199 | .96741 / 4.192 | .97037 / 6.722 |
| synthetic | `GASearchCV` | .88833 / 4.600 | .87000 / 1.856 | .87167 / 3.304 |
| synthetic | `GAFeatureSelectionCV` | .90667 / 2.130 | .89667 / .957 | .88667 / 1.546 |

One fold plus eight finalists is **2.10–2.74× faster** across these six validation groups, with mean accuracy changes from zero to −.01833. The four-finalist pilot yielded 2.35–2.97× speedups, but its mean feature-selection accuracy fell on digits and synthetic data. Adding finalists helped some validation groups, yet the synthetic SVC search still lost 1.83 percentage points. Two preliminary folds cut the speedup to 1.36–1.48× and did not eliminate synthetic losses. Selected feature counts stay near baseline on average; this workflow is a speed/quality tradeoff, not a dimensionality-reduction improvement.

The [raw artifact](results/performance-v2/fold-promotion-svm.json) contains all per-seed scores, times, candidate CV counts, finalist counts, selected feature counts, source hashes, dependency versions and reproduction scripts. At most 72 candidate CV calls were made in every fit (some cached duplicates made fewer); finalist calls are recorded separately. The groups share data and outer splits within each seed. Three overlapping validation seeds are descriptive rather than proof of equivalent accuracy. The [guide](../docs-vitepress/versions/latest/guide/advanced-optimizer-control.md) shows the opt-in API pattern; full three-fold search remains the default.

## Five-fold, larger-budget promotion versus Optuna TPE

A further 72 fits on the same three datasets test whether the earlier speed gain persists with five original folds and ten evolutionary generations. Seeds 907–909 were not used in the three-fold experiment. Every run has a stratified 25% outer holdout, the same training folds within each seed, one CPU and a StandardScaler/RBF-SVC pipeline. The GA runs use population eight and ten generations; all methods have up to 168 proposed candidates. `GASearchCV` and Optuna TPE search the same log-uniform C/gamma space; feature selection fixes C=10 and selects at most half the columns. The reduced workflow searches on the first fold, then scores its eight best distinct finalists on all five folds before full-training refit. Full workflows search on all five folds and refit the winner. All reported times include promotion and refit. Variant execution order alternates by seed.

Mean held-out accuracy / total fitting seconds over three seeds:

| Dataset | GA HPO, five folds | GA HPO, one fold + eight | Optuna, five folds | Optuna, one fold + eight |
|---|---:|---:|---:|---:|
| breast | .97902 / 9.878 | .97669 / 2.391 | .98135 / 4.593 | .97436 / 1.492 |
| digits | .97704 / 124.401 | .98000 / 26.585 | .97778 / 40.746 | .98222 / 9.793 |
| synthetic | .88833 / 21.433 | .87333 / 4.832 | .89333 / 11.166 | .87833 / 2.888 |

| Dataset | Feature selection, five folds | Feature selection, one fold + eight | Speed ratio | Mean accuracy change |
|---|---:|---:|---:|---:|
| breast | .97203 / 3.813 | .96037 / 1.052 | 3.63× | −.01166 |
| digits | .97556 / 34.544 | .97259 / 8.737 | 3.95× | −.00296 |
| synthetic | .92833 / 8.530 | .90833 / 2.190 | 3.90× | −.02000 |

For HPO, one fold plus eight finalists is **4.13–4.68× faster** than the full GA workflow; a four-finalist variant reaches 4.33–4.80×. The reduced GA gains .00296 accuracy on digits but loses .00233 on breast and .01500 on synthetic. Feature selection remains slightly below 4× with eight finalists and loses held-out accuracy on all three datasets; the worst individual loss is .035 on synthetic. The 4× time objective is thus met for this HPO configuration, not for a general speed-and-quality improvement.

Optuna is **1.60–3.05× faster** than GA under the matching HPO fold policy in these measurements and has higher mean held-out accuracy in five of the six dataset/policy comparisons. Both methods' reduced workflow loses .01500 accuracy on synthetic. This is evidence that lowering CV fidelity cannot simply be made the new default, and that GA candidate generation remains an important speed target. Optuna made 168 CV candidate calls per fit; `GASearchCV` made 158–167 after cache reuse. The three outer splits overlap, so small accuracy differences are descriptive. These fixed-model results do not compare full AutoML pipelines.

The [raw 72-fit artifact](results/performance-v2/five-fold-promotion-svm.json) retains each paired score, time, candidate/finalist count, selected feature count, source hash, dependency version and exact reproduction scripts. The one-fold pattern is documented in the [advanced optimizer guide](../docs-vitepress/versions/latest/guide/advanced-optimizer-control.md); evaluating candidates on all configured CV folds remains the default.

## Further evolutionary proposal screening

Ten additional paired experiments (99 pairs) tested whether changing how individuals are generated improves the quality/time tradeoff. Most HPO runs used one-fold evolution followed by eight finalists on the original five folds, with population eight and at most 168 proposals. Post-search local refinement adds three neighbors (at most 171 proposals). The feature-mask mutation experiment used 20/60/120-variable synthetic classification, a tree and logistic regression, three folds and 72 proposals. The feature-ranking experiment used full five-fold search; its supervised ranking sees all outer-training labels, so its inner search CV is not a nested performance estimate. All runs used one CPU and untouched outer holdouts. These experiments are exploratory; their seeds and exact scripts differ by proposal.

| Proposal | Pairs | Holdout wins/ties/losses | Median paired speed ratio | Worst accuracy change |
|---|---:|---:|---:|---:|
| Mixed local/global HPO mutation | 9 | 1/4/4 | 1.01× | −.0150 |
| Differential HPO, new validation seeds | 9 | 1/7/1 | 1.22× | −.0200 |
| Differential feature selection | 9 | 2/0/7 | 1.04× | −.0222 |
| Ranked feature-mask initialization | 9 | 5/1/3 | 1.03× | −.0210 |
| One-bit mask mutation with higher mutation probability | 18 | 7/1/10 | 1.03× | −.0900 |
| Tournament-selected parents | 9 | 2/4/3 | 1.00× | −.0100 |
| Bounded numeric blend crossover | 9 | 4/2/3 | 1.10× | −.0250 |
| Post-search local refinement | 9 | 0/9/0 | .99× | .0000 |
| Elite-neighborhood offspring | 9 | 3/5/1 | 1.09× | −.0210 |

The tenth experiment was a nine-pair differential-HPO pilot: 3 wins, 4 ties, 2 losses, median speed ratio 1.30× and worst accuracy change −.0050. Its baseline and proposed runs used separate processes, so the alternating validation above is the more reliable timing check. Differential evolution improves mean candidate scores on the SVC tasks, but cannot be recommended as the new default after its holdout loss and the feature-selection regressions. The other proposals likewise fail to show a repeatable improvement in both outcomes. None changed library behavior. The [raw proposal artifact](results/performance-v2/evolution-proposal-pilots.json) includes all per-fit records, versions, source hashes, paired summaries and exact reproduction scripts. A slow baseline outlier inflates the numeric-blend aggregate timing ratio; the table uses the median paired ratio.

## Does promoting more finalists recover accuracy?

A further 63 fits on fresh seeds 943–945 kept the same datasets, outer split, five original folds, one CPU, RBF-SVC models and maximum of 168 search proposals. They compared full five-fold GA, one-fold GA with eight or 24 five-fold finalists, and full five-fold Optuna TPE. Timings include finalist validation and final refit. For `GAFeatureSelectionCV`, C=10 and at most half the columns were selected. Task order reversed on odd seeds. Mean held-out accuracy / total fitting seconds:

| Dataset | GA HPO full | GA HPO top 8 | GA HPO top 24 | Optuna full | Features full | Features top 8 | Features top 24 |
|---|---:|---:|---:|---:|---:|---:|---:|
| breast | .96037 / 11.537 | .96737 / 2.845 | .96737 / 3.793 | .96270 / 6.541 | .95338 / 4.909 | .95804 / 1.456 | .95338 / 2.363 |
| digits | .98222 / 136.647 | .98519 / 27.646 | .98370 / 31.618 | .98148 / 40.878 | .98074 / 37.597 | .97778 / 9.683 | .97852 / 13.212 |
| synthetic | .86333 / 21.207 | .86667 / 4.815 | .86833 / 5.722 | .87000 / 10.693 | .92000 / 8.513 | .90667 / 2.178 | .90167 / 2.891 |

Twenty-four finalists did not consistently recover external accuracy. Against eight, they tied on breast, lost .00148 on digits and gained .00167 on synthetic for HPO, while taking longer in every group. For feature selection they helped digits by only .00074, tied or lost on the other datasets, and cost about 33–62% more. A larger finalist pool can increase selection bias even with five-fold finalist scores; more validation candidates are not a monotonic quality guarantee.

The source hashes and settings match the earlier [72-fit five-fold cohort](results/performance-v2/five-fold-promotion-svm.json), so its seeds 907–909 can be pooled descriptively with these new seeds. Across all six splits, the eight-finalist HPO workflow is **4.09–4.81× faster** than full five-fold GA search. Its mean held-out accuracy changes by +.00233 on breast, +.00296 on digits and −.00583 on synthetic; the worst single loss is .025. Against full five-fold Optuna, it is faster in these workloads, but this comparison uses fewer preliminary folds for GA. Under matching fold policies, the earlier cohort found Optuna faster. The feature-selection workflow is **3.48–3.92× faster**, with mean accuracy losses of .00350, .00296 and .01667 respectively; its worst single loss is .040. These six overlapping outer splits do not establish a general speed/quality guarantee or SOTA equivalence.

The [raw 63-fit artifact](results/performance-v2/top24-fivefold-validation.json) contains every score and timing, actual CV calls, finalist counts, split and source hashes, dependency versions, the exact script, per-cohort summaries and the six-seed synthesis. Each split hash is SHA-256 of the concatenated little-endian int64 training indices, test indices, then the training and validation indices for each of the five CV folds. No finalist count or fold policy was changed in the library; users can make this tradeoff through the existing opt-in API.

To reproduce those 63 fits from the repository root in PowerShell, install the benchmark dependencies in `.venv`, create the output directory, extract the recorded script, then run it. It writes the raw JSON and JSONL files under `.agent-runs/performance-v2`:

```powershell
New-Item -ItemType Directory -Force -Path .agent-runs/performance-v2 | Out-Null
.\.venv\Scripts\python.exe -c "import json,pathlib; a=json.loads(pathlib.Path('benchmarks/results/performance-v2/top24-fivefold-validation.json').read_text(encoding='utf-8')); pathlib.Path('.agent-runs/performance-v2/top24-fivefold-validation.py').write_text(a['reproduction_script'], encoding='utf-8')"
.\.venv\Scripts\python.exe .agent-runs/performance-v2/top24-fivefold-validation.py
```

## Evolutionary feature masks versus random search

A separate 48-fit control tests whether the default feature-search evolution adds value beyond random masks at the same proposal budget. Breast Cancer, Digits and the 800-row synthetic classification problem above each use eight fresh stratified 75/25 outer splits (seeds 951–958), three fixed CV folds, a scaled RBF SVC with C=10, one CPU and at most half the columns. `GAFeatureSelectionCV` uses its default `eaMuPlusLambda` algorithm, smart initialization, population eight and four generations: 72 mask proposals. The random control also proposes 72 masks, drawing the number of selected columns uniformly from one to the cap and then choosing that many columns uniformly; it caches duplicate CV scores and breaks equal scores toward fewer columns. Both refit the CV winner on the full outer-training partition. Timings include search and refit; execution order alternates by seed. The first three seeds were exploratory and the next five were reserved for validation.

| Dataset | GA accuracy / CV / seconds | Random accuracy / CV / seconds | GA − random accuracy | Validation-only GA − random |
|---|---:|---:|---:|---:|
| Breast Cancer | .97378 / .97653 / 1.052 | .96853 / .97477 / 1.062 | +.00524 | +.00420 |
| Digits | .97278 / .97197 / 9.243 | .96833 / .96446 / 9.618 | +.00444 | +.00044 |
| Synthetic | .90625 / .89833 / 2.243 | .88000 / .87854 / 2.377 | +.02625 | +.01600 |

The GA wins four, five and six of eight paired outer splits respectively, with two, one and zero ties. After duplicate caching, it makes 71–72 actual candidate CV calls per fit; the random control makes 70–72. Mean selected counts are 13.25 versus 12.63 on Breast Cancer, 30.00 versus 30.88 on Digits and 9.13 versus 8.88 on synthetic data. The held-out gains are therefore not accompanied by a consistent reduction in selected columns; two losses occur on each dataset. Validation-only Digits accuracy is nearly tied. This fixed-model control shows useful evolutionary search pressure on these workloads, not a general quality guarantee, statistical significance, a fourfold quality gain or superiority over stronger optimizers.

The [48-fit artifact](results/performance-v2/random-mask-control.json) includes all raw rows, pilot/validation summaries, source and split hashes, dependency versions, and the search and summary scripts. From the repository root, extract and run them with PowerShell after installing the benchmark dependencies in `.venv`:

```powershell
New-Item -ItemType Directory -Force -Path .agent-runs/performance-v2 | Out-Null
.\.venv\Scripts\python.exe -c "import json,pathlib; a=json.loads(pathlib.Path('benchmarks/results/performance-v2/random-mask-control.json').read_text(encoding='utf-8')); w=pathlib.Path('.agent-runs/performance-v2'); (w/'random-mask-ablation.py').write_text(a['reproduction_script'], encoding='utf-8'); (w/'summarize-random-mask.py').write_text(a['summary_script'], encoding='utf-8')"
.\.venv\Scripts\python.exe .agent-runs/performance-v2/random-mask-ablation.py
.\.venv\Scripts\python.exe .agent-runs/performance-v2/random-mask-ablation.py --seed-start 954 --seeds 5 --output .agent-runs/performance-v2/random-mask-ablation-validation.jsonl
.\.venv\Scripts\python.exe .agent-runs/performance-v2/summarize-random-mask.py
```

## Diverse feature finalists: exploratory tradeoff

The eight-finalist, one-fold search can spend its full-validation budget on similar masks. To test a diversity adjustment without changing the library, each paired run used one `GAFeatureSelectionCV` trajectory (population eight, ten generations, at most 168 one-fold candidate CV calls), then evaluated the union of two eight-mask finalist sets on the original five folds. The baseline uses the existing top-eight ranking. The exploratory selector starts with the top mask and greedily combines preliminary rank with minimum normalized Hamming distance from masks already chosen. Both choose a winner solely by five-fold accuracy, breaking ties toward fewer features, then fit on the complete outer-training data. No outer holdout information selects finalists or winners. The datasets, SVC model, half-column cap and 25% holdout match the five-fold experiments above.

A moderate setting (top-24 pool, distance weight 0.5) selected almost the same masks as the baseline across three pilot seeds 971–973. A stronger setting (top-64 pool, weight 2.0) was tried on seeds 974–976 and then held fixed for five new validation seeds 977–981. Its validation results are:

| Dataset | Top-eight holdout accuracy | Diverse-eight holdout accuracy | Mean change | Diverse wins/ties/losses |
|---|---:|---:|---:|---:|
| Breast Cancer | .95385 | .95804 | +.00420 | 2/2/1 |
| Digits | .97644 | .97511 | −.00133 | 1/2/2 |
| Synthetic | .89600 | .90600 | +.01000 | 2/3/0 |

The stronger selector changes most finalist masks, but loses accuracy on two Digits splits and one Breast Cancer split; its worst Digits change is −.01111. The default finalist ranking is therefore unchanged. These exploratory timings include an extra preliminary-model refit before the manual final-selection simulation and are not presented as production workflow speedups. The [paired artifact](results/performance-v2/diverse-finalists-pilot.json) records all 33 trajectories, finalist indices, held-out results, pilot/validation summaries, source hashes and reproduction scripts. As with the other comparisons, the overlapping splits do not establish a general quality guarantee.

The recorded scripts can be extracted and rerun from the repository root in PowerShell after installing the benchmark dependencies in `.venv`:

```powershell
New-Item -ItemType Directory -Force -Path .agent-runs/performance-v2 | Out-Null
.\.venv\Scripts\python.exe -c "import json,pathlib; a=json.loads(pathlib.Path('benchmarks/results/performance-v2/diverse-finalists-pilot.json').read_text(encoding='utf-8')); w=pathlib.Path('.agent-runs/performance-v2'); (w/'diverse-finalists-pilot.py').write_text(a['reproduction_script'], encoding='utf-8'); (w/'summarize-diverse-finalists.py').write_text(a['summary_script'], encoding='utf-8')"
.\.venv\Scripts\python.exe .agent-runs/performance-v2/diverse-finalists-pilot.py
.\.venv\Scripts\python.exe .agent-runs/performance-v2/diverse-finalists-pilot.py --seed-start 974 --seeds 3 --pool 64 --diversity-weight 2 --output .agent-runs/performance-v2/diverse-finalists-strong.jsonl
.\.venv\Scripts\python.exe .agent-runs/performance-v2/diverse-finalists-pilot.py --seed-start 977 --seeds 5 --pool 64 --diversity-weight 2 --output .agent-runs/performance-v2/diverse-finalists-validation.jsonl
.\.venv\Scripts\python.exe .agent-runs/performance-v2/summarize-diverse-finalists.py
```

## Does feature evolution improve after initialization?

The [random-mask control above](#evolutionary-feature-masks-versus-random-search) compares final solutions. A separate 24-fit experiment measures progress *within* the default evolutionary search: Breast Cancer, Digits and the same 800-row synthetic problem use eight fresh seeds 991–998, a stratified 25% outer holdout, three CV folds, one CPU, the same scaled RBF SVC with C=10, and a cap of half the columns. Every `GAFeatureSelectionCV` fit proposes 72 masks through population eight and four generations. The initial comparator is the best of the eight generation-zero masks by CV score (fewer columns break ties); it is separately refit on all outer-training rows. The final winner comes from the ordinary fitted search. The holdout is scored only after those choices.

| Dataset | Initial → final mean CV accuracy | Initial → final mean held-out accuracy | Held-out wins/ties/losses |
|---|---:|---:|---:|
| Breast Cancer | .97154 → .97975 | .96678 → .96503 | 1/4/3 |
| Digits | .95147 → .97253 | .95917 → .97444 | 7/0/1 |
| Synthetic | .82646 → .89958 | .83000 → .90500 | 8/0/0 |

The best-so-far CV score improves in 23 of 24 runs. On synthetic data, mean best CV accuracy progresses `.82646 → .85479 → .86875 → .88125 → .89958` from generation zero through four, and held-out accuracy gains .07500 on average. Digits gains .01528 held-out accuracy; Breast Cancer instead loses .00175 on average despite its CV gain. The evolutionary process is measurably improving its training objective on these workloads, and often improves external quality, but additional search can also overfit CV. This is not a fourfold quality gain or a guarantee on unseen datasets. The initial-model refit used for this comparison is outside the reported search fitting time.

The [24-fit artifact](results/performance-v2/feature-progress.json) contains every per-generation best score, initial and final external score, fit count and time, split and source hashes, dependency versions, and reproduction scripts. To rerun it from the repository root in PowerShell after installing benchmark dependencies in `.venv`:

```powershell
New-Item -ItemType Directory -Force -Path .agent-runs/performance-v2 | Out-Null
.\.venv\Scripts\python.exe -c "import json,pathlib; a=json.loads(pathlib.Path('benchmarks/results/performance-v2/feature-progress.json').read_text(encoding='utf-8')); w=pathlib.Path('.agent-runs/performance-v2'); (w/'feature-progress.py').write_text(a['reproduction_script'], encoding='utf-8'); (w/'summarize-feature-progress.py').write_text(a['summary_script'], encoding='utf-8')"
.\.venv\Scripts\python.exe .agent-runs/performance-v2/feature-progress.py
.\.venv\Scripts\python.exe .agent-runs/performance-v2/summarize-feature-progress.py
```

## Does hyperparameter evolution improve after initialization?

A matching 24-fit trajectory check measures the default `GASearchCV` algorithm on eight fresh seeds 1001–1008 for each dataset. The scaled RBF SVC searches log-uniform `C` in `[.01, 100]` and `gamma` in `[.0001, 1]`; the outer 25% holdout, three CV folds, one CPU, population eight, four generations and 72 proposed candidates match the feature-progress budget. The initial comparator is the highest-CV candidate among the eight generation-zero proposals, separately refit on all outer-training rows. The final comparator is the search's refitted winner. Holdout labels are used only after selection.

| Dataset | Initial → final mean CV accuracy | Initial → final mean held-out accuracy | Held-out wins/ties/losses |
|---|---:|---:|---:|
| Breast Cancer | .96948 → .97741 | .97290 → .97465 | 3/3/2 |
| Digits | .97764 → .98033 | .98167 → .98361 | 5/2/1 |
| Synthetic | .86583 → .87375 | .85813 → .85687 | 2/2/4 |

Best-so-far CV accuracy improves in 23 of 24 runs. On synthetic data, its mean progression is `.86583 → .86896 → .87104 → .87333 → .87375` over the five recorded generations, yet the mean held-out accuracy falls .00125, with one split losing .030. Mean held-out gains on Breast Cancer and Digits are .00175 and .00194. Thus the default HPO evolutionary process improves the selection objective on these tasks, but may overfit CV and does not show a general external-quality gain. The separate initial-model refit is excluded from the reported search fitting time. The three datasets and overlapping outer splits are a limited check, not statistical proof of superiority over random search or AutoML.

The [24-fit HPO artifact](results/performance-v2/hpo-progress.json) contains the initial and final candidate scores and parameters, per-generation best scores, holdout accuracies, fit counts and times, split and source hashes, dependency versions, and exact reproduction scripts. To rerun from the repository root in PowerShell after installing benchmark dependencies in `.venv`:

```powershell
New-Item -ItemType Directory -Force -Path .agent-runs/performance-v2 | Out-Null
.\.venv\Scripts\python.exe -c "import json,pathlib; a=json.loads(pathlib.Path('benchmarks/results/performance-v2/hpo-progress.json').read_text(encoding='utf-8')); w=pathlib.Path('.agent-runs/performance-v2'); (w/'hpo-progress.py').write_text(a['reproduction_script'], encoding='utf-8'); (w/'summarize-hpo-progress.py').write_text(a['summary_script'], encoding='utf-8')"
.\.venv\Scripts\python.exe .agent-runs/performance-v2/hpo-progress.py
.\.venv\Scripts\python.exe .agent-runs/performance-v2/summarize-hpo-progress.py
```

## Default HPO evolution versus equal-budget random search

The [HPO trajectory cohort above](#does-hyperparameter-evolution-improve-after-initialization) also permits an external optimizer control. On the identical 24 dataset/split/fold combinations, `RandomizedSearchCV` sampled 72 configurations from the same log-uniform `C` and `gamma` intervals and refit its CV winner. It used the same scaled RBF SVC, one CPU thread and accuracy metric. The GA proposed 72 candidates per run and made 69–72 actual CV calls after duplicate caching; random search made 72. The train/test split and CV folds match by seed, but the two cohorts were timed sequentially with every GA run first, so elapsed-time ratios are descriptive rather than clean paired timing evidence.

| Dataset | GA / random mean CV accuracy | GA / random mean held-out accuracy | GA wins/ties/losses on holdout | GA / random mean seconds |
|---|---:|---:|---:|---:|
| Breast Cancer | .97741 / .97770 | .97465 / .97640 | 1/5/2 | 2.385 / 1.519 |
| Digits | .98033 / .98033 | .98361 / .98417 | 3/2/3 | 28.083 / 16.725 |
| Synthetic | .87375 / .87146 | .85687 / .86438 | 1/3/4 | 4.681 / 2.800 |

On these fixed SVC tasks, default GA evolution does **not** beat independent log-uniform random sampling in mean held-out accuracy. Its mean score is lower by .00175, .00056 and .00750, respectively, despite a higher mean CV score on synthetic data. Random search also completed the timed cohorts faster, but different parameter values can have different SVC fitting cost and the fixed run order confounds exact speed ratios. The negative result is consistent with CV overfitting in the synthetic trajectory; it does not establish that random search is universally better, or explain performance of feature selection, where a separate equal-budget mask control favored the GA.

The [24-pair random-control artifact](results/performance-v2/hpo-random-control.json) contains all raw random-search scores, times and winning parameters, paired summaries, split/source hashes and reproduction scripts. The GA raw records are in the [trajectory artifact](results/performance-v2/hpo-progress.json). To rerun the random cohort and summaries from the repository root in PowerShell:

```powershell
New-Item -ItemType Directory -Force -Path .agent-runs/performance-v2 | Out-Null
.\.venv\Scripts\python.exe -c "import json,pathlib; a=json.loads(pathlib.Path('benchmarks/results/performance-v2/hpo-random-control.json').read_text(encoding='utf-8')); w=pathlib.Path('.agent-runs/performance-v2'); (w/'hpo-random-control.py').write_text(a['reproduction_script'], encoding='utf-8'); (w/'summarize-hpo-random-control.py').write_text(a['summary_script'], encoding='utf-8')"
.\.venv\Scripts\python.exe .agent-runs/performance-v2/hpo-random-control.py
.\.venv\Scripts\python.exe .agent-runs/performance-v2/summarize-hpo-random-control.py
```

## Does a forced one-coordinate crossover transfer to other models?

The random-search control exposed a weak HPO quality/time tradeoff on the RBF SVC tasks. A narrowly scoped crossover pilot replaced the ordinary uniform crossover with a forced swap of exactly one parameter between parents, leaving candidate budgets, initialization, selection, mutation, CV, caching and final refit unchanged. This is an experiment implemented through a `GASearchCV` subclass in the recorded script; **the library default has not changed**. Each paired fit used population eight, four generations, at most 72 proposed candidates, three CV folds, a stratified 25% classification holdout, one CPU thread and alternating run order by seed.

On fresh SVC seeds 1009–1011, the pilot gained .00233 mean held-out accuracy on Breast Cancer, tied on Digits and gained .01333 on synthetic data, but one synthetic split lost .015. The frozen variant was then evaluated on five further SVC seeds 1012–1016:

| SVC dataset | Baseline → forced mean held-out accuracy | Forced wins/ties/losses | Baseline → forced mean seconds |
|---|---:|---:|---:|
| Breast Cancer | .96224 → .96643 | 2/3/0 | 2.484 → 2.407 |
| Digits | .97644 → .97733 | 2/3/0 | 28.446 → 27.498 |
| Synthetic | .86600 → .87500 | 3/2/0 | 5.046 → 5.178 |

To check transfer beyond the motivating SVC tasks, the same frozen variant was compared on new seeds 1021–1025 across decision-tree classification on Wine (integer/categorical space), KNN classification on Digits (integer/categorical), ElasticNet regression on Diabetes (two continuous parameters, negative MSE) and random-forest classification on separate synthetic data (two integers). Each task used its own untouched 25% outer holdout, three folds, one CPU and the same 72-proposal maximum. Repeated candidates reduced actual CV calls in the finite classifier spaces; the counts do not imply complete space coverage.

| Transfer task | Baseline → forced mean held-out score | Forced wins/ties/losses | Baseline → forced mean seconds |
|---|---:|---:|---:|
| Wine decision tree (accuracy) | .89333 → .89333 | 0/5/0 | .271 → .259 |
| Digits KNN (accuracy) | .97067 → .97111 | 1/4/0 | 1.769 → 1.818 |
| Diabetes ElasticNet (negative MSE) | −2798.67 → −2799.08 | 2/1/2 | .793 → .781 |
| Synthetic random forest (accuracy) | .88400 → .88533 | 1/4/0 | 11.903 → 11.393 |

The validation is directionally positive for SVC classification, but most transfer splits tie, ElasticNet loses up to 8.44 negative-MSE points on one split, and time differences are small and mixed. This does not justify a global crossover change or a general SOTA/4× claim. The [88-fit artifact](results/performance-v2/hpo-crossover-transfer.json) retains raw pairs, pilot and validation summaries, source and split hashes, versions, and exact scripts. To reproduce from the repository root in PowerShell after installing benchmark dependencies in `.venv`:

```powershell
New-Item -ItemType Directory -Force -Path .agent-runs/performance-v2 | Out-Null
.\.venv\Scripts\python.exe -c "import json,pathlib; a=json.loads(pathlib.Path('benchmarks/results/performance-v2/hpo-crossover-transfer.json').read_text(encoding='utf-8')); w=pathlib.Path('.agent-runs/performance-v2'); (w/'hpo-forced-crossover-pilot.py').write_text(a['svc_reproduction_script'], encoding='utf-8'); (w/'hpo-crossover-transfer.py').write_text(a['transfer_reproduction_script'], encoding='utf-8'); (w/'summarize-hpo-crossover-transfer.py').write_text(a['summary_script'], encoding='utf-8')"
.\.venv\Scripts\python.exe .agent-runs/performance-v2/hpo-forced-crossover-pilot.py
.\.venv\Scripts\python.exe .agent-runs/performance-v2/hpo-forced-crossover-pilot.py --seed-start 1012 --seeds 5 --output .agent-runs/performance-v2/hpo-forced-crossover-validation.jsonl
.\.venv\Scripts\python.exe .agent-runs/performance-v2/hpo-crossover-transfer.py
.\.venv\Scripts\python.exe .agent-runs/performance-v2/summarize-hpo-crossover-transfer.py
```

## Feature-mask evolution across model families

The earlier [feature-mask random control](#evolutionary-feature-masks-versus-random-search) used one RBF SVC family. A new 40-fit control tests transfer to a decision tree on Wine, scaled KNN on Digits, scaled Ridge regression on Diabetes, and a random forest on a separate synthetic classification problem. Five new outer-split seeds 1041–1045 use a stratified 25% holdout for classification and an ordinary 25% holdout for regression. Both methods propose 72 masks, select one to half the columns, use the same three CV folds and one CPU, cache duplicate mask scores, refit the CV winner, and alternate execution order. The random control samples mask cardinality uniformly from one to the cap, then samples columns uniformly. Accuracy is reported for classifiers and negative MSE for Ridge (larger is better). The holdout is never used to choose a mask.

| Model/dataset | GA / random mean CV score | GA / random mean held-out score | GA wins/ties/losses | GA / random mean seconds |
|---|---:|---:|---:|---:|
| Wine decision tree | .94778 / .93993 | .89333 / .86667 | 2/2/1 | .560 / .532 |
| Digits KNN | .96110 / .95160 | .96178 / .95422 | 3/0/2 | 1.651 / 1.794 |
| Diabetes Ridge | −3126.14 / −3161.71 | −3117.52 / −3058.21 | 2/0/3 | .676 / .622 |
| Synthetic random forest | .89422 / .88267 | .89333 / .87733 | 3/1/1 | 14.905 / 13.721 |

The GA improves mean held-out accuracy by .02667, .00756 and .01600 for the three classifiers, but loses 59.31 negative-MSE points on Ridge despite a better mean CV score; one Ridge split loses 257.75 points. The GA is faster only for KNN in these timings. Classification results extend the evidence that evolutionary mask search is useful beyond SVC, while the regression loss shows that neither solution quality nor time generalizes uniformly. Five overlapping splits per task are descriptive, not statistical proof. The [40-fit artifact](results/performance-v2/feature-random-transfer.json) records raw masks, scores, times, proposal and CV-call counts, source and split hashes, dependencies, and the exact scripts.

To rerun from the repository root in PowerShell after installing benchmark dependencies in `.venv`:

```powershell
New-Item -ItemType Directory -Force -Path .agent-runs/performance-v2 | Out-Null
.\.venv\Scripts\python.exe -c "import json,pathlib; a=json.loads(pathlib.Path('benchmarks/results/performance-v2/feature-random-transfer.json').read_text(encoding='utf-8')); w=pathlib.Path('.agent-runs/performance-v2'); (w/'feature-random-transfer.py').write_text(a['reproduction_script'], encoding='utf-8'); (w/'summarize-feature-random-transfer.py').write_text(a['summary_script'], encoding='utf-8')"
.\.venv\Scripts\python.exe .agent-runs/performance-v2/feature-random-transfer.py
.\.venv\Scripts\python.exe .agent-runs/performance-v2/summarize-feature-random-transfer.py
```
