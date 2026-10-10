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
