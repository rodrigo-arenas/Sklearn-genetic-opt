---
title: "Hyperparameter Tuning Backend Benchmarks"
description: Compare DEAP, SciPy, Nevergrad, CatCMAwM and pymoo for scikit-learn hyperparameter tuning and feature selection at the same cross-validation cost.
---

<script setup>
import pilot32 from '../../../../benchmarks/results/optimizer-catalog/pilot32.csv?url'
import pilot96 from '../../../../benchmarks/results/optimizer-catalog/pilot96.csv?url'
import budget32 from '../../../../benchmarks/results/optimizer-catalog/budget32.csv?url'
import budget96 from '../../../../benchmarks/results/optimizer-catalog/budget96.csv?url'
</script>

# Comparing optimization backends

For a first fit, follow the [backend tutorial](../tutorials/optimizer-backends). The [API reference](../api/optimizer-backends) lists the controls and checkpoint support for each engine.

Choose an engine with the `optimizer_backend` keyword in either estimator. Use the
same estimator, folds, budget and random state to compare engines on your own data.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn_genetic import GAFeatureSelectionCV

X, y = load_breast_cancer(return_X_y=True)
search = GAFeatureSelectionCV(
    DecisionTreeClassifier(random_state=41), cv=3, scoring="balanced_accuracy",
    optimizer_backend="pymoo", max_features=10, population_size=8,
    generations=3, random_state=41, n_jobs=1, verbose=False,
)
search.fit(X, y)
print(search.get_support(indices=True), search.fit_stats_["cross_validate_calls"])
```

## Results at equal CV cost

The following scores are means over five held-out splits. The table uses the
cache-disabled runs with a nominal target of 96 real CV calls. In 38 of the 45
case/seed groups all engines reached 96. In seven tree-tuning groups SciPy converged
earlier and every engine is compared at the same 64, 80 or 88 calls. Each CV call
fits three folds. Higher balanced accuracy and lower positive RMSE are better.

| Task | Metric | DEAP | SciPy | Nevergrad | CatCMAwM | pymoo |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Breast cancer, tree parameters | Balanced accuracy | 0.905 | 0.917 | 0.909 | 0.914 | 0.914 |
| Breast cancer, tree mask | Balanced accuracy | 0.917 | 0.899 | 0.912 | 0.919 | 0.919 |
| Synthetic classification, easy mask | Balanced accuracy | 0.885 | 0.899 | 0.896 | 0.891 | 0.901 |
| Synthetic classification, hard mask | Balanced accuracy | 0.683 | 0.699 | 0.701 | 0.704 | 0.690 |
| Diabetes, tree parameters | RMSE | 64.37 | 64.47 | 64.49 | 64.39 | 64.50 |
| Diabetes, SVR parameters | RMSE | 57.03 | 56.66 | 56.95 | 56.77 | 56.50 |
| Friedman #1, tree parameters | RMSE | 3.77 | 3.89 | 3.82 | 3.88 | 3.89 |
| Synthetic linear regression, easy mask | RMSE | 69.91 | 75.37 | 61.46 | 44.56 | 23.05 |
| Synthetic linear regression, hard mask | RMSE | 9.04 | 9.17 | 9.13 | 9.01 | 8.76 |

On the synthetic linear masks, pymoo reduced holdout RMSE relative to
predicting the training mean by 87.0% on easy data and 9.1% on hard data.
The difficult case offers considerably less recoverable signal. Its smaller
raw RMSE reflects a different target scale.

Across the 25 engine/seed combinations per scenario, all 25 improved their CV
best after the first eight calls on both linear mask problems. Only 14 of 25
improved on Diabetes tree tuning at the larger target. Equal budgets still distinguish
the engines on scenarios where choice of features or continuous scale matters.

### Runtime and reuse

Median times and duplicate rates below use the 38 complete case/seed groups. Prefix timings use
96 real calls and exclude holdout refit. The last column comes from separate
cache-enabled runs across all 45 case/seed groups at the same nominal generation budget.

| Engine | Time to 96th CV result (ms) | Prefix time outside CV (ms) | Repeated proposals in 96 calls¹ | Median CV calls, full run with cache |
| --- | ---: | ---: | ---: | ---: |
| DEAP | 933 | 27 | 1.0% | 181 |
| SciPy | 887 | 31 | 2.4% | 94 |
| Nevergrad | 1050 | 146 | 20.7% | 88 |
| CatCMAwM | 912 | 48 | 0.7% | 96 |
| pymoo | 990 | 40 | 16.2% | 80 |

¹ Mean fraction of decoded candidates already evaluated in the matched uncached prefix.
The estimator cache and within-batch deduplication can avoid those CV fits in production.
With cache enabled, paired prefixes shrink to the smallest realized count in each group.
Use the CSV columns for full-run time and CV calls when comparing production cost.

### Opportunities to measure next

- **pymoo** had the lowest mean holdout RMSE on both linear mask scenarios and on SVR. Test bounded retries for duplicate offspring on large spaces while retaining a finite-space stopping rule.
- **CatCMAwM** performed well on the noisy classification mask and generated few repeated proposals. Measure wall time and memory as the number of numeric coordinates grows, and compare diagonal covariance for high dimensions.
- **Nevergrad** used the most time outside CV and revisited candidates frequently. Profile ask/tell and mask decoding separately, then test budget-aware search settings on independent data.
- **SciPy** combined low overhead with the highest mean balanced accuracy for breast-cancer parameter tuning. Compare its continuous mask encoding with a cardinality-aware proposal on sparse feature selection.
- **DEAP** reached competitive holdout scores on the Friedman and Diabetes trees. Its default offspring count gives a larger natural CV budget than the other engines at equal generation counts. An explicit budget in real CV calls would help users control cost across backends.

Five seeds per scenario support a workload-specific comparison. Stronger evidence would
include more independent tasks, larger and nested CV budgets, a random-search reference
and replicated runtime measurements on additional hardware.

## Configure a search

For pymoo, set the estimator's `crossover_probability`. For CatCMAwM and
Nevergrad, configure the installed optimizer callable through the standard
entry point. This example changes the initial numeric step size and keeps the
estimator's usual CV, callbacks and results.

```python
from functools import partial
from importlib.metadata import entry_points

from sklearn.base import clone
from sklearn_genetic import GASearchCV
from sklearn_genetic.space import Continuous, Integer

entries = entry_points(group="sklearn_genetic.optimizers")
cma_search = GASearchCV(
    DecisionTreeClassifier(random_state=41), cv=3,
    param_grid={"ccp_alpha": Continuous(1e-6, 0.1, distribution="log-uniform"),
                "max_depth": Integer(2, 12)},
    optimizer_backend=partial(entries["cmaes"].load(), sigma=0.15),
    population_size=8, generations=11, random_state=41, verbose=False,
)
ng_search = clone(cma_search).set_params(
    optimizer_backend=partial(entries["nevergrad"].load(), strategy="TwoPointsDE"),
)
```

CatCMAwM accepts `0 < sigma <= 1` in normalized numeric coordinates. It only
affects numeric variation. Nevergrad accepts `OnePlusOne` and `TwoPointsDE`.
Live plotting and stopping callbacks work on reported batches. The same configured adapters can
resume from a completed generation with [ModelCheckpoint](../recipes/advanced/checkpointing).

The installed adapters use general-purpose engines. Nevergrad and cmaes are
MIT-licensed. pymoo uses Apache-2.0. See the original
[Nevergrad optimizer documentation](https://facebookresearch.github.io/nevergrad/optimization.html),
[CatCMA with Margin implementation](https://github.com/CyberAgentAILab/cmaes/blob/v0.13.1/cmaes/_catcmawm.py)
and [pymoo mixed-variable guide](https://pymoo.org/customization/mixed.html).

Download the paired measurements for
<a :href="pilot32" download="pilot32.csv">calibration at 32</a>,
<a :href="pilot96" download="pilot96.csv">calibration at 96</a>,
<a :href="budget32" download="budget32.csv">comparison at 32</a> and
<a :href="budget96" download="budget96.csv">comparison at 96</a> calls.

## Reproduce the comparison

From the repository root, install the optional engines and run the following commands.
Each program warms imports and runs the engines sequentially with one native thread.

```bash
python -m pip install -e '.[scipy,nevergrad,cmaes,pymoo]'
python -m benchmarks.benchmark_optimizer_catalog --pilot --output pilot32.csv
python -m benchmarks.benchmark_optimizer_catalog --pilot --budget 96 --output pilot96.csv
python -m benchmarks.benchmark_optimizer_catalog --budget 32 --output budget32.csv
python -m benchmarks.benchmark_optimizer_catalog --budget 96 --output budget96.csv
python -m benchmarks.benchmark_optimizer_catalog --budget 32 --public-refit --output public32.csv
```

### Reproduce selected cases and splits

Use `--cases`, `--engines`, and `--seeds` to rerun specific comparisons or
check a change on additional splits. This example compares tuning on Wine and
feature selection on Breast Cancer using DEAP and the random search reference.
It works with the base package installed from the checkout.

```bash
python -m benchmarks.benchmark_optimizer_catalog --cases wine cancer --engines deap --seeds 41 107 --references random --stop-at-budget --public-refit --output subset.csv --pairs-output subset-pairs.csv
```

Use `--help` to list case and engine names. Install the optional engines or
Optuna only when your command uses them. With `--pilot`, selecting an engine
runs all its configured calibration variants.

Each seed controls the search, the inner folds, and the holdout split. Keep
the same case, engine, and seed lists in the same order when comparing two
versions. The CSV's cohort-wide shared CV count can change if you add engines.
Use `subset-pairs.csv` for engine/reference comparisons at the same number of
actual CV calls and `completed_target` to see which pairs reached the budget.

### Compare with random search and Optuna TPE

Run these references through the same estimators, search spaces, three inner
folds, and outer test split as the five engines:

```bash
python -m pip install -e '.[benchmark]'
python -m benchmarks.benchmark_optimizer_catalog --budget 32 --references random tpe --time-ms 250 1000 --public-refit --output references32.csv --pairs-output pairs32.csv
python -m benchmarks.benchmark_optimizer_catalog --budget 96 --references random tpe --time-ms 250 1000 --public-refit --output references96.csv --pairs-output pairs96.csv
```

The references start with the same eight initial proposals as the engines.
Random search draws tuning values from the defined distributions. For feature
selection it draws a mask size uniformly from one through the feature limit,
then chooses that many columns uniformly without replacement. Optuna TPE runs
on tuning tasks and can update its model after each CV result.

With `--references`, the optional `--pairs-output` CSV has one row per engine,
reference, task, seed and cache setting. `matched_cv_calls` gives the largest shared count of actual
CV calls up to the target. This count depends only on the two runs in the row.
Filter `completed_target` to compare pairs at the full target. For each side,
`*_best_cv` gives the best CV score at the matched count, `*_prefix_holdout`
scores that candidate on the test split after refitting on the training split,
and `*_prefix_ms` measures the search time to return the matched CV result.

### Compare pairs of optimization backends

Use `--pairs-output` with at least two default engines to compare them
directly. Omit `--references` to write one row for each unordered engine
pair, task, seed, and cache setting. In this mode `reference` names the
second engine in the pair.

```bash
python -m benchmarks.benchmark_optimizer_catalog --cases diabetes wine-mask --engines deap scipy pymoo --seeds 41 --budget 32 --cv-checkpoints 8 32 --output engines32.csv --pairs-output engine-pairs32.csv
```

The main CSV still reports the CV prefix shared by the whole group.
The pairs CSV uses each pair's own shared prefix, so an engine that stops
early affects its own pairs. Filter `completed_target` for comparisons
at the requested budget and check the JSON completion counts for each
pair before aggregating scores. Keep the held-out scores in
`*_prefix_holdout` separate from the search CV scores in `*_best_cv`.

### Compare several CV budgets in one run

Use `--cv-checkpoints` to score prefixes of the same search at several fresh
CV call counts. Choose positive counts up to `--budget`. This uses the same
folds, seed, and proposal stream at every checkpoint.

```bash
python -m benchmarks.benchmark_optimizer_catalog --budget 96 --stop-at-budget --cv-checkpoints 8 16 32 64 96 --references random tpe --output checkpoints.csv --pairs-output checkpoint-pairs.csv
```

For a checkpoint at 32 calls, the main CSV reports `cv_32_best_cv`,
`cv_32_holdout`, and `cv_32_elapsed_ms`. The holdout score comes from refitting
that prefix's CV winner on the training partition. The elapsed time stops
when the 32nd CV result returns. A run that ends before that call has blank
values for the checkpoint. If the run reaches 32 calls and none produced
a finite score, the elapsed time is reported and the quality fields are blank.

In the pairs CSV, filter `cv_32_completed` before comparing the `backend_cv_32_*`
and `reference_cv_32_*` columns. Both sides need 32 calls for any of those score
and time fields to be populated. Use `completed_by_cv_calls` and `eligible`
from the JSON metadata to report the sample size for each requested checkpoint
by backend, reference, and cache setting. When reading a trend across budgets,
check how the number of completed pairs changes.

### Stop the benchmark at a fixed CV budget

Use `--stop-at-budget` to stop each fit with
[CVBudgetStopping](../api/callbacks#cvbudgetstopping). Give the engines extra
proposal capacity with `--proposal-budget` so they can continue past cache
hits. The nominal proposal budget sets the generation limit using an
eight-member population. It defaults to twice the CV target in this mode.
Actual proposals per generation vary by engine. Use `full_proposals` and
`full_cv_calls` for the measured counts.

```bash
python -m benchmarks.benchmark_optimizer_catalog --budget 32 --proposal-budget 64 --stop-at-budget --references random tpe --public-refit --output capped32.csv --pairs-output capped-pairs32.csv
```

`full_ms` measures time until `fit` returns. With `--public-refit`, this
includes the full-data refit. `returned_cv` and `returned_holdout` describe
the candidate selected by the estimator. `returned_matches_trace_winner`
shows whether it matches the harness's best candidate across the recorded
trace. Equal CV scores can yield different selections and holdout scores.
Holdout scoring happens outside the measured fit.

Filter `completed_target` before comparing paired durations at the requested
budget. Each side has `*_fit_ms` for the full fit, `*_search_ms` for the completed
optimizer run, and `*_refit_ms` for the full-data refit with `--public-refit`.
The refit fields are blank when refit is disabled. To compare with a longer
search on the same settings, retain `--proposal-budget` and omit `--stop-at-budget`.

Read `budget_status` in the primary CSV and `backend_budget_status` or
`reference_budget_status` in the pairs CSV to see the outcome for each fit.

| Status | Meaning |
| --- | --- |
| `target_reached` | The run completed the requested number of fresh CV calls |
| `space_exhausted` | The run finished below the target with confirmed coverage of its finite search space |
| `below_target` | The run finished below the target with the reason unconfirmed |

`space_exhausted` comes from the estimator's
[`search_space_exhausted_`](../api/optimizer-backends#domain-and-budget) flag.
Built-in engines check coverage on eligible finite cached searches. Random and
TPE references report `below_target` when they finish short. Use the measured
`full_cv_calls` and `matched_cv_calls` for comparison.

The primary CSV also reports each run's `target_best_cv`, `target_holdout`
and `target_elapsed_ms`. Blank target fields indicate that a run finished
before the requested number of actual CV calls. Its `cv_calls` column gives
the common count across the whole group. The JSON printed to stdout counts
eligible and completed pairs for each engine, reference and cache setting.

For equal wall-time comparisons, use the `time_<ms>ms_*` columns. Each checkpoint
uses the CV results returned by that deadline, up to the requested CV count.
It can include a completed run's incumbent after the run ends. A run without
a result by the deadline has no score for that checkpoint. The holdout score
comes from refitting the CV winner on the training partition. The search clock
ends before that separate refit. Use `public_holdout` for the score of the
estimator returned by the full run.

**Calibration.** Wine classification, Wine feature selection, synthetic classification CART tuning and synthetic correlated linear-regression mask selection used three seeds
(7, 19, 31). The pilot compared Nevergrad TwoPointsDE with OnePlusOne, CatCMAwM numeric
step sizes 0.15, 0.25 and 0.4, and pymoo crossover probabilities 0.5, 0.8 and
1.0. TwoPointsDE initializes at least 30 internal proposals, so the 32-call
pilot covers its sampling phase and the 96-call pilot also covers differential variation.
Results on the comparison datasets did not determine these defaults.

For each task and seed, rank the tested settings within each engine using the best CV
score. Average tied ranks. This compares classification and regression without averaging
their raw scores. Across both budgets and all four pilot cases:

| Engine | Default selected | Calibration evidence |
| --- | --- | --- |
| Nevergrad | OnePlusOne | On masks the mean best-score rank was 1.29 against 1.71 for TwoPointsDE. The parameter-search ranks tied at 1.50. OnePlusOne also had lower mean full-run overhead on parameter searches. |
| CatCMAwM | Normalized numeric step 0.25 | On parameter searches, 0.25 had the best mean running-best rank (1.83). Step 0.4 had a small lead on final-score rank (1.92 vs 1.96). Sigma does not affect purely categorical masks. |
| pymoo | Crossover 0.8 | Its mean final-score rank was 1.71 on masks and 1.83 on parameters, ahead of 0.5 and 1.0 in both tasks. |

The original Wine-only pilot at 32 evaluations slightly favored TwoPointsDE on
masks. At 96 calls and on the correlated regression masks the ordering changed.
NGOpt was removed during validation because its upstream metamodel can fail on
current NumPy in long continuous and mixed-domain runs.

**Comparison.** Five other split/search seeds (41, 59, 73, 97, 113) were used for
breast-cancer CART tuning and feature selection, Diabetes CART and RBF SVR tuning, synthetic sparse classification with logistic-regression mask selection (easy/hard), synthetic linear regression with Ridge mask selection (easy/hard), and nonlinear Friedman #1 CART tuning. CART uses integer, categorical and log-scaled real parameters. SVR
uses three log-scaled positive reals and standardizes the raw Diabetes features
inside each training fold via a pipeline. The mask cap is one third of the input width.
Paired seeds align the data splits and search budgets. Each synthetic scenario generates a new dataset for each seed using scikit-learn's
`make_classification`, `make_regression` or `make_friedman1`. Classification varies
class separation (1.6 or 0.65) and label noise (1% or 12%). Linear regression compares
independent features and noise 5 with effective rank 8 and noise 8. Friedman adds
19 irrelevant features and Gaussian noise 1.5. The pilot uses intermediate settings
and disjoint seeds. Each engine retains its own proposal
sequence. Each split holds out 25 percent for final scoring and uses three shuffled folds
inside the training partition. Classification uses balanced accuracy. Regression
uses negative RMSE so larger scores are better throughout.

**Budget and time.** The population has eight members. Each paired group uses the
first 32 or 96 calls to scikit-learn's actual `cross_validate`, or the smallest
realized count in the group if an engine finishes sooner. Each call fits three
folds. DEAP uses the default `eaMuPlusLambda`. Variation and survival can give
it a different natural evaluation count from the other engines. We also run with
the cache enabled to measure reuse; distinct proposals, duplicate evaluations
and cache hits are recorded separately. Non-finite scores consume their CV
calls and are ineligible for prefix selection. At the highest finite score,
the search prefers smaller masks and then the earliest observation. The
harness refits that candidate on the training data and scores it on the held-out
partition. `finite_cv_calls` counts finite scores in the reported prefix and
`first_finite_cv_call` gives the call number of the first finite score.

`prefix_ms` ends when the last matching candidate CV call returns. Its
non-CV component includes proposal generation and bookkeeping to that point.
The harness keeps the built-in candidate evaluators active and links each
timed call to its recorded candidate. Cache hits add no trace point.
The metadata for these runs records `trace_timing: candidate_cv_return`.
`full_search_ms` measures the entire optimizer run using
[`search_time_`](../api/optimizer-backends#fit-time-by-phase), including population
setup, candidate evaluation, callbacks, and the last population update. This
field is available in CSVs generated with the current harness.
`full_ms` covers the entire call to `fit`, and `full_outside_cv_ms` subtracts the
measured search CV time from that total. Both clocks stop before the separate
refit used to score the prefix winner on holdout. `mean_running_best` averages
the best finite score after each real call, starting at `first_finite_cv_call`.
A later non-finite result keeps the last finite best in the average.

**Check the fitted estimator.** Add `--public-refit` to also call `fit` with
`refit=True` and score the estimator's predictions on the same holdout.
`public_holdout` is the test score of the refitted estimator, and `public_cv`
records its `best_score_`. For repeated evaluations of the same feature mask,
`public_cv` follows the evaluation selected by the estimator. The matched-prefix
columns use the best score observed by the common budget.
`public_winner_in_prefix` reports whether the selected configuration or
mask was evaluated in that prefix. A repeated evaluation can assign it a
different score. When comparing public scores across engines, match
`full_cv_calls`. With this option, the full-run clocks include the estimator's
refit, also reported separately as `public_refit_ms`.

To inspect a fitted search, [plot the work by phase](../examples/plotting-gallery#fit-work-by-phase)
and compare fresh fold fits with reused finalist scores. If you enabled
finalist validation, [compare the finalists' search and validation scores](../examples/plotting-gallery#finalist-score-comparison)
and check which candidate was refitted.

The CSV also gives a holdout baseline and relative skill. For classification,
skill is the fraction of the gap above chance balanced accuracy closed by the
model. For regression, skill is one minus the ratio of its RMSE to the RMSE of
predicting the training mean. The effective-rank generator changes the target scale,
so compare raw RMSE within one case and use relative skill across difficulties.
