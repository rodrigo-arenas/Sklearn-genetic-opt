---
title: Comparing optimization backends
description: Compare five optimizer backends at matched cross-validation budgets for feature selection and hyperparameter search.
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
print(search.best_score_, search.support_.sum(), search.fit_stats_["cross_validate_calls"])
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
```

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
and cache hits are recorded separately. The first highest-CV candidate wins ties,
with smaller masks preferred for equal CV. We refit that candidate once on the
training data and score it on the held-out partition.

`prefix_ms` ends when the last matching CV result becomes available. Its
non-CV component includes proposal generation and bookkeeping to that point.
`full_ms` and `full_outside_cv_ms` also include the final population update and
estimator result assembly for the full run. Neither clock includes the separate
holdout refit. `mean_running_best` averages the best score seen after each real
evaluation. This captures how quickly the engine found good candidates.

The CSV also gives a holdout baseline and relative skill. For classification,
skill is the fraction of the gap above chance balanced accuracy closed by the
model. For regression, skill is one minus the ratio of its RMSE to the RMSE of
predicting the training mean. The effective-rank generator changes the target scale,
so compare raw RMSE within one case and use relative skill across difficulties.
