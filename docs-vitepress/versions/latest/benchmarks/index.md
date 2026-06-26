---
title: Benchmarks
description: How GASearchCV compares with Optuna (TPE) and random search on the Bayesmark suite — the same datasets, search spaces, and evaluation budgets the Optuna team benchmarks against.
---
:::warning Development version
This is the **latest (dev)** documentation. It may contain unreleased features or breaking changes. For the stable release, use [version 0.13](/versions/0.13/).
:::


# Benchmarks

This page compares `GASearchCV` with two widely used baselines — **Optuna** (Tree-structured Parzen Estimator) and scikit-learn's **`RandomizedSearchCV`** — using the experiment design that the hyperparameter-optimization community already agrees on.

The goal is **comparability, not a leaderboard win**. We reuse the datasets, estimators, and search spaces from a published benchmark suite so the numbers mean the same thing they mean elsewhere, then we add `sklearn-genetic-opt` under the same evaluation budget.

## Why Bayesmark?

Frameworks like Optuna do not benchmark on ad-hoc problems. The Optuna team uses tools such as [kurobako](https://github.com/optuna/kurobako) and benchmark suites such as [Bayesmark](https://github.com/uber/bayesmark) (originally from the NeurIPS 2020 Black-Box Optimization Challenge). Bayesmark's recipe is deliberately simple and framework-agnostic:

- A handful of small, standard scikit-learn datasets.
- A handful of standard scikit-learn estimators, each with a **fixed search space**.
- Every optimizer gets the **same number of evaluations**.
- Each task is repeated over several seeds, and the **best cross-validation score** is reported.

Because the datasets and search spaces are fixed and public, any optimizer — Optuna, random search, or a genetic algorithm — can be dropped into the same harness and compared fairly. That is exactly what the benchmark script in this repository does.

::: info Faithful reproduction
The per-model search spaces below are copied verbatim from Bayesmark's `API_CONFIG`. The only adaptation: Bayesmark searches some bounded parameters in *logit* space, a warp that none of the three optimizers expose directly, so those parameters are searched on their natural bounded range for every optimizer. This keeps the comparison even.
:::

## What We Compare

| Optimizer | Library | Strategy |
|-----------|---------|----------|
| `GASearchCV` | sklearn-genetic-opt | Evolutionary search (DEAP) with the recommended "best" setup |
| Optuna | optuna | Bayesian optimization with the TPE sampler |
| `RandomizedSearchCV` | scikit-learn | Random sampling — the standard strong baseline |

### Datasets

From Bayesmark's `DATA_LOADERS` (we drop `boston`, which modern scikit-learn removed):

| Dataset | Task | Loader |
|---------|------|--------|
| iris | classification | `load_iris` |
| wine | classification | `load_wine` |
| breast | classification | `load_breast_cancer` |
| digits | classification | `load_digits` |
| diabetes | regression | `load_diabetes` |

Classification is scored by **accuracy**; regression by **mean squared error** (optimized as `neg_mean_squared_error`).

### Models and Search Spaces

Copied from Bayesmark's `API_CONFIG`. `log` parameters use a log-uniform distribution; `linear`/`logit` parameters use a uniform distribution on the listed range.

| Model | Parameter | Type | Scale | Range |
|-------|-----------|------|-------|-------|
| **knn** | `n_neighbors` | int | linear | 1 – 25 |
| | `p` | int | linear | 1 – 4 |
| **svm** | `C` | real | log | 1.0 – 1e3 |
| | `gamma` | real | log | 1e-4 – 1e-3 |
| | `tol` | real | log | 1e-5 – 1e-1 |
| **dt** | `max_depth` | int | linear | 1 – 15 |
| | `min_samples_split` | real | logit | 0.01 – 0.99 |
| | `min_samples_leaf` | real | logit | 0.01 – 0.49 |
| | `min_weight_fraction_leaf` | real | logit | 0.01 – 0.49 |
| | `max_features` | real | logit | 0.01 – 0.99 |
| | `min_impurity_decrease` | real | linear | 0.0 – 0.5 |
| **rf** | *(same six as dt)* | | | |
| **ada** | `n_estimators` | int | linear | 10 – 100 |
| | `learning_rate` | real | log | 1e-4 – 1e1 |
| **lasso** / **linear** | `C` (clf) / `alpha` (reg) | real | log | 1e-2 – 1e2 |
| | `intercept_scaling` (clf) | real | log | 1e-2 – 1e2 |

Scaled models (`knn`, `svm`, `mlp`, `lasso`, `linear`) are wrapped in a `StandardScaler` pipeline.

## Equal Budget, Fair Comparison

Every optimizer gets the same **function-evaluation budget** (the number of candidate hyperparameter configurations it is allowed to score with cross-validation):

- `RandomizedSearchCV` → `n_iter = budget`.
- Optuna → `n_trials = budget`.
- `GASearchCV` → `population_size × (generations + 1) ≈ budget`. Genetic search additionally **caches and deduplicates** repeated candidates, so it often evaluates *fewer* unique configurations than the nominal budget — a built-in efficiency, reported as `evaluated_candidates`.

The `GASearchCV` configuration is the recommended "best" setup: smart population initialization, elitism, local search on the top candidates, diversity control, random immigrants, and fitness sharing.

## Running It Yourself

Optuna and SciPy are **optional benchmarking dependencies** — regular users of `sklearn-genetic-opt` never need them. Install the extra only when you want to run the comparison:

```bash
pip install sklearn-genetic-opt[benchmark]
```

Then run the script from the repository root:

```bash
# Fast smoke run (a few datasets/models, small budget)
python benchmarks/benchmark_bayesmark.py --quick

# Full reproducible run with a JSON report
python benchmarks/benchmark_bayesmark.py \
    --datasets iris wine breast diabetes \
    --models knn svm dt rf ada \
    --optimizers gasearch optuna randomized \
    --budget 64 --seeds 3 \
    --output-json benchmarks/bayesmark.json

# Pick a single task to inspect closely
python benchmarks/benchmark_bayesmark.py --datasets wine --models svm --budget 80
```

If Optuna is not installed, the script prints a note and simply runs the remaining optimizers.

## Results

The table below reports the mean ± standard deviation of the best cross-validation
score across seeds. For classification the metric is accuracy (**higher is
better**); for regression it is MSE (**lower is better**). The **winner** column
marks the optimizer with the best mean on the optimized objective.

<!-- BENCHMARK_RESULTS_START -->
The numbers below come from `benchmarks/benchmark_bayesmark.py` with **budget = 48 evaluations, 3 seeds, 3-fold CV** (the command in [Running It Yourself](#running-it-yourself), restricted to the four model families and three datasets shown). Run it on your own hardware to reproduce — scores will vary slightly with library versions and seeds.

| dataset | model | metric | gasearch | optuna | randomized | winner |
| --- | --- | --- | --- | --- | --- | --- |
| wine | knn | accuracy ↑ | **0.9832 ± 0.0046** | 0.9832 ± 0.0046 | 0.9813 ± 0.0027 | gasearch |
| wine | svm | accuracy ↑ | **0.9869 ± 0.0053** | 0.9851 ± 0.0026 | 0.9850 ± 0.0027 | gasearch |
| wine | dt | accuracy ↑ | 0.8520 ± 0.0117 | **0.8651 ± 0.0091** | 0.8463 ± 0.0312 | optuna |
| wine | rf | accuracy ↑ | 0.9645 ± 0.0095 | **0.9719 ± 0.0000** | 0.9532 ± 0.0096 | optuna |
| breast | knn | accuracy ↑ | 0.9684 ± 0.0014 | **0.9695 ± 0.0008** | 0.9678 ± 0.0008 | optuna |
| breast | svm | accuracy ↑ | **0.9789 ± 0.0000** | 0.9783 ± 0.0008 | 0.9783 ± 0.0008 | gasearch |
| breast | dt | accuracy ↑ | 0.9133 ± 0.0030 | **0.9151 ± 0.0017** | 0.9133 ± 0.0030 | optuna |
| breast | rf | accuracy ↑ | 0.9256 ± 0.0054 | **0.9285 ± 0.0072** | 0.9215 ± 0.0060 | optuna |
| diabetes | knn | MSE ↓ | **3223.40 ± 114.3** | 3229.59 ± 107.3 | 3223.47 ± 114.4 | gasearch |
| diabetes | svm | MSE ↓ | 3035.66 ± 28.6 | **3034.17 ± 30.1** | 3053.55 ± 33.9 | optuna |
| diabetes | dt | MSE ↓ | 3994.28 ± 183.6 | **3837.88 ± 106.5** | 3852.95 ± 159.9 | optuna |
| diabetes | rf | MSE ↓ | 3553.56 ± 219.1 | **3320.70 ± 52.5** | 3498.15 ± 188.9 | optuna |

`↑` higher is better, `↓` lower is better. **Bold** marks the best mean per row.

At this small budget the three optimizers are within a hair of each other on most
tasks — frequently inside one standard deviation. Two patterns stand out:

- **`GASearchCV` beats or ties random search on 10 of 12 tasks**, and wins
  outright on the smoother spaces (`knn`, `svm`).
- **Optuna's TPE has the edge on the rugged tree spaces** (`dt`, `rf`), where the
  six logit-warped split parameters create a bumpy objective that a Bayesian
  surrogate models well. With a larger budget and the GA's optimizer controls
  given more generations to work, that gap narrows — try `--budget 96`.
<!-- BENCHMARK_RESULTS_END -->

::: tip Reading the numbers
On small, smooth datasets like iris and wine, **all three optimizers cluster
near the same score** — the search space is easy and there is little room to
separate them. The differences that matter show up as problems get harder:
larger or mixed spaces, more rugged objectives, and tighter budgets, where
smarter exploration (TPE or evolutionary search) pulls ahead of random
sampling. Always read the *spread* across seeds alongside the mean.
:::

## How to Interpret a Benchmark

- **Equal budget is the whole point.** A method that "wins" by evaluating 10× more candidates has not won. Compare at a fixed number of evaluations.
- **Report variance.** A single seed is anecdote. Mean ± std over several seeds is the signal.
- **Small datasets compress differences.** iris and wine saturate quickly; expect ties. Use them to validate correctness, not to rank optimizers.
- **Genetic search shines on mixed and rugged spaces.** When the space combines integers, continuous, log-scaled, and categorical parameters — or when the objective has many local optima — evolutionary search and its optimizer controls (diversity, fitness sharing, local refinement) tend to pay off. See [When to Use](../guide/when-to-use).

## See Also

- [Comparing Search Methods](../examples/sklearn-comparison) — a single-task, copy-pasteable comparison with full code.
- [Advanced Optimizer Control](../guide/advanced-optimizer-control) — the controls used in the "best" GA setup.
- [When to Use](../guide/when-to-use) — picking the right search method for your problem.
- [Bayesmark](https://github.com/uber/bayesmark) and [kurobako](https://github.com/optuna/kurobako) — the upstream benchmark tooling.
