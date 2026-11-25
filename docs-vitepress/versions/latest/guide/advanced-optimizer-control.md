---
title: "Advanced Optimizer Control: Diversity, Fitness Sharing, and Local Search"
description: Control population diversity, refine candidates with local search, and validate finalists for scikit-learn hyperparameter tuning and feature selection.
---

<script setup>
import shortlistTimingsUrl from '../../../../benchmarks/results/performance-v2/shortlist-fit-times.json?url'
</script>

:::warning Development version
You are reading the **latest (dev)** docs. For the stable version, see [stable](/stable/).
:::

# Advanced Optimizer Control: Diversity, Fitness Sharing, and Local Search

When default settings produce premature convergence or poor search quality, these controls give you finer-grained command over the evolutionary process. They are **optional** — the defaults are conservative and work well for most searches. Add them one at a time when telemetry shows the optimizer needs them.

Diversity-driven variation, fitness sharing, random immigrants and local refinement in this guide apply to DEAP. Shared telemetry and finalist selection also work with the other [optimizer backends](../api/optimizer-backends).

## Prerequisites

- Completed [Basic Usage](./basic-usage)
- Familiar with [Understanding Cross-Validation](./understand-cv)

## When to Use These Controls

Check `pd.DataFrame(search.history)` after a run. Add controls if you see:

| Symptom | Likely cause | Remedy |
|---------|-------------|--------|
| `unique_individual_ratio` drops to near 0 in a few generations | Population collapsed | Enable `diversity_control` |
| `genotype_diversity` is low while score still improves | Slow convergence, not stuck | Let it run or increase `population_size` |
| `stagnation_generations` grows for 5+ generations | Local optimum | Enable `diversity_control`, fitness sharing, or local search |
| Multiple high-scoring but similar candidates | Single dominant region | Enable `fitness_sharing` to keep multiple niches alive |
| Final solution is good but nearby configs might be better | Under-exploited region | Enable `local_search` |

## Configuration Objects

Advanced settings live in `OptimizationConfig`. All parameters are disabled (or set to low-impact defaults) unless explicitly changed.

```python
from sklearn_genetic import OptimizationConfig
```

## Full Example: Hyperparameter Search

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from sklearn_genetic import (
    EvolutionConfig,
    GASearchCV,
    OptimizationConfig,
    PopulationConfig,
    RuntimeConfig,
)
from sklearn_genetic.plots import plot_diversity, plot_optimizer_events
from sklearn_genetic.schedules import ExponentialAdapter, InverseAdapter
from sklearn_genetic.space import Categorical, Integer

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

param_grid = {
    "n_estimators": Integer(50, 250),
    "max_depth": Integer(2, 20),
    "min_samples_split": Integer(2, 20),
    "min_samples_leaf": Integer(1, 10),
    "max_features": Categorical(["sqrt", "log2", None]),
    "criterion": Categorical(["gini", "entropy", "log_loss"]),
}

crossover_schedule = InverseAdapter(
    initial_value=0.8,
    end_value=0.6,
    adaptive_rate=0.05,
)
mutation_schedule = ExponentialAdapter(
    initial_value=0.1,
    end_value=0.25,
    adaptive_rate=0.08,
)

search = GASearchCV(
    estimator=RandomForestClassifier(random_state=42, n_jobs=1),
    random_state=42,
    param_grid=param_grid,
    cv=cv,
    scoring="roc_auc",
    evolution_config=EvolutionConfig(
        population_size=24,
        generations=18,
        crossover_probability=crossover_schedule,
        mutation_probability=mutation_schedule,
        tournament_size=3,
        elitism=True,
        keep_top_k=4,
    ),
    population_config=PopulationConfig(initializer="smart"),
    runtime_config=RuntimeConfig(n_jobs=-1, parallel_backend="auto", verbose=True),
    optimization_config=OptimizationConfig(
        # Diversity control
        diversity_control=True,
        diversity_threshold=0.18,
        diversity_stagnation_generations=4,
        diversity_mutation_boost=1.8,
        random_immigrants_fraction=0.15,
        # Fitness sharing
        fitness_sharing=True,
        sharing_radius=0.25,
        sharing_alpha=1.0,
        # Local search
        local_search=True,
        local_search_top_k=2,
        local_search_steps=4,
        local_search_radius=0.08,
    ),
)

search.fit(X_train, y_train)

y_pred = search.predict(X_test)
y_proba = search.predict_proba(X_test)[:, 1]

print(search.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_proba))
```

## Reading Optimizer Telemetry

After fitting, convert `history` to a DataFrame to diagnose the search:

```python
history = pd.DataFrame(search.history)

columns = [
    "gen",
    "fitness_best",
    "fitness_max",
    "unique_individual_ratio",
    "genotype_diversity",
    "stagnation_generations",
    "mutation_probability",
    "diversity_control_triggered",
    "random_immigrants",
    "duplicate_replacements",
    "fitness_sharing_applied",
    "mean_niche_count",
    "max_niche_count",
    "local_refinements",
]

print(history[columns])
print(search.fit_stats_)
```

The same telemetry is easier to reason about visually. `plot_diversity` shows whether the population collapsed or recovered, while `plot_optimizer_events` marks the generations where diversity control, immigrant injection, sharing, or local search changed the search process.

```python
import matplotlib.pyplot as plt

plot_diversity(search)
plt.show()

plot_optimizer_events(search)
plt.show()
```

![Advanced optimizer diversity](/images/advanced_optimizer_diversity.png)

![Advanced optimizer events](/images/advanced_optimizer_events.png)

**Key telemetry fields:**

| Field | Meaning |
|-------|---------|
| `finite_fitness_count` | Population members with a finite primary score. Compare with `population_size`. |
| `unique_individual_ratio` | Fraction of distinct individuals. Low → population collapsed |
| `genotype_diversity` | Average probability that two different population members disagree at a coordinate |
| `stagnation_generations` | Generations since `fitness_best` last improved |
| `diversity_control_triggered` | Whether diversity control fired this generation |
| `random_immigrants` | Number of random candidates injected |
| `duplicate_replacements` | Duplicate offspring replaced before evaluation |
| `fitness_sharing_applied` | Whether niche-aware selection was active |
| `mean_niche_count` / `max_niche_count` | Crowding during selection |
| `local_refinements` | Neighbor candidates evaluated by local search (usually non-zero only in the final row) |

For feature selection, each coordinate is the decision to include an input
column. Diversity includes every member of the current population, including
repeated masks. The selector computes this statistic from masks one at a time
on wide inputs.

Population mean, standard deviation, minimum and maximum use the finite
primary scores. A generation with zero finite scores reports `NaN` for
these summaries. Callbacks that read population statistics use these same
values. Candidate rows and cumulative CV calls include failed evaluations.
Earlier checkpoint rows may lack `finite_fitness_count`.

`fitness_improvement` records the signed gap from the previous best score in
the search direction. If the gap exceeds the score dtype's range, the gap is
reported as `inf` or `-inf`. `fitness_best_observed` retains the original score.

Fitness sharing applies the crowding penalty during selection. The CV scores
in `cv_results_` and the reported population statistics retain their original
scale, even when sharing rescales extreme scores internally to avoid overflow.
On wide feature sets, DEAP packs binary masks and reuses each pairwise
distance for both candidates. The number of comparisons still grows
quadratically with population size.

## Feature Selection Example

The same controls work with `GAFeatureSelectionCV`. In feature selection, `local_search_radius` controls the fraction of feature bits flipped when creating local neighbors.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split

from sklearn_genetic import (
    EvolutionConfig,
    GAFeatureSelectionCV,
    OptimizationConfig,
    PopulationConfig,
    RuntimeConfig,
)
from sklearn_genetic.schedules import ExponentialAdapter

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

selector = GAFeatureSelectionCV(
    estimator=RandomForestClassifier(random_state=42, n_jobs=1),
    random_state=42,
    cv=cv,
    scoring="roc_auc",
    max_features=18,
    evolution_config=EvolutionConfig(
        population_size=30,
        generations=16,
        crossover_probability=0.8,
        mutation_probability=ExponentialAdapter(
            initial_value=0.1, end_value=0.25, adaptive_rate=0.08
        ),
        keep_top_k=4,
    ),
    population_config=PopulationConfig(initializer="smart"),
    runtime_config=RuntimeConfig(n_jobs=-1, parallel_backend="auto", verbose=True),
    optimization_config=OptimizationConfig(
        diversity_control=True,
        diversity_threshold=0.2,
        diversity_stagnation_generations=4,
        random_immigrants_fraction=0.15,
        fitness_sharing=True,
        sharing_radius=0.2,
        local_search=True,
        local_search_top_k=2,
        local_search_steps=5,
        local_search_radius=0.1,
    ),
)

selector.fit(X_train, y_train)

print("Selected features:", selector.best_features_)
print("Test score:", selector.score(X_test, y_test))
```

## Recommended Workflow

Start simple and add controls based on telemetry:

1. **Default run** — use `PopulationConfig(initializer="smart")`, `crossover_probability=0.8`, `mutation_probability=0.1`. Inspect history.
2. **If diversity collapses early** — enable `OptimizationConfig(diversity_control=True)`.
3. **If one candidate family dominates** — enable `fitness_sharing=True`.
4. **If the final region looks close but not fully refined** — enable `local_search=True`.
5. **If stagnation persists** — add adaptive schedules: slowly increase mutation over generations.

## Tuning Guidelines

| Parameter | Practical starting range | Notes |
|-----------|------------------------|-------|
| `diversity_threshold` | 0.1–0.3 | Higher triggers earlier |
| `diversity_mutation_boost` | 1.5–2.5 | Too high = next generation is random |
| `random_immigrants_fraction` | 0.05–0.2 | Larger helps rugged spaces; may slow convergence |
| `sharing_radius` | 0.15–0.35 | Smaller = only penalize very similar candidates |
| `local_search_radius` | 0.05–0.15 | Larger = behaves more like an extra mutation phase |
| `local_search_steps` | 1–5 | Each step = extra CV calls |

## Practical Notes

- Fitness sharing only changes temporary selection pressure; it does not alter `best_score_`, `cv_results_`, or raw cross-validation scores.
- Local search evaluates extra candidates when evolution finishes. A
  [stopping callback](./callbacks#using-callbacks) can end search before
  refinement starts.
- Always keep a final holdout set separate from the CV folds for honest model assessment.

## Next Steps

- [Adaptive Schedules](./adapters) — schedule mutation and crossover rates over generations
- [Troubleshooting](./troubleshooting) — diagnose convergence problems using `fit_stats_` and `history`
- [Config Objects API](../api/config) — full parameter reference for `OptimizationConfig`

## Cheaper search with full-training finalist validation

Both searches can explore using fewer training rows per CV fold, then re-evaluate a small number of distinct finalists on the original folds. `subsample_cv` changes only the training indices; every validation row stays in place. The evolutionary operators continue to select and generate candidates using these cheaper scores.

`subsample_cv` names the fold if its training partition is empty. Check the
original CV splits when this happens. The helper rounds the retained
training count down, so increase `train_size` if sampling would leave any
fold with zero training rows. For stratified sampling, also retain every
class present in that fold's original training rows.

Use a separate outer test set to measure whether the speed/quality tradeoff suits your data:

```python
from sklearn.base import clone
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn_genetic import GASearchCV, GAFeatureSelectionCV, subsample_cv
from sklearn_genetic.space import Continuous

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)
full_cv = list(StratifiedKFold(3, shuffle=True, random_state=42).split(X_train, y_train))
search_cv = subsample_cv(full_cv, 0.25, stratify=y_train, random_state=42)
model = make_pipeline(StandardScaler(), SVC())
common = dict(
    cv=search_cv,
    scoring="accuracy",
    population_size=8,
    generations=4,
    random_state=42,
    n_jobs=1,
    verbose=False,
    final_selection=True,
    final_selection_top_k=4,
    final_selection_cv=full_cv,
)
hpo = GASearchCV(
    clone(model),
    param_grid={
        "svc__C": Continuous(0.01, 1000, distribution="log-uniform"),
        "svc__gamma": Continuous(0.0001, 1, distribution="log-uniform"),
    },
    **common,
).fit(X_train, y_train)
features = GAFeatureSelectionCV(clone(model), **common).fit(X_train, y_train)
print(hpo.best_score_, hpo.score(X_test, y_test))
print(features.best_score_, features.score(X_test, y_test), features.support_.sum())
```

For models where each fold is expensive, another option is to replace the `subsample_cv(...)` line with `search_cv = full_cv[:1]` and set `final_selection_top_k=8` in `common`. The search then scores candidates on one original fold and validates the eight best distinct candidates on all three folds. This can save more time than shortening each training fold, but preliminary rankings use less validation data and can miss the best candidate. In a [six-split five-fold SVC benchmark](https://github.com/rodrigo-arenas/Sklearn-genetic-opt/blob/master/benchmarks/PERFORMANCE_V2.md#does-promoting-more-finalists-recover-accuracy), this pattern made HPO 4.09–4.81× faster with a mean accuracy loss of 0.00583 on synthetic data. Feature selection was 3.48–3.92× faster but lost mean accuracy on all three datasets. Expanding from eight to 24 finalists did not consistently recover quality. Compare held-out accuracy and the complete fitting time on your own data before choosing either shortcut.

`cv_results_`, `history` and the hall of fame retain the original search scores and candidates. `final_selection_results_` records each completed evaluation, its fold scores and the validation time. `best_score_` describes the final-validation winner and can differ from the corresponding search score in `cv_results_`. Feature selection breaks equal final scores by preferring fewer columns. If every finalist score is non-finite, fitting raises with [the completed evaluations available for inspection](./troubleshooting#finalist-validation-ends-without-a-winner).

Use [the finalist score plot](../examples/plotting-gallery#finalist-score-comparison)
to compare the two scores for each candidate and identify the row chosen
for refit.

### Reuse scores from matching folds

For a deterministic estimator, preprocessing pipeline and scorer, use
`final_selection="reuse_folds"` and `error_score="raise"` when the search and
finalist validation share folds. Both training and validation indices
must match in order. Subsampling the training rows makes a fold a fresh fit.
Leave `final_selection_cv=None` to use the folds captured during the search.
Matching Boolean masks also support reuse. A full-length mask can match its
positional indices when they list the same rows in the same order for both
training and validation. Negative indices count from the end of the training
dataset, so `-1` matches the last row. Reuse requires valid row positions.
See the
[Boolean mask example](../tutorials/optimizer-backends#reuse-folds-defined-by-boolean-masks).
Score reuse compares plain NumPy arrays, lists, and tuples. Folds with
custom selector containers or subclasses are scored afresh using the
installed scikit-learn version's indexing rules.
Use either integer positions or a full Boolean mask in each selector.
When comparing separately supplied folds, a mixed list or tuple on either
side requires a fresh finalist fit. Some scikit-learn versions can interpret
a mixed sequence differently from an integer array when indexing a DataFrame.
An explicit finalist splitter or iterable must yield at least one fold. An
empty list or tuple raises before the search starts.
Folds supplied as lists, tuples, or NumPy arrays must contain one training
selector and one validation selector. NumPy selectors must have an integer
or Boolean dtype. A one-dimensional selector in any of these forms must
contain valid row positions or a Boolean mask with one entry per row.
These checks apply to folds yielded by iterables and splitters
as well. The error names the setting, the fold number and, for a bad selector,
the partition. Search folds and explicit finalist lists or tuples are checked
before candidate evaluation. Other finalist iterables and splitters are
checked when their folds are collected for finalist validation. See the
[split validation example](../tutorials/optimizer-backends#validate-explicit-cv-splits).
The matcher can recognize shared index arrays without comparing their
contents again. Explicitly supplied folds are checked by their ordered
indices. See [fit work by phase](../examples/plotting-gallery#fit-work-by-phase)
for an example that reports the reused scores.
Each entry of `final_selection_results_["candidates"]` reports `cv_calls`,
`fold_fits`, and `reused_folds`. The first two count work performed for
that finalist, including a retry when a partial evaluation cannot be combined
with saved scores. `reused_folds` counts scores from the search that were
used in the final result. See the
[per-finalist example](../tutorials/optimizer-backends#inspect-actual-work-for-each-finalist).
Reuse can carry over an undefined `NaN` score. A finalist with an undefined
mean score remains ineligible.
When every search fold matches one-to-one in the original order, finalist
validation reuses the original score array and preserves its numeric dtype.
When the search and final validation share the same training or validation
selector, the matcher converts that NumPy array, list, or tuple once per comparison.
In the original order, each occurrence of a repeated fold keeps its
corresponding score. In a reordered sequence, matching occurrences are reused
in the order they appeared in the search. If finalist validation requests more
copies, the matcher cycles through those scores again.
See the [repeated-fold example](../tutorials/optimizer-backends#preserve-repeated-folds-in-the-same-order).
When finalist CV contains every occurrence from the search in a different
order, finalist validation calculates the ordering once and applies it to each
candidate's stored score array. This retains the original numeric dtype, including
when scores are large.
Finalist validation can reuse a subset of large integers when the search
stored the scores in an integer array. It combines those exact integers with
new integer or floating-point scores and infers the dtype as fresh validation
would. Scores stored as Python objects can require a retry. Large scores
stored as floats may require fresh validation to recover rounded integers.
See the
[large integer score example](../tutorials/optimizer-backends#reuse-large-integer-scores-with-new-folds).
If every finalist fold matches a search fold, a retained finite score
with magnitude above `2**64` can establish the floating-point dtype of the
subset. For extended precision, the score must also lose precision or
range when converted to a Python float. See the
[large floating-point score example](../tutorials/optimizer-backends#reuse-very-large-floating-point-scores-from-a-subset).
When the new folds already produce a `float64` score array, finalist
validation can combine large new scores with saved integer or `float64`
values whose magnitudes are below `2**53`. The combined array stays in
double precision.
See the [example with a new fold](../tutorials/optimizer-backends#reuse-double-precision-scores-with-a-new-fold).
New folds scored in extended precision can also establish the combined array's
dtype when the saved scores are small and were stored as integers, doubles,
or in the same extended dtype. If only the search array uses extended
precision, a retained small value must lose precision when converted to a
Python float for the wider dtype to carry over. If it cannot infer the wider
dtype, finalist validation can fit the matching folds again. It keeps newly
computed `float64` scores of magnitude below `2**53` because any integers
promoted to those values still fit exactly. Large new scores can require a
complete evaluation to recover integers that were rounded on promotion. See the
[precision recovery example](../tutorials/optimizer-backends#recover-precision-after-excluding-a-search-fold).
When matching scores are stored as small integers and the new folds also
produce an integer array, validation can combine them even when the new
integers are large.
A retained fractional double-precision score can also allow reuse when the
new folds produce an integer array. The fraction determines that the combined
scores use floating-point precision, as in fresh validation. An integer-valued score
stored as a float can come from an integer promoted during search. If the
finalist folds supply no evidence for a floating-point dtype, validation
evaluates the matching folds to recover their original score types. It can
keep already computed integer or Boolean scores from new folds when the
recovered scores can be combined exactly. Multiple new folds collected as
unsigned integers can hide Boolean scores, so validation evaluates the
complete list again if the original types are still ambiguous. See the
[integer score example](../tutorials/optimizer-backends#preserve-integer-scores-when-a-fractional-fold-is-excluded).

When a custom scorer supplies extended NumPy precision such as
`numpy.longdouble`, search ranks and finalist comparisons use the precision
retained by cross-validation. `best_score_` and finalist score fields retain
that precision on platforms that support it. Ordinary double-precision
scores are exposed as Python floats. The mean of a short CV score vector
uses a lower-overhead check for ordinary values. When all scores are finite
and their sum could overflow, the mean uses a range-preserving reduction.

When [resuming from a checkpoint](../recipes/advanced/checkpointing), the
five bundled backends can reuse matching fold scores from the saved search.
DEAP requires a verified training-data fingerprint
and independently captured search folds. Older DEAP checkpoints that lack
either piece of information validate finalists afresh. Use
`final_selection=True` when fresh scores are needed for stochastic models
or scorers.

Final selection runs with `refit=True`. Its extra CV evaluations and final refit are additional to the evolutionary budget; generation stopping callbacks do not cap those stages. Include them when timing a workflow. An explicit non-None `final_selection_cv` overrides the corresponding `OptimizationConfig` value. Both settings and explicit fold arrays survive cloning and checkpoints.

Each fresh fit captures a custom splitter or iterable's train and validation
indices as each fold is produced. A splitter can reuse its index arrays without
overwriting earlier folds. If an older checkpoint was created with a splitter
that reused arrays, start a fresh fit to apply this behavior to the search.

Use `stratify=y_train` for classification, and omit it for regression. Sampling that drops a class raises an error; increase the fraction in that case. The discarded rows do not need to contain every class. A fraction of 1 keeps the original training folds. This is opt-in: ordinary searches still use full training folds. Smaller training sets can change candidate rankings, and full validation of finalists cannot recover a good candidate discarded earlier. Measure held-out quality and total time; small datasets or models with high fixed training cost may gain little.

### How many finalists fit my compute budget?

With five final folds and one reusable preliminary fold, each distinct finalist costs four additional fits. Five finalists cost 20 fits and eight cost 32, plus the search and final refit. To try a budget of 20 with the deterministic SVC example above, reuse its unfitted configuration and training split:

```python
folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(X_train, y_train))
budgeted = clone(hpo).set_params(
    cv=folds[:1],
    final_selection="reuse_folds",
    final_selection_cv=folds,
    final_selection_top_k=5,
    error_score="raise",
).fit(X_train, y_train)

counts = budgeted.fit_counts_
print(counts["finalist_fold_fits"], counts["finalist_reused_folds"])
```

The same counters work for `GAFeatureSelectionCV` and for searches that produce fewer distinct finalists than requested. They also include additional fits if finalist validation repeats a fold. At a tied preliminary cutoff, the feature selector can include a denser mask. Changing the shortlist size can therefore change which masks qualify.

When several masks tie at that cutoff, the shortlist considers each distinct
mask once at that score. A later cache hit for one of those masks leaves the
finalists in place. When many rows reuse cached evaluations, feature selection
can skip packing a shared Boolean mask again while filling the shortlist or
checking denser masks at a tied cutoff. The scan follows the original row
order within each score and mask size.

On a large tied cutoff, the selector can restrict mask packing to the
largest size group. Distinctness and the order of first occurrences within
that group still determine which candidate reaches finalist validation.

In 24 paired comparisons on Breast Cancer (logistic regression) and Diabetes (Ridge), reducing the shortlist from eight to five saved 12 fold fits in every pair. Mean end-to-end time fell by 15–23% across the four dataset/search combinations. Mean held-out ROC AUC for feature selection on Breast Cancer stayed at 0.993501, while mean held-out R² for Diabetes hyperparameter search went from 0.527600 to 0.527351. Each comparison used six outer splits per dataset and a single worker. <a :href="shortlistTimingsUrl" download="shortlist-fit-times.json">Download per-split timings, finalist lists and holdout scores (JSON)</a>.

## Choose a refit candidate with a function

Pass a callable as `refit` to choose a candidate after the evolutionary
search. The function receives `cv_results_` and returns an integer row index.
Use `final_selection=False` so the callable makes the final choice. With
multiple metrics, set `objective_metric` to name the scorer used during search
and by the refitted estimator's `score` method.

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

from sklearn_genetic import EvolutionConfig, GASearchCV
from sklearn_genetic.space import Integer


def prefer_smaller_tree(results):
    scores = np.asarray(results["mean_test_balanced_accuracy"])
    eligible = np.flatnonzero(np.isfinite(scores) & (scores >= np.nanmax(scores) - 0.01))
    depths = np.asarray(results["param_max_depth"], dtype=int)
    f1 = results["mean_test_f1"]
    return int(min(eligible, key=lambda index: (depths[index], -f1[index])))


X, y = load_iris(return_X_y=True)
search = GASearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid={"max_depth": Integer(2, 8), "min_samples_leaf": Integer(1, 5)},
    scoring={"balanced_accuracy": "balanced_accuracy", "f1": "f1_macro"},
    objective_metric="balanced_accuracy",
    refit=prefer_smaller_tree,
    evolution_config=EvolutionConfig(population_size=8, generations=2),
    random_state=42,
    error_score="raise",
    verbose=False,
).fit(X, y)
ranked = search.cv_results_dataframe(preserve_index=True)
chosen = ranked.loc[search.best_index_]
chosen_cv_score = chosen["mean_test_balanced_accuracy"]
```

Both search estimators expose the chosen row as `best_index_` and fit that
candidate on all training data. The preserved index lets you use
`.loc[search.best_index_]` directly in the rank-sorted table.

### Refit the best observed feature mask

With `use_cache=False`, a stochastic model or scorer can give a repeated
feature mask different scores. Standard refit without finalist validation
follows the elite archive, which can keep an earlier observation. When
several rows match a finite archived score for the same mask, the first
matching row supplies `best_index_`. Use a callable to choose the
highest-ranked finite observation directly from
`cv_results_`. This example prefers smaller masks at equal ranks. The
rank column already reflects the configured search direction.

```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sklearn_genetic import GAFeatureSelectionCV


def best_observed_mask(results):
    usable = np.isfinite(results["mean_test_score"]) & ~results["all_fits_failed"]
    eligible = np.flatnonzero(usable)
    if eligible.size == 0:
        raise ValueError("No feature mask has a finite search CV score.")
    ranks = np.asarray(results["rank_test_score"])[eligible]
    sizes = np.asarray(results["n_features"])[eligible]
    return int(eligible[np.lexsort((sizes, ranks))[0]])


X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.25, random_state=42
)
selector = GAFeatureSelectionCV(
    RandomForestClassifier(n_estimators=16, n_jobs=1),
    cv=3,
    scoring="balanced_accuracy",
    use_cache=False,
    refit=best_observed_mask,
    population_size=6,
    generations=2,
    random_state=42,
    n_jobs=1,
    verbose=False,
).fit(X_train, y_train)

print(selector.cv_results_["mean_test_score"][selector.best_index_])
print(selector.score(X_test, y_test))
```

Callable refit reads the recorded scores and runs one full-training fit.
Use the selected row for its CV score. A high score from one repeat can
reflect training randomness. For a decision based on fresh finalist
scores, use `final_selection=True` with standard refit.

### Break an exact CV tie by fold dispersion

With a single scoring metric, pass `refit=prefer_lower_dispersion` if you
want the candidate with the lowest fold-score standard deviation among
the top-ranked rows. For feature selection, the function first restricts
those rows to the smallest feature sets.

```python
import numpy as np


def prefer_lower_dispersion(cv_results):
    ranks = np.asarray(cv_results["rank_test_score"])
    tied = np.flatnonzero(ranks == ranks.min())
    if "n_features" in cv_results:
        counts = np.asarray(cv_results["n_features"])[tied]
        tied = tied[counts == counts.min()]
    std = np.asarray(cv_results["std_test_score"])[tied]
    return int(tied[np.where(np.isfinite(std), std, np.inf).argmin()])
```

This rule uses the existing fold scores. Evaluate the refitted estimator on
held-out data to assess how it performs beyond those folds.

## Inspect the selected candidate without refitting

Use `refit=False` to compare candidates using search CV alone. Both search
estimators skip finalist validation and the full-data fit. In a multi-metric
search, use `objective_metric` to choose the score that guides the optimizer.

```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

from sklearn_genetic import EvolutionConfig, GASearchCV
from sklearn_genetic.space import Integer

X, y = load_iris(return_X_y=True)
search = GASearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid={"max_depth": Integer(2, 8), "min_samples_leaf": Integer(1, 5)},
    scoring={"accuracy": "accuracy", "f1": "f1_macro"},
    objective_metric="f1",
    refit=False,
    evolution_config=EvolutionConfig(population_size=8, generations=2),
    random_state=42,
    verbose=False,
).fit(X, y)

selected = search.cv_results_dataframe(preserve_index=True).loc[search.best_index_]
print(search.best_params_, search.best_score_, selected["mean_test_f1"])
print(search.fit_counts_["refit_fits"])  # 0
print(len(search), search[-1]["cross_validate_calls"])
```

With `GAFeatureSelectionCV`, use `support_` to inspect the selected mask
and `best_index_` to find its row in `cv_results_`. Both estimators report
the selected candidate's CV objective score as `best_score_`.

Use `len(search)` for the number of recorded generations, `search[-1]` for
the last report, or `list(search)` for all reports. After early stopping,
only recorded reports count. To compare progress by evaluation cost, use the
[candidate CV calls axis](../api/plots#plot-fitness-evolution).
