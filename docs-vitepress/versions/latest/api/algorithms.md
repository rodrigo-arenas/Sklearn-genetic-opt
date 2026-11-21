---
title: Algorithms
description: API reference for the DEAP-based evolutionary algorithm implementations used internally by GASearchCV.
---

:::warning Development version
You are reading the **latest (dev)** docs. For the stable version, see [stable](/stable/).
:::

# Algorithms

The `sklearn_genetic.algorithms` module provides the DEAP-based evolutionary algorithm implementations used internally by `GASearchCV` and `GAFeatureSelectionCV`.

These are low-level functions — you typically do not call them directly. They are exposed for advanced users who want to inspect or extend the evolutionary logic.

```python
from sklearn_genetic.algorithms import eaMuPlusLambda, eaMuCommaLambda, eaSimple, eaDifferentialEvolution
```

## eaMuPlusLambda

The (μ + λ) evolutionary strategy. The next generation is selected from the union of parents and offspring. This is the default algorithm used by `GASearchCV`.

- **Pros:** Parents compete with offspring; with `elitism=True`, archived best candidates retain survivor slots.
- **Cons:** Can converge prematurely if the initial population is too small.

## eaMuCommaLambda

The (μ, λ) evolutionary strategy. The next generation is selected from offspring only. Parents do not survive directly.

- **Pros:** Better exploration — avoids stagnation around a local optimum.
- **Cons:** Without `elitism=True`, offspring-only survival can lose the best individual between generations.

## eaSimple

A generational genetic algorithm with a fixed population size. With
`elitism=True`, each distinct archived elite occupies one population slot, up
to `keep_top_k` and the size of the population. The remaining slots are available
for offspring. For example, a population of six with four archived elites has
room for two offspring. Choose `keep_top_k` below `population_size` to leave
room for variation once the archive fills. With `elitism=False`, every slot is
available for offspring.

## eaDifferentialEvolution

An optional differential-evolution strategy that generates one trial per
parent and retains the better candidate. It requires a population of at least
four. Both search classes accept `algorithm="eaDifferentialEvolution"`.
Numeric dimensions use their declared linear or log domain; categorical
choices and feature masks stay within their valid domains. It is an
alternative to benchmark for your task, not a generally faster or more
accurate replacement for the default algorithm.

## SciPy backend

Install the supported solver from this development checkout with `pip install ".[scipy]"`
(SciPy 1.15+). The base installation already includes SciPy through scikit-learn.
Both estimators use DEAP by default. Select SciPy when a bounded, single-objective
differential evolution search fits your problem.

```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn_genetic import GASearchCV, GAFeatureSelectionCV
from sklearn_genetic.space import Integer, Categorical

X, y = load_iris(return_X_y=True)
options = dict(optimizer_backend="scipy", cv=3, population_size=8,
               generations=4, random_state=7, verbose=False)
tree = DecisionTreeClassifier(random_state=7)
search = GASearchCV(tree, param_grid={"max_depth": Integer(1, 8),
                    "criterion": Categorical(["gini", "entropy"])}, **options).fit(X, y)
selector = GAFeatureSelectionCV(tree, max_features=2, **options).fit(X, y)
print(search.best_params_, selector.get_support())
```

- SciPy proposes one trial per target using deferred `best1bin`, its default
  dithering scale (0.5–1.0) and the numeric `crossover_probability`. Start with at
  least five individuals. The maximum is `population_size * (generations + 1)`
  proposals. Equal population energies can stop the solver early. Polishing is off.
  Cache hits avoid CV. Final selection and refit can add work after optimization.
- The existing random/smart initializer and warm starts provide the first population.
  Continuous values use linear or logarithmic coordinates. Variable integer bounds
  must fit int64 and span less than 2**52. Fixed integers remain exact. Unweighted
  categories use equal-width bins in the supplied order, so reordering choices can
  change the trajectory. Weighted categories require DEAP.
- Feature coordinates at or above 0.5 select columns. Overflow keeps the largest
  coordinates, with ties resolved in column order. An empty mask selects the largest
  coordinate. SciPy compares scalar scores; the archive breaks score ties in favor
  of smaller masks. CV receives the decoded mask in every case.
- `parallel_backend` still selects fold or population evaluation. Both engines
  use the same scoring, cache, finalist selection, candidate records, plots and
  refit. Generation summaries describe survivors; the archive includes discarded
  trials. Seed the estimator, search and any shuffled splitter for repeatability
  within the same software environment.
- The legacy `algorithm`, mutation, tournament, elitism and optimization controls
  retain their constructor defaults when selecting SciPy. Changing these DEAP-only
  settings raises an error. Diversity control is inactive for SciPy under either
  Boolean setting. Final selection and fixed crossover are supported.
- LivePlot, progress loggers and stopping callbacks run at the same generation
  boundaries as external plugins. `ModelCheckpoint` saves an inspectable progress
  snapshot. SciPy's public solver API does not restore its internal state for
  exact continuation. See [callbacks and checkpoints](#callbacks-and-checkpoints).

SciPy was selected because scikit-learn already installs it and its recent releases
publish Linux, macOS and Windows wheels for all four supported Python minors.
The public vectorized objective lets the estimator keep ownership of CV and
parallelism. See the [SciPy solver documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html).

## See Also

- [DEAP documentation](https://deap.readthedocs.io/en/master/api/algo.html) — upstream algorithm documentation
- [Advanced Optimizer Control](../guide/advanced-optimizer-control) — configure algorithm behavior via `EvolutionConfig`

## Diversity telemetry

`history["genotype_diversity"]` is the mean probability that two distinct population
members have different values at a gene. Identical populations score zero;
populations with distinct values at every gene score one. Balanced binary genes
score approximately 0.5 even in large populations. `unique_individual_ratio`
continues to describe the fraction of distinct complete candidates.

This corrects the older distinct-allele normalization that made large feature
populations appear collapsed. Custom diversity thresholds may need recalibration.
Local integer refinement now samples directly from the bounded neighborhood,
excluding the parent's value when alternatives exist, with constant memory.

## Elites and progress

With `elitism=True`, the mu/lambda algorithms reserve up to `keep_top_k` survivor
slots for copies of the best archived candidates. With `elitism=False`, all
survivors are selected from the algorithm's candidate pool. In both cases,
`fitness_best` tracks the best evaluated candidate in the hall of fame, while
`fitness_min`, `fitness_max` and `fitness` describe the current population.

## Optimizer plugins

Pass a function as `optimizer_backend` to try a search strategy with either
estimator. This example evaluates the supplied population, then proposes new
candidates by recombining coordinates. For feature selection it repeats valid
initial masks, preserving the feature-count limit.

```python
from random import Random

from sklearn.tree import DecisionTreeClassifier
from sklearn_genetic import GASearchCV
from sklearn_genetic.space import Integer


def recombine(problem, evaluate, report):
    rng = Random(problem.seed)
    parents = problem.initial_population
    for generation in range(problem.generations + 1):
        if generation == 0:
            candidates = parents
        elif problem.task == "parameters":
            candidates = [
                [rng.choice(parents)[i] for i in range(len(parents[0]))]
                for _ in parents
            ]
        else:
            candidates = [rng.choice(parents) for _ in parents]
        scored = evaluate(candidates)
        report([handle for loss, handle in scored])
        ranked = sorted(zip(candidates, scored), key=lambda item: item[1][0])
        parents = [row for row, _ in ranked[: max(1, len(ranked) // 2)]]


search = GASearchCV(
    DecisionTreeClassifier(random_state=7), cv=3,
    param_grid={"max_depth": Integer(1, 10), "min_samples_leaf": Integer(1, 5)},
    optimizer_backend=recombine, population_size=8, generations=2, random_state=7,
)
```

The public signature is `optimize(problem, evaluate, report) -> None`.
`sklearn_genetic.backends.Optimizer` describes the callable type.
`OptimizationProblem` supplies the task, initial rows, parameter names and
dimensions in coordinate order, feature limit, generation limit, fixed crossover
probability and a per-fit seed. Treat dimensions and category values as read-only.
Use the initial rows to honor warm starts. Check any solver-specific restrictions
before calling `evaluate`.

- Pass a nonempty batch of valid decoded rows to `evaluate`. It returns ordered
  `(loss, handle)` pairs. Minimize the scalar loss for either scoring direction.
  Feature masks must be binary, nonempty and respect `max_features`.
- Pass surviving handles to `report` after initialization and once per generation.
  Report between one and `population_size` survivors. A handle identifies one evaluation and may survive several generations. Keep it
  opaque. The estimator records the best candidates across all evaluated proposals.
- Stop early if appropriate. The upper limits are `problem.max_evaluations`
  proposals (including cache hits) and `problem.generations + 1` reports.
  The last evaluation must be followed by a report. Use these callbacks
  synchronously within the call. The estimator owns CV parallelism and caching.
- Predictions, refit, fitted history, post-fit plots and candidate results share
  the estimator's normal API. DEAP remains an installation dependency for the
  default engine. Plugin fits do not import it. Explicit access to the historical
  `logbook` or feature `hof` materializes the corresponding DEAP object.
  LivePlot and stopping callbacks use the shared runtime. Custom callbacks that
  access DEAP operators or the toolbox still require the DEAP engine.

To distribute this function independently, put it in a Python package and add
the following entry point to that package's `pyproject.toml`:

```toml
[project.entry-points."sklearn_genetic.optimizers"]
recombine = "my_optimizer:recombine"
```

After installation, select `optimizer_backend="recombine"` in either estimator.
Only the selected plugin is loaded. The names `deap` and `scipy` are reserved.
Use a unique name and declare solver dependencies in the plugin package. For
serialization, use its registered name or an importable module-level callable.
See the [PyPA plugin guide](https://packaging.python.org/en/latest/guides/creating-and-discovering-plugins/).

### Callbacks and checkpoints

For either estimator, pass callbacks to `fit`. This example displays the best
observed fitness, stops on a plateau and saves progress at each reported boundary:

```python
from sklearn_genetic.callbacks import DeltaThreshold, ModelCheckpoint
from sklearn_genetic.plots import LivePlot

checkpoint = ModelCheckpoint("search-progress.pkl")  # Use a fresh path for a new search.
search.fit(X, y, callbacks=[
    LivePlot(metrics=("fitness_best_observed",)),
    DeltaThreshold(threshold=0.001, generations=3),
    checkpoint,
])
progress = checkpoint.load()
print(len(progress["candidate_records"]), progress["resumable"])
```

The example uses a plugin or SciPy. DEAP keeps its historical checkpoint format.
On the shared runtime, a custom callback receives the estimator at `on_start`,
the generation row and accumulated rows at `on_step`, and the rows at `on_end`.
The log supports indexing, iteration and `select("fitness")`. Returning `True`
from `on_step` stops after all callbacks for that row have run. The shared
runtime writes checkpoints after the other observers finish that step. A plugin can
release resources in `finally`; let exceptions from `evaluate` and `report`
propagate to the runtime.

`TimerStopping` checks between CV batches. The in-flight batch can finish and
the first batch is allowed to complete, so refit has evaluated candidates. When
a deadline interrupts a generation, its final row has `partial=True` and
summarizes the completed proposals since the last report. Refitting uses the
normal selection rules. A partial row preserves the previous complete resumable
checkpoint when one exists.

**Optional exact resume for plugins.** A callable may implement
`resume(problem, evaluate, report, state)` as described by
`sklearn_genetic.backends.ResumableOptimizer`. When
`problem.checkpoint_requested` is true, such a plugin calls
`report(survivors, state=...)` with non-None, picklable continuation state at
each completed boundary. Save everything needed to propose the **next**
generation, including the solver's RNG state, iteration number and any retained
evaluation handles. Give the solver state its own version and validate it in
`resume`. For example, an engine using a Python `random.Random` generator can
save `(1, next_generation, rng.getstate(), parents)` and reconstruct its
generator with `rng.setstate(saved_rng_state)`.

To continue, supply the same checkpoint to `fit` with the same inputs and
search settings. The runtime restores the original CV folds, remaining proposal
budget, cache, observed candidates, elites and generation numbers before calling
`resume`. Comparisons require the same software versions and reproducible
model/scoring behavior. Checkpoint-enabled exact resume requires fingerprintable
settings, including importable scorers. Callback instances start a new lifecycle
and can consult the accumulated generation log. A completed generation budget
can be refitted directly. SciPy and plugins implementing only the base protocol
can save and inspect progress; starting a new search uses a fresh path. Load
pickle checkpoints only from trusted sources.

## Nevergrad

Install `sklearn-genetic-opt[nevergrad]` and pass `optimizer_backend="nevergrad"` to either
estimator. For parameter searches, NGOpt chooses an optimizer from its portfolio using the remaining budget, typed
search space and batch width. The best initial candidate seeds its parameterization. Feature searches use OnePlusOne on a vector of ordered binary choices. In the
Wine calibration at 32 real CV calls, this reduced average non-CV time and improved
mean best mask CV over NGOpt across three seeds. The generation history summarizes each evaluated
batch. The default crossover setting leaves variation to Nevergrad. Use unweighted categories,
finite linear or logarithmic intervals and integer spans below 2**52. Fixed coordinates are
removed, and an entirely fixed search finishes after initialization. LivePlot, early stopping
and progress snapshots are supported. Start a fresh search to generate new trials.
