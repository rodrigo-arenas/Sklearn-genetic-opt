---
title: "Switch optimization backends in a scikit-learn search"
description: "Tune a classifier, select features and resume from a generation checkpoint with interchangeable optimizers."
---

:::warning Development version
This is the **latest (dev)** documentation. It may contain unreleased features or breaking changes. For the stable release, use [stable](/stable/).
:::

# Switch optimization backends in a scikit-learn search

Use the same folds and scorer when changing engines. This walkthrough uses all five named backends and keeps a test partition for the final model.

## Install the development checkout

From the repository root, run `python -m pip install -e ".[scipy,nevergrad,cmaes,pymoo]"`.
For a single engine, install just its [extra](../guide/installation#optimization-backends).
The examples use built-in scikit-learn data and run serially.

```python
from sklearn.base import clone
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn_genetic import GAFeatureSelectionCV, GASearchCV
from sklearn_genetic.space import Categorical, Continuous, Integer

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.25, random_state=7
)
model = DecisionTreeClassifier(random_state=7)
folds = StratifiedKFold(3, shuffle=True, random_state=7)
settings = dict(cv=folds, scoring="balanced_accuracy", refit=True, random_state=7,
                population_size=8, generations=2, n_jobs=1, verbose=False)
search = GASearchCV(
    model,
    param_grid={"max_depth": Integer(2, 10), "criterion": Categorical(["gini", "entropy"]),
                "ccp_alpha": Continuous(1e-6, 0.1, distribution="log-uniform")},
    **settings,
)
```

## Change the engine

Clone the template to start a fresh search. Inspect the best CV score and the number
of real CV calls. A cache hit reuses a score, and DEAP's default algorithm can generate
more offspring per generation. Each real CV call here fits three folds.

```python
fitted = {}
for engine in ("deap", "scipy", "nevergrad", "cmaes", "pymoo"):
    run = clone(search).set_params(optimizer_backend=engine).fit(X_train, y_train)
    fitted[engine] = run
    print(engine, round(run.best_score_, 3), run.fit_stats_["cross_validate_calls"])
```

```text
deap 0.941 39
scipy 0.941 23
nevergrad 0.941 23
cmaes 0.941 24
pymoo 0.928 20
```

Inspect one fitted model on the held-out data. The choice of SciPy here is fixed in
advance. Use `cv_results_` for individual candidates and `history` for generations.
For a performance comparison across datasets and seeds, see the
[matched-budget benchmark](../benchmarks/optimizer-catalog).

```python
run = fitted["scipy"]
print("candidate rows", len(run.cv_results_["mean_test_score"]))
print("CV calls by generation", list(run.history["cross_validate_calls"]))
print("test balanced accuracy", round(balanced_accuracy_score(y_test, run.predict(X_test)), 3))
```

```text
candidate rows 24
CV calls by generation [8, 16, 23]
test balanced accuracy 0.917
```

## Select features and inspect progress

Pass the same keyword to `GAFeatureSelectionCV`. pymoo now evolves binary masks.
The estimator applies the selected mask for prediction. A stopping callback checks
each reported generation. Here the callback stops after generation 1 so we can
resume the original two-generation budget and measure additional CV calls.

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from sklearn_genetic.callbacks import BaseCallback, ModelCheckpoint

class StopAfterOne(BaseCallback):
    def on_step(self, record=None, logbook=None, estimator=None):
        return record["gen"] == 1

selector = GAFeatureSelectionCV(model, optimizer_backend="pymoo", max_features=8, **settings)
with TemporaryDirectory() as directory:
    checkpoint = ModelCheckpoint(Path(directory) / "progress.pkl")
    clone(selector).fit(X_train, y_train, callbacks=[StopAfterOne(), checkpoint])
    saved = checkpoint.load()
    selector.fit(X_train, y_train, callbacks=[checkpoint])

print("checkpoint", saved["backend"], saved["logbook"][-1]["gen"], saved["resumable"])
print("final generation", selector.history["gen"][-1])
print("additional CV calls",
      selector.fit_stats_["cross_validate_calls"] - saved["fit_stats"]["cross_validate_calls"])
print("selected columns", selector.get_support(indices=True).tolist())
print("test balanced accuracy", round(balanced_accuracy_score(y_test, selector.predict(X_test)), 3))
```

```text
checkpoint pymoo 1 True
final generation 2
additional CV calls 6
selected columns [6, 11, 12, 15, 23, 27]
test balanced accuracy 0.901
```

SciPy and pymoo verify the recorded proposals while reconstructing the solver,
using saved results for generations already completed. Nevergrad and CatCMAwM
restore native solver state. Use the same data, search settings and environment
when resuming. Work in a partially completed generation can require CV again.
The [checkpoint recipe](../recipes/advanced/checkpointing) also covers DEAP's
historical format and older progress-only files. Third-party engines can implement
continuation through the [optimizer protocol](../api/algorithms#callbacks-and-checkpoints).

## Configure a solver

The [backend API](../api/optimizer-backends) lists supported settings. The following
examples create configured estimators. Call `fit` as above to run them.

```python
from functools import partial
from importlib.metadata import entry_points

engines = entry_points(group="sklearn_genetic.optimizers")
cma = clone(search).set_params(optimizer_backend=partial(engines["cmaes"].load(), sigma=0.15))
ng = clone(search).set_params(optimizer_backend=partial(engines["nevergrad"].load(), strategy="TwoPointsDE"))
ga = clone(selector).set_params(crossover_probability=0.7)
```
