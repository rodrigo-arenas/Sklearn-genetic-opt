# Conda recipe

This directory contains the conda recipe used to build `sklearn-genetic-opt` as
a `noarch: python` conda package.

## Why this recipe exists

`sklearn-genetic-opt` is published on
[conda-forge](https://anaconda.org/conda-forge/sklearn-genetic-opt), so the
recommended way to install it is:

```bash
conda install -c conda-forge sklearn-genetic-opt
```

The previous conda-forge recipe listed the **optional** integrations
(`seaborn`, `mlflow`, `tensorflow`, `tensorboard`) as **hard runtime
requirements**. Those packages carry tight, mutually incompatible constraints,
which made the environment unsolvable for many users and Python versions
(see [issue #138](https://github.com/rodrigo-arenas/Sklearn-genetic-opt/issues/138)).

In the codebase those dependencies are imported lazily (`try/except
ModuleNotFoundError`), so they are genuinely optional. This recipe lists only
the core dependencies — matching the `dependencies` table in `pyproject.toml` —
so the package installs cleanly. Users who want the extras can add them
explicitly, for example:

```bash
conda install -c conda-forge sklearn-genetic-opt seaborn mlflow
```

## Building locally

From the repository root, with `conda-build` installed:

```bash
conda build conda.recipe
```

The recipe reads the version directly from `sklearn_genetic/_version.py` and
builds from the local checkout (`source: path: ..`).

## Updating the conda-forge feedstock

When releasing, the
[feedstock](https://github.com/conda-forge/sklearn-genetic-opt-feedstock) should
be updated to:

- bump `version` and the sdist `sha256`,
- set `python >=3.12`,
- keep the `run` requirements limited to the core dependencies listed here
  (do **not** re-add `seaborn`, `mlflow`, `tensorflow`, or `tensorboard`).
