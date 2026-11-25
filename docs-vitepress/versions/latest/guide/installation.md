---
title: Installation
description: Install sklearn-genetic-opt with pip or conda, with optional optimizer engines, plotting and MLflow.
---

:::warning Development version
You are reading the **latest (dev)** docs. For the stable version, see [stable](/stable/).
:::

# Installation

## Prerequisites

- Python ≥ 3.12
- pip or conda

The test matrix covers standard CPython 3.12–3.15 on Linux, macOS, and Windows.
Free-threaded builds are not covered.

## pip

Install the core package:

```bash
pip install sklearn-genetic-opt
```

Install the runtime extras for optimization, plotting and MLflow:

```bash
python -m pip install "sklearn-genetic-opt[all]"
```

Install individual extras:

```bash
python -m pip install "sklearn-genetic-opt[plot]"      # seaborn for plot_fitness_evolution / plot_search_space
python -m pip install "sklearn-genetic-opt[mlflow]"    # MLflow 3 logging
```

## Optimization backends

Install one engine and plotting together, or install all four extra engines:

```bash
python -m pip install "sklearn-genetic-opt[pymoo,plot]>=0.16.0"
python -m pip install "sklearn-genetic-opt[scipy,nevergrad,cmaes,pymoo]>=0.16.0"
```

DEAP is included in the base package. The SciPy extra enforces the solver's
minimum version even when scikit-learn can use an older SciPy. Some platforms
may build pymoo from source. See the [engine reference](../api/optimizer-backends)
and [run your first backend comparison](../tutorials/optimizer-backends).

## conda

```bash
conda install -c conda-forge sklearn-genetic-opt
```

The conda package ships only the core dependencies. Install optional extras alongside it:

```bash
conda install -c conda-forge sklearn-genetic-opt seaborn mlflow
```

## Requirements

| Package | Minimum version | Required |
|---------|----------------|----------|
| Python | 3.12 | Yes |
| scikit-learn | 1.5.0 | Yes |
| NumPy | 1.26.0 | Yes |
| pandas | 1.3.0 | Yes |
| DEAP | 1.4.4 | Yes |
| tqdm | 4.60.0 | Yes |
| SciPy | 1.15 | For the SciPy optimization backend |
| Nevergrad | >=1.0.12, <2 | `nevergrad` extra |
| cmaes | >=0.13.1, <0.14 | `cmaes` extra |
| pymoo | >=0.6.2, <0.7 | `pymoo` extra |
| Seaborn | 0.13.2 | No (plots) |
| MLflow | 3.14.0 | No (MLflow logging) |

On Python 3.15, pip requires scikit-learn ≥ 1.9.1, NumPy ≥ 2.5.2,
pandas ≥ 3.0.6, and SciPy ≥ 1.18.1. These versions have upstream wheels
for CPython 3.15. Update any older pins in your environment before installing;
the general lower bounds above remain unchanged on older interpreters. Conda
instead resolves compatible builds from its channels.

## Verify the installation

```python
import sklearn_genetic
print(sklearn_genetic.__version__)
```

## Next Steps

- [Basic Usage](./basic-usage) — start here if you're new to the library.
- [When to Use](./when-to-use) — decide if GA search fits your problem.
