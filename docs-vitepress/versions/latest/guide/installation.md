---
title: Installation
description: Install sklearn-genetic-opt with pip or conda, including optional extras for plotting, MLflow, and TensorBoard.
---

:::warning Development version
You are reading the **latest (dev)** docs. For the stable version, see [0.13](/versions/0.13/).
:::

# Installation

## Prerequisites

- Python ≥ 3.12
- pip or conda

## pip

Install the core package:

```bash
pip install sklearn-genetic-opt
```

Install all optional extras (plotting, MLflow, TensorBoard):

```bash
pip install sklearn-genetic-opt[all]
```

Install individual extras:

```bash
pip install sklearn-genetic-opt[plot]      # seaborn for plot_fitness_evolution / plot_search_space
pip install sklearn-genetic-opt[mlflow]    # MLflow 3 logging
```

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
| scikit-learn | 1.9.0 | Yes |
| NumPy | 2.4.6 | Yes |
| DEAP | 1.4.4 | Yes |
| tqdm | 4.68.3 | Yes |
| Seaborn | 0.13.2 | No (plots) |
| MLflow | 3.14.0 | No (MLflow logging) |
| TensorFlow | 2.21.0 | No (TensorBoard logging, Python < 3.14) |
| TensorBoard | 2.20.0 | No (TensorBoard logging) |

## Verify the installation

```python
import sklearn_genetic
print(sklearn_genetic.__version__)
```

## Next Steps

- [Basic Usage](./basic-usage) — start here if you're new to the library.
- [When to Use](./when-to-use) — decide if GA search fits your problem.
