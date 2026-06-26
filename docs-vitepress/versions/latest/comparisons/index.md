---
title: "Hyperparameter Optimization: Method Comparisons"
description: "Honest, benchmark-backed comparisons between GridSearchCV, RandomizedSearchCV, Bayesian Optimization, and genetic algorithm search."
---

:::warning Development version
You are reading the **latest (dev)** docs. For stable documentation, see [stable](/stable/).
:::

# Hyperparameter Optimization: Method Comparisons

Not every hyperparameter search method is the right tool for every problem. This section collects honest, benchmark-backed comparisons so you can make an informed choice rather than a default one.

Each page shows **when sklearn-genetic-opt wins and when it doesn't**. If a competing method is the better fit for your problem, we say so.

## Comparisons

| Page | What it answers |
|------|----------------|
| [Grid Search vs Random Search vs Bayesian vs Genetic Algorithms](./grid-search-vs-genetic-algorithms) | Which method to use and why — with a fair benchmark, code for all four methods, and an honest breakdown of each method's failure modes |
| [Optuna vs sklearn-genetic-opt](./optuna-vs-sklearn-genetic-opt) | Head-to-head: Bayesian optimization (TPE) vs genetic algorithms — with benchmarks, code examples for the same problem, and an honest decision guide including when Optuna wins |

## A Note on Honest Comparisons

Most tool documentation shows the tool at its best. We try to do better than that.

Every comparison page in this section includes:

- **A scenario where sklearn-genetic-opt wins** — with numbers to back it up
- **A scenario where sklearn-genetic-opt loses** — because it does lose, and knowing when saves you time
- **Equal-budget benchmarks** — comparing methods that ran the same number of evaluations, not the same wall-clock time
- **Runnable code** — every example uses scikit-learn's built-in datasets and runs without modification

The benchmark data on the [Benchmarks](../benchmarks/) page was collected with the same philosophy: the numbers that didn't go our way are published alongside the ones that did.

## See Also

- [When to Use sklearn-genetic-opt](../guide/when-to-use) — the quick decision guide
- [Benchmarks](../benchmarks/) — full Bayesmark suite results across multiple datasets and models
- [Comparing Search Methods (Example)](../examples/sklearn-comparison) — a copy-pasteable side-by-side with full output
