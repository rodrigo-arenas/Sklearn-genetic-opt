# GridSearchCV vs GASearchCV: When Does Genetic Search Actually Pay Off?

Hyperparameter tuning usually comes down to a tradeoff between how thoroughly you search
and how much time you're willing to spend. `GridSearchCV` searches exhaustively; genetic
algorithm-based approaches like `GASearchCV` (from
[sklearn-genetic-opt](https://github.com/rodrigo-arenas/Sklearn-genetic-opt)) search
adaptively, using evolutionary principles to focus evaluations on promising regions of
the hyperparameter space.

The common pitch for genetic search is "fewer evaluations, comparable results." I wanted
to check that claim directly, using the project's own benchmarking tool
(`benchmarks/benchmark_search_methods.py`) rather than a hand-rolled comparison — so the
numbers below come straight from that script, unmodified.

## Setup

Two scenarios, both built into the benchmark suite:

- **`classification_lr`** — logistic regression, classification task
- **`regression_ridge`** — ridge regression, regression task

Three search methods compared: `GASearchCV`, `RandomizedSearchCV`, and `GridSearchCV`.
`RandomizedSearchCV`'s candidate budget is auto-matched to GA's total candidate
generation budget, so all three methods get a fair, comparable number of evaluation
opportunities where possible.

I ran the comparison twice — once with the default grid density (`--grid-points 5`) and
once with a denser grid (`--grid-points 10`) — to see whether the size of the search
space changes which method comes out ahead.

## Results: default grid (`--grid-points 5`)

**Classification (`classification_lr`)**

| Method | Fit time | Candidates evaluated | Best CV score | Test accuracy | ROC AUC |
|---|---|---|---|---|---|
| GASearchCV | 20.71s | 253 (759 CV evals) | 0.9950 | 0.9766 | 0.9966 |
| RandomizedSearchCV | 16.36s | 252 (756 CV evals) | 0.9950 | 0.9825 | 0.9966 |
| GridSearchCV | 0.35s | 10 (30 CV evals) | 0.9944 | 0.9825 | 0.9860 |

**Regression (`regression_ridge`)**

| Method | Fit time | Candidates evaluated | Best CV score (R²) | RMSE |
|---|---|---|---|---|
| GASearchCV | 10.97s | 253 (759 CV evals) | 0.4648 | 41.8357 |
| RandomizedSearchCV | 3.34s | 252 (756 CV evals) | 0.4639 | 41.8394 |
| GridSearchCV | 0.17s | 10 (30 CV evals) | 0.4645 | 41.8435 |

At this grid density, `GridSearchCV` is competitive on both scenarios — matching or
nearly matching the CV score of GA and randomized search, in a fraction of the wall
time. With only 10 grid combinations, exhaustive search simply isn't expensive yet.

## Results: denser grid (`--grid-points 10`)

**Classification (`classification_lr`)**

| Method | Fit time | Candidates evaluated | Best CV score | Test accuracy |
|---|---|---|---|---|
| GASearchCV | 21.08s | 253 (759 CV evals) | 0.9949 | 0.9825 |
| RandomizedSearchCV | 19.07s | 252 (756 CV evals) | 0.9950 | 0.9825 |
| GridSearchCV | 0.59s | 20 (60 CV evals) | 0.9949 | 0.9825 |

**Regression (`regression_ridge`)**

| Method | Fit time | Candidates evaluated | Best CV score (R²) | RMSE |
|---|---|---|---|---|
| GASearchCV | 12.12s | 253 (759 CV evals) | 0.4648 | 41.8357 |
| RandomizedSearchCV | 2.89s | 252 (756 CV evals) | 0.4639 | 41.8394 |
| GridSearchCV | 0.29s | 20 (60 CV evals) | 0.4647 | 41.8363 |

Doubling the grid density roughly doubled grid search's candidate count (10 → 20), but
its result quality barely moved and it *still* matched GA and randomized search on
accuracy and CV score — in under a second, versus roughly 20 seconds for the other two.

## The honest takeaway

Based on these numbers, the answer to "does GASearchCV beat GridSearchCV?" is: **not
here, and not by much even when it's ahead.** Logistic and ridge regression have small,
well-behaved hyperparameter spaces — a handful of dimensions, mostly independent effects,
no sharp interactions between parameters. That's exactly the setting where exhaustive
search is cheap and effective, and where an adaptive search strategy has little
inefficiency to reclaim.

This isn't a knock against `GASearchCV` — it's a boundary condition. Genetic search
earns its keep as a strategy for *avoiding combinatorial explosion*, not as a strategy
for beating a well-chosen small grid. The scenarios where it should pull ahead are ones
where:

- The hyperparameter space has **many dimensions** (5+), so a grid with reasonable
  resolution per dimension becomes combinatorially enormous
- Hyperparameters **interact** — e.g. tree-ensemble models like Gradient Boosting or
  Random Forest, where `max_depth`, `min_samples_split`, `n_estimators`, and
  `learning_rate` all influence each other, so a coarse grid risks missing good
  combinations that a grid simply doesn't sample
- You genuinely cannot afford an exhaustive search at the resolution you'd like,
  and need a method that spends its evaluation budget adaptively

For a linear model with 2-3 tunable hyperparameters, reach for `GridSearchCV` — it's
faster, simpler, and in this benchmark, just as accurate. For a high-dimensional,
interaction-heavy model, that's where it's worth testing `GASearchCV` against your
baseline, the same way this benchmark did here.

## Reproducing this

All numbers above come directly from the repository's own benchmark script:

```bash
python benchmarks/benchmark_search_methods.py \
  --methods gasearch randomized grid \
  --scenarios classification_lr regression_ridge \
  --output-json benchmarks/my_comparison.json

python benchmarks/benchmark_search_methods.py \
  --methods gasearch randomized grid \
  --scenarios classification_lr regression_ridge \
  --grid-points 10 \
  --output-json benchmarks/my_large_grid_comparison.json
```

No custom scripts, no cherry-picked scenarios — just the tooling already in the repo,
run twice with one parameter changed.

---

*Written by Ayush Aman, B.Tech AI & Data Science, Dr. Akhilesh Das Gupta Institute of
Professional Studies, as part of exploring
[sklearn-genetic-opt](https://github.com/rodrigo-arenas/Sklearn-genetic-opt) issue
[#267](https://github.com/rodrigo-arenas/Sklearn-genetic-opt/issues/267).*