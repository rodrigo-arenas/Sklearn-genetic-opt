"""Evaluate fold-local filter proposals with correlated regression predictors."""

import argparse
import json
from functools import partial
from pathlib import Path
from runpy import run_path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor
from threadpoolctl import threadpool_limits


def run(seed, compare, tree=False):
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(1100, 2))
    replica_noise = rng.normal(size=(1100, 8))
    copies = latent[:, [0]] + 0.15 * replica_noise
    X = np.column_stack((copies, latent[:, 1], rng.normal(size=(1100, 23))))
    order = rng.permutation(X.shape[1])
    X = X[:, order]
    noise = rng.normal(size=len(X))
    targets = {
        "redundant": 1.5 * latent[:, 0] + latent[:, 1] + 0.5 * noise,
        "contrast": replica_noise[:, 0] - replica_noise[:, 1] + 0.5 * noise,
    }
    folds = list(KFold(3, shuffle=True, random_state=seed).split(X[:240]))
    if tree:
        targets["quadratic"] = latent[:, 1] ** 2 + 0.5 * noise
    rows = []
    for task, y in targets.items():
        for cap in ((4,) if tree else (4, 8)):
            result = compare(
                X[:240],
                y[:240],
                X[240:],
                y[240:],
                folds,
                (
                    DecisionTreeRegressor(max_depth=5, min_samples_leaf=5, random_state=seed)
                    if tree
                    else Ridge()
                ),
                "r2",
                seed,
                cap=cap,
                one_slot=True,
            )
            rows.append(dict(task=task, cap=cap, column_order=order.tolist(), **result))
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--residual", action="store_true")
    modes.add_argument("--tree", action="store_true")
    args = parser.parse_args()
    compare = run_path(str(Path(__file__).with_name("filter-portfolio.py")))["compare"]
    seeds = range(9571, 9577) if args.residual else range(9461, 9467)
    if args.tree:
        seeds = range(9681, 9687)
    compare_run = partial(compare, residual=args.residual or args.tree, time_slot=args.tree)
    with threadpool_limits(1):
        rows = [row for seed in seeds for row in run(seed, compare_run, tree=args.tree)]
        if args.residual:
            real = run_path(str(Path(__file__).with_name("known-support.py")))["run_real"]
            sparse = run_path(str(Path(__file__).with_name("partition-initialization.py")))[
                "sparse_regression"
            ]
            for builder in (real, sparse):
                builder.__globals__["compare"] = partial(compare, residual=True, one_slot=True)
            rows += [dict(cap=4, **real("diabetes", seed, seed + 10)) for seed in seeds]
            rows += [dict(task="sparse", cap=4, **sparse(64, seed)) for seed in seeds]
    print(json.dumps(rows, indent=2, allow_nan=False))
