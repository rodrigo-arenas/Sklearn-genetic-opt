"""Evaluate fold-local filter proposals with correlated regression predictors."""

import json
from pathlib import Path
from runpy import run_path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits


def run(seed, compare):
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
    rows = []
    for task, y in targets.items():
        for cap in (4, 8):
            result = compare(
                X[:240],
                y[:240],
                X[240:],
                y[240:],
                folds,
                Ridge(),
                "r2",
                seed,
                cap=cap,
                one_slot=True,
            )
            rows.append(dict(task=task, cap=cap, column_order=order.tolist(), **result))
    return rows


if __name__ == "__main__":
    compare = run_path(str(Path(__file__).with_name("filter-portfolio.py")))["compare"]
    with threadpool_limits(1):
        rows = [row for seed in range(9461, 9467) for row in run(seed, compare)]
    print(json.dumps(rows, indent=2, allow_nan=False))
