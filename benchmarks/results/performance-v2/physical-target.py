"""Transfer the diabetes CV target to held-out airfoil conditions and hull geometries."""

import argparse
import hashlib
import json
from io import BytesIO
from itertools import combinations, permutations
from math import comb
from pathlib import Path
from runpy import run_path
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

run = run_path(str(Path(__file__).with_name("diabetes-target.py")))["run"]


def exhaustive(X, y, test, labels, cv, seed, genetic):
    model = make_pipeline(StandardScaler(), Ridge())
    scores = {}
    for k in range(1, 5):
        for mask in combinations(range(X.shape[1]), k):
            scores[mask] = cross_val_score(
                model, X[:, mask], y, cv=cv, scoring="r2", error_score="raise"
            ).mean()
    best = max(scores, key=scores.get)
    # For the mask at ascending rank i to win, all other draws must rank below it.
    draws, remainder = divmod(genetic["folds"], len(cv))
    assert remainder == 0 and 1 <= draws <= len(scores)
    ranked = sorted(reversed(scores), key=scores.get)
    ways = [comb(i, draws - 1) for i in range(len(scores))]
    total = comb(len(scores), draws)
    assert sum(ways) == total
    expected_cv = expected_holdout = match = 0.0
    for mask, count in zip(ranked, ways, strict=True):
        if count:
            weight = count / total
            expected_cv += weight * scores[mask]
            model.fit(X[:, mask], y)
            expected_holdout += weight * model.score(test[:, mask], labels)
            if scores[mask] >= genetic["cv"] - 1e-12:
                match += weight
    model.fit(X[:, best], y)
    return dict(
        policy="exhaustive",
        seed=seed,
        features=best,
        cv=scores[best],
        folds=len(scores) * len(cv),
        holdout=model.score(test[:, best], labels),
        random_at_genetic_budget=dict(
            draws=draws,
            expected_cv=expected_cv,
            expected_holdout=expected_holdout,
            probability_matching_or_beating_genetic=match,
        ),
    )


def compare(name, archive, member, group_columns, reference=False):
    with urlopen(f"https://archive.ics.uci.edu/static/public/{archive}.zip", timeout=20) as stream:
        with ZipFile(BytesIO(stream.read())) as zipped:
            raw = zipped.read(member)
    data = np.loadtxt(BytesIO(raw))
    X, y = data[:, :-1], data[:, -1]
    _, groups = np.unique(X[:, group_columns], axis=0, return_inverse=True)
    rows = []
    for index, order in enumerate(permutations(("genetic", "univariate", "residual"))):
        seed = 10011 + index
        split = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
        train, test = next(split.split(X, y, groups))
        cv = list(GroupKFold(3).split(X[train], y[train], groups[train]))
        assert not np.intersect1d(groups[train], groups[test]).size
        for a, b in cv:
            assert not np.intersect1d(groups[train[a]], groups[train[b]]).size
        args = X[train], y[train], X[test], y[test], cv, seed
        if reference:
            genetic = run("genetic", *args, 1.0)
            baseline = exhaustive(*args, genetic)
            assert not genetic["reached"] and genetic["cv"] <= baseline["cv"] + 1e-12
            rows.extend([baseline, genetic])
        else:
            for policy in order:
                rows.append(run(policy, *args, 0.45))
    return dict(
        dataset=name,
        sha256=hashlib.sha256(raw).hexdigest(),
        groups=len(set(groups)),
        target=1.0 if reference else 0.45,
        runs=rows,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", action="store_true")
    reference = parser.parse_args().reference
    with threadpool_limits(1):
        result = [
            compare(
                "airfoil",
                "291/airfoil+self+noise",
                "airfoil_self_noise.dat",
                slice(1, None),
                reference,
            ),
            compare(
                "yacht",
                "243/yacht+hydrodynamics",
                "yacht_hydrodynamics.data",
                slice(0, 5),
                reference,
            ),
        ]
    print(json.dumps(result, indent=2, allow_nan=False))
