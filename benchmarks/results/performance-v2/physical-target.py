"""Transfer the diabetes CV target to held-out airfoil conditions and hull geometries."""

import hashlib
import json
from io import BytesIO
from itertools import permutations
from pathlib import Path
from runpy import run_path
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from threadpoolctl import threadpool_limits

run = run_path(str(Path(__file__).with_name("diabetes-target.py")))["run"]


def compare(name, archive, member, group_columns):
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
        for policy in order:
            rows.append(run(policy, X[train], y[train], X[test], y[test], cv, seed, 0.45))
    return dict(
        dataset=name,
        sha256=hashlib.sha256(raw).hexdigest(),
        groups=len(set(groups)),
        target=0.45,
        runs=rows,
    )


if __name__ == "__main__":
    with threadpool_limits(1):
        result = [
            compare("airfoil", "291/airfoil+self+noise", "airfoil_self_noise.dat", slice(1, None)),
            compare("yacht", "243/yacht+hydrodynamics", "yacht_hydrodynamics.data", slice(0, 5)),
        ]
    print(json.dumps(result, indent=2, allow_nan=False))
