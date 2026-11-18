"""Pilot feasible unions against ordinary feature crossover at equal CV budgets."""

import json
from functools import partial
from pathlib import Path
from runpy import run_path

from sklearn_genetic import GAFeatureSelectionCV
from threadpoolctl import threadpool_limits


class FeasibleUnion(GAFeatureSelectionCV):
    union_count_ = 0

    def mate(self, first, second):
        union = [a | b for a, b in zip(first, second)]
        eligible = max(sum(first), sum(second)) < sum(union) <= self.max_features
        first, second = super().mate(first, second)
        if eligible:
            first[:] = union
            self.union_count_ += 1
        return first, second


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    harness = run_path(str(here / "partition-initialization.py"))
    methods = (GAFeatureSelectionCV, FeasibleUnion)
    run_real = run_path(str(here / "known-support.py"))["run_real"]
    run_real.__globals__["compare"] = partial(harness["compare"], methods=methods)
    with threadpool_limits(1):
        rows = [
            harness["sparse_regression"](width, seed, methods=methods)
            for width in (16, 64)
            for seed in range(8511, 8519)
        ]
        rows.extend(
            run_real(task, split, seed)
            for task in ("wine", "diabetes")
            for split in (8521, 8522)
            for seed in (8531, 8532)
        )
    print(json.dumps(rows, indent=2, allow_nan=False))
