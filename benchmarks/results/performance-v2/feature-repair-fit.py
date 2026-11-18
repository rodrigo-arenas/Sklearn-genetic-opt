"""Compare two equivalent feature-mask repair loops during complete fits."""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from runpy import run_path
from time import perf_counter

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

from sklearn_genetic import GAFeatureSelectionCV
from sklearn_genetic.evaluation import evaluate_population


class BulkRepair(GAFeatureSelectionCV):
    def _repair_individual(self, individual):
        individual[:] = [1 if value else 0 for value in individual]
        count = sum(individual)
        if self.max_features and count > self.max_features:
            selected = [index for index, value in enumerate(individual) if value]
            random.shuffle(selected)
            for index in selected[self.max_features :]:
                individual[index] = 0
        elif count == 0:
            individual[random.randrange(len(individual))] = 1
        return individual


class InternalBatch(GAFeatureSelectionCV):
    """Benchmark the internal path on built-in operators and fresh populations."""

    def _register(self):
        super()._register()
        self.toolbox.register(
            "evaluate_population",
            evaluate_population,
            self,
            cache_record_key="current_generation_features",
        )


class AuditedRepair(GAFeatureSelectionCV):
    def _repair_individual(self, individual):
        stage = sys._getframe(1).f_code.co_name
        counts = self.repair_counts[stage]
        counts["calls"] += 1
        before = list(individual)
        needs_conversion = any(type(v) is not int for v in before)
        start = perf_counter()
        result = super()._repair_individual(individual)
        counts["seconds"] += perf_counter() - start
        counts["changed_values"] += before != result
        counts["changed_types"] += needs_conversion
        return result


def audit(seed, mode):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(160, 4096))
    y = X[:, 0] > 0
    options = dict(
        random=dict(population_initializer="random"),
        differential=dict(algorithm="eaDifferentialEvolution"),
        local=dict(local_search=True),
        default={},
    )[mode]
    search = AuditedRepair(
        DecisionTreeClassifier(max_depth=3, random_state=seed),
        cv=2,
        population_size=8,
        generations=3,
        max_features=16,
        scoring="accuracy",
        n_jobs=1,
        random_state=seed,
        verbose=False,
        **options,
    )
    search.repair_counts = defaultdict(Counter)
    search.fit(X, y)
    return dict(seed=seed, mode=mode, stages=dict(search.repair_counts))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--audit", action="store_true")
    modes.add_argument("--internal-batch", action="store_true")
    args = parser.parse_args()
    paired_fit = run_path(str(Path(__file__).with_name("finalist-mask-keys.py")))["paired_fit"]
    with threadpool_limits(1):
        if args.audit:
            rows = [
                audit(seed, mode)
                for mode in ("default", "random", "differential", "local")
                for seed in range(9161, 9164)
            ]
        else:
            variant = InternalBatch if args.internal_batch else BulkRepair
            seeds = range(9171, 9179) if args.internal_batch else range(9151, 9157)
            rows = [
                paired_fit(width, seed, (GAFeatureSelectionCV, variant))
                for width in (16, 256, 4096)
                for seed in seeds
            ]
    print(json.dumps(rows, indent=2))
