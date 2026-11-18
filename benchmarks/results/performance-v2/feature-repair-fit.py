"""Compare two equivalent feature-mask repair loops during complete fits."""

import json
import random
from pathlib import Path
from runpy import run_path

from threadpoolctl import threadpool_limits

from sklearn_genetic import GAFeatureSelectionCV


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


if __name__ == "__main__":
    paired_fit = run_path(str(Path(__file__).with_name("finalist-mask-keys.py")))["paired_fit"]
    with threadpool_limits(1):
        rows = [
            paired_fit(width, seed, (GAFeatureSelectionCV, BulkRepair))
            for width in (16, 256, 4096)
            for seed in range(9151, 9157)
        ]
    print(json.dumps(rows, indent=2))
