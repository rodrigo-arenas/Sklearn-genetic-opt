"""Time public finalist budgets on two additional Bayesmark model families."""

import json
import sys
from pathlib import Path
from runpy import run_path

from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from benchmarks import benchmark_bayesmark as bm
from sklearn_genetic import GAFeatureSelectionCV, GASearchCV

if __name__ == "__main__":
    time_shortlists = run_path(str(Path(__file__).with_name("bounded-finalists.py")))[
        "time_shortlists"
    ]
    rows = []
    with threadpool_limits(1):
        for dataset, family in (("wine", "knn"), ("diabetes", "dt")):
            task = bm.build_task(bm.DATASETS[dataset], bm.MODELS[family])
            X, y = task.dataset.loader()
            for seed in range(8811, 8817):
                train, test, labels, target = train_test_split(
                    X,
                    y,
                    test_size=0.25,
                    random_state=seed,
                    stratify=y if task.dataset.task == "classification" else None,
                )
                folds = list(bm.make_cv(task, seed, 5).split(train, labels))
                options = dict(
                    cv=folds[:1],
                    final_selection="reuse_folds",
                    final_selection_cv=folds,
                    scoring=task.scoring,
                    error_score="raise",
                    population_size=6,
                    generations=2,
                    n_jobs=1,
                    random_state=seed,
                    verbose=False,
                )
                for features in (False, True):
                    model, prefix = bm.make_pipeline(task, seed)
                    if features:
                        search = GAFeatureSelectionCV(
                            model, max_features=train.shape[1] // 2, **options
                        )
                    else:
                        space = bm.to_sklearn_genetic_space(bm.task_space(task), prefix)
                        search = GASearchCV(model, param_grid=space, **options)
                    runs = time_shortlists(
                        search, train, labels, test, target, task.scoring, seed, features
                    )
                    rows.append(
                        dict(
                            dataset=dataset,
                            family=family,
                            metric=task.scoring,
                            seed=seed,
                            features=features,
                            runs=runs,
                        )
                    )
    print(json.dumps(rows, indent=2, allow_nan=False))
