import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits
from benchmarks.benchmark_fit import SCENARIOS, make_cv
from sklearn_genetic import GASearchCV, GAFeatureSelectionCV

with threadpool_limits(1):
    for name in ("classification_lr", "regression_ridge"):
        scenario = SCENARIOS[name]
        X, y = scenario.loader()
        X, _, y, _ = train_test_split(
            X,
            y,
            test_size=0.25,
            random_state=1407,
            stratify=y if scenario.task == "classification" else None,
        )
        folds = list(make_cv(scenario, 3, 1407).split(X, y))
        # Same test observations, different training set on the second search fold.
        preliminary = [folds[0], (folds[1][0][::2], folds[1][1])]
        final = [folds[2], folds[0], folds[1]]
        for features in (False, True):
            options = dict(
                cv=preliminary,
                final_selection=True,
                final_selection_top_k=3,
                final_selection_cv=final,
                population_size=6,
                generations=1,
                n_jobs=1,
                random_state=1407,
                scoring=scenario.scoring,
                error_score="raise",
                verbose=False,
            )
            model = scenario.estimator_builder(1407)
            search = (
                GAFeatureSelectionCV(model, max_features=X.shape[1] // 2, **options)
                if features
                else GASearchCV(model, param_grid=scenario.param_grid_builder(), **options)
            )
            search.fit(X, y)
            finalists = search.final_selection_results_["candidates"]
            equal = changed = 0
            for row in finalists:
                idx = row["index"]
                equal += search.cv_results_["split0_test_score"][idx] == row["cv_scores"][1]
                changed += search.cv_results_["split1_test_score"][idx] != row["cv_scores"][2]
            print(name, "features" if features else "hpo", len(finalists), int(equal), int(changed))
