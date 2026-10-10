r"""Compare native CPU AutoML workflows with fixed-model evolutionary searches.

All methods share outer train/test splits and a requested time budget. Native
AutoML validation, model families and ensembles differ: this is an end-to-end
comparison, not an optimizer-only or identical-CV comparison. Soft time limits
can overshoot; actual elapsed fitting time is always reported.
"""

import argparse
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import (
    load_breast_cancer,
    load_diabetes,
    load_digits,
    load_iris,
    make_classification,
)
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from threadpoolctl import threadpool_limits


def dataset(name):
    loaders = {
        "iris": load_iris,
        "breast": load_breast_cancer,
        "digits": load_digits,
        "diabetes": load_diabetes,
    }
    if name == "synthetic":
        X, y = make_classification(
            n_samples=2000,
            n_features=80,
            n_informative=12,
            n_redundant=8,
            flip_y=0.05,
            random_state=113,
        )
    else:
        X, y = loaders[name](return_X_y=True)
    return X, y, name != "diabetes"


def run(args):
    from lightgbm import LGBMClassifier, LGBMRegressor
    from sklearn_genetic import GASearchCV, GAFeatureSelectionCV
    from sklearn_genetic.callbacks import TimerStopping
    from sklearn_genetic.space import Integer, Continuous

    if "autogluon" in args.methods:
        from autogluon.tabular import TabularPredictor
    if "flaml" in args.methods:
        from flaml import AutoML
    packages = ["numpy", "scipy", "scikit-learn", "lightgbm", "deap"]
    packages += ["autogluon.tabular"] if "autogluon" in args.methods else []
    packages += ["flaml"] if "flaml" in args.methods else []
    payload = dict(
        label=args.label,
        requested_seconds=args.seconds,
        seed_start=args.seed_start,
        seeds=args.seeds,
        python=platform.python_version(),
        platform=platform.platform(),
        versions={name: importlib.metadata.version(name) for name in packages},
        settings={
            "native_families": ["lightgbm", "random_forest", "extra_trees"],
            "evolutionary_model": "lightgbm",
            "cv_folds": 3,
            "population": 8,
            "native_threads": 1,
            "autogluon_bag_folds": 3,
            "autogluon_stack_levels": 0,
        },
        results=[],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1):
        for name in args.datasets:
            X, y, classification = dataset(name)
            for seed in range(args.seed_start, args.seed_start + args.seeds):
                Xtr, Xte, ytr, yte = train_test_split(
                    X, y, test_size=0.25, random_state=seed, stratify=y if classification else None
                )
                for method in args.methods:
                    model_cls = LGBMClassifier if classification else LGBMRegressor
                    model = model_cls(
                        n_estimators=100, num_leaves=31, n_jobs=1, verbosity=-1, random_state=seed
                    )
                    cv_cls = StratifiedKFold if classification else KFold
                    cv = list(cv_cls(3, shuffle=True, random_state=seed).split(Xtr, ytr))
                    scoring = "accuracy" if classification else "neg_mean_squared_error"
                    extra = {}
                    start = time.perf_counter()
                    if method in ("gasearch", "features"):
                        common = dict(
                            cv=cv,
                            scoring=scoring,
                            population_size=8,
                            generations=1000,
                            n_jobs=1,
                            verbose=False,
                            random_state=seed,
                            error_score="raise",
                        )
                        if method == "gasearch":
                            search = GASearchCV(
                                model,
                                param_grid={
                                    "n_estimators": Integer(30, 200),
                                    "num_leaves": Integer(7, 63),
                                    "learning_rate": Continuous(
                                        0.01, 0.2, distribution="log-uniform"
                                    ),
                                    "min_child_samples": Integer(5, 50),
                                    "reg_lambda": Continuous(
                                        1e-4, 10.0, distribution="log-uniform"
                                    ),
                                },
                                **common,
                            )
                        else:
                            search = GAFeatureSelectionCV(model, **common)
                        search.fit(Xtr, ytr, callbacks=TimerStopping(args.seconds))
                        fit_seconds = time.perf_counter() - start
                        prediction = search.predict(Xte)
                        extra = dict(
                            cv_score=(
                                float(search.best_score_)
                                if method == "gasearch"
                                else float(max(search.cv_results_["mean_test_score"]))
                            ),
                            evaluations=search.fit_stats_["cross_validate_calls"],
                            generations=len(search.history["gen"]),
                            selected_features=(
                                int(search.support_.sum()) if method == "features" else X.shape[1]
                            ),
                        )
                    elif method == "flaml":
                        search = AutoML()
                        search.fit(
                            X_train=Xtr,
                            y_train=ytr,
                            task="classification" if classification else "regression",
                            metric="accuracy" if classification else "mse",
                            time_budget=args.seconds,
                            estimator_list=["lgbm", "rf", "extra_tree"],
                            eval_method="cv",
                            n_splits=3,
                            n_jobs=1,
                            seed=seed,
                            verbose=0,
                            use_ray=False,
                            mlflow_logging=False,
                            mem_thres=1024**3,
                        )
                        fit_seconds = time.perf_counter() - start
                        prediction = search.predict(Xte)
                        extra = dict(
                            best_family=search.best_estimator, selected_features=X.shape[1]
                        )
                    else:
                        train = pd.DataFrame(Xtr)
                        train.columns = [f"f{i}" for i in range(X.shape[1])]
                        test = pd.DataFrame(Xte, columns=train.columns)
                        train["target"] = ytr
                        path = args.model_dir / args.label / name / str(seed)
                        search = TabularPredictor(
                            label="target",
                            problem_type=(
                                ("binary" if len(np.unique(ytr)) == 2 else "multiclass")
                                if classification
                                else "regression"
                            ),
                            eval_metric="accuracy" if classification else "mean_squared_error",
                            path=str(path),
                            verbosity=0,
                            learner_kwargs={"random_state": seed},
                        )
                        search.fit(
                            train,
                            time_limit=args.seconds,
                            hyperparameters={"GBM": {}, "RF": {}, "XT": {}},
                            num_bag_folds=3,
                            num_bag_sets=1,
                            num_stack_levels=0,
                            dynamic_stacking=False,
                            fit_strategy="sequential",
                            ag_args_ensemble={"fold_fitting_strategy": "sequential_local"},
                            num_cpus=1,
                            num_gpus=0,
                            memory_limit=1.0,
                        )
                        fit_seconds = time.perf_counter() - start
                        prediction = search.predict(test)
                        extra = dict(
                            best_family=search.model_best,
                            model_count=len(search.model_names()),
                            selected_features=X.shape[1],
                        )
                    row = dict(
                        dataset=name,
                        seed=seed,
                        method=method,
                        metric=scoring,
                        holdout_score=float(
                            accuracy_score(yte, prediction)
                            if classification
                            else -mean_squared_error(yte, prediction)
                        ),
                        seconds=fit_seconds,
                        requested_seconds=args.seconds,
                        train_samples=len(ytr),
                        test_samples=len(yte),
                        **extra,
                    )
                    payload["results"].append(row)
                    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
                    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                    temporary.replace(args.output)
                    print(json.dumps(row), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--label", default="native")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["iris", "breast", "digits", "diabetes", "synthetic"],
        default=["breast", "digits", "diabetes", "synthetic"],
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["gasearch", "features", "flaml", "autogluon"],
        default=["gasearch", "features", "flaml", "autogluon"],
    )
    parser.add_argument("--seconds", type=float, default=15)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--seed-start", type=int, default=901)
    args = parser.parse_args()
    if args.seconds <= 0 or not np.isfinite(args.seconds) or args.seeds < 1:
        parser.error("seconds must be finite and positive, and seeds must be >=1")
    run(args)


if __name__ == "__main__":
    main()
