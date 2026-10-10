r"""Paired, bounded search and feature-selection benchmark.

PowerShell: .venv\Scripts\python.exe benchmarks/benchmark_evolution.py --output results.json
Use --source-root PATH to run the same harness against an exported older package.
Hyperparameter methods share train/test data, CV folds, spaces and --budget.
The optional feature-selection workload is a separate before/after comparison
with a fixed maximum of 208 CV evaluations (16 initial + 6 * 32 offspring).
"""

import argparse
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.metrics import get_scorer
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

CATEGORICAL_SPACES = {
    "dt": [
        ("criterion", "cat", "linear", ("gini", "entropy")),
        ("splitter", "cat", "linear", ("best", "random")),
        ("max_depth", "int", "linear", (1, 15)),
        ("min_samples_leaf", "int", "linear", (1, 10)),
        ("ccp_alpha", "real", "log", (1e-5, 0.1)),
    ],
    "svm": [
        ("kernel", "cat", "linear", ("rbf", "poly", "sigmoid")),
        ("shrinking", "cat", "linear", (True, False)),
        ("C", "real", "log", (0.01, 100)),
        ("gamma", "real", "log", (1e-4, 1)),
    ],
    "knn": [
        ("weights", "cat", "linear", ("uniform", "distance")),
        ("metric", "cat", "linear", ("euclidean", "manhattan")),
        ("n_neighbors", "int", "linear", (1, 30)),
        ("leaf_size", "int", "linear", (10, 50)),
    ],
}


def native_params(params):
    # Some tuners return NumPy scalar categories. Estimators with Cython
    # parameter dispatch (e.g. KNN metrics) require the original Python types.
    return {
        name: value.item() if isinstance(value, np.generic) else value
        for name, value in params.items()
    }


def run(args):
    if args.source_root:
        sys.path.insert(0, str(args.source_root.resolve()))
    from sklearn_genetic import GASearchCV, GAFeatureSelectionCV
    from benchmark_bayesmark import (
        ga_budget,
        DATASETS,
        MODELS,
        build_task,
        make_pipeline,
        task_space,
        to_sklearn_genetic_space,
        to_scipy_distributions,
        suggest_with_optuna,
    )

    if "optuna" in args.methods:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)

    if "flaml" in args.methods:
        from flaml import CFO, tune

    rows = []
    population, generations = ga_budget(args.budget)
    for dataset_name in args.datasets:
        dataset = DATASETS[dataset_name]
        X, y = dataset.loader()
        for model_name in args.models:
            task = build_task(dataset, MODELS[model_name])
            for seed in range(args.seed_start, args.seed_start + args.seeds):
                Xtr, Xte, ytr, yte = train_test_split(
                    X,
                    y,
                    test_size=0.25,
                    random_state=seed,
                    stratify=y if dataset.task == "classification" else None,
                )
                from benchmark_bayesmark import make_cv

                cv = list(make_cv(task, seed, 3).split(Xtr, ytr))
                for method in args.methods:
                    model, prefix = make_pipeline(task, seed)
                    specs = CATEGORICAL_SPACES[model_name] if args.categorical else task_space(task)
                    started = time.perf_counter()
                    if method in ("gasearch", "differential"):
                        de = method == "differential"
                        search = GASearchCV(
                            model,
                            param_grid=to_sklearn_genetic_space(specs, prefix),
                            cv=cv,
                            scoring=task.scoring,
                            population_size=population,
                            generations=(args.budget // population - 1) if de else generations,
                            algorithm="eaDifferentialEvolution" if de else "eaMuPlusLambda",
                            n_jobs=1,
                            verbose=False,
                            random_state=seed,
                            error_score="raise",
                        )
                        search.fit(Xtr, ytr)
                        score, best = search.best_score_, search.best_estimator_
                        evaluations = search.fit_stats_["cross_validate_calls"]
                        extra = {
                            "fitness_best": search.history["fitness_best"],
                            "fit_stats": search.fit_stats_,
                        }
                    elif method == "randomized":
                        search = RandomizedSearchCV(
                            model,
                            to_scipy_distributions(specs, prefix),
                            n_iter=args.budget,
                            scoring=task.scoring,
                            cv=cv,
                            random_state=seed,
                            n_jobs=1,
                            error_score="raise",
                        )
                        search.fit(Xtr, ytr)
                        score, best = search.best_score_, search.best_estimator_
                        evaluations = len(search.cv_results_["params"])
                        extra = {}
                    elif method == "optuna":
                        params_by_trial = {}

                        def objective(trial):
                            params = suggest_with_optuna(trial, specs, prefix)
                            params_by_trial[trial.number] = params
                            return float(
                                np.mean(
                                    cross_val_score(
                                        clone(model).set_params(**params),
                                        Xtr,
                                        ytr,
                                        cv=cv,
                                        scoring=task.scoring,
                                        n_jobs=1,
                                        error_score="raise",
                                    )
                                )
                            )

                        study = optuna.create_study(
                            direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
                        )
                        study.optimize(objective, n_trials=args.budget)
                        score = study.best_value
                        best = model.set_params(**params_by_trial[study.best_trial.number]).fit(
                            Xtr, ytr
                        )
                        evaluations, extra = len(study.trials), {}
                    else:
                        space = {}
                        for name, kind, distribution, payload in specs:
                            key = prefix + name
                            if kind == "cat":
                                space[key] = tune.choice(list(payload))
                            elif kind == "int":
                                space[key] = tune.randint(int(payload[0]), int(payload[1]) + 1)
                            elif distribution == "log":
                                space[key] = tune.loguniform(*payload)
                            else:
                                space[key] = tune.uniform(*payload)

                        def objective(config):
                            return {
                                "score": float(
                                    np.mean(
                                        cross_val_score(
                                            clone(model).set_params(**native_params(config)),
                                            Xtr,
                                            ytr,
                                            cv=cv,
                                            scoring=task.scoring,
                                            n_jobs=1,
                                            error_score="raise",
                                        )
                                    )
                                )
                            }

                        analysis = tune.run(
                            objective,
                            config=space,
                            metric="score",
                            mode="max",
                            search_alg=CFO(space=space, metric="score", mode="max", seed=seed),
                            num_samples=args.budget,
                            use_ray=False,
                            verbose=0,
                        )
                        score = analysis.best_result["score"]
                        best = model.set_params(**native_params(analysis.best_config)).fit(Xtr, ytr)
                        evaluations, extra = len(analysis.trials), {}
                    seconds = time.perf_counter() - started
                    if evaluations > args.budget:
                        raise RuntimeError("CV budget exceeded")
                    row = dict(
                        dataset=dataset_name,
                        model=model_name,
                        seed=seed,
                        method=method,
                        cv_score=float(score),
                        holdout_score=float(get_scorer(task.scoring)(best, Xte, yte)),
                        seconds=seconds,
                        evaluations=evaluations,
                        evaluation_budget=args.budget,
                        **extra,
                    )
                    rows.append(row)
                    print(json.dumps(row), flush=True)
    if args.features:
        X, y = make_classification(
            n_samples=240, n_features=20, n_informative=5, n_redundant=5, random_state=42
        )
        for seed in range(args.seed_start, args.seed_start + args.seeds):
            Xtr, Xte, ytr, yte = train_test_split(
                X, y, test_size=0.25, stratify=y, random_state=seed
            )
            search = GAFeatureSelectionCV(
                DecisionTreeClassifier(max_depth=4, random_state=seed),
                cv=StratifiedKFold(3, shuffle=True, random_state=seed),
                population_size=16,
                generations=6,
                n_jobs=1,
                verbose=False,
                random_state=seed,
                max_features=10,
                error_score="raise",
            )
            started = time.perf_counter()
            search.fit(Xtr, ytr)
            row = dict(
                dataset="synthetic_features",
                model="dt",
                seed=seed,
                method="features",
                seconds=time.perf_counter() - started,
                holdout_score=float(search.score(Xte, yte)),
                cv_score=float(max(search.cv_results_["mean_test_score"])),
                evaluations=search.fit_stats_["cross_validate_calls"],
                evaluation_budget=208,
                selected_features=int(search.get_support().sum()),
                fitness_best=search.history["fitness_best"],
                fit_stats=search.fit_stats_,
                diversity=search.history["genotype_diversity"],
                recovery_generations=sum(search.history["diversity_control_triggered"]),
            )
            if row["evaluations"] > row["evaluation_budget"]:
                raise RuntimeError("Feature CV budget exceeded")
            if args.source_root is None and not np.isclose(
                row["fitness_best"][-1], row["cv_score"]
            ):
                raise RuntimeError("Best-so-far history disagrees with evaluated scores")
            rows.append(row)
            print(json.dumps(row), flush=True)
    payload = dict(
        label=args.label,
        budget=args.budget,
        seeds=args.seeds,
        seed_start=args.seed_start,
        space_profile="categorical" if args.categorical else "bayesmark",
        feature_evaluation_budget=208 if args.features else None,
        python=platform.python_version(),
        platform=platform.platform(),
        source_root=str(args.source_root),
        versions={
            name: importlib.metadata.version(name)
            for name in (
                ["numpy", "scipy", "scikit-learn", "deap"]
                + (["optuna"] if "optuna" in args.methods else [])
                + (["flaml"] if "flaml" in args.methods else [])
            )
        },
        results=rows,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", default="current")
    parser.add_argument("--budget", type=int, default=40)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--datasets", nargs="+", default=["wine", "breast"])
    parser.add_argument("--models", nargs="+", default=["svm", "dt"])
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["gasearch", "differential", "optuna", "randomized", "flaml"],
        default=["gasearch", "optuna", "randomized"],
    )
    parser.add_argument("--features", action="store_true")
    parser.add_argument(
        "--categorical",
        action="store_true",
        help="Use mixed spaces with multiple categorical dimensions (classification only)",
    )
    args = parser.parse_args()
    if args.budget < 2 or args.seeds < 1:
        parser.error("budget must be >=2 and seeds >=1")
    if "differential" in args.methods and args.budget < 4:
        parser.error("differential evolution requires budget >=4")
    if args.categorical and (
        not set(args.models) <= CATEGORICAL_SPACES.keys() or "diabetes" in args.datasets
    ):
        parser.error("categorical profile supports dt/svm/knn classification workloads")
    with threadpool_limits(limits=1):
        run(args)


if __name__ == "__main__":
    main()
