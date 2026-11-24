r"""Paired finite-space and non-exhausting controls; export an older package with
git archive and pass --source-root to run the identical workload against it.

PowerShell: .venv\Scripts\python.exe benchmarks/benchmark_exhaustion.py --output results.json
"""

import argparse
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path

from sklearn.datasets import load_diabetes, load_iris, make_classification
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from threadpoolctl import threadpool_limits

BACKENDS = ("deap", "scipy", "nevergrad", "cmaes", "pymoo")
CASES = (
    "hpo-small",
    "hpo-medium",
    "hpo-continuous",
    "features-small",
    "features-medium",
    "features-wide",
)
REGRESSION_CASES = ("hpo-regression", "features-regression")


def run(args):
    if args.source_root:
        sys.path.insert(0, str(args.source_root.resolve()))
    from sklearn_genetic import GASearchCV, GAFeatureSelectionCV
    from sklearn_genetic.space import Categorical, Continuous, Integer

    rows = []
    with threadpool_limits(limits=1):
        for case in args.cases or CASES:
            for seed in range(args.seed_start, args.seed_start + args.seeds):
                regression = case in REGRESSION_CASES
                if case in ("features-medium", "features-wide"):
                    width = 8 if case == "features-medium" else 20
                    X, y = make_classification(
                        n_samples=240, n_features=width, n_informative=4, random_state=seed
                    )
                elif regression:
                    X, y = load_diabetes(return_X_y=True)
                    X = X[:, :5]
                else:
                    X, y = load_iris(return_X_y=True)
                Xtr, Xte, ytr, yte = train_test_split(
                    X, y, stratify=None if regression else y, random_state=seed
                )
                cv = KFold if regression else StratifiedKFold
                common = dict(
                    cv=list(cv(3, shuffle=True, random_state=seed).split(Xtr, ytr)),
                    scoring="neg_root_mean_squared_error" if regression else "accuracy",
                    population_size=args.population_size,
                    generations=args.generations,
                    n_jobs=1,
                    random_state=seed,
                    use_cache=not args.no_cache,
                    verbose=False,
                )
                estimator = DecisionTreeRegressor if regression else DecisionTreeClassifier
                start_at = seed % len(args.backends)
                order = args.backends[start_at:] + args.backends[:start_at]
                for backend in order:
                    model = estimator(max_depth=5, random_state=seed)
                    options = (
                        common if backend == "deap" else dict(common, optimizer_backend=backend)
                    )
                    if case.startswith("hpo"):
                        criteria = (
                            ["squared_error", "absolute_error"]
                            if regression
                            else ["gini", "entropy"]
                        )
                        space = {
                            "max_depth": Integer(1, 3),
                            "criterion": Categorical(criteria),
                        }
                        if case == "hpo-medium":
                            space["max_depth"] = Integer(1, 8)
                            space["min_samples_split"] = Integer(2, 4)
                        elif case == "hpo-continuous":
                            space["ccp_alpha"] = Continuous(0, 0.1)
                        search = GASearchCV(model, param_grid=space, **options)
                    else:
                        maximum = {
                            "features-small": 2,
                            "features-regression": 2,
                            "features-medium": 3,
                            "features-wide": 10,
                        }[case]
                        search = GAFeatureSelectionCV(model, max_features=maximum, **options)
                    start = time.perf_counter()
                    search.fit(Xtr, ytr)
                    seconds = time.perf_counter() - start
                    row = dict(
                        case=case,
                        backend=backend,
                        seed=seed,
                        cache=not args.no_cache,
                        seconds=seconds,
                        cv_score=float(max(search.cv_results_["mean_test_score"])),
                        holdout_score=float(search.score(Xte, yte)),
                        predictions=search.predict(Xte).tolist(),
                        evaluations=search.fit_stats_["cross_validate_calls"],
                        generations=len(search.history["gen"]),
                        exhausted=getattr(search, "search_space_exhausted_", None),
                        selected=(
                            search.best_params_
                            if case.startswith("hpo")
                            else search.support_.tolist()
                        ),
                    )
                    rows.append(row)
                    print(json.dumps(row), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            dict(
                results=rows,
                population_size=args.population_size,
                generations=args.generations,
                source_root=str(args.source_root),
                python=platform.python_version(),
                platform=platform.platform(),
                versions={
                    name: importlib.metadata.version(name)
                    for name in dict.fromkeys(("numpy", "scikit-learn", "deap", *args.backends))
                },
            ),
            indent=2,
        ),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--seed-start", type=int, default=11)
    parser.add_argument("--population-size", type=int, default=20)
    parser.add_argument("--generations", type=int, default=40)
    parser.add_argument("--backends", nargs="+", choices=BACKENDS, default=["deap"])
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=(*CASES, *REGRESSION_CASES),
    )
    args = parser.parse_args()
    if args.seeds < 1 or args.seed_start < 0 or args.population_size < 2 or args.generations < 0:
        parser.error("use seeds >=1, seed-start >=0, population-size >=2 and generations >=0")
    run(args)


if __name__ == "__main__":
    main()
