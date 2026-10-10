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

from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits


def run(args):
    if args.source_root:
        sys.path.insert(0, str(args.source_root.resolve()))
    from sklearn_genetic import GASearchCV, GAFeatureSelectionCV
    from sklearn_genetic.space import Categorical, Continuous, Integer

    rows = []
    with threadpool_limits(limits=1):
        for case in args.cases or (
            "hpo-small",
            "hpo-medium",
            "hpo-continuous",
            "features-small",
            "features-medium",
            "features-wide",
        ):
            for seed in range(11, 11 + args.seeds):
                if case in ("features-medium", "features-wide"):
                    width = 8 if case == "features-medium" else 20
                    X, y = make_classification(
                        n_samples=240, n_features=width, n_informative=4, random_state=seed
                    )
                else:
                    X, y = load_iris(return_X_y=True)
                Xtr, Xte, ytr, yte = train_test_split(X, y, stratify=y, random_state=seed)
                cv = list(StratifiedKFold(3, shuffle=True, random_state=seed).split(Xtr, ytr))
                common = dict(
                    cv=cv,
                    scoring="accuracy",
                    population_size=args.population_size,
                    generations=args.generations,
                    n_jobs=1,
                    random_state=seed,
                    verbose=False,
                )
                model = DecisionTreeClassifier(max_depth=5, random_state=seed)
                if case.startswith("hpo"):
                    space = {
                        "max_depth": Integer(1, 3),
                        "criterion": Categorical(["gini", "entropy"]),
                    }
                    if case == "hpo-medium":
                        space["max_depth"] = Integer(1, 8)
                        space["min_samples_split"] = Integer(2, 4)
                    elif case == "hpo-continuous":
                        space["ccp_alpha"] = Continuous(0, 0.1)
                    search = GASearchCV(model, param_grid=space, **common)
                else:
                    maximum = {"features-small": 2, "features-medium": 3, "features-wide": 10}[case]
                    search = GAFeatureSelectionCV(model, max_features=maximum, **common)
                start = time.perf_counter()
                search.fit(Xtr, ytr)
                seconds = time.perf_counter() - start
                row = dict(
                    case=case,
                    seed=seed,
                    seconds=seconds,
                    cv_score=float(max(search.cv_results_["mean_test_score"])),
                    holdout_score=float(search.score(Xte, yte)),
                    predictions=search.predict(Xte).tolist(),
                    evaluations=search.fit_stats_["cross_validate_calls"],
                    generations=len(search.history["gen"]),
                    selected=(
                        search.best_params_ if case.startswith("hpo") else search.support_.tolist()
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
                    for name in ("numpy", "scikit-learn", "deap")
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
    parser.add_argument("--population-size", type=int, default=20)
    parser.add_argument("--generations", type=int, default=40)
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=[
            "hpo-small",
            "hpo-medium",
            "hpo-continuous",
            "features-small",
            "features-medium",
            "features-wide",
        ],
    )
    args = parser.parse_args()
    if args.seeds < 1 or args.population_size < 2 or args.generations < 0:
        parser.error("seeds must be >=1, population-size >=2 and generations >=0")
    run(args)


if __name__ == "__main__":
    main()
