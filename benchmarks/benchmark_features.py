r"""Paired feature-selection benchmark with 72 proposed candidates per run.

Both evolutionary algorithms share data, folds, estimator, population of 8,
and a maximum of half the input features. Default seeds are held apart from
exploratory seeds 101-103. A single thread bounds native model training.
PowerShell: .venv\Scripts\python.exe benchmarks/benchmark_features.py --output results.json
"""

import argparse
import platform
import importlib.metadata
import json
import time
from pathlib import Path
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from sklearn_genetic import GAFeatureSelectionCV


def run(args):
    rows = []
    with threadpool_limits(limits=1):
        for width in (20, 60, 120):
            for seed in range(args.seed_start, args.seed_start + args.seeds):
                X, y = make_classification(
                    n_samples=400,
                    n_features=width,
                    n_informative=8,
                    n_redundant=4,
                    random_state=seed,
                )
                Xtr, Xte, ytr, yte = train_test_split(X, y, stratify=y, random_state=seed)
                cv = list(StratifiedKFold(3, shuffle=True, random_state=seed).split(Xtr, ytr))
                for modelname in ("dt", "logistic"):
                    for method, algorithm, generations in [
                        ("genetic", "eaMuPlusLambda", 4),
                        ("differential", "eaDifferentialEvolution", 8),
                    ]:
                        model = (
                            DecisionTreeClassifier(max_depth=5, random_state=seed)
                            if modelname == "dt"
                            else make_pipeline(
                                StandardScaler(),
                                LogisticRegression(max_iter=500, random_state=seed),
                            )
                        )
                        search = GAFeatureSelectionCV(
                            model,
                            cv=cv,
                            population_size=8,
                            generations=generations,
                            random_state=seed,
                            verbose=False,
                            n_jobs=1,
                            max_features=width // 2,
                            algorithm=algorithm,
                        )
                        start = time.perf_counter()
                        search.fit(Xtr, ytr)
                        seconds = time.perf_counter() - start
                        row = dict(
                            width=width,
                            seed=seed,
                            model=modelname,
                            method=method,
                            cv_score=float(max(search.cv_results_["mean_test_score"])),
                            holdout_score=float(search.score(Xte, yte)),
                            seconds=seconds,
                            evaluations=search.fit_stats_["cross_validate_calls"],
                            features=int(search.support_.sum()),
                        )
                        assert row["evaluations"] <= 72
                        rows.append(row)
                        print(json.dumps(row), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        results=rows,
        candidate_budget=72,
        seed_start=args.seed_start,
        seeds=args.seeds,
        python=platform.python_version(),
        platform=platform.platform(),
        versions={
            name: importlib.metadata.version(name) for name in ("numpy", "scikit-learn", "deap")
        },
    )
    args.output.write_text(json.dumps(payload, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=301)
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("seeds must be >=1")
    run(args)


if __name__ == "__main__":
    main()
