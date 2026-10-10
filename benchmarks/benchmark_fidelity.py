r"""Paired full-training versus subsampled-training evolutionary workflows.

All methods preserve the outer holdout and CV validation rows. Reduced-training
searches promote distinct finalists to full-training CV before final refit.
Reported fitting time includes promotion and refit, excluding holdout prediction.
PowerShell: .venv\Scripts\python.exe benchmarks\benchmark_fidelity.py --output fidelity.json
"""

import argparse
import json
import importlib.metadata
import platform
import time
from pathlib import Path

from sklearn.base import clone
from sklearn.datasets import load_breast_cancer, load_diabetes, load_digits, make_classification
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from threadpoolctl import threadpool_limits
from sklearn_genetic import GASearchCV, GAFeatureSelectionCV, subsample_cv
from sklearn_genetic.space import Continuous, Integer


def task(name, modelname, seed, samples):
    if name == "synthetic":
        X, y = make_classification(
            n_samples=samples,
            n_features=40,
            n_informative=10,
            n_redundant=8,
            flip_y=0.05,
            random_state=113,
        )
    else:
        X, y = {"breast": load_breast_cancer, "digits": load_digits, "diabetes": load_diabetes}[
            name
        ](return_X_y=True)
    classification = name != "diabetes"
    if modelname == "svm":
        model = make_pipeline(
            StandardScaler(),
            (SVC(C=1, gamma="scale") if classification else SVR(C=100, gamma="scale")),
        )
        prefix = "svc" if classification else "svr"
        space = {
            prefix + "__C": Continuous(0.01, 1000, distribution="log-uniform"),
            prefix + "__gamma": Continuous(0.0001, 1, distribution="log-uniform"),
        }
    else:
        cls = RandomForestClassifier if classification else RandomForestRegressor
        model = cls(n_estimators=80, n_jobs=1, random_state=seed)
        space = {
            "max_depth": Integer(2, 20),
            "min_samples_leaf": Integer(1, 15),
            "max_features": Continuous(0.1, 1.0),
        }
    return X, y, classification, model, space


def run(args):
    payload = dict(
        label=args.label,
        config=vars(args) | {"output": str(args.output)},
        python=platform.python_version(),
        platform=platform.platform(),
        versions={
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "scikit-learn", "deap")
        },
        results=[],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1):
        for name in args.datasets:
            for modelname in args.models:
                for seed in range(args.seed_start, args.seed_start + args.seeds):
                    X, y, classification, model, space = task(name, modelname, seed, args.samples)
                    Xtr, Xte, ytr, yte = train_test_split(
                        X,
                        y,
                        test_size=0.25,
                        random_state=seed,
                        stratify=y if classification else None,
                    )
                    splitter = StratifiedKFold if classification else KFold
                    full_cv = list(splitter(3, shuffle=True, random_state=seed).split(Xtr, ytr))
                    scoring = "accuracy" if classification else "neg_mean_squared_error"
                    for kind in args.kinds:
                        for fraction in args.fractions:
                            start = time.perf_counter()
                            cv = subsample_cv(
                                full_cv,
                                fraction,
                                stratify=ytr if classification else None,
                                random_state=seed,
                            )
                            common = dict(
                                cv=cv,
                                population_size=8,
                                generations=args.generations,
                                random_state=seed,
                                n_jobs=1,
                                verbose=False,
                                scoring=scoring,
                                error_score="raise",
                                final_selection=fraction < 1,
                                final_selection_top_k=args.finalists,
                                final_selection_cv=full_cv,
                            )
                            search = (
                                GASearchCV(clone(model), param_grid=space, **common)
                                if kind == "hpo"
                                else GAFeatureSelectionCV(clone(model), **common)
                            )
                            search.fit(Xtr, ytr)
                            seconds = time.perf_counter() - start
                            pred = search.predict(Xte)
                            selected_features = (
                                Xtr.shape[1] if kind == "hpo" else int(search.support_.sum())
                            )
                            proxy_cv = float(max(search.cv_results_["mean_test_score"]))
                            full_score = (
                                float(search.best_score_)
                                if kind == "hpo" or fraction < 1
                                else proxy_cv
                            )
                            promotion_calls = (
                                len(search.final_selection_results_["candidates"])
                                if fraction < 1
                                else 0
                            )
                            row = dict(
                                dataset=name,
                                model=modelname,
                                kind=kind,
                                seed=seed,
                                fraction=fraction,
                                seconds=seconds,
                                cv_score=full_score,
                                holdout_score=float(
                                    accuracy_score(yte, pred)
                                    if classification
                                    else -mean_squared_error(yte, pred)
                                ),
                                selected_features=selected_features,
                                proxy_cv=proxy_cv,
                                selected_params=search.best_params_ if kind == "hpo" else None,
                                selected_mask=(
                                    search.support_.tolist() if kind == "features" else None
                                ),
                                search_cv_calls=search.fit_stats_["cross_validate_calls"],
                                promotion_cv_calls=promotion_calls,
                                train_fold_sizes=[len(train) for train, _ in cv],
                                full_fold_sizes=[len(train) for train, _ in full_cv],
                            )
                            payload["results"].append(row)
                            temporary = args.output.with_suffix(args.output.suffix + ".tmp")
                            temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                            temporary.replace(args.output)
                            print(json.dumps(row), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--datasets",
        nargs="+",
        choices=["synthetic", "breast", "digits", "diabetes"],
        default=["synthetic", "breast", "digits", "diabetes"],
    )
    p.add_argument("--models", nargs="+", choices=["svm", "rf"], default=["svm"])
    p.add_argument("--kinds", nargs="+", choices=["hpo", "features"], default=["hpo", "features"])
    p.add_argument("--fractions", nargs="+", type=float, default=[1, 0.25, 0.125])
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--seed-start", type=int, default=1201)
    p.add_argument("--samples", type=int, default=2000)
    p.add_argument("--generations", type=int, default=4)
    p.add_argument("--finalists", type=int, default=4)
    p.add_argument("--label", default="public-fidelity")
    args = p.parse_args()
    if (
        args.seeds < 1
        or args.generations < 0
        or args.finalists < 1
        or any(not 0 < fraction <= 1 for fraction in args.fractions)
    ):
        p.error(
            "seeds/finalists must be positive, generations nonnegative, and fractions in (0, 1]"
        )
    run(args)


if __name__ == "__main__":
    main()
