r"""Paired feature-selection benchmark with 72 proposed candidates per run.

Both evolutionary algorithms share data, folds, estimator, population of 8,
and a maximum of half the input features. Default seeds are held apart from
exploratory seeds 101-103. A single thread bounds native model training.
Use --methods genetic differential optuna flaml --unconstrained to compare
all optimizers over nonempty masks without a feature-count cap. Optuna and
FLAML are used as binary-mask optimizers, not their full AutoML workflows.
PowerShell: .venv\Scripts\python.exe benchmarks/benchmark_features.py --output results.json
"""

import argparse
import platform
import importlib.metadata
import json
import time
from pathlib import Path
import numpy as np
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from sklearn_genetic import GAFeatureSelectionCV


def optimize_mask(method, model, X, y, cv, seed, budget):
    """Apply an HPO optimizer to the same unconstrained binary mask domain."""
    cache = {}
    calls = 0
    invalid = 0

    def evaluate(config):
        nonlocal calls, invalid
        calls += 1
        mask = tuple(bool(config[f"f{i}"]) for i in range(X.shape[1]))
        if not any(mask):
            invalid += 1
            return -float("inf")
        if mask not in cache:
            cache[mask] = float(
                np.mean(
                    cross_val_score(
                        clone(model),
                        X[:, mask],
                        y,
                        cv=cv,
                        scoring="accuracy",
                        n_jobs=1,
                        error_score="raise",
                    )
                )
            )
        return cache[mask]

    if method == "optuna":
        import optuna

        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
        )
        study.optimize(
            lambda trial: evaluate(
                {f"f{i}": trial.suggest_categorical(f"f{i}", [0, 1]) for i in range(X.shape[1])}
            ),
            n_trials=budget,
        )
    else:
        from flaml import CFO, tune

        space = {f"f{i}": tune.choice([0, 1]) for i in range(X.shape[1])}
        tune.run(
            lambda config: {"score": evaluate(config)},
            config=space,
            metric="score",
            mode="max",
            search_alg=CFO(space=space, metric="score", mode="max", seed=seed),
            num_samples=budget,
            use_ray=False,
            verbose=0,
        )
    mask = max(cache, key=cache.get)
    best = clone(model).fit(X[:, mask], y)
    return best, np.asarray(mask), cache[mask], len(cache), calls, invalid


def run(args):
    if "optuna" in args.methods:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
    if "flaml" in args.methods:
        import flaml  # Import before timing any method.

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
                    for method in args.methods:
                        model = (
                            DecisionTreeClassifier(max_depth=5, random_state=seed)
                            if modelname == "dt"
                            else make_pipeline(
                                StandardScaler(),
                                LogisticRegression(max_iter=500, random_state=seed),
                            )
                        )
                        start = time.perf_counter()
                        if method in ("genetic", "differential"):
                            search = GAFeatureSelectionCV(
                                model,
                                cv=cv,
                                population_size=8,
                                generations=4 if method == "genetic" else 8,
                                random_state=seed,
                                verbose=False,
                                n_jobs=1,
                                max_features=None if args.unconstrained else width // 2,
                                algorithm=(
                                    "eaMuPlusLambda"
                                    if method == "genetic"
                                    else "eaDifferentialEvolution"
                                ),
                            )
                            search.fit(Xtr, ytr)
                            best, mask = search.best_estimator_, search.support_
                            score = float(max(search.cv_results_["mean_test_score"]))
                            evaluations = search.fit_stats_["cross_validate_calls"]
                            proposals = search.fit_stats_["evaluated_candidates"]
                            invalid = search.fit_stats_["skipped_invalid_candidates"]
                        else:
                            best, mask, score, evaluations, proposals, invalid = optimize_mask(
                                method, model, Xtr, ytr, cv, seed, 72
                            )
                        seconds = time.perf_counter() - start
                        row = dict(
                            width=width,
                            seed=seed,
                            model=modelname,
                            method=method,
                            cv_score=score,
                            holdout_score=float(best.score(Xte[:, mask], yte)),
                            seconds=seconds,
                            evaluations=evaluations,
                            proposals=proposals,
                            invalid_proposals=invalid,
                            features=int(mask.sum()),
                        )
                        assert row["evaluations"] <= row["proposals"] <= 72
                        rows.append(row)
                        print(json.dumps(row), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        results=rows,
        candidate_budget=72,
        max_features="unconstrained" if args.unconstrained else "half",
        seed_start=args.seed_start,
        seeds=args.seeds,
        python=platform.python_version(),
        platform=platform.platform(),
        versions={
            name: importlib.metadata.version(name)
            for name in ["numpy", "scikit-learn", "deap"]
            + [name for name in ("optuna", "flaml") if name in args.methods]
        },
    )
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=301)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["genetic", "differential", "optuna", "flaml"],
        default=["genetic", "differential"],
    )
    parser.add_argument("--unconstrained", action="store_true")
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("seeds must be >=1")
    if any(method in args.methods for method in ("optuna", "flaml")) and not args.unconstrained:
        parser.error("HPO mask comparisons require --unconstrained for the same binary domain")
    run(args)


if __name__ == "__main__":
    main()
