"""Run from the checkout: python -m benchmarks.benchmark_optimizer_catalog --help."""

import argparse
import csv
import json
import platform
import sys
from collections import Counter
from contextlib import ExitStack, redirect_stdout
from functools import partial
from importlib.metadata import version
from importlib.util import find_spec
from itertools import combinations, product
from math import isfinite
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np
from sklearn.base import clone
from sklearn.datasets import (
    load_breast_cancer,
    load_diabetes,
    load_wine,
    make_classification,
    make_regression,
    make_friedman1,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import get_scorer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from threadpoolctl import threadpool_limits

from sklearn_genetic import GAFeatureSelectionCV, GASearchCV
from sklearn_genetic.callbacks import CVBudgetStopping
from sklearn_genetic.evaluation import candidate_parameters, candidate_records
from sklearn_genetic.space import Categorical, Continuous, Integer
from sklearn_genetic.utils._score_statistics import score_mean

_ENGINE_NAMES = ("deap", "scipy", "nevergrad", "cmaes", "pymoo")
_PILOT_VARIANTS = {
    "nevergrad": ("TwoPointsDE", "OnePlusOne"),
    "cmaes": (".15", ".25", ".4"),
    "pymoo": (".5", ".8", "1"),
}
_CATALOG_CASES = (
    "tumor-params",
    "cancer",
    "diabetes",
    "diabetes-svr",
    "sparse-easy",
    "sparse-hard",
    "linear-easy",
    "linear-hard",
    "friedman",
)
_PILOT_CASES = ("wine", "wine-mask", "calibration-cls", "linear-calibration")
# Synthetic data use offset seeds to keep generation separate from splitting.
_MAX_DATA_SEED_OFFSET = 40_000


def random_search(problem, evaluate, report):
    """Keep the shared initial population and draw independent later proposals."""
    rng = np.random.default_rng(problem.seed)
    size = len(problem.initial_population)
    width = len(problem.initial_population[0])
    cap = problem.max_features or width

    def sample():
        if problem.task == "parameters":
            return [dim.sample() for dim in problem.dimensions]
        mask = np.zeros(width, dtype=int)
        count = rng.integers(1, cap + 1)
        mask[rng.choice(width, count, replace=False)] = 1
        return mask.tolist()

    report([handle for _, handle in evaluate(problem.initial_population)])
    for _ in range(problem.generations):
        report([handle for _, handle in evaluate([sample() for _ in range(size)])])


def tpe_search(problem, evaluate, report):
    """Run sequential Optuna TPE on the catalog's tuning spaces."""
    import optuna

    previous_verbosity = optuna.logging.get_verbosity()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    try:
        study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=problem.seed))
    finally:
        optuna.logging.set_verbosity(previous_verbosity)

    for row in problem.initial_population:
        study.enqueue_trial(dict(zip(problem.parameter_names, row, strict=True)))

    def suggest(trial, name, dim):
        if isinstance(dim, Categorical):
            return trial.suggest_categorical(name, dim.choices)
        if isinstance(dim, Integer):
            return trial.suggest_int(name, dim.lower, dim.upper)
        return trial.suggest_float(
            name, dim.lower, dim.upper, log=dim.distribution == "log-uniform"
        )

    # CV order and timings remain visible to the catalog, and TPE can learn
    # from each completed trial before asking for the next one.
    for _ in range(problem.generations + 1):
        handles = []
        for _ in problem.initial_population:
            trial = study.ask()
            candidate = [
                suggest(trial, name, dim)
                for name, dim in zip(problem.parameter_names, problem.dimensions, strict=True)
            ]
            ((loss, handle),) = evaluate([candidate])
            study.tell(trial, loss)
            handles.append(handle)
        report(handles)


def is_mask_case(case):
    return case.startswith(("sparse", "linear")) or case in ("cancer", "wine-mask")


def make_search(case, backend, variant, seed, cache, pilot, proposal_budget, public_refit):
    classification = not case.startswith(("diabetes", "linear", "friedman"))
    masks = is_mask_case(case)
    if case.startswith(("sparse", "calibration-cls")):
        difficulty = {
            "sparse-easy": (1.6, 0.01),
            "sparse-hard": (0.65, 0.12),
            "calibration-cls": (1.0, 0.06),
        }
        separation, label_noise = difficulty[case]
        X, y = make_classification(
            n_samples=300,
            n_features=40,
            n_informative=6,
            n_redundant=4,
            class_sep=separation,
            flip_y=label_noise,
            random_state=20_000 + seed,
        )
    elif case.startswith("linear"):
        rank, noise = {
            "linear-easy": (None, 5),
            "linear-hard": (8, 8),
            "linear-calibration": (16, 5),
        }[case]
        X, y = make_regression(
            n_samples=280,
            n_features=36,
            n_informative=8,
            effective_rank=rank,
            noise=noise,
            random_state=30_000 + seed,
        )
    elif case == "friedman":
        X, y = make_friedman1(
            n_samples=300, n_features=24, noise=1.5, random_state=_MAX_DATA_SEED_OFFSET + seed
        )
    elif case.startswith("diabetes"):
        X, y = load_diabetes(return_X_y=True, scaled=False)
    elif case in ("wine", "wine-mask"):
        X, y = load_wine(return_X_y=True)
    else:
        X, y = load_breast_cancer(return_X_y=True)
    X, hold_X, y, hold_y = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y if classification else None
    )
    scoring = "balanced_accuracy" if classification else "neg_root_mean_squared_error"
    model = (DecisionTreeClassifier if classification else DecisionTreeRegressor)(random_state=seed)
    if case.startswith("sparse"):
        model = LogisticRegression(solver="liblinear", random_state=seed)
    elif case.startswith("linear"):
        model = Ridge(alpha=0.1)
    elif case == "diabetes-svr":
        model = make_pipeline(StandardScaler(), SVR())
    engine = {"random": random_search, "tpe": tpe_search}.get(backend, backend)
    prob = 0.8
    if pilot:
        if backend == "cmaes":
            from sklearn_genetic._cmaes_backend import optimize as cmaes

            engine = partial(cmaes, sigma=float(variant))
        elif backend == "nevergrad":
            from sklearn_genetic._nevergrad_backend import optimize as nevergrad

            engine = partial(nevergrad, strategy=variant)
        elif backend == "pymoo":
            from sklearn_genetic._pymoo_backend import optimize as pymoo

            prob, engine = float(variant), pymoo
    size = 8
    generations = proposal_budget // size - 1
    options = dict(
        estimator=model,
        cv=(StratifiedKFold if classification else KFold)(3, shuffle=True, random_state=seed),
        random_state=seed,
        scoring=scoring,
        optimizer_backend=engine,
        n_jobs=1,
        population_size=size,
        generations=generations,
        crossover_probability=prob,
        use_cache=cache,
        refit=public_refit,
        verbose=False,
    )
    if masks:
        search = GAFeatureSelectionCV(max_features=max(2, X.shape[1] // 3), **options)
    elif case == "diabetes-svr":
        grid = {
            name: Continuous(lo, hi, distribution="log-uniform")
            for name, lo, hi in (
                ("svr__C", 0.1, 1000),
                ("svr__gamma", 0.0001, 1),
                ("svr__epsilon", 0.01, 10),
            )
        }
        search = GASearchCV(param_grid=grid, **options)
    else:
        grid = dict(
            max_depth=Integer(2, 12),
            min_samples_leaf=Integer(1, 12),
            ccp_alpha=Continuous(1e-6, 0.1, distribution="log-uniform"),
            criterion=Categorical(
                ["gini", "entropy"] if classification else ["squared_error", "absolute_error"]
            ),
        )
        search = GASearchCV(param_grid=grid, **options)
    return search, X, y, hold_X, hold_y, scoring, masks


def measure(
    case,
    backend,
    variant,
    seed,
    cache,
    pilot,
    target=32,
    *,
    public_refit=False,
    proposal_budget=None,
    stop_at_budget=False,
):
    if proposal_budget is None:
        proposal_budget = 2 * target if stop_at_budget else target
    search, X, y, hold_X, hold_y, scoring, masks = make_search(
        case, backend, variant, seed, cache, pilot, proposal_budget, public_refit
    )
    callbacks = CVBudgetStopping(target) if stop_at_budget else None
    measured = []
    cv_seconds = 0
    evaluate_cv = search._cross_validate_candidate

    def clock_cv(*args, **kwargs):
        nonlocal cv_seconds
        before = perf_counter()
        result = evaluate_cv(*args, **kwargs)
        finished = perf_counter()
        cv_seconds += finished - before
        measured.append((result[f"test_{search.refit_metric}"], finished - start, cv_seconds))
        return result

    # Keep the native candidate evaluator in place so its optimized paths run.
    with patch.object(search, "_cross_validate_candidate", clock_cv):
        start = perf_counter()
        search.fit(X, y, callbacks=callbacks)
        total = perf_counter() - start

    cv_calls = len(measured)
    if not cv_calls or cv_calls != search.fit_stats_["cross_validate_calls"]:
        raise RuntimeError(f"CV count mismatch: {case}, {backend}, {seed}")
    if stop_at_budget and cv_calls > target:
        raise RuntimeError(f"CV budget exceeded: {case}, {backend}, {seed}")

    # Cache hits retain their original fold-score array. Link each measured
    # call to its first candidate row and recover the coordinates sent to CV.
    recorded = {}
    for row in candidate_records(search):
        recorded.setdefault(id(row["cv_scores"]), row)
    trace = []
    for fold_scores, elapsed, cv_time in measured:
        row = recorded.pop(id(fold_scores), None)
        if row is None:
            raise RuntimeError(f"CV result missing its candidate record: {case}, {backend}, {seed}")
        if masks:
            coords = tuple(map(bool, row["features"]))
        else:
            params = candidate_parameters(row)
            coords = tuple(params[name] for name in search.param_grid)
        trace.append((float(row["score"]), coords, elapsed, cv_time))
    return (trace, search.estimator, X, y, hold_X, hold_y, scoring, masks, search, total)


def prefix_winner(run, budget):
    trace, _, _, _, _, _, _, masks, _, _ = run
    candidates = (row for row in trace[:budget] if isfinite(row[0]))
    return max(candidates, key=lambda row: (row[0], -sum(row[1]) if masks else 0), default=None)


def holdout_score(run, coords):
    _, model, X, y, hold_X, hold_y, scoring, masks, search, _ = run
    model = clone(model)
    if masks:
        columns = np.asarray(coords, dtype=bool)
        X, hold_X = X[:, columns], hold_X[:, columns]
    else:
        model.set_params(**dict(zip(search.param_grid, coords, strict=True)))
    return get_scorer(scoring)(model.fit(X, y), hold_X, hold_y), X.shape[1]


def summarize(run, budget):
    trace, _, _, y, _, hold_y, scoring, masks, search, total = run
    observed = trace[:budget]
    best = held_out = selected = first_best = first_finite = running_best = None
    scores = np.asarray([row[0] for row in observed])
    finite = np.isfinite(scores)
    winner = prefix_winner(run, budget)
    if winner is not None:
        best, coords, _, _ = winner
        held_out, selected = holdout_score(run, coords)
        first_best = 1 + int(np.argmax(scores == best))
        first_finite = 1 + int(np.argmax(finite))
        best_so_far = np.maximum.accumulate(np.where(finite, scores, -np.inf))
        running_best = float(score_mean(best_so_far[first_finite - 1 :]))
    if scoring == "balanced_accuracy":
        baseline = 1.0 / len(np.unique(y))
        skill = (held_out - baseline) / (1 - baseline) if winner is not None else None
    else:
        baseline = -np.sqrt(np.mean((hold_y - np.mean(y)) ** 2))
        skill = 1 - held_out / baseline if winner is not None else None
    elapsed, cv_time = observed[-1][2:]
    return dict(
        cv_calls=budget,
        full_cv_calls=len(trace),
        full_proposals=search.fit_stats_["evaluated_candidates"],
        full_cache_hits=search.fit_stats_["cache_hits"],
        full_duplicates=search.fit_stats_["duplicate_candidates"],
        finite_cv_calls=int(finite.sum()),
        first_finite_cv_call=first_finite,
        first_best_call=first_best,
        unique_prefix=len({x[1] for x in observed}),
        best_cv=best,
        best_first_eight=max(scores[:8][finite[:8]], default=None),
        mean_running_best=running_best,
        holdout=held_out,
        holdout_baseline=baseline,
        holdout_skill=skill,
        selected=selected if masks else "",
        prefix_ms=1000 * elapsed,
        prefix_outside_cv_ms=1000 * (elapsed - cv_time),
        full_outside_cv_ms=1000 * (total - trace[-1][3]),
        full_ms=1000 * total,
        full_search_ms=1000 * search.search_time_,
    )


def prefix_scores(run, calls, refits):
    if not calls:
        return {"best_cv": None, "holdout": None, "elapsed_ms": None}
    elapsed_ms = 1000 * run[0][calls - 1][2]
    winner = prefix_winner(run, calls)
    if winner is None:
        return dict(best_cv=None, holdout=None, elapsed_ms=elapsed_ms)
    best, coords, _, _ = winner
    if coords not in refits:
        refits[coords] = holdout_score(run, coords)[0]
    return dict(best_cv=best, holdout=refits[coords], elapsed_ms=elapsed_ms)


def _prefix_checkpoint_fields(run, checkpoints, available, refits, *, prefix=""):
    fields = {}
    for calls in checkpoints:
        scores = prefix_scores(run, calls if calls <= available else 0, refits)
        fields.update({f"{prefix}cv_{calls}_{key}": value for key, value in scores.items()})
    return fields


def comparison_checkpoints(run, cohort, target, deadlines, refits=None, *, cv_checkpoints=()):
    trace = run[0]
    if refits is None:
        refits = {}
    winner = prefix_winner(run, cohort["cv_calls"])
    if winner is not None:
        refits[winner[1]] = cohort["holdout"]

    scores = prefix_scores(run, target if len(trace) >= target else 0, refits)
    fields = {f"target_{key}": val for key, val in scores.items()}
    fields.update(_prefix_checkpoint_fields(run, cv_checkpoints, len(trace), refits))
    for deadline in deadlines:
        calls = sum(elapsed * 1000 <= deadline for _, _, elapsed, _ in trace[:target])
        fields[f"time_{deadline}ms_cv_calls"] = calls
        scores = prefix_scores(run, calls, refits)
        fields.update({f"time_{deadline}ms_{key}": val for key, val in scores.items()})
    return fields


def _budget_status(search, calls, target):
    if calls == target:
        return "target_reached"
    if search.search_space_exhausted_:
        return "space_exhausted"
    return "below_target"


def _comparison_names(engines, references, *, engine_pairs):
    return combinations(engines, 2) if engine_pairs else product(engines, references)


def pair_comparisons(
    runs,
    engines,
    references,
    target,
    refits,
    *,
    stopped=False,
    cv_checkpoints=(),
    engine_pairs=False,
):
    """Compare two optimizers at their largest shared prefix up to target."""
    for engine, reference in _comparison_names(engines, references, engine_pairs=engine_pairs):
        budget = min(target, len(runs[engine][0]), len(runs[reference][0]))
        row = dict(
            backend=engine,
            reference=reference,
            matched_cv_calls=budget,
            completed_target=budget == target,
        )
        row.update({f"cv_{calls}_completed": budget >= calls for calls in cv_checkpoints})
        for side, name in (("backend", engine), ("reference", reference)):
            run = runs[name]
            run_refits = refits.setdefault(name, {})
            scores = prefix_scores(run, budget, run_refits)
            row[f"{side}_cv_calls"] = len(run[0])
            row[f"{side}_best_cv"] = scores["best_cv"]
            row[f"{side}_prefix_holdout"] = scores["holdout"]
            row[f"{side}_prefix_ms"] = scores["elapsed_ms"]
            row.update(
                _prefix_checkpoint_fields(
                    run, cv_checkpoints, budget, run_refits, prefix=f"{side}_"
                )
            )
            if stopped:
                search, total = run[-2:]
                row[f"{side}_budget_status"] = _budget_status(search, len(run[0]), target)
                row[f"{side}_fit_ms"] = 1000 * total
                row[f"{side}_search_ms"] = 1000 * search.search_time_
                row[f"{side}_refit_ms"] = 1000 * search.refit_time_ if search.refit else None
        yield row


def _selected_coordinates(search, masks):
    if masks:
        return tuple(search.get_support())
    return tuple(search.best_params_[name] for name in search.param_grid)


def summarize_stopped_search(run, target):
    trace, _, _, _, hold_X, hold_y, scoring, masks, search, _ = run
    coords = _selected_coordinates(search, masks)
    returned_cv = search.best_score_ if isfinite(search.best_score_) else None
    held_out = None
    if search.refit:
        held_out = get_scorer(scoring)(search, hold_X, hold_y)
    elif returned_cv is not None:
        held_out = holdout_score(run, coords)[0]
    winner = prefix_winner(run, len(trace))
    return dict(
        completed_target=len(trace) == target,
        budget_status=_budget_status(search, len(trace), target),
        returned_cv=returned_cv,
        returned_holdout=held_out,
        returned_matches_trace_winner=coords == winner[1] if winner is not None else None,
    )


def summarize_public_refit(run, budget):
    trace, _, _, _, hold_X, hold_y, scoring, masks, search, _ = run
    coords = _selected_coordinates(search, masks)
    return dict(
        public_holdout=get_scorer(scoring)(search, hold_X, hold_y),
        public_cv=search.best_score_ if isfinite(search.best_score_) else None,
        public_selected=int(sum(coords)) if masks else "",
        public_winner_in_prefix=any(candidate == coords for _, candidate, *_ in trace[:budget]),
        public_refit_ms=1000 * search.refit_time_,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=(*_CATALOG_CASES, *_PILOT_CASES),
        help="Run the selected datasets and tasks",
    )
    parser.add_argument(
        "--engines",
        nargs="+",
        choices=_ENGINE_NAMES,
        help="Run only these optimization engines, in addition to requested references",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, help="Use these seeds for data splits and searches"
    )
    parser.add_argument("--budget", type=int, choices=(32, 96), default=32)
    parser.add_argument(
        "--stop-at-budget",
        action="store_true",
        help="Stop each fit at the specified number of fresh search CV calls",
    )
    parser.add_argument(
        "--proposal-budget",
        type=int,
        help="Nominal proposal count used to set the generation limit; defaults to twice the CV target when capped",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--public-refit",
        action="store_true",
        help="Also score the winner from the fitted estimator's public prediction API",
    )
    parser.add_argument("--references", nargs="+", choices=("random", "tpe"), default=[])
    parser.add_argument(
        "--pairs-output",
        type=Path,
        help="Write matched CV prefixes for engine pairs or engine/reference pairs to a CSV",
    )
    parser.add_argument("--time-ms", nargs="+", type=int, default=[])
    parser.add_argument(
        "--cv-checkpoints",
        nargs="+",
        type=int,
        default=[],
        help="Record additional score and time checkpoints at these fresh CV call counts",
    )
    args = parser.parse_args()
    proposal_budget = args.proposal_budget
    if proposal_budget is None:
        proposal_budget = 2 * args.budget if args.stop_at_budget else args.budget
    if proposal_budget < args.budget or proposal_budget % 8:
        parser.error("--proposal-budget must be a multiple of 8 and at least --budget")
    references = list(dict.fromkeys(args.references))
    deadlines = sorted(set(args.time_ms))
    if any(ms <= 0 for ms in deadlines):
        parser.error("--time-ms needs positive milliseconds")
    checkpoints = sorted(set(args.cv_checkpoints))
    if any(not 1 <= calls <= args.budget for calls in checkpoints):
        parser.error("--cv-checkpoints must be between 1 and --budget")
    if args.pilot and references:
        parser.error("--references compares the default engines; omit --pilot")
    variants = _PILOT_VARIANTS if args.pilot else dict.fromkeys(_ENGINE_NAMES, ("default",))
    engine_names = list(dict.fromkeys(args.engines or variants))
    if any(engine not in variants for engine in engine_names):
        parser.error(f"--pilot can only use these engines: {', '.join(_PILOT_VARIANTS)}")
    engines = [(engine, variant) for engine in engine_names for variant in variants[engine]]
    engine_pairs = args.pairs_output is not None and not references
    if engine_pairs and (args.pilot or len(engine_names) < 2):
        parser.error("Engine pairs require at least two default engines without --pilot")
    pair_names = tuple(_comparison_names(engine_names, references, engine_pairs=engine_pairs))
    default_seeds = (7, 19, 31) if args.pilot else (41, 59, 73, 97, 113)
    seeds = list(dict.fromkeys(args.seeds or default_seeds))
    max_seed = np.iinfo(np.uint32).max - _MAX_DATA_SEED_OFFSET
    if any(seed < 0 or seed > max_seed for seed in seeds):
        parser.error(f"--seeds must be between 0 and {max_seed}")
    cases = list(dict.fromkeys(args.cases or (_PILOT_CASES if args.pilot else _CATALOG_CASES)))
    if "tpe" in references:
        if all(map(is_mask_case, cases)):
            parser.error("TPE needs at least one tuning case")
        if find_spec("optuna") is None:
            parser.error("Install sklearn-genetic-opt[benchmark] for the TPE reference")
    partial_path = args.output.with_suffix(args.output.suffix + ".partial")
    pairs_path = None
    if args.pairs_output is not None:
        pairs_path = args.pairs_output.with_suffix(args.pairs_output.suffix + ".partial")
        outputs = (args.output, partial_path, args.pairs_output, pairs_path)
        if len({path.resolve() for path in outputs}) != len(outputs):
            parser.error(
                "--output and --pairs-output need distinct paths, including .partial files"
            )
    count = 0
    cohort_completions = Counter()
    pair_completions = Counter()
    writer = None
    pair_writer = None
    with threadpool_limits(1), ExitStack() as files:
        if args.stop_at_budget:
            # Generation callbacks can report progress while fit is running.
            files.enter_context(redirect_stdout(sys.stderr))
        out = files.enter_context(partial_path.open("w", newline=""))
        pairs_out = files.enter_context(pairs_path.open("w", newline="")) if pairs_path else None
        # Pay import and solver setup costs once outside the timed runs.
        for engine, variant in [*engines, *((name, "reference") for name in references)]:
            measure("wine", engine, variant, 3, False, args.pilot)
        for cache in ((False,) if args.pilot else (False, True)):
            for case in cases:
                active_references = [
                    name for name in references if name != "tpe" or not is_mask_case(case)
                ]
                active = [*engines, *((name, "reference") for name in active_references)]
                for i, seed in enumerate(seeds):
                    offset = i % len(active)
                    order = active[offset:] + active[:offset]
                    runs = [
                        (
                            name,
                            v,
                            measure(
                                case,
                                name,
                                v,
                                seed,
                                cache,
                                args.pilot,
                                args.budget,
                                public_refit=args.public_refit,
                                proposal_budget=proposal_budget,
                                stop_at_budget=args.stop_at_budget,
                            ),
                        )
                        for name, v in order
                    ]
                    refits = {}
                    budget = min(args.budget, *(len(run[0]) for _, _, run in runs))
                    if pair_names:
                        cohort_completions[cache, budget == args.budget] += 1
                        call_counts = {name: len(run[0]) for name, _, run in runs}
                        for engine, reference in pair_names:
                            if reference not in call_counts:
                                continue
                            key = (cache, engine, reference)
                            shared = min(call_counts[engine], call_counts[reference])
                            pair_completions[(*key, "eligible")] += 1
                            pair_completions[(*key, "complete")] += shared >= args.budget
                            for calls in checkpoints:
                                pair_completions[(*key, calls)] += shared >= calls
                    for backend, variant, run in runs:
                        row = dict(
                            case=case,
                            target=args.budget,
                            seed=seed,
                            cache=cache,
                            backend=backend,
                            variant=variant,
                            **summarize(run, budget),
                        )
                        if args.public_refit:
                            row.update(summarize_public_refit(run, budget))
                        if args.stop_at_budget:
                            row.update(summarize_stopped_search(run, args.budget))
                        if pair_names or deadlines or checkpoints:
                            row.update(
                                comparison_checkpoints(
                                    run,
                                    row,
                                    args.budget,
                                    deadlines,
                                    refits.setdefault(backend, {}),
                                    cv_checkpoints=checkpoints,
                                )
                            )
                        if writer is None:
                            writer = csv.DictWriter(out, fieldnames=list(row), lineterminator="\n")
                            writer.writeheader()
                        writer.writerow(row)
                        count += 1
                    if pairs_out is not None:
                        runs_by_name = {name: run for name, _, run in runs}
                        for pair in pair_comparisons(
                            runs_by_name,
                            (name for name, _ in engines),
                            active_references,
                            args.budget,
                            refits,
                            stopped=args.stop_at_budget,
                            cv_checkpoints=checkpoints,
                            engine_pairs=engine_pairs,
                        ):
                            row = dict(
                                case=case, target=args.budget, seed=seed, cache=cache, **pair
                            )
                            if pair_writer is None:
                                pair_writer = csv.DictWriter(
                                    pairs_out, fieldnames=list(row), lineterminator="\n"
                                )
                                pair_writer.writeheader()
                            pair_writer.writerow(row)
                        pairs_out.flush()
                    out.flush()
    partial_path.replace(args.output)
    if pairs_path is not None:
        pairs_path.replace(args.pairs_output)
    packages = ["scikit-learn", "numpy", "deap", "scipy"]
    packages.extend(engine for engine in _PILOT_VARIANTS if engine in engine_names)
    if "tpe" in references:
        packages.append("optuna")
    metadata = dict(
        python=platform.python_version(),
        platform=platform.platform(),
        packages={package: version(package) for package in packages},
        rows=count,
        trace_timing="candidate_cv_return",
    )
    if args.stop_at_budget or args.proposal_budget is not None:
        metadata.update(
            evaluation_mode="capped" if args.stop_at_budget else "prefix",
            proposal_budget=proposal_budget,
            target_cv_calls=args.budget,
        )
    if checkpoints:
        metadata.update(cv_checkpoints=checkpoints, target_cv_calls=args.budget)
    if pair_names:
        metadata["target_cv_calls"] = args.budget
        metadata["cohorts"] = [
            dict(
                cache=cache,
                complete=cohort_completions[cache, True],
                eligible=cohort_completions[cache, True] + cohort_completions[cache, False],
            )
            for cache in (False, True)
        ]
        metadata["pairs"] = [
            dict(
                cache=cache,
                backend=engine,
                reference=reference,
                complete=pair_completions[cache, engine, reference, "complete"],
                eligible=pair_completions[cache, engine, reference, "eligible"],
            )
            for cache in (False, True)
            for engine, reference in pair_names
        ]
        if checkpoints:
            for row in metadata["pairs"]:
                key = (row["cache"], row["backend"], row["reference"])
                row["completed_by_cv_calls"] = {
                    calls: pair_completions[(*key, calls)] for calls in checkpoints
                }
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()
