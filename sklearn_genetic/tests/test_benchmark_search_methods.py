import csv
import json
import sys
from importlib.metadata import PackageNotFoundError
from itertools import groupby
from pathlib import Path
from runpy import run_path

import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier

from sklearn_genetic import GAFeatureSelectionCV

# benchmark_search_methods.py does `from benchmark_fit import ...` (a bare,
# sibling-relative import meant for running the script directly, where Python
# puts its own directory on sys.path). Importing it as benchmarks.* for
# testing does not add benchmarks/ itself to sys.path, so that import would
# otherwise fail with ModuleNotFoundError.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))

from benchmarks.benchmark_search_methods import comparison_key, print_comparison_table  # noqa: E402


def test_catalog_separates_prefix_holdout_from_the_public_winner():
    from benchmarks.benchmark_optimizer_catalog import (
        pair_comparisons,
        summarize,
        summarize_public_refit,
        summarize_stopped_search,
    )

    def both_masks(problem, evaluate, report):
        report([handle for _, handle in evaluate([[1, 0], [0, 1]])])

    y = np.tile([0, 1], 4)
    X = np.column_stack((y, y))
    hold_X = np.array([[0, 1], [1, 0]])
    hold_y = np.array([0, 1])
    model = DecisionTreeClassifier(random_state=5)
    search = GAFeatureSelectionCV(
        model,
        cv=2,
        scoring="balanced_accuracy",
        max_features=1,
        optimizer_backend=both_masks,
        population_size=2,
        generations=0,
        keep_top_k=2,
        random_state=5,
        verbose=False,
    ).fit(X, y)
    trace = [
        (score, tuple(mask), 0, 0)
        for score, mask in zip(
            search.cv_results_["mean_test_score"], search.cv_results_["features"], strict=True
        )
    ]
    run = (trace, model, X, y, hold_X, hold_y, "balanced_accuracy", True, search, 0)

    assert summarize(run, 1)["holdout"] == 1
    public = summarize_public_refit(run, 1)
    assert public["public_cv"] == 1
    assert public["public_holdout"] == 0
    assert not public["public_winner_in_prefix"]
    stopped = summarize_stopped_search(run, 2)
    assert stopped == {
        "completed_target": True,
        "budget_status": "target_reached",
        "returned_cv": 1,
        "returned_holdout": 0,
        "returned_matches_trace_winner": False,
    }
    assert not summarize_stopped_search(run, 3)["completed_target"]
    short = (trace[:1], *run[1:])
    pairs = pair_comparisons({"short": short, "reference": run}, ["short"], ["reference"], 2, {})
    assert next(pairs)["reference_prefix_holdout"] == 1


@pytest.mark.parametrize("cache", [False, True])
def test_catalog_public_cv_uses_selected_observation_when_a_mask_repeats(monkeypatch, cache):
    from benchmarks import benchmark_optimizer_catalog as catalog

    def make_search(case, engine, variant, seed, cache, pilot, proposal_budget, public_refit):
        y = np.tile([0, 1], 4)
        X = np.column_stack((y, y + 10))
        repeated_scores = iter((0.75, 0.75, 1.0, 1.0))

        def objective(estimator, X, y):
            return next(repeated_scores) if X[0, 0] < 10 else 0.5

        def optimize(problem, evaluate, report):
            report([handle for _, handle in evaluate([[1, 0], [1, 0], [0, 1]])])

        search = GAFeatureSelectionCV(
            DecisionTreeClassifier(random_state=seed),
            cv=2,
            scoring=objective,
            max_features=1,
            optimizer_backend=optimize,
            population_size=3,
            generations=0,
            use_cache=cache,
            refit=public_refit,
            random_state=seed,
            n_jobs=1,
            verbose=False,
        )
        hold_X = np.array([[0, 11], [1, 10]])
        hold_y = np.array([0, 1])
        return search, X, y, hold_X, hold_y, "balanced_accuracy", True

    monkeypatch.setattr(catalog, "make_search", make_search)
    run = catalog.measure(
        "wine-mask", "deap", "default", 41, cache, False, target=3, public_refit=True
    )
    trace, search = run[0], run[-2]
    assert len(trace) == search.fit_counts_["search_cv_calls"] == (2 if cache else 3)
    assert search.best_score_ == 0.75

    prefix = catalog.summarize(run, len(trace))
    assert prefix["best_cv"] == (0.75 if cache else 1.0)
    public = catalog.summarize_public_refit(run, len(trace))
    stopped = catalog.summarize_stopped_search(run, 3)
    assert public["public_cv"] == stopped["returned_cv"] == search.best_score_
    assert public["public_holdout"] == stopped["returned_holdout"] == 1.0
    assert public["public_winner_in_prefix"]


@pytest.mark.parametrize(
    "case, backend, variant, public_refit",
    [
        ("wine", "deap", "default", False),
        ("wine-mask", "deap", "default", True),
        ("wine", "random", "reference", True),
        ("wine-mask", "random", "reference", False),
    ],
)
def test_catalog_can_stop_at_a_fresh_cv_budget(case, backend, variant, public_refit):
    from benchmarks.benchmark_optimizer_catalog import measure, summarize_stopped_search

    reference = measure(case, backend, variant, 41, False, False, 9, proposal_budget=32)
    capped = measure(
        case,
        backend,
        variant,
        41,
        False,
        False,
        9,
        public_refit=public_refit,
        proposal_budget=32,
        stop_at_budget=True,
    )
    assert len(reference[0]) > 9
    assert len(capped[0]) == capped[-2].fit_counts_["search_cv_calls"] == 9
    assert capped[-2].fit_counts_["refit_fits"] == int(public_refit)
    assert [row[:2] for row in capped[0]] == [row[:2] for row in reference[0][:9]]
    assert capped[-1] >= capped[0][-1][2]
    summary = summarize_stopped_search(capped, 9)
    assert summary["completed_target"]
    assert summary["budget_status"] == "target_reached"
    assert summary["returned_cv"] == capped[-2].best_score_
    assert 0 <= summary["returned_holdout"] <= 1


@pytest.mark.parametrize("case", ["wine", "wine-mask"])
def test_capped_catalog_distinguishes_confirmed_coverage_from_an_early_return(case, monkeypatch):
    from benchmarks import benchmark_optimizer_catalog as catalog
    from sklearn_genetic.space import Integer

    original = catalog.make_search

    def make_finite(*args, **kwargs):
        search, X, y, hold_X, hold_y, scoring, masks = original(*args, **kwargs)
        if masks:
            search.set_params(max_features=1)
            X, hold_X = X[:, :2], hold_X[:, :2]
        else:
            search.set_params(param_grid={"max_depth": Integer(1, 2)})
        return search, X, y, hold_X, hold_y, scoring, masks

    monkeypatch.setattr(catalog, "make_search", make_finite)
    target = 32
    runs = {
        engine: catalog.measure(case, engine, variant, 41, True, False, target, stop_at_budget=True)
        for engine, variant in (("deap", "default"), ("random", "reference"))
    }
    assert len(runs["deap"][0]) == len(runs["random"][0]) == 2
    assert runs["deap"][-2].search_space_exhausted_
    assert not runs["random"][-2].search_space_exhausted_
    assert (
        catalog.summarize_stopped_search(runs["deap"], target)["budget_status"] == "space_exhausted"
    )
    assert (
        catalog.summarize_stopped_search(runs["random"], target)["budget_status"] == "below_target"
    )
    pair = next(catalog.pair_comparisons(runs, ["deap"], ["random"], target, {}, stopped=True))
    assert not pair["completed_target"]
    assert pair["matched_cv_calls"] == 2
    assert pair["backend_budget_status"] == "space_exhausted"
    assert pair["reference_budget_status"] == "below_target"


@pytest.mark.parametrize("proposal_budget", ["24", "33"])
def test_catalog_rejects_a_proposal_budget_below_the_target_or_off_batch(
    proposal_budget,
    monkeypatch,
    tmp_path,
    capsys,
):
    from benchmarks.benchmark_optimizer_catalog import main

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_optimizer_catalog",
            "--stop-at-budget",
            "--budget",
            "32",
            "--proposal-budget",
            proposal_budget,
            "--output",
            str(tmp_path / "capped.csv"),
        ],
    )
    with pytest.raises(SystemExit, match="2"):
        main()
    assert (
        "--proposal-budget must be a multiple of 8 and at least --budget" in capsys.readouterr().err
    )


@pytest.mark.parametrize("capped", [False, True])
def test_catalog_cli_runs_selected_cases_engines_and_seeds(monkeypatch, tmp_path, capsys, capped):
    from benchmarks import benchmark_optimizer_catalog as catalog

    def package_version(name):
        if name in {"nevergrad", "cmaes", "pymoo"}:
            raise PackageNotFoundError(name)
        return "test-version"

    output = tmp_path / "subset.csv"
    pairs = tmp_path / "pairs.csv"
    mode = ["--stop-at-budget"] if capped else []
    monkeypatch.setattr(catalog, "version", package_version)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_optimizer_catalog",
            "--cases",
            "wine",
            "wine-mask",
            "wine",
            "--engines",
            "deap",
            "deap",
            "--seeds",
            "107",
            "107",
            "--references",
            "random",
            "--cv-checkpoints",
            "32",
            "8",
            "8",
            "--output",
            str(output),
            "--pairs-output",
            str(pairs),
            *mode,
        ],
    )
    catalog.main()

    with output.open(newline="") as file:
        rows = list(csv.DictReader(file))
    with pairs.open(newline="") as file:
        paired = list(csv.DictReader(file))
    metadata = json.loads(capsys.readouterr().out)
    assert metadata["rows"] == len(rows) == 8
    assert len(paired) == 4
    assert metadata["cv_checkpoints"] == [8, 32]
    for group in metadata["pairs"]:
        subset = [row for row in paired if row["cache"] == str(group["cache"])]
        assert group["completed_by_cv_calls"] == {
            str(calls): sum(row[f"cv_{calls}_completed"] == "True" for row in subset)
            for calls in (8, 32)
        }
    assert all(float(row["cv_8_best_cv"]) == float(row["best_first_eight"]) for row in rows)
    assert all(0 < float(row["full_search_ms"]) <= float(row["full_ms"]) for row in rows)
    assert set(metadata["packages"]) == {"scikit-learn", "numpy", "deap", "scipy"}
    assert {row["backend"] for row in rows} == {"deap", "random"}
    assert {row["backend"] for row in paired} == {"deap"}
    assert {row["reference"] for row in paired} == {"random"}
    assert {(row["case"], row["seed"], row["cache"]) for row in paired} == {
        ("wine", "107", "False"),
        ("wine", "107", "True"),
        ("wine-mask", "107", "False"),
        ("wine-mask", "107", "True"),
    }
    assert all(0 < int(row["matched_cv_calls"]) <= 32 for row in paired)
    assert all(row["backend_prefix_holdout"] and row["reference_prefix_holdout"] for row in paired)
    if capped:
        statuses = {
            (row["case"], row["cache"], row["backend"]): row["budget_status"] for row in rows
        }
        for row in paired:
            for side in ("backend", "reference"):
                assert (
                    row[f"{side}_budget_status"] == statuses[row["case"], row["cache"], row[side]]
                )
                assert 0 < float(row[f"{side}_search_ms"]) <= float(row[f"{side}_fit_ms"])
                assert row[f"{side}_refit_ms"] == ""


@pytest.mark.parametrize(
    "extra, message",
    [
        (["--pilot", "--engines", "deap"], "--pilot can only use"),
        (["--seeds", "-1"], "--seeds must be between"),
        (["--seeds", "4294967295"], "--seeds must be between"),
        (["--cv-checkpoints", "0"], "--cv-checkpoints must be between 1 and --budget"),
        (["--cv-checkpoints", "33"], "--cv-checkpoints must be between 1 and --budget"),
        (["--cases", "cancer", "--references", "tpe"], "TPE needs at least one tuning case"),
    ],
)
def test_catalog_rejects_invalid_subset_before_opening_outputs(
    extra, message, monkeypatch, tmp_path, capsys
):
    from benchmarks.benchmark_optimizer_catalog import main

    output = tmp_path / "invalid.csv"
    monkeypatch.setattr(
        sys, "argv", ["benchmark_optimizer_catalog", "--output", str(output), *extra]
    )
    with pytest.raises(SystemExit, match="2"):
        main()
    assert message in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("case", ["wine", "wine-mask"])
def test_catalog_random_reference_uses_shared_folds_and_real_cv_budget(case):
    from benchmarks.benchmark_optimizer_catalog import comparison_checkpoints, measure, summarize

    baseline = measure(case, "deap", "default", 41, False, False, 16)
    random = measure(case, "random", "reference", 41, False, False, 16)
    assert len(random[0]) == random[-2].fit_stats_["cross_validate_calls"] == 16
    assert [row[:2] for row in baseline[0][:8]] == [row[:2] for row in random[0][:8]]
    for before, after in zip(baseline[-2]._cv_splits, random[-2]._cv_splits, strict=True):
        np.testing.assert_equal(before, after)

    cohort = summarize(random, 8)
    reached = comparison_checkpoints(random, cohort, 16, [10_000_000])
    assert reached["target_best_cv"] == max(row[0] for row in random[0])
    assert reached["time_10000000ms_cv_calls"] == 16
    assert reached["time_10000000ms_holdout"] == reached["target_holdout"]

    incomplete = comparison_checkpoints(
        random, cohort, 32, [10_000_000], cv_checkpoints=(8, 16, 32)
    )
    assert incomplete["target_holdout"] is None
    assert incomplete["time_10000000ms_cv_calls"] == 16
    assert incomplete["cv_8_best_cv"] == cohort["best_cv"]
    assert incomplete["cv_16_holdout"] == reached["target_holdout"]
    assert incomplete["cv_32_best_cv"] is incomplete["cv_32_holdout"] is None


@pytest.mark.parametrize("case", ["wine", "wine-mask"])
@pytest.mark.parametrize("target", [16, 32])
def test_catalog_pair_comparison_matches_each_pair_independently(case, target):
    from benchmarks.benchmark_optimizer_catalog import (
        measure,
        pair_comparisons,
        summarize,
        summarize_public_refit,
    )

    deap = measure(case, "deap", "default", 41, False, False, 16, public_refit=True)
    reference = measure(case, "random", "reference", 41, False, False, 16, public_refit=True)
    short = measure(case, "deap", "default", 41, False, False, 8, public_refit=True)
    runs = {"deap": deap, "short": short, "random": reference}
    pairs = {
        row["backend"]: row
        for row in pair_comparisons(
            runs, ["deap", "short"], ["random"], target, {}, cv_checkpoints=(8, 16)
        )
    }
    assert pairs["deap"]["matched_cv_calls"] == 16
    assert pairs["deap"]["completed_target"] == (target == 16)
    assert pairs["short"]["matched_cv_calls"] == 8
    assert not pairs["short"]["completed_target"]
    assert pairs["short"]["cv_8_completed"] and pairs["deap"]["cv_16_completed"]
    assert not pairs["short"]["cv_16_completed"]
    assert pairs["short"]["reference_cv_16_best_cv"] is None
    for backend, prefix in [("deap", 16), ("short", 8)]:
        for side, run in [("backend", runs[backend]), ("reference", reference)]:
            expected = summarize(run, prefix)
            row = pairs[backend]
            assert row[f"{side}_cv_calls"] == len(run[0])
            assert row[f"{side}_best_cv"] == expected["best_cv"]
            assert row[f"{side}_prefix_holdout"] == expected["holdout"]
            assert row[f"{side}_prefix_ms"] == expected["prefix_ms"]
            assert row[f"{side}_cv_{prefix}_best_cv"] == expected["best_cv"]
            assert row[f"{side}_cv_{prefix}_holdout"] == expected["holdout"]
            assert row[f"{side}_cv_{prefix}_elapsed_ms"] == expected["prefix_ms"]
    stopped_pair = next(pair_comparisons(runs, ["deap"], ["random"], target, {}, stopped=True))
    for side, run in [("backend", deap), ("reference", reference)]:
        search, total = run[-2:]
        assert stopped_pair[f"{side}_fit_ms"] == 1000 * total
        assert stopped_pair[f"{side}_search_ms"] == 1000 * search.search_time_
        assert stopped_pair[f"{side}_refit_ms"] == 1000 * search.refit_time_


def test_forward_feature_benchmark_compares_distinct_extensions_at_equal_budget():
    runner = (
        Path(__file__).resolve().parents[2] / "benchmarks/results/performance-v2/known-support.py"
    )
    result = run_path(str(runner))["run_pair"](2101, interaction=True, forward=True)
    history = result["forward"]["proposals"]
    assert len({tuple(p["columns"]) for p in history}) == len(result["genetic"]["proposals"]) == 24
    parent = set()
    for size, candidates in groupby(history, key=lambda p: len(p["columns"])):
        candidates = list(candidates)
        assert size == len(parent) + 1
        assert all(parent < set(p["columns"]) for p in candidates)
        parent = set(max(candidates, key=lambda p: p["cv"])["columns"])
    assert [len(p["columns"]) for p in history] == [1] * 16 + [2] * 8


def _summary(*, scenario="scenario_a", method="ga", n_jobs=1, fit_seconds_mean=1.0):
    return {
        "scenario": scenario,
        "method": method,
        "n_jobs": n_jobs,
        "fit_seconds_mean": fit_seconds_mean,
    }


def test_comparison_key_uses_scenario_method_n_jobs():
    summary = _summary(scenario="s", method="m", n_jobs=2)

    assert comparison_key(summary) == ("s", "m", "2")


def test_comparison_table_matches_same_key(capsys):
    current = _summary(fit_seconds_mean=10.0)
    baseline = _summary(fit_seconds_mean=2.0)

    print_comparison_table([current], [baseline])

    rows = capsys.readouterr().out.splitlines()
    data_rows = [r for r in rows if r.startswith("scenario_a")]
    assert len(data_rows) == 1
    assert data_rows[0].split("\t")[3] == "8.0000"


def test_comparison_table_reports_no_comparable_rows(capsys):
    current = _summary(scenario="s1")
    baseline = _summary(scenario="s2")

    print_comparison_table([current], [baseline])

    assert "No comparable baseline rows found." in capsys.readouterr().out


def test_duplicate_comparison_keys_raise():
    """Two baseline rows with the same (scenario, method, n_jobs) must not
    silently let one win over the other (#336)."""
    baseline_a = _summary(fit_seconds_mean=2.0)
    baseline_b = _summary(fit_seconds_mean=9.0)

    with pytest.raises(ValueError, match="Duplicate baseline comparison key"):
        print_comparison_table([_summary()], [baseline_a, baseline_b])


@pytest.mark.parametrize("budget", [2, 8, 24, 40, 80])
def test_bayesmark_ga_respects_candidate_budget(budget):
    from benchmark_bayesmark import ga_budget

    population, generations = ga_budget(budget)
    assert population >= 2
    assert generations >= 0
    assert population * (1 + 2 * generations) <= budget


@pytest.mark.parametrize("budget", [8, 24])
def test_bayesmark_runner_reports_reproducible_bounded_evaluations(monkeypatch, budget):
    from benchmark_bayesmark import DATASETS, MODELS, build_task, run_gasearch

    from sklearn_genetic import genetic_search

    calls = []
    original = genetic_search.cross_validate

    def counting_cv(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(genetic_search, "cross_validate", counting_cv)
    task = build_task(DATASETS["iris"], MODELS["dt"])
    X, y = task.dataset.loader()
    first = run_gasearch(task, X, y, seed=3, budget=budget, cv_splits=2)
    assert first.evaluations == len(calls)
    calls.clear()
    second = run_gasearch(task, X, y, seed=3, budget=budget, cv_splits=2)
    assert second.evaluations == len(calls)
    assert 0 < first.evaluations <= budget
    assert first.evaluations == second.evaluations
    assert first.best_score == second.best_score
    assert 0 <= first.best_score <= 1


def test_randomized_benchmark_counts_exhausted_finite_space():
    from dataclasses import replace
    from benchmark_bayesmark import DATASETS, MODELS, build_task, run_randomized

    model = replace(MODELS["knn"], space=[("n_neighbors", "cat", "linear", [1, 2])])
    task = build_task(DATASETS["iris"], model)
    X, y = task.dataset.loader()
    result = run_randomized(task, X, y, seed=3, budget=8, cv_splits=2)
    assert result.evaluations == 2


@pytest.mark.parametrize("method", ["optuna", "flaml"])
def test_feature_optimizer_rejects_empty_caches_duplicates_and_refits_best(monkeypatch, method):
    from types import SimpleNamespace
    import numpy as np
    from sklearn.tree import DecisionTreeClassifier
    from benchmarks import benchmark_features

    configs = [dict(f0=0, f1=0), dict(f0=1, f1=0), dict(f0=1, f1=0), dict(f0=0, f1=1)]

    class Study:
        def optimize(self, objective, n_trials):
            for config in configs[:n_trials]:
                objective(SimpleNamespace(suggest_categorical=lambda name, choices: config[name]))

    def tune_run(objective, num_samples, **kwargs):
        for config in configs[:num_samples]:
            objective(config)

    monkeypatch.setitem(
        sys.modules,
        "optuna",
        SimpleNamespace(
            create_study=lambda **kwargs: Study(),
            samplers=SimpleNamespace(TPESampler=lambda **kwargs: None),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "flaml",
        SimpleNamespace(
            CFO=lambda **kwargs: None,
            tune=SimpleNamespace(choice=lambda choices: choices, run=tune_run),
        ),
    )
    labels = np.tile([0, 1], 8)
    X = np.column_stack([labels, np.zeros(len(labels))])
    original_cv = benchmark_features.cross_val_score
    evaluated = []

    def counted_cv(model, features, y, **kwargs):
        evaluated.append(features.copy())
        return original_cv(model, features, y, **kwargs)

    monkeypatch.setattr(benchmark_features, "cross_val_score", counted_cv)
    best, mask, score, evaluations, proposals, invalid = benchmark_features.optimize_mask(
        method, DecisionTreeClassifier(random_state=0), X, labels, 2, 0, 4
    )
    assert (evaluations, proposals, invalid) == (2, 4, 1)
    assert len(evaluated) == 2
    assert mask.tolist() == [True, False]
    assert score == 1.0
    assert np.array_equal(best.predict(X[:, mask]), labels)


@pytest.mark.parametrize("method", ["optuna", "flaml"])
def test_feature_optimizer_cli_requires_unconstrained_domain(monkeypatch, capsys, method):
    from benchmarks import benchmark_features

    monkeypatch.setattr(
        sys, "argv", ["benchmark_features", "--output", "unused.json", "--methods", method]
    )
    with pytest.raises(SystemExit) as error:
        benchmark_features.main()
    assert error.value.code == 2
    assert "HPO mask comparisons require --unconstrained" in capsys.readouterr().err


def test_categorical_benchmark_runs_models_without_optional_optimizers(monkeypatch, tmp_path):
    import json
    from benchmarks import benchmark_evolution

    from sklearn_genetic import genetic_search

    candidates = []
    original_cv = genetic_search.cross_validate

    def record_candidate(model, *args, **kwargs):
        candidates.append(model.get_params(deep=True))
        return original_cv(model, *args, **kwargs)

    monkeypatch.setattr(genetic_search, "cross_validate", record_candidate)
    output = tmp_path / "categorical.json"
    monkeypatch.setitem(sys.modules, "optuna", None)
    monkeypatch.setitem(sys.modules, "flaml", None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_evolution",
            "--output",
            str(output),
            "--datasets",
            "iris",
            "--models",
            "dt",
            "svm",
            "knn",
            "--methods",
            "gasearch",
            "--budget",
            "8",
            "--seeds",
            "1",
            "--categorical",
        ],
    )
    benchmark_evolution.main()
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["space_profile"] == "categorical"
    assert {row["model"] for row in result["results"]} == {"dt", "svm", "knn"}
    for row in result["results"]:
        assert 0 < row["evaluations"] <= row["evaluation_budget"] == 8
        assert 0 <= row["cv_score"] <= 1
        assert 0 <= row["holdout_score"] <= 1
    assert "optuna" not in result["versions"]
    assert "flaml" not in result["versions"]
    expected = {
        "criterion": {"gini", "entropy"},
        "splitter": {"best", "random"},
        "model__kernel": {"rbf", "poly", "sigmoid"},
        "model__shrinking": {True, False},
        "model__weights": {"uniform", "distance"},
        "model__metric": {"euclidean", "manhattan"},
    }
    for name, choices in expected.items():
        assert {params[name] for params in candidates if name in params} == choices


@pytest.mark.parametrize("dataset,model", [("diabetes", "dt"), ("iris", "rf")])
def test_categorical_benchmark_rejects_incompatible_workload(monkeypatch, capsys, dataset, model):
    from benchmarks import benchmark_evolution

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_evolution",
            "--output",
            "unused.json",
            "--datasets",
            dataset,
            "--models",
            model,
            "--categorical",
        ],
    )
    with pytest.raises(SystemExit) as error:
        benchmark_evolution.main()
    assert error.value.code == 2
    assert (
        "categorical profile supports dt/svm/knn classification workloads"
        in capsys.readouterr().err
    )


def test_categorical_flaml_benchmark_normalizes_numpy_choices(monkeypatch, tmp_path):
    import json
    import numpy as np
    from types import SimpleNamespace
    from benchmarks import benchmark_evolution

    def tune_run(objective, config, num_samples, **kwargs):
        assert config["model__metric"] == ["euclidean", "manhattan"]
        assert config["model__weights"] == ["uniform", "distance"]
        results = []
        for index in range(num_samples):
            params = {
                key: (
                    np.str_(value[(index + (key == "model__metric")) % len(value)])
                    if isinstance(value, list)
                    else value
                )
                for key, value in config.items()
            }
            results.append((objective(params), params))
        result, params = max(results, key=lambda entry: entry[0]["score"])
        return SimpleNamespace(
            best_result=result, best_config=params, trials=list(range(num_samples))
        )

    monkeypatch.setitem(
        sys.modules,
        "flaml",
        SimpleNamespace(
            CFO=lambda **kwargs: None,
            tune=SimpleNamespace(
                choice=list,
                randint=lambda low, high: low,
                loguniform=lambda low, high: low,
                uniform=lambda low, high: low,
                run=tune_run,
            ),
        ),
    )
    original_version = benchmark_evolution.importlib.metadata.version
    monkeypatch.setattr(
        benchmark_evolution.importlib.metadata,
        "version",
        lambda name: "test" if name == "flaml" else original_version(name),
    )
    output = tmp_path / "flaml.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_evolution",
            "--output",
            str(output),
            "--datasets",
            "digits",
            "--models",
            "knn",
            "--methods",
            "flaml",
            "--budget",
            "8",
            "--seeds",
            "1",
            "--categorical",
        ],
    )
    benchmark_evolution.main()
    result = json.loads(output.read_text(encoding="utf-8"))["results"][0]
    assert result["evaluations"] == 8
    assert 0 <= result["cv_score"] <= 1
    assert 0 <= result["holdout_score"] <= 1


@pytest.mark.parametrize(
    "option,value",
    [
        ("--seconds", "0"),
        ("--seconds", "-1"),
        ("--seconds", "nan"),
        ("--seconds", "inf"),
        ("--seeds", "0"),
    ],
)
def test_native_automl_benchmark_rejects_invalid_budget(monkeypatch, capsys, option, value):
    from benchmarks import benchmark_native_automl

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_native_automl",
            "--output",
            "unused.json",
            "--model-dir",
            "unused-models",
            option,
            value,
        ],
    )
    monkeypatch.setattr(benchmark_native_automl, "run", lambda args: None)
    with pytest.raises(SystemExit) as error:
        benchmark_native_automl.main()
    assert error.value.code == 2
    assert "seconds must be finite and positive" in capsys.readouterr().err


def test_native_automl_benchmark_accepts_minimum_seed_count(monkeypatch):
    from benchmarks import benchmark_native_automl

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_native_automl",
            "--output",
            "unused.json",
            "--model-dir",
            "unused-models",
            "--seconds",
            "0.5",
            "--seeds",
            "1",
        ],
    )
    calls = []
    monkeypatch.setattr(benchmark_native_automl, "run", calls.append)
    benchmark_native_automl.main()
    assert len(calls) == 1
    assert calls[0].seeds == 1
    assert calls[0].seconds == 0.5
    assert (calls[0].evolutionary_model, calls[0].train_fraction, calls[0].generations) == (
        "lightgbm",
        1,
        1000,
    )


@pytest.mark.parametrize(
    "option,value",
    [("--seeds", "0"), ("--generations", "-1"), ("--finalists", "0"), ("--fractions", "nan")],
)
def test_fidelity_benchmark_rejects_invalid_workload(monkeypatch, capsys, option, value):
    from benchmarks import benchmark_fidelity

    monkeypatch.setattr(
        sys, "argv", ["benchmark_fidelity", "--output", "unused.json", option, value]
    )
    monkeypatch.setattr(benchmark_fidelity, "run", lambda args: None)
    with pytest.raises(SystemExit) as error:
        benchmark_fidelity.main()
    assert error.value.code == 2
    assert "fractions in (0, 1]" in capsys.readouterr().err


def test_fidelity_benchmark_dispatches_legal_boundaries(monkeypatch):
    from benchmarks import benchmark_fidelity

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_fidelity",
            "--output",
            "unused.json",
            "--seeds",
            "1",
            "--generations",
            "0",
            "--finalists",
            "1",
            "--fractions",
            "1",
        ],
    )
    calls = []
    monkeypatch.setattr(benchmark_fidelity, "run", calls.append)
    benchmark_fidelity.main()
    assert len(calls) == 1
    args = calls[0]
    assert (args.seeds, args.generations, args.finalists, args.fractions) == (1, 0, 1, [1.0])


@pytest.mark.parametrize(
    "dataset,fraction,generations", [("iris", 0.5, 0), ("diabetes", 0.5, 1), ("iris", 1, 1)]
)
def test_native_benchmark_runs_both_svm_searches_with_full_finalists(
    monkeypatch, tmp_path, dataset, fraction, generations
):
    import json
    import numpy as np
    from benchmarks import benchmark_native_automl
    from sklearn_genetic import genetic_search

    from sklearn.model_selection import train_test_split

    X, y, classification = benchmark_native_automl.dataset(dataset)
    expected_X, _, expected_y, _ = train_test_split(
        X, y, test_size=0.25, random_state=901, stratify=y if classification else None
    )
    monkeypatch.setitem(sys.modules, "lightgbm", None)
    original = genetic_search.cross_validate
    calls = []

    def recording_cv(estimator, *args, **kwargs):
        np.testing.assert_array_equal(args[1], expected_y)
        assert args[0].shape[0] == expected_X.shape[0]
        for column in args[0].T:
            assert any(np.array_equal(column, original_column) for original_column in expected_X.T)
        calls.append((estimator, [len(train) for train, _ in kwargs["cv"]]))
        return original(estimator, *args, **kwargs)

    monkeypatch.setattr(genetic_search, "cross_validate", recording_cv)
    output = tmp_path / "native-svm.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_native_automl",
            "--output",
            str(output),
            "--model-dir",
            str(tmp_path / "models"),
            "--datasets",
            dataset,
            "--methods",
            "gasearch",
            "features",
            "--seeds",
            "1",
            "--seconds",
            "60",
            "--evolutionary-model",
            "svm",
            "--train-fraction",
            str(fraction),
            "--generations",
            str(generations),
            "--finalists",
            "2",
        ],
    )
    benchmark_native_automl.main()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert "lightgbm" not in payload["versions"]
    assert payload["settings"]["evolutionary_model"] == "svm"
    assert payload["settings"]["train_fraction"] == fraction
    assert {row["method"] for row in payload["results"]} == {"gasearch", "features"}
    for row in payload["results"]:
        assert row["generations"] == generations + 1
        assert row["promotion_cv_calls"] == (2 if fraction < 1 else 0)
        assert row["train_fold_sizes"] == [int(fraction * n) for n in row["full_fold_sizes"]]
        assert 0 < row["selected_features"] <= (4 if dataset == "iris" else 10)
        assert np.isfinite(row["cv_score"]) and np.isfinite(row["holdout_score"])
        assert row["seconds"] > 0
    assert len(calls) == sum(
        row["evaluations"] + row["promotion_cv_calls"] for row in payload["results"]
    )
    full = payload["results"][0]["full_fold_sizes"]
    assert sum(sizes == full for _, sizes in calls) == (4 if fraction < 1 else len(calls))
    assert all(
        type(est.steps[-1][1]).__name__ == ("SVC" if dataset == "iris" else "SVR")
        for est, _ in calls
    )


@pytest.mark.parametrize(
    "option,value",
    [
        ("--train-fraction", "0"),
        ("--train-fraction", "1.1"),
        ("--train-fraction", "nan"),
        ("--generations", "-1"),
        ("--finalists", "0"),
    ],
)
def test_native_benchmark_rejects_invalid_fidelity(monkeypatch, capsys, option, value):
    from benchmarks import benchmark_native_automl

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_native_automl",
            "--output",
            "unused.json",
            "--model-dir",
            "unused-models",
            option,
            value,
        ],
    )
    with pytest.raises(SystemExit) as error:
        benchmark_native_automl.main()
    assert error.value.code == 2
    assert "train-fraction in (0, 1]" in capsys.readouterr().err
