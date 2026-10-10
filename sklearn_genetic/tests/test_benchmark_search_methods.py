import sys
from pathlib import Path

import pytest

# benchmark_search_methods.py does `from benchmark_fit import ...` (a bare,
# sibling-relative import meant for running the script directly, where Python
# puts its own directory on sys.path). Importing it as benchmarks.* for
# testing does not add benchmarks/ itself to sys.path, so that import would
# otherwise fail with ModuleNotFoundError.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))

from benchmarks.benchmark_search_methods import comparison_key, print_comparison_table  # noqa: E402


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
