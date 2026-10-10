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
