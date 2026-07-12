from benchmarks.benchmark_fit import (
    SCENARIOS,
    aggregate_results,
    comparison_key,
    print_comparison_table,
)
from sklearn_genetic.space import Categorical, Integer


def _benchmark_result(*, use_cache, cache_hits, duplicate_candidates, cv_calls):
    return {
        "label": "cache-comparison",
        "scenario": "classification_lr_collisions",
        "estimator": "GASearchCV",
        "n_jobs": 1,
        "parallel_backend": "auto",
        "population_initializer": "smart",
        "use_cache": use_cache,
        "fit_seconds": 1.0,
        "actual_cross_validate_calls": cv_calls,
        "cache_hits": cache_hits,
        "duplicate_candidates": duplicate_candidates,
        "duplicate_or_cache_reuses": cache_hits + duplicate_candidates,
        "skipped_invalid_candidates": 0,
        "population_parallel_batches": 0,
        "holdout_metrics": {"roc_auc": 0.9},
        "train_metrics": {"roc_auc": 0.9},
        "generalization_gap": {"roc_auc_gap": 0.0},
    }


def test_collision_scenario_has_four_discrete_candidates():
    scenario = SCENARIOS["classification_lr_collisions"]
    param_grid = scenario.param_grid_builder()

    integer = param_grid["clf__C"]
    categorical = param_grid["clf__class_weight"]

    assert isinstance(integer, Integer)
    assert integer.upper - integer.lower + 1 == 2
    assert isinstance(categorical, Categorical)
    assert len(categorical.choices) == 2


def test_aggregate_results_separates_cache_modes_and_reports_cache_counters():
    results = [
        _benchmark_result(use_cache=True, cache_hits=3, duplicate_candidates=1, cv_calls=4),
        _benchmark_result(use_cache=False, cache_hits=0, duplicate_candidates=4, cv_calls=7),
    ]

    summaries = aggregate_results(results)

    assert len(summaries) == 2
    summaries_by_cache = {summary["use_cache"]: summary for summary in summaries}
    assert summaries_by_cache[True]["cache_hits_mean"] == 3
    assert summaries_by_cache[True]["duplicate_candidates_mean"] == 1
    assert summaries_by_cache[True]["actual_cross_validate_calls_mean"] == 4
    assert summaries_by_cache[False]["cache_hits_mean"] == 0
    assert summaries_by_cache[False]["duplicate_candidates_mean"] == 4
    assert summaries_by_cache[False]["actual_cross_validate_calls_mean"] == 7


def test_comparison_matches_same_cache_mode_when_baseline_contains_both(capsys):
    cache_on = aggregate_results(
        [_benchmark_result(use_cache=True, cache_hits=3, duplicate_candidates=1, cv_calls=4)]
    )[0]
    cache_off = aggregate_results(
        [_benchmark_result(use_cache=False, cache_hits=0, duplicate_candidates=4, cv_calls=7)]
    )[0]

    current_on = dict(cache_on, fit_seconds_mean=10.0)
    current_off = dict(cache_off, fit_seconds_mean=10.0)
    baseline_on = dict(cache_on, fit_seconds_mean=2.0)
    baseline_off = dict(cache_off, fit_seconds_mean=5.0)

    print_comparison_table([current_on, current_off], [baseline_on, baseline_off])

    rows = capsys.readouterr().out.splitlines()
    assert rows[-2].split("\t")[5:9] == ["True", "True", "8.0000", "5.0000"]
    assert rows[-1].split("\t")[5:9] == ["False", "False", "5.0000", "2.0000"]


def test_comparison_supports_old_summaries_without_cache_mode(capsys):
    cache_on = aggregate_results(
        [_benchmark_result(use_cache=True, cache_hits=3, duplicate_candidates=1, cv_calls=4)]
    )[0]
    cache_off = aggregate_results(
        [_benchmark_result(use_cache=False, cache_hits=0, duplicate_candidates=4, cv_calls=7)]
    )[0]
    old_baseline = dict(cache_off)
    old_baseline.pop("use_cache")

    assert comparison_key(cache_on) == comparison_key(cache_off)
    assert comparison_key(cache_on) == comparison_key(old_baseline)

    print_comparison_table([cache_on, cache_off], [old_baseline])

    output = capsys.readouterr().out
    assert "current_use_cache" in output
    assert "baseline_use_cache" in output
    rows = output.splitlines()
    assert rows[-1].split("\t")[5:7] == ["True", "True"]
    assert len(rows) == 5
