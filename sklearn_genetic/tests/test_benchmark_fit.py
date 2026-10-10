import pytest

from benchmarks.benchmark_fit import (
    SCENARIOS,
    aggregate_results,
    build_baseline_lookup,
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


def test_comparison_key_includes_use_cache():
    cache_on = aggregate_results(
        [_benchmark_result(use_cache=True, cache_hits=3, duplicate_candidates=1, cv_calls=4)]
    )[0]
    cache_off = aggregate_results(
        [_benchmark_result(use_cache=False, cache_hits=0, duplicate_candidates=4, cv_calls=7)]
    )[0]

    assert comparison_key(cache_on) != comparison_key(cache_off)
    assert comparison_key(cache_on)[-1] is True
    assert comparison_key(cache_off)[-1] is False


def test_comparison_supports_old_summaries_without_cache_mode(capsys):
    cache_on = aggregate_results(
        [_benchmark_result(use_cache=True, cache_hits=3, duplicate_candidates=1, cv_calls=4)]
    )[0]
    old_baseline = dict(cache_on)
    old_baseline.pop("use_cache")

    assert comparison_key(old_baseline)[-1] is None

    print_comparison_table([cache_on], [old_baseline])

    output = capsys.readouterr().out
    assert "current_use_cache" in output
    assert "baseline_use_cache" in output
    rows = output.splitlines()
    data_rows = [r for r in rows if r.startswith("classification_lr_collisions")]
    assert len(data_rows) == 1
    assert data_rows[0].split("\t")[5:7] == ["True", "True"]


def test_duplicate_cache_mode_keys_do_not_silently_overwrite(capsys):
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
    data_rows = [r for r in rows if r.startswith("classification_lr_collisions")]
    assert len(data_rows) == 2

    on_row = [r for r in data_rows if "\tTrue\tTrue\t" in r][0]
    off_row = [r for r in data_rows if "\tFalse\tFalse\t" in r][0]

    assert on_row.split("\t")[7] == "8.0000"
    assert off_row.split("\t")[7] == "5.0000"


def test_legacy_baseline_only_matches_cache_on_current(capsys):
    cache_on = aggregate_results(
        [_benchmark_result(use_cache=True, cache_hits=3, duplicate_candidates=1, cv_calls=4)]
    )[0]
    cache_off = aggregate_results(
        [_benchmark_result(use_cache=False, cache_hits=0, duplicate_candidates=4, cv_calls=7)]
    )[0]

    current_on = dict(cache_on, fit_seconds_mean=10.0)
    current_off = dict(cache_off, fit_seconds_mean=10.0)

    old_baseline = dict(cache_on)
    old_baseline.pop("use_cache")
    old_baseline["fit_seconds_mean"] = 2.0

    print_comparison_table([current_on, current_off], [old_baseline])

    rows = capsys.readouterr().out.splitlines()
    data_rows = [r for r in rows if r.startswith("classification_lr_collisions")]
    assert len(data_rows) == 1
    assert data_rows[0].split("\t")[5:7] == ["True", "True"]


def test_build_baseline_lookup_raises_on_duplicate_key():
    with pytest.raises(ValueError, match="Duplicate baseline comparison key"):
        build_baseline_lookup(
            [{"scenario": "a", "n": 1}, {"scenario": "a", "n": 2}],
            key_func=lambda summary: (summary["scenario"],),
        )


def test_build_baseline_lookup_allows_distinct_keys():
    lookup = build_baseline_lookup(
        [{"scenario": "a"}, {"scenario": "b"}],
        key_func=lambda summary: (summary["scenario"],),
    )

    assert set(lookup) == {("a",), ("b",)}


def test_duplicate_true_comparison_keys_raise():
    """Two genuinely identical comparison keys (#336) must not silently pick
    whichever baseline row happens to be built last."""
    cache_on = aggregate_results(
        [_benchmark_result(use_cache=True, cache_hits=3, duplicate_candidates=1, cv_calls=4)]
    )[0]
    baseline_a = dict(cache_on, fit_seconds_mean=2.0)
    baseline_b = dict(cache_on, fit_seconds_mean=9.0)

    with pytest.raises(ValueError, match="Duplicate baseline comparison key"):
        print_comparison_table([cache_on], [baseline_a, baseline_b])


def test_duplicate_legacy_comparison_keys_raise():
    """Same as above, for the legacy (no use_cache) baseline lookup (#336)."""
    cache_on = aggregate_results(
        [_benchmark_result(use_cache=True, cache_hits=3, duplicate_candidates=1, cv_calls=4)]
    )[0]
    old_baseline_a = dict(cache_on)
    old_baseline_a.pop("use_cache")
    old_baseline_a["fit_seconds_mean"] = 2.0
    old_baseline_b = dict(old_baseline_a, fit_seconds_mean=9.0)

    with pytest.raises(ValueError, match="Duplicate baseline comparison key"):
        print_comparison_table([cache_on], [old_baseline_a, old_baseline_b])
