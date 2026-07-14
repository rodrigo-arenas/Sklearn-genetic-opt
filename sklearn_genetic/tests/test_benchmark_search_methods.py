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
