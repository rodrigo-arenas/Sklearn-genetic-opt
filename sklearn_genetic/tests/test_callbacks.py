import pytest

from ..callbacks import check_callback, check_stats, ThresholdStopping


def test_check_metrics():
    assert check_stats('fitness') is None

    with pytest.raises(Exception) as excinfo:
        check_stats('accuracy')
    assert str(
        excinfo.value) == "metric must be one of ['fitness', 'fitness_std', 'fitness_max', 'fitness_min'], " \
                          "but got accuracy instead"


def test_check_callback():
    assert check_callback(sum) == [sum]
    assert check_callback(None) == []
    assert check_callback([sum, min]) == [sum, min]

    with pytest.raises(Exception) as excinfo:
        check_callback(1)
    assert str(excinfo.value) == "callback should be either a callable or a list of callables."


def test_threshold_callback():
    callback = ThresholdStopping(threshold=0.8)
    assert not callback(record={'fitness': 0.5})
    assert callback(record={'fitness': 0.9})
