import pytest

from deap.tools import Logbook

from ..callbacks import (
    check_callback,
    check_stats,
    ThresholdStopping,
    ConsecutiveStopping,
    DeltaThreshold,
)


def test_check_metrics():
    assert check_stats("fitness") is None

    with pytest.raises(Exception) as excinfo:
        check_stats("accuracy")
    assert (
        str(excinfo.value)
        == "metric must be one of ['fitness', 'fitness_std', 'fitness_max', 'fitness_min'], "
        "but got accuracy instead"
    )


def test_check_callback():
    assert check_callback(sum) == [sum]
    assert check_callback(None) == []
    assert check_callback([sum, min]) == [sum, min]

    with pytest.raises(Exception) as excinfo:
        check_callback(1)
    assert (
        str(excinfo.value)
        == "callback should be either a callable or a list of callables."
    )


def test_threshold_callback():
    callback = ThresholdStopping(threshold=0.8)
    assert check_callback(callback) == [callback]
    assert not callback(record={"fitness": 0.5})
    assert callback(record={"fitness": 0.9})

    # test callback using LogBook instead of a record
    logbook = Logbook()
    logbook.record(fitness=0.93)
    logbook.record(fitness=0.4)

    assert not callback(logbook=logbook)

    logbook.record(fitness=0.95)

    assert callback(logbook=logbook)

    with pytest.raises(Exception) as excinfo:
        callback()
    assert (
        str(excinfo.value)
        == "At least one of record or logbook parameters must be provided"
    )


def test_consecutive_callback():
    callback = ConsecutiveStopping(generations=3)
    assert check_callback(callback) == [callback]

    logbook = Logbook()

    logbook.record(fitness=0.9)
    logbook.record(fitness=0.8)
    logbook.record(fitness=0.83)

    # Not enough records to decide
    assert not callback(logbook=logbook)

    logbook.record(fitness=0.85)
    logbook.record(fitness=0.81)

    # Current record is better that at least of of the previous 3 records
    assert not callback(logbook=logbook)

    logbook.record(fitness=0.8)

    # Current record is worst that the 3 previous ones
    assert callback(logbook=logbook)
    assert callback(logbook=logbook, record={"fitness": 0.8})

    with pytest.raises(Exception) as excinfo:
        callback()
    assert str(excinfo.value) == "logbook parameter must be provided"


def test_delta_callback():
    callback = DeltaThreshold(0.001)
    assert check_callback(callback) == [callback]

    logbook = Logbook()

    logbook.record(fitness=0.9)

    # Not enough records to decide
    assert not callback(logbook=logbook)

    logbook.record(fitness=0.923)
    logbook.record(fitness=0.914)

    # Abs difference is not bigger than the threshold
    assert not callback(logbook=logbook)

    logbook.record(fitness=0.9141)

    # Abs difference is bigger than the threshold
    assert callback(logbook=logbook)
    assert callback(logbook=logbook, record={"fitness": 0.9141})

    with pytest.raises(Exception) as excinfo:
        callback()
    assert str(excinfo.value) == "logbook parameter must be provided"
