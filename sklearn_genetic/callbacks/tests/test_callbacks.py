import pytest
import logging
import importlib.util
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
from deap.tools import Logbook
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from ... import GAFeatureSelectionCV, GASearchCV
from ...space import Integer, Continuous
from .. import (
    ProgressBar,
    ThresholdStopping,
    ConsecutiveStopping,
    DeltaThreshold,
    TimerStopping,
    LogbookSaver,
    TensorBoard,
)
from ..validations import check_stats, check_callback, eval_callbacks
from ..base import BaseCallback

tensorflow_available = importlib.util.find_spec("tensorflow") is not None


def test_callbacks_import_does_not_attempt_tensorflow():
    script = """
import builtins
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == 'tensorflow' or name.startswith('tensorflow.'):
        raise AssertionError('TensorFlow was imported by callbacks')
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
import sklearn_genetic.callbacks
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr


def test_legacy_tensorboard_requires_explicit_tensorflow(monkeypatch):
    monkeypatch.setitem(sys.modules, "tensorflow", None)
    callback = TensorBoard()
    logbook = Logbook()
    logbook.record(gen=0, fitness_best=0.5)
    with pytest.raises(ImportError, match="TensorFlow"):
        callback.on_step(logbook=logbook)


data = load_digits()
label_names = data["target_names"]
y = data["target"]
X = data["data"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


def test_base_callback_attributes():
    assert hasattr(BaseCallback, "on_start")
    assert hasattr(BaseCallback, "on_step")
    assert hasattr(BaseCallback, "on_end")


def test_check_metrics_and_methods():
    assert check_stats("fitness") is None

    with pytest.raises(Exception) as excinfo:
        check_stats("accuracy")
    assert (
        str(excinfo.value) == "metric must be one of "
        "['fitness', 'fitness_std', 'fitness_best', 'fitness_max', 'fitness_min', "
        "'fitness_best_observed'], "
        "but got accuracy instead"
    )

    with pytest.raises(Exception) as excinfo:
        eval_callbacks(callbacks=None, record=None, logbook=None, estimator=None, method="on_epoch")
    assert (
        str(excinfo.value)
        == "The callback method must be one of ['on_start', 'on_step', 'on_end'], but got on_epoch instead"
    )


@pytest.mark.parametrize(
    "callback",
    [
        ProgressBar,
        ThresholdStopping,
        ConsecutiveStopping,
        DeltaThreshold,
        TimerStopping,
        LogbookSaver,
        TensorBoard,
    ],
)
def test_check_at_least_one_method(callback):
    assert any(
        [
            hasattr(callback, "on_start"),
            hasattr(callback, "on_step"),
            hasattr(callback, "on_end"),
        ]
    )


def test_check_callback():
    callback_threshold = ThresholdStopping(threshold=0.8)
    callback_consecutive = ConsecutiveStopping(generations=3)
    assert not BaseCallback().on_step()
    assert check_callback(callback_threshold) == [callback_threshold]
    assert check_callback(None) == []
    assert check_callback([callback_threshold, callback_consecutive]) == [
        callback_threshold,
        callback_consecutive,
    ]

    with pytest.raises(Exception) as excinfo:
        check_callback(1)
    assert (
        str(excinfo.value)
        == "callback should be either a class or a list of classes with inheritance from "
        "callbacks.base.BaseCallback"
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
    assert str(excinfo.value) == "At least one of record or logbook parameters must be provided"


@pytest.mark.parametrize("criteria", ["min", "max"])
@pytest.mark.parametrize(
    "metric", ["fitness", "fitness_best", "fitness_min", "fitness_max", "fitness_std"]
)
def test_threshold_callback_uses_search_direction(criteria, metric):
    callback = ThresholdStopping(threshold=2.0, metric=metric)
    search = SimpleNamespace(criteria=criteria)
    for value in (1.0, 2.0, 3.0):
        record = {metric: value}
        expected = value <= 2.0 if criteria == "min" and metric != "fitness_std" else value >= 2.0
        assert callback(record=record, estimator=search) == expected
        assert callback(logbook=[record], estimator=search) == expected


@pytest.mark.parametrize("unavailable", [None, float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("criteria", ["min", "max"])
def test_score_callbacks_wait_for_finite_values(unavailable, criteria):
    search = SimpleNamespace(criteria=criteria)
    threshold = ThresholdStopping(threshold=0)
    assert not threshold(record={"fitness": unavailable}, estimator=search)
    assert not threshold(logbook=[{"fitness": unavailable}], estimator=search)

    for callback in (ConsecutiveStopping(1), DeltaThreshold(0.0)):
        healthy = {"fitness": 1.0}
        bad = {"fitness": unavailable}
        assert not callback(logbook=[healthy, bad], estimator=search)
        assert not callback(logbook=[bad, healthy], estimator=search)
        assert not callback(logbook=[bad, bad], estimator=search)
        assert callback(logbook=[bad, healthy, healthy], estimator=search)


@pytest.mark.parametrize("criteria", ["min", "max"])
@pytest.mark.parametrize("extended", [False, True], ids=["double", "extended"])
def test_score_callbacks_can_use_observed_fitness_with_legacy_history(criteria, extended):
    if extended and np.finfo(np.longdouble).maxexp <= 1026:
        pytest.skip("requires extended exponent range")
    metric = "fitness_best_observed"
    search = SimpleNamespace(criteria=criteria)
    direction = 1 if criteria == "max" else -1
    scale = np.ldexp(np.longdouble(1), 1025) if extended else 1.0
    target = direction * 0.8 * scale
    old = {"fitness_best": direction * 0.4 * scale}
    reached = {**old, metric: target}

    threshold = ThresholdStopping(target, metric=metric)
    assert not threshold(logbook=[old], estimator=search)
    assert not threshold(record={**old, metric: direction * 0.6 * scale}, estimator=search)
    assert threshold(record=reached, estimator=search)
    assert threshold(logbook=[old, reached], estimator=search)

    for callback in (ConsecutiveStopping(1, metric=metric), DeltaThreshold(0.0, metric=metric)):
        assert not callback(logbook=[old, reached], estimator=search)
        assert callback(logbook=[old, reached, reached], estimator=search)


@pytest.mark.parametrize("unsigned", [np.uint8, np.uint32, np.uint64])
@pytest.mark.parametrize("name", ["consecutive", "delta"])
def test_score_callbacks_accept_unsigned_windows(unsigned, name):
    generations = unsigned(2)
    if name == "consecutive":
        callback = ConsecutiveStopping(generations)
        expected = [False, False, False, False, False, True]
    else:
        callback = DeltaThreshold(0.0, generations=generations)
        expected = [False, False, False, False, True, True]

    logbook = Logbook()
    with np.errstate(over="raise", invalid="raise"):
        for score, stop in zip((0.1, np.nan, 0.3, 0.4, 0.4, 0.4), expected, strict=True):
            logbook.record(fitness=score)
            assert callback(logbook=logbook) == stop
    assert callback.generations is generations


@pytest.mark.parametrize("threshold", ["0.8", None, object()])
def test_threshold_callback_rejects_non_numeric_threshold(threshold):
    with pytest.raises((TypeError, ValueError), match="threshold"):
        ThresholdStopping(threshold=threshold)


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


@pytest.mark.parametrize("criteria", ["min", "max"])
@pytest.mark.parametrize(
    "metric", ["fitness", "fitness_best", "fitness_min", "fitness_max", "fitness_std"]
)
def test_consecutive_callback_uses_search_direction(criteria, metric):
    callback = ConsecutiveStopping(generations=2, metric=metric)
    searches = [
        GASearchCV(
            DecisionTreeClassifier(), param_grid={"max_depth": Integer(1, 3)}, criteria=criteria
        ),
        GAFeatureSelectionCV(DecisionTreeClassifier(), criteria=criteria),
    ]
    for current in (1, 2, 3, 4):
        logbook = Logbook()
        for value in (2, 3, current):
            logbook.record(**{metric: value})
        expected = current >= 3 if criteria == "min" and metric != "fitness_std" else current <= 2
        for search in searches:
            assert callback(logbook=logbook, estimator=search) == expected
            assert callback(record=logbook[-1], logbook=logbook, estimator=search) == expected


@pytest.mark.parametrize("generations", ["3", None, object()])
def test_consecutive_callback_rejects_non_numeric_generations(generations):
    with pytest.raises((TypeError, ValueError), match="generations"):
        ConsecutiveStopping(generations=generations)


@pytest.mark.parametrize("generations", [0, -1])
def test_consecutive_callback_rejects_non_positive_generations(generations):
    with pytest.raises((TypeError, ValueError), match="generations"):
        ConsecutiveStopping(generations=generations)


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

    callback = DeltaThreshold(0.1, generations=4)

    assert callback(logbook=logbook)

    with pytest.raises(Exception) as excinfo:
        callback()
    assert str(excinfo.value) == "logbook parameter must be provided"


@pytest.mark.parametrize(
    "low, high, too_small, large_enough",
    [
        (np.float32(-2e38), np.float32(2e38), 3e38, 5e38),
        (np.float64(-1e308), np.float64(1e308), 10**308, 10**309),
        (np.int64(-(2**62)), np.int64(2**62), 2**63 - 1, 2**63),
        (np.uint64(2**63), np.uint64(2**63 + 1), 0, 1),
    ],
    ids=["float32", "float64", "int64", "uint64"],
)
def test_delta_callback_compares_extreme_ranges_without_overflow(
    low, high, too_small, large_enough
):
    logbook = Logbook()
    logbook.record(fitness=low)
    logbook.record(fitness=high)

    with np.errstate(over="raise", invalid="raise"):
        assert not DeltaThreshold(too_small)(logbook=logbook)
        assert DeltaThreshold(large_enough)(logbook=logbook)
        assert DeltaThreshold(float("inf"))(logbook=logbook)
        assert not DeltaThreshold(float("-inf"))(logbook=logbook)


@pytest.mark.parametrize("threshold", ["0.001", None, object()])
def test_delta_callback_rejects_non_numeric_threshold(threshold):
    with pytest.raises((TypeError, ValueError), match="threshold"):
        DeltaThreshold(threshold=threshold)


@pytest.mark.parametrize("generations", ["2", None, object()])
def test_delta_callback_rejects_non_numeric_generations(generations):
    with pytest.raises((TypeError, ValueError), match="generations"):
        DeltaThreshold(threshold=0.001, generations=generations)


@pytest.mark.parametrize("generations", [0, -1])
def test_delta_callback_rejects_non_positive_generations(generations):
    with pytest.raises((TypeError, ValueError), match="generations"):
        DeltaThreshold(threshold=0.001, generations=generations)


@pytest.mark.parametrize(
    "callback,parameter",
    [
        (TimerStopping, "total_seconds"),
        (ThresholdStopping, "threshold"),
        (DeltaThreshold, "threshold"),
    ],
)
@pytest.mark.parametrize("value", [float("nan"), np.float32("nan"), np.float64("nan")])
def test_stopping_callback_rejects_nan_configuration(callback, parameter, value):
    with pytest.raises(ValueError, match=rf"{parameter}.*NaN"):
        callback(**{parameter: value})


@pytest.mark.parametrize("total_seconds", [0.1, np.float32(0.5), float("inf"), 10**400])
def test_timer_callback_accepts_numeric_total_seconds(total_seconds):
    callback = TimerStopping(total_seconds=total_seconds)

    assert callback.total_seconds == total_seconds


@pytest.mark.parametrize("total_seconds", ["10", None, object()])
def test_timer_callback_rejects_non_numeric_total_seconds(total_seconds):
    with pytest.raises((TypeError, ValueError), match="total_seconds"):
        TimerStopping(total_seconds=total_seconds)


@pytest.mark.parametrize("total_seconds", [0, -1])
def test_timer_callback_rejects_non_positive_total_seconds(total_seconds):
    with pytest.raises((TypeError, ValueError), match="total_seconds"):
        TimerStopping(total_seconds=total_seconds)


def test_logbook_saver_callback(caplog, tmp_path):
    path = tmp_path / "logbook.pkl"
    callback = LogbookSaver(path)
    assert check_callback(callback) == [callback]

    clf = DecisionTreeClassifier()
    evolved_estimator = GASearchCV(
        clf,
        cv=3,
        scoring="accuracy",
        generations=2,
        param_grid={
            "min_weight_fraction_leaf": Continuous(0, 0.5),
            "max_depth": Integer(2, 20),
            "max_leaf_nodes": Integer(2, 30),
        },
        verbose=False,
    )

    evolved_estimator.fit(X_train, y_train, callbacks=callback)

    assert path.is_file()

    with caplog.at_level(logging.ERROR):
        callback = LogbookSaver(checkpoint_path=tmp_path / "no_folder" / "logbook.pkl", estimator=4)
        callback()
    assert "Could not save the Logbook in the checkpoint" in caplog.text


def test_logbook_saver_supports_binary_streams():
    from io import BytesIO
    from joblib import load

    records = [{"index": 0, "score": 0.75}]
    buffer = BytesIO()
    saver = LogbookSaver(buffer, compress=("gzip", 3))
    saver.on_step(estimator=SimpleNamespace(_candidate_rows=records))

    assert buffer.getvalue().startswith(b"\x1f\x8b")
    buffer.seek(0)
    assert load(buffer) == records


def test_logbook_saver_keeps_file_permissions_and_recovers_from_replace_failure(
    tmp_path, monkeypatch, caplog
):
    from pathlib import Path
    from stat import S_IMODE
    from joblib import load

    path = tmp_path / "logbook.pkl"
    saver = LogbookSaver(str(path))
    estimator = SimpleNamespace(_candidate_rows=[{"index": 0, "score": 0.75}])
    saver.on_step(estimator=estimator)
    previous = path.read_bytes()
    path.chmod(0o600)
    mode = S_IMODE(path.stat().st_mode)
    estimator._candidate_rows.append({"index": 1, "score": 0.8})

    def fail_replace(source, target):
        raise OSError("simulated rename failure")

    with monkeypatch.context() as scoped:
        scoped.setattr(Path, "replace", fail_replace)
        saver.on_step(estimator=estimator)
    assert path.read_bytes() == previous
    assert "simulated rename failure" in caplog.text
    assert set(tmp_path.iterdir()) == {path}

    saver.on_step(estimator=estimator)
    assert load(path) == estimator._candidate_rows
    assert S_IMODE(path.stat().st_mode) == mode
    assert set(tmp_path.iterdir()) == {path}


@pytest.mark.parametrize(
    "callback, path",
    [
        (TensorBoard(), "./logs"),
        (TensorBoard(log_dir="./sklearn_logs"), "./sklearn_logs"),
        (TensorBoard(log_dir="./logs", run_id="0"), "./logs/0"),
        (TensorBoard(log_dir="./logs", run_id="1"), "./logs/1"),
    ],
)
@pytest.mark.skipif(
    not tensorflow_available,
    reason="TensorFlow is not available for this Python version",
)
def test_tensorboard_callback(callback, path, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert check_callback(callback) == [callback]

    clf = DecisionTreeClassifier()
    evolved_estimator = GASearchCV(
        clf,
        cv=3,
        scoring="accuracy",
        generations=2,
        param_grid={
            "min_weight_fraction_leaf": Continuous(0, 0.5),
            "max_depth": Integer(2, 20),
            "max_leaf_nodes": Integer(2, 30),
        },
        verbose=False,
    )

    evolved_estimator.fit(X_train, y_train, callbacks=callback)

    assert (tmp_path / path).is_dir()
