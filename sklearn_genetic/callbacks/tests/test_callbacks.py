import pytest
import os
import shutil
import logging

from deap.tools import Logbook
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from ... import GASearchCV
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
        str(excinfo.value)
        == "metric must be one of ['fitness', 'fitness_std', 'fitness_max', 'fitness_min'], "
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

    callback = DeltaThreshold(0.1, generations=4)

    assert callback(logbook=logbook)

    with pytest.raises(Exception) as excinfo:
        callback()
    assert str(excinfo.value) == "logbook parameter must be provided"


def test_logbook_saver_callback(caplog):
    callback = LogbookSaver("./logbook.pkl")
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

    assert os.path.exists("./logbook.pkl")

    os.remove("./logbook.pkl")

    with caplog.at_level(logging.ERROR):
        callback = LogbookSaver(checkpoint_path="./no_folder/logbook.pkl", estimator=4)
        callback()
    assert "Could not save the Logbook in the checkpoint" in caplog.text


@pytest.mark.parametrize(
    "callback, path",
    [
        (TensorBoard(), "./logs"),
        (TensorBoard(log_dir="./sklearn_logs"), "./sklearn_logs"),
        (TensorBoard(log_dir="./logs", run_id="0"), "./logs/0"),
        (TensorBoard(log_dir="./logs", run_id="1"), "./logs/1"),
    ],
)
def test_tensorboard_callback(callback, path):
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

    assert os.path.exists(path)

    shutil.rmtree(path)
