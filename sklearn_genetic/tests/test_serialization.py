import pytest
from sklearn.datasets import load_iris, load_diabetes
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import make_scorer
import numpy as np

from .. import GAFeatureSelectionCV
from ..callbacks import (
    ThresholdStopping,
    DeltaThreshold,
    ConsecutiveStopping,
    TimerStopping,
    ProgressBar,
)
from ..schedules import ExponentialAdapter, InverseAdapter
from joblib import dump, load
import os


data = load_iris()
label_names = data["target_names"]
y = data["target"]
X = data["data"]

noise = np.random.uniform(1, 4, size=(X.shape[0], 10))

X = np.hstack((X, noise))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


def test_estimator_serialization():
    clf = SGDClassifier(loss="modified_huber", fit_intercept=True)
    generations = 6
    evolved_estimator = GAFeatureSelectionCV(
        clf,
        cv=3,
        scoring="accuracy",
        population_size=6,
        generations=generations,
        tournament_size=3,
        elitism=False,
        keep_top_k=4,
        verbose=False,
        algorithm="eaSimple",
        n_jobs=-1,
        return_train_score=True,
    )

    evolved_estimator.fit(X_train, y_train)
    dump_file = "evolved_estimator.pkl"

    # test dump
    assert dump(evolved_estimator, dump_file)[0] == dump_file

    # load
    dumped_estimator = load(dump_file)
    features = dumped_estimator.support_

    assert check_is_fitted(dumped_estimator) is None
    assert features.shape[0] == X.shape[1]
    assert len(dumped_estimator) == generations + 1  # +1 random initial population
    assert len(dumped_estimator.predict(X_test)) == len(X_test)
    assert dumped_estimator.score(X_train, y_train) >= 0
    assert len(dumped_estimator.decision_function(X_test)) == len(X_test)
    assert len(dumped_estimator.predict_proba(X_test)) == len(X_test)
    assert len(dumped_estimator.predict_log_proba(X_test)) == len(X_test)
    assert dumped_estimator.score(X_test, y_test) == accuracy_score(
        y_test, dumped_estimator.predict(X_test)
    )
    assert bool(dumped_estimator.get_params())
    assert len(dumped_estimator.hof) == dumped_estimator.keep_top_k
    assert "gen" in dumped_estimator[0]
    assert "fitness_max" in dumped_estimator[0]
    assert "fitness" in dumped_estimator[0]
    assert "fitness_std" in dumped_estimator[0]
    assert "fitness_min" in dumped_estimator[0]

    cv_results_ = dumped_estimator.cv_results_
    cv_result_keys = set(cv_results_.keys())

    assert "split0_test_score" in cv_result_keys
    assert "split1_test_score" in cv_result_keys
    assert "split2_test_score" in cv_result_keys
    assert "split0_train_score" in cv_result_keys
    assert "split1_train_score" in cv_result_keys
    assert "split2_train_score" in cv_result_keys
    assert "mean_test_score" in cv_result_keys
    assert "std_test_score" in cv_result_keys
    assert "rank_test_score" in cv_result_keys
    assert "mean_train_score" in cv_result_keys
    assert "std_train_score" in cv_result_keys
    assert "rank_train_score" in cv_result_keys
    assert "std_fit_time" in cv_result_keys
    assert "mean_score_time" in cv_result_keys
    assert "rank_n_features" in cv_result_keys
    assert "features" in cv_result_keys

    # delete dumped estimator
    os.remove(dump_file)
