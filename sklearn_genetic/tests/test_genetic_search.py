import pytest
from sklearn.datasets import load_digits, load_diabetes
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted
from sklearn.tree import DecisionTreeRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import make_scorer

import numpy as np
import os

from .. import GASearchCV
from ..space import Integer, Categorical, Continuous
from ..callbacks import (
    ThresholdStopping,
    DeltaThreshold,
    ConsecutiveStopping,
    TimerStopping,
    ProgressBar,
    ModelCheckpoint,
)

from ..schedules import ExponentialAdapter, InverseAdapter

data = load_digits()
label_names = data["target_names"]
y = data["target"]
X = data["data"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


def test_expected_ga_results():
    clf = SGDClassifier(loss="modified_huber", fit_intercept=True)
    generations = 6
    evolved_estimator = GASearchCV(
        clf,
        cv=3,
        scoring="accuracy",
        population_size=6,
        generations=generations,
        tournament_size=3,
        elitism=False,
        keep_top_k=4,
        param_grid={
            "l1_ratio": Continuous(0, 1),
            "alpha": Continuous(1e-4, 1, distribution="log-uniform"),
            "average": Categorical([True, False]),
            "max_iter": Integer(700, 1000),
        },
        verbose=False,
        algorithm="eaSimple",
        n_jobs=-1,
        return_train_score=True,
    )

    evolved_estimator.fit(X_train, y_train)

    assert check_is_fitted(evolved_estimator) is None
    assert "l1_ratio" in evolved_estimator.best_params_
    assert "alpha" in evolved_estimator.best_params_
    assert "average" in evolved_estimator.best_params_
    assert len(evolved_estimator) == generations + 1  # +1 random initial population
    assert len(evolved_estimator.predict(X_test)) == len(X_test)
    assert evolved_estimator.score(X_train, y_train) >= 0
    assert len(evolved_estimator.decision_function(X_test)) == len(X_test)
    assert len(evolved_estimator.predict_proba(X_test)) == len(X_test)
    assert len(evolved_estimator.predict_log_proba(X_test)) == len(X_test)
    assert evolved_estimator.score(X_test, y_test) == accuracy_score(
        y_test, evolved_estimator.predict(X_test)
    )
    assert bool(evolved_estimator.get_params())
    assert len(evolved_estimator.hof) == evolved_estimator.keep_top_k
    assert "gen" in evolved_estimator[0]
    assert "fitness_max" in evolved_estimator[0]
    assert "fitness" in evolved_estimator[0]
    assert "fitness_std" in evolved_estimator[0]
    assert "fitness_min" in evolved_estimator[0]

    cv_results_ = evolved_estimator.cv_results_
    cv_result_keys = set(cv_results_.keys())

    assert "param_l1_ratio" in cv_result_keys
    assert "param_alpha" in cv_result_keys
    assert "param_average" in cv_result_keys
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
    assert "params" in cv_result_keys


@pytest.mark.parametrize(
    "algorithm, callback",
    [
        ("eaSimple", ThresholdStopping(threshold=0.01)),
        ("eaMuPlusLambda", ThresholdStopping(threshold=0.01)),
        ("eaMuCommaLambda", ThresholdStopping(threshold=0.01)),
        ("eaSimple", TimerStopping(total_seconds=0.5)),
        ("eaMuPlusLambda", TimerStopping(total_seconds=2)),
        ("eaMuCommaLambda", TimerStopping(total_seconds=5)),
        ("eaSimple", ConsecutiveStopping(generations=3, metric="fitness")),
        ("eaMuPlusLambda", ConsecutiveStopping(generations=3, metric="fitness")),
        ("eaMuCommaLambda", ConsecutiveStopping(generations=3, metric="fitness")),
        ("eaSimple", DeltaThreshold(threshold=0.001, metric="fitness")),
        ("eaMuPlusLambda", DeltaThreshold(threshold=0.001, metric="fitness")),
        ("eaMuCommaLambda", DeltaThreshold(threshold=0.001, metric="fitness")),
        ("eaSimple", ProgressBar()),
        (
            "eaMuPlusLambda",
            ProgressBar(**{"desc": "my_custom_desc", "mininterval": 0.5}),
        ),
        ("eaMuCommaLambda", ProgressBar()),
        (
            "eaSimple",
            [
                ThresholdStopping(threshold=0.01),
                ConsecutiveStopping(generations=5, metric="fitness"),
                DeltaThreshold(threshold=0.001, metric="fitness"),
            ],
        ),
        (
            "eaMuPlusLambda",
            [
                ThresholdStopping(threshold=0.01),
                ConsecutiveStopping(generations=5, metric="fitness"),
                DeltaThreshold(threshold=0.001, metric="fitness"),
            ],
        ),
        (
            "eaMuCommaLambda",
            [
                ThresholdStopping(threshold=0.01),
                ConsecutiveStopping(generations=5, metric="fitness"),
                DeltaThreshold(threshold=0.001, metric="fitness"),
            ],
        ),
    ],
)
def test_expected_algorithms_callbacks(algorithm, callback):
    clf = SGDClassifier(loss="modified_huber", fit_intercept=True)
    generations = 8
    evolved_estimator = GASearchCV(
        clf,
        cv=2,
        scoring="accuracy",
        population_size=6,
        generations=generations,
        tournament_size=3,
        elitism=False,
        keep_top_k=4,
        param_grid={
            "l1_ratio": Continuous(0, 1),
            "alpha": Continuous(1e-4, 1, distribution="log-uniform"),
            "average": Categorical([True, False]),
            "max_iter": Integer(700, 1000),
        },
        verbose=True,
        algorithm=algorithm,
        n_jobs=-1,
    )

    evolved_estimator.fit(X_train, y_train, callbacks=callback)

    assert check_is_fitted(evolved_estimator) is None
    assert "l1_ratio" in evolved_estimator.best_params_
    assert "alpha" in evolved_estimator.best_params_
    assert "average" in evolved_estimator.best_params_
    assert len(evolved_estimator) <= generations + 1  # +1 random initial population
    assert len(evolved_estimator.predict(X_test)) == len(X_test)
    assert evolved_estimator.score(X_train, y_train) >= 0
    assert len(evolved_estimator.decision_function(X_test)) == len(X_test)
    assert len(evolved_estimator.predict_proba(X_test)) == len(X_test)
    assert len(evolved_estimator.predict_log_proba(X_test)) == len(X_test)
    assert evolved_estimator.score(X_test, y_test) == accuracy_score(
        y_test, evolved_estimator.predict(X_test)
    )
    assert bool(evolved_estimator.get_params())
    assert len(evolved_estimator.hof) <= evolved_estimator.keep_top_k
    assert "gen" in evolved_estimator[0]
    assert "fitness_max" in evolved_estimator[0]
    assert "fitness" in evolved_estimator[0]
    assert "fitness_std" in evolved_estimator[0]
    assert "fitness_min" in evolved_estimator[0]


@pytest.mark.parametrize(
    "param_grid",
    [
        (
            {
                "criterion": Categorical(["gini", "entropy"]),
                "max_depth": Integer(2, 20),
                "max_leaf_nodes": Integer(2, 30),
            }
        ),
        ({"ccp_alpha": Continuous(0.01, 0.5), "max_depth": Integer(2, 20)}),
        (
            {
                "ccp_alpha": Continuous(0.01, 0.5),
                "criterion": Categorical(["gini", "entropy"]),
            }
        ),
        (
            {
                "max_depth": Integer(2, 20),
                "max_leaf_nodes": Integer(2, 30),
            }
        ),
    ],
)
def test_missing_data_types(param_grid):
    clf = DecisionTreeClassifier()
    generations = 6
    evolved_estimator = GASearchCV(
        clf,
        cv=2,
        scoring="accuracy",
        population_size=3,
        generations=generations,
        tournament_size=3,
        elitism=True,
        param_grid=param_grid,
        verbose=False,
        n_jobs=-1,
    )

    evolved_estimator.fit(X_train, y_train)

    assert check_is_fitted(evolved_estimator) is None
    assert set(param_grid.keys()) == set(evolved_estimator.best_params_.keys())


def test_negative_criteria():
    data_boston = load_diabetes()

    y_diabetes = data_boston["target"]
    X_diabetes = data_boston["data"]

    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
        X_diabetes, y_diabetes, test_size=0.33, random_state=42
    )

    clf = DecisionTreeRegressor()
    generations = 6
    evolved_estimator = GASearchCV(
        clf,
        cv=3,
        scoring="neg_max_error",
        population_size=5,
        generations=generations,
        tournament_size=3,
        elitism=True,
        crossover_probability=0.9,
        mutation_probability=0.05,
        param_grid={
            "ccp_alpha": Continuous(0, 1),
            "criterion": Categorical(["squared_error", "absolute_error"]),
            "max_depth": Integer(2, 20),
            "min_samples_split": Integer(2, 30),
        },
        criteria="min",
        n_jobs=-1,
    )

    evolved_estimator.fit(X_train_b, y_train_b)

    assert check_is_fitted(evolved_estimator) is None
    assert "ccp_alpha" in evolved_estimator.best_params_
    assert "criterion" in evolved_estimator.best_params_
    assert "max_depth" in evolved_estimator.best_params_
    assert "min_samples_split" in evolved_estimator.best_params_
    assert len(evolved_estimator.predict(X_test_b)) == len(X_test_b)
    assert evolved_estimator.score(X_train_b, y_train_b) <= 0


def test_wrong_criteria():
    clf = SGDClassifier(loss="modified_huber", fit_intercept=True)
    generations = 8
    with pytest.raises(Exception) as excinfo:
        evolved_estimator = GASearchCV(
            clf,
            cv=3,
            scoring="accuracy",
            population_size=5,
            generations=generations,
            tournament_size=3,
            elitism=False,
            param_grid={
                "l1_ratio": Continuous(0, 1),
                "alpha": Continuous(1e-4, 1),
                "average": Categorical([True, False]),
            },
            verbose=False,
            criteria="maximization",
        )
    assert str(excinfo.value) == "Criteria must be one of ['max', 'min'], got maximization instead"


def test_wrong_estimator():
    clf = KMeans()
    generations = 8
    with pytest.raises(Exception) as excinfo:
        evolved_estimator = GASearchCV(
            clf,
            cv=3,
            scoring="accuracy",
            population_size=5,
            generations=generations,
            tournament_size=3,
            elitism=False,
            param_grid={
                "l1_ratio": Continuous(0, 1),
                "alpha": Continuous(1e-4, 1),
                "average": Categorical([True, False]),
            },
            verbose=False,
            criteria="maximization",
        )
    assert str(excinfo.value) == "KMeans() is not a valid Sklearn classifier or regressor"


def test_wrong_get_item():
    clf = SGDClassifier(loss="modified_huber", fit_intercept=True)
    generations = 8
    evolved_estimator = GASearchCV(
        clf,
        cv=3,
        scoring="accuracy",
        population_size=5,
        generations=generations,
        tournament_size=3,
        elitism=False,
        param_grid={
            "l1_ratio": Continuous(0, 1),
            "alpha": Continuous(1e-4, 1),
            "average": Categorical([True, False]),
        },
        verbose=False,
        criteria="max",
    )
    with pytest.raises(Exception) as excinfo:
        value = evolved_estimator[0]
    assert (
        str(excinfo.value)
        == "This GASearchCV instance is not fitted yet or used refit=False. Call 'fit' with "
        "appropriate arguments before using this estimator."
    )


def test_iterator():
    clf = DecisionTreeClassifier()
    generations = 4
    evolved_estimator = GASearchCV(
        clf,
        cv=3,
        scoring="accuracy",
        population_size=3,
        generations=generations,
        tournament_size=3,
        elitism=True,
        param_grid={
            "min_weight_fraction_leaf": Continuous(0, 0.5),
            "max_depth": Integer(2, 20),
            "max_leaf_nodes": Integer(2, 30),
        },
        verbose=False,
        n_jobs=-1,
    )
    evolved_estimator.fit(X_train, y_train)

    i = iter(evolved_estimator)
    assert next(i) == evolved_estimator[0]
    assert next(i) == evolved_estimator[1]


def test_wrong_algorithm():
    clf = SGDClassifier(loss="modified_huber", fit_intercept=True)
    generations = 6
    evolved_estimator = GASearchCV(
        clf,
        cv=3,
        scoring="accuracy",
        population_size=5,
        generations=generations,
        tournament_size=3,
        elitism=False,
        param_grid={
            "l1_ratio": Continuous(0, 1),
            "alpha": Continuous(1e-4, 1),
            "average": Categorical([True, False]),
        },
        verbose=False,
        criteria="max",
        algorithm="genetic",
    )
    with pytest.raises(Exception) as excinfo:
        evolved_estimator.fit(X_train, y_train)
    assert (
        str(excinfo.value)
        == "The algorithm genetic is not supported, please select one from ['eaSimple', 'eaMuPlusLambda', 'eaMuCommaLambda']"
    )


def test_no_param_grid():
    clf = SGDClassifier(loss="modified_huber", fit_intercept=True)
    generations = 8
    with pytest.raises(Exception) as excinfo:
        evolved_estimator = GASearchCV(
            clf,
            cv=3,
            scoring="accuracy",
            population_size=12,
            generations=generations,
            tournament_size=3,
            elitism=False,
            verbose=False,
            criteria="max",
        )

    assert str(excinfo.value) == "param_grid can not be empty"


def test_expected_ga_multimetric():
    clf = SGDClassifier(loss="modified_huber", fit_intercept=True)
    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": make_scorer(balanced_accuracy_score),
    }

    generations = 6
    evolved_estimator = GASearchCV(
        clf,
        cv=3,
        scoring=scoring,
        population_size=6,
        generations=generations,
        tournament_size=3,
        elitism=True,
        keep_top_k=4,
        param_grid={
            "l1_ratio": Continuous(0, 1),
            "alpha": Continuous(1e-4, 1, distribution="log-uniform"),
            "average": Categorical([True, False]),
            "max_iter": Integer(700, 1000),
        },
        verbose=False,
        algorithm="eaSimple",
        n_jobs=-1,
        return_train_score=True,
        refit="accuracy",
    )

    evolved_estimator.fit(X_train, y_train)

    assert check_is_fitted(evolved_estimator) is None
    assert "l1_ratio" in evolved_estimator.best_params_
    assert "alpha" in evolved_estimator.best_params_
    assert "average" in evolved_estimator.best_params_
    assert len(evolved_estimator) == generations + 1  # +1 random initial population
    assert len(evolved_estimator.predict(X_test)) == len(X_test)
    assert evolved_estimator.score(X_train, y_train) >= 0
    assert len(evolved_estimator.decision_function(X_test)) == len(X_test)
    assert len(evolved_estimator.predict_proba(X_test)) == len(X_test)
    assert len(evolved_estimator.predict_log_proba(X_test)) == len(X_test)
    assert evolved_estimator.score(X_test, y_test) == accuracy_score(
        y_test, evolved_estimator.predict(X_test)
    )
    assert bool(evolved_estimator.get_params())
    assert len(evolved_estimator.hof) == evolved_estimator.keep_top_k
    assert "gen" in evolved_estimator[0]
    assert "fitness_max" in evolved_estimator[0]
    assert "fitness" in evolved_estimator[0]
    assert "fitness_std" in evolved_estimator[0]
    assert "fitness_min" in evolved_estimator[0]

    cv_results_ = evolved_estimator.cv_results_
    cv_result_keys = set(cv_results_.keys())

    assert "param_l1_ratio" in cv_result_keys
    assert "param_alpha" in cv_result_keys
    assert "param_average" in cv_result_keys
    assert "std_fit_time" in cv_result_keys
    assert "mean_score_time" in cv_result_keys
    assert "params" in cv_result_keys

    for metric in scoring.keys():
        assert f"split0_test_{metric}" in cv_result_keys
        assert f"split1_test_{metric}" in cv_result_keys
        assert f"split2_test_{metric}" in cv_result_keys
        assert f"split0_train_{metric}" in cv_result_keys
        assert f"split1_train_{metric}" in cv_result_keys
        assert f"split2_train_{metric}" in cv_result_keys
        assert f"mean_test_{metric}" in cv_result_keys
        assert f"std_test_{metric}" in cv_result_keys
        assert f"rank_test_{metric}" in cv_result_keys
        assert f"mean_train_{metric}" in cv_result_keys
        assert f"std_train_{metric}" in cv_result_keys
        assert f"rank_train_{metric}" in cv_result_keys


def test_expected_ga_callable_score():
    clf = SGDClassifier(loss="modified_huber", fit_intercept=True)
    generations = 6
    scoring = make_scorer(accuracy_score)
    evolved_estimator = GASearchCV(
        clf,
        cv=3,
        scoring=scoring,
        population_size=6,
        generations=generations,
        tournament_size=3,
        elitism=False,
        keep_top_k=4,
        param_grid={
            "l1_ratio": Continuous(0, 1),
            "alpha": Continuous(1e-4, 1, distribution="log-uniform"),
            "average": Categorical([True, False]),
            "max_iter": Integer(700, 1000),
        },
        verbose=False,
        algorithm="eaSimple",
        n_jobs=-1,
        return_train_score=True,
    )

    evolved_estimator.fit(X_train, y_train)

    assert check_is_fitted(evolved_estimator) is None
    assert "l1_ratio" in evolved_estimator.best_params_
    assert "alpha" in evolved_estimator.best_params_
    assert "average" in evolved_estimator.best_params_
    assert len(evolved_estimator) == generations + 1  # +1 random initial population
    assert len(evolved_estimator.predict(X_test)) == len(X_test)
    assert evolved_estimator.score(X_train, y_train) >= 0
    assert len(evolved_estimator.decision_function(X_test)) == len(X_test)
    assert len(evolved_estimator.predict_proba(X_test)) == len(X_test)
    assert len(evolved_estimator.predict_log_proba(X_test)) == len(X_test)
    assert evolved_estimator.score(X_test, y_test) == accuracy_score(
        y_test, evolved_estimator.predict(X_test)
    )
    assert bool(evolved_estimator.get_params())
    assert len(evolved_estimator.hof) == evolved_estimator.keep_top_k
    assert "gen" in evolved_estimator[0]
    assert "fitness_max" in evolved_estimator[0]
    assert "fitness" in evolved_estimator[0]
    assert "fitness_std" in evolved_estimator[0]
    assert "fitness_min" in evolved_estimator[0]

    cv_results_ = evolved_estimator.cv_results_
    cv_result_keys = set(cv_results_.keys())

    assert "param_l1_ratio" in cv_result_keys
    assert "param_alpha" in cv_result_keys
    assert "param_average" in cv_result_keys
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
    assert "params" in cv_result_keys


def test_expected_ga_schedulers():
    clf = SGDClassifier(loss="modified_huber", fit_intercept=True)
    generations = 6

    mutation_scheduler = ExponentialAdapter(initial_value=0.6, adaptive_rate=0.01, end_value=0.2)
    crossover_scheduler = InverseAdapter(initial_value=0.4, adaptive_rate=0.01, end_value=0.3)

    evolved_estimator = GASearchCV(
        clf,
        cv=3,
        scoring="accuracy",
        population_size=6,
        generations=generations,
        tournament_size=3,
        mutation_probability=mutation_scheduler,
        crossover_probability=crossover_scheduler,
        elitism=False,
        keep_top_k=4,
        param_grid={
            "l1_ratio": Continuous(0, 1),
            "alpha": Continuous(1e-4, 1, distribution="log-uniform"),
            "average": Categorical([True, False]),
            "max_iter": Integer(700, 1000),
        },
        warm_start_configs=[
            {"l1_ratio": 0.5, "alpha": 0.5, "average": False, "max_iter": 400},
            {"l1_ratio": 0.2, "alpha": 0.8, "average": True, "max_iter": 400},
        ],
        verbose=False,
        algorithm="eaSimple",
        n_jobs=-1,
        return_train_score=True,
    )

    evolved_estimator.fit(X_train, y_train)

    assert check_is_fitted(evolved_estimator) is None
    assert "l1_ratio" in evolved_estimator.best_params_
    assert "alpha" in evolved_estimator.best_params_
    assert "average" in evolved_estimator.best_params_
    assert len(evolved_estimator) == generations + 1  # +1 random initial population
    assert len(evolved_estimator.predict(X_test)) == len(X_test)
    assert evolved_estimator.score(X_train, y_train) >= 0
    assert len(evolved_estimator.decision_function(X_test)) == len(X_test)
    assert len(evolved_estimator.predict_proba(X_test)) == len(X_test)
    assert len(evolved_estimator.predict_log_proba(X_test)) == len(X_test)
    assert evolved_estimator.score(X_test, y_test) == accuracy_score(
        y_test, evolved_estimator.predict(X_test)
    )
    assert bool(evolved_estimator.get_params())
    assert len(evolved_estimator.hof) == evolved_estimator.keep_top_k
    assert "gen" in evolved_estimator[0]
    assert "fitness_max" in evolved_estimator[0]
    assert "fitness" in evolved_estimator[0]
    assert "fitness_std" in evolved_estimator[0]
    assert "fitness_min" in evolved_estimator[0]

    cv_results_ = evolved_estimator.cv_results_
    cv_result_keys = set(cv_results_.keys())

    assert "param_l1_ratio" in cv_result_keys
    assert "param_alpha" in cv_result_keys
    assert "param_average" in cv_result_keys
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
    assert "params" in cv_result_keys

    assert crossover_scheduler.current_value + mutation_scheduler.current_value <= 1


def test_checkpoint_functionality():
    clf = SGDClassifier(loss="modified_huber", fit_intercept=True)
    gen = 5
    evolved_estimator = GASearchCV(
        clf,
        cv=3,
        scoring="accuracy",
        population_size=6,
        generations=gen,
        tournament_size=3,
        param_grid={
            "l1_ratio": Continuous(0, 1),
            "alpha": Continuous(1e-4, 1),
            "average": Categorical([True, False]),
        },
    )
    checkpoint_path = "test_checkpoint.pkl"
    checkpoint = ModelCheckpoint(checkpoint_path=checkpoint_path)  # noqa
    evolved_estimator.fit(X_train, y_train, callbacks=checkpoint)

    checkpoint_data = checkpoint.load()

    assert "estimator" in checkpoint_data["estimator_state"]
    assert "algorithm" in checkpoint_data["estimator_state"]
    assert "logbook" in checkpoint_data

    restored_estimator = GASearchCV(**checkpoint_data["estimator_state"])

    assert restored_estimator.algorithm == checkpoint_data["estimator_state"]["algorithm"]  # noqa

    assert len(checkpoint_data["logbook"]) == gen + 1

    restored_estimator.save(checkpoint_path)

    test_estimator = GASearchCV(
        clf,
        param_grid={
            "l1_ratio": Continuous(0, 1),
            "alpha": Continuous(1e-1, 1),
            "average": Categorical([False, True]),
        },
    )

    test_estimator.load(checkpoint_path)

    assert restored_estimator.algorithm == test_estimator.algorithm  # noqa
    assert restored_estimator.scoring == test_estimator.scoring  # noqa
    assert restored_estimator.generations == test_estimator.generations  # noqa

    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
