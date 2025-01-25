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

data = load_iris()
label_names = data["target_names"]
y = data["target"]
X = data["data"]

noise = np.random.uniform(1, 4, size=(X.shape[0], 10))

X = np.hstack((X, noise))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


def test_expected_ga_results():
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
    features = evolved_estimator.support_

    assert check_is_fitted(evolved_estimator) is None
    assert features.shape[0] == X.shape[1]
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
    evolved_estimator = GAFeatureSelectionCV(
        clf,
        cv=2,
        scoring="accuracy",
        population_size=6,
        generations=generations,
        tournament_size=3,
        elitism=False,
        keep_top_k=4,
        verbose=True,
        algorithm=algorithm,
        n_jobs=-1,
    )

    evolved_estimator.fit(X_train, y_train, callbacks=callback)
    features = evolved_estimator.support_

    assert check_is_fitted(evolved_estimator) is None
    assert features.shape[0] == X.shape[1]
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
    assert len(evolved_estimator.hof) == evolved_estimator.keep_top_k
    assert "gen" in evolved_estimator[0]
    assert "fitness_max" in evolved_estimator[0]
    assert "fitness" in evolved_estimator[0]
    assert "fitness_std" in evolved_estimator[0]
    assert "fitness_min" in evolved_estimator[0]


def test_negative_criteria():
    data_boston = load_diabetes()

    y_diabetes = data_boston["target"]
    X_diabetes = data_boston["data"]

    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
        X_diabetes, y_diabetes, test_size=0.33, random_state=42
    )

    clf = DecisionTreeRegressor()
    generations = 6
    evolved_estimator = GAFeatureSelectionCV(
        clf,
        cv=3,
        scoring="neg_max_error",
        population_size=5,
        generations=generations,
        tournament_size=3,
        elitism=True,
        crossover_probability=0.9,
        mutation_probability=0.05,
        criteria="min",
        n_jobs=-1,
    )

    evolved_estimator.fit(X_train_b, y_train_b)

    assert check_is_fitted(evolved_estimator) is None
    assert len(evolved_estimator.predict(X_test_b)) == len(X_test_b)


def test_wrong_criteria():
    clf = SGDClassifier(loss="modified_huber", fit_intercept=True)
    generations = 8
    with pytest.raises(Exception) as excinfo:
        evolved_estimator = GAFeatureSelectionCV(
            clf,
            cv=3,
            scoring="accuracy",
            population_size=5,
            generations=generations,
            tournament_size=3,
            elitism=False,
            verbose=False,
            criteria="maximization",
        )
    assert str(excinfo.value) == "Criteria must be one of ['max', 'min'], got maximization instead"


def test_wrong_estimator():
    clf = KMeans()
    generations = 8
    with pytest.raises(Exception) as excinfo:
        evolved_estimator = GAFeatureSelectionCV(
            clf,
            cv=3,
            scoring="accuracy",
            population_size=5,
            generations=generations,
            tournament_size=3,
            elitism=False,
            verbose=False,
            criteria="maximization",
        )
    assert str(excinfo.value) == "KMeans() is not a valid Sklearn classifier or regressor"


def test_wrong_get_item():
    clf = SGDClassifier(loss="modified_huber", fit_intercept=True)
    generations = 8
    evolved_estimator = GAFeatureSelectionCV(
        clf,
        cv=3,
        scoring="accuracy",
        population_size=5,
        generations=generations,
        tournament_size=3,
        elitism=False,
        verbose=False,
        criteria="max",
    )
    with pytest.raises(Exception) as excinfo:
        value = evolved_estimator[0]
    assert (
        str(excinfo.value)
        == "This GAFeatureSelectionCV instance is not fitted yet or used refit=False. Call 'fit' with "
        "appropriate arguments before using this estimator."
    )


def test_iterator():
    clf = DecisionTreeClassifier()
    generations = 4
    evolved_estimator = GAFeatureSelectionCV(
        clf,
        cv=3,
        scoring="accuracy",
        population_size=3,
        generations=generations,
        tournament_size=3,
        elitism=True,
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
    evolved_estimator = GAFeatureSelectionCV(
        clf,
        cv=3,
        scoring="accuracy",
        population_size=5,
        generations=generations,
        tournament_size=3,
        elitism=False,
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


def test_expected_ga_max_features():
    clf = SGDClassifier(loss="modified_huber", fit_intercept=True)
    noise_train = np.random.uniform(0, 16, size=(X_train.shape[0], 16))
    noise_test = np.random.uniform(0, 16, size=(X_test.shape[0], 16))
    X_noise_train = np.hstack((X_train, noise_train))
    X_noise_test = np.hstack((X_test, noise_test))
    generations = 8
    max_features = 10
    evolved_estimator = GAFeatureSelectionCV(
        clf,
        cv=3,
        scoring="accuracy",
        population_size=10,
        generations=generations,
        tournament_size=3,
        elitism=False,
        keep_top_k=4,
        max_features=max_features,
        verbose=False,
        algorithm="eaSimple",
        n_jobs=-1,
        return_train_score=True,
    )

    evolved_estimator.fit(X_noise_train, y_train)
    features = evolved_estimator.support_

    assert check_is_fitted(evolved_estimator) is None
    assert features.shape[0] == X_noise_train.shape[1]
    assert sum(features) <= max_features
    assert len(evolved_estimator) == generations + 1  # +1 random initial population
    assert len(evolved_estimator.predict(X_noise_test)) == len(X_noise_test)
    assert evolved_estimator.score(X_noise_train, y_train) >= 0
    assert len(evolved_estimator.decision_function(X_noise_test)) == len(X_noise_test)
    assert len(evolved_estimator.predict_proba(X_noise_test)) == len(X_noise_test)
    assert len(evolved_estimator.predict_log_proba(X_noise_test)) == len(X_noise_test)
    assert evolved_estimator.score(X_noise_test, y_test) == accuracy_score(
        evolved_estimator.predict(X_noise_test), y_test
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


def test_expected_ga_multimetric():
    clf = SGDClassifier(loss="modified_huber", fit_intercept=True)
    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": make_scorer(balanced_accuracy_score),
    }

    generations = 6
    evolved_estimator = GAFeatureSelectionCV(
        clf,
        cv=3,
        scoring=scoring,
        population_size=6,
        generations=generations,
        tournament_size=3,
        elitism=False,
        keep_top_k=4,
        verbose=False,
        algorithm="eaSimple",
        n_jobs=-1,
        return_train_score=True,
        refit="accuracy",
    )

    evolved_estimator.fit(X_train, y_train)
    features = evolved_estimator.support_

    assert check_is_fitted(evolved_estimator) is None
    assert features.shape[0] == X.shape[1]
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

    assert "std_fit_time" in cv_result_keys
    assert "features" in cv_result_keys

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
    scoring = make_scorer(accuracy_score)
    generations = 6
    evolved_estimator = GAFeatureSelectionCV(
        clf,
        cv=3,
        scoring=scoring,
        population_size=6,
        generations=generations,
        tournament_size=3,
        elitism=False,
        keep_top_k=4,
        verbose=False,
        algorithm="eaSimple",
        n_jobs=-1,
        return_train_score=True,
        refit="accuracy",
    )

    evolved_estimator.fit(X_train, y_train)
    features = evolved_estimator.support_

    assert check_is_fitted(evolved_estimator) is None
    assert features.shape[0] == X.shape[1]
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


def test_expected_ga_schedulers():
    clf = SGDClassifier(loss="modified_huber", fit_intercept=True)
    generations = 6
    mutation_scheduler = ExponentialAdapter(initial_value=0.6, adaptive_rate=0.01, end_value=0.2)
    crossover_scheduler = InverseAdapter(initial_value=0.4, adaptive_rate=0.01, end_value=0.3)

    evolved_estimator = GAFeatureSelectionCV(
        clf,
        cv=3,
        scoring="accuracy",
        population_size=6,
        generations=generations,
        mutation_probability=mutation_scheduler,
        crossover_probability=crossover_scheduler,
        tournament_size=3,
        elitism=False,
        keep_top_k=4,
        verbose=False,
        algorithm="eaSimple",
        n_jobs=-1,
        return_train_score=True,
        refit="accuracy",
    )

    evolved_estimator.fit(X_train, y_train)
    features = evolved_estimator.support_

    assert check_is_fitted(evolved_estimator) is None
    assert features.shape[0] == X.shape[1]
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

    assert crossover_scheduler.current_value + mutation_scheduler.current_value <= 1
