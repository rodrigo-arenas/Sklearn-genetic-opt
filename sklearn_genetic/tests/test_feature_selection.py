import pytest
from deap import tools
from sklearn.base import clone
from sklearn.datasets import load_iris, load_diabetes
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import make_scorer
import numpy as np

from .. import (
    EvolutionConfig,
    GAFeatureSelectionCV,
    OptimizationConfig,
    PopulationConfig,
    RuntimeConfig,
)
from .. import genetic_search
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


def test_default_n_jobs_is_none():
    estimator = GAFeatureSelectionCV(
        SGDClassifier(loss="modified_huber", fit_intercept=True),
        generations=1,
        population_size=2,
        verbose=False,
    )

    assert estimator.n_jobs is None
    assert estimator.get_params()["n_jobs"] is None
    assert estimator.parallel_backend == "auto"
    assert estimator.get_params()["parallel_backend"] == "auto"
    assert estimator.population_initializer == "smart"
    assert estimator.get_params()["population_initializer"] == "smart"


def test_feature_selection_accepts_grouped_config_objects():
    evolution_config = EvolutionConfig(population_size=5, generations=2, tournament_size=2)
    population_config = PopulationConfig(initializer="smart")
    runtime_config = RuntimeConfig(n_jobs=1, parallel_backend="auto", verbose=False)
    optimization_config = OptimizationConfig(
        local_search=True,
        local_search_top_k=2,
        diversity_control=True,
        adaptive_selection=True,
        offspring_diversity_retries=2,
        fitness_sharing=True,
    )

    estimator = GAFeatureSelectionCV(
        DecisionTreeClassifier(random_state=42),
        cv=2,
        scoring="accuracy",
        max_features=5,
        evolution_config=evolution_config,
        population_config=population_config,
        runtime_config=runtime_config,
        optimization_config=optimization_config,
    )

    assert estimator.population_size == 5
    assert estimator.generations == 2
    assert estimator.tournament_size == 2
    assert estimator.population_initializer == "smart"
    assert estimator.n_jobs == 1
    assert estimator.verbose is False
    assert estimator.local_search is True
    assert estimator.local_search_top_k == 2
    assert estimator.diversity_control is True
    assert estimator.adaptive_selection is True
    assert estimator.offspring_diversity_retries == 2
    assert estimator.fitness_sharing is True
    assert estimator.get_params()["evolution_config"] is evolution_config
    assert estimator.get_params()["runtime_config"] is runtime_config


def test_feature_selection_grouped_config_is_sklearn_clone_compatible():
    estimator = GAFeatureSelectionCV(
        DecisionTreeClassifier(random_state=42),
        cv=2,
        scoring="accuracy",
        evolution_config=EvolutionConfig(population_size=4, generations=1),
        population_config=PopulationConfig(initializer="smart"),
        runtime_config=RuntimeConfig(verbose=False),
    )

    cloned = clone(estimator)

    assert cloned.population_size == 4
    assert cloned.generations == 1
    assert cloned.population_initializer == "smart"
    assert cloned.verbose is False
    assert "runtime_config" in cloned.get_params()


def test_feature_selection_rejects_invalid_error_score():
    with pytest.raises(ValueError) as excinfo:
        GAFeatureSelectionCV(
            DecisionTreeClassifier(),
            error_score="warn",
        )

    assert str(excinfo.value) == "error_score must be numeric or 'raise', got 'warn' instead"


@pytest.mark.parametrize("error_score", ["raise", np.nan, 0.0])
def test_feature_selection_accepts_valid_error_score(error_score):
    estimator = GAFeatureSelectionCV(
        DecisionTreeClassifier(),
        error_score=error_score,
    )

    if isinstance(error_score, float) and np.isnan(error_score):
        assert np.isnan(estimator.error_score)
    else:
        assert estimator.error_score == error_score


def test_smart_population_initializer_creates_diverse_feature_masks():
    estimator = GAFeatureSelectionCV(
        DecisionTreeClassifier(random_state=42),
        cv=2,
        scoring="accuracy",
        population_size=5,
        generations=1,
        max_features=3,
        verbose=False,
    )
    estimator.n_features = 6
    estimator.features_proportion = estimator.max_features / estimator.n_features

    estimator._register()
    population = [list(individual) for individual in estimator._pop]
    selected_counts = [sum(individual) for individual in population]

    try:
        assert len(population) == estimator.population_size
        assert len({tuple(individual) for individual in population}) == len(population)
        assert max(selected_counts) <= estimator.max_features
        assert min(selected_counts) >= 1
        assert len(set(selected_counts)) > 1
    finally:
        del genetic_search.creator.FitnessMax
        del genetic_search.creator.Individual


def test_optimizer_telemetry_is_recorded_for_feature_selection():
    generations = 2
    estimator = GAFeatureSelectionCV(
        DecisionTreeClassifier(random_state=42),
        cv=2,
        scoring=lambda estimator, X, y: 1.0,
        population_size=4,
        generations=generations,
        algorithm="eaSimple",
        verbose=False,
        n_jobs=1,
    )

    estimator.fit(X_train, y_train)

    telemetry_fields = [
        "population_size",
        "fitness_best",
        "unique_individuals",
        "unique_individual_ratio",
        "genotype_diversity",
        "fitness_improvement",
        "fitness_improved",
        "stagnation_generations",
        "best_generation",
    ]

    for field in telemetry_fields:
        assert field in estimator.history
        assert field in estimator[0]
        assert len(estimator.history[field]) == generations + 1

    assert estimator.history["population_size"][-1] == estimator.population_size
    assert 0 <= estimator.history["unique_individual_ratio"][-1] <= 1
    assert 0 <= estimator.history["genotype_diversity"][-1] <= 1
    assert estimator.history["best_generation"][-1] == 0
    assert estimator.history["stagnation_generations"][-1] == generations


def test_feature_masks_are_repaired_before_evaluation():
    estimator = GAFeatureSelectionCV(
        DecisionTreeClassifier(),
        cv=3,
        scoring="accuracy",
        max_features=1,
        verbose=False,
        return_train_score=True,
    )
    estimator.X_ = X_train[:6]
    estimator.y_ = y_train[:6]
    estimator.n_splits_ = 3
    estimator.refit_metric = "score"
    estimator.metrics_list = ["score"]
    estimator.logbook = tools.Logbook()
    estimator.fit_stats_ = genetic_search._create_fit_stats()

    individual = [1, 1]
    estimator._repair_individual(individual)

    assert sum(individual) == 1
    assert set(individual).issubset({0, 1})


def test_feature_masks_are_repaired_when_max_features_is_missing():
    estimator = GAFeatureSelectionCV(
        DecisionTreeClassifier(),
        cv=3,
        scoring="accuracy",
        verbose=False,
    )
    if hasattr(estimator, "max_features"):
        delattr(estimator, "max_features")

    individual = [1, 1, 0]
    repaired = estimator._repair_individual(individual)

    assert repaired is individual
    assert sum(individual) >= 1
    assert set(individual).issubset({0, 1})


def test_feature_selection_genetic_operations_respect_max_features():
    estimator = GAFeatureSelectionCV(
        DecisionTreeClassifier(random_state=42),
        cv=2,
        scoring="accuracy",
        population_size=4,
        generations=1,
        max_features=2,
        verbose=False,
    )
    estimator.n_features = 6
    estimator.features_proportion = estimator.max_features / estimator.n_features

    estimator._register()

    try:
        first = genetic_search.creator.Individual([1, 1, 1, 0, 0, 0])
        second = genetic_search.creator.Individual([0, 0, 0, 1, 1, 1])

        offspring = estimator.mate(first, second)
        mutated = estimator.mutate(genetic_search.creator.Individual([1, 1, 1, 1, 1, 1]))

        for individual in (*offspring, *mutated, estimator.toolbox.individual()):
            assert 1 <= sum(individual) <= estimator.max_features
            assert set(individual).issubset({0, 1})
    finally:
        del genetic_search.creator.FitnessMax
        del genetic_search.creator.Individual


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
        scoring="neg_mean_squared_error",
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
    assert (
        str(excinfo.value)
        == "KMeans() is not a valid Sklearn classifier, regressor, or outlier detector"
    )


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


def test_feature_selection_fit_with_groups_supports_group_kfold():
    """#338: GAFeatureSelectionCV.fit(..., groups=...) enables group-aware CV.

    Every materialized split must keep train and test groups disjoint, and the
    selector must still produce a valid support mask.
    """
    from sklearn.model_selection import GroupKFold

    groups = np.arange(X_train.shape[0]) % 3
    selector = GAFeatureSelectionCV(
        DecisionTreeClassifier(random_state=0),
        cv=GroupKFold(n_splits=3),
        scoring="accuracy",
        population_size=4,
        generations=2,
        verbose=False,
    )

    selector.fit(X_train, y_train, groups=groups)

    assert selector.support_.shape[0] == X_train.shape[1]
    assert len(selector._cv_splits) == 3
    for train_idx, test_idx in selector._cv_splits:
        assert set(groups[train_idx]).isdisjoint(set(groups[test_idx]))


def test_feature_selection_forwards_sample_weight_and_slices_only_features():
    """#339: GAFeatureSelectionCV forwards sample_weight while only X is
    feature-sliced.

    Candidate fits see a feature-sliced X but per-fold sample_weight slices;
    the final refit sees the full weight vector.
    """
    recorded = []

    class WeightRecordingTree(DecisionTreeClassifier):
        def fit(self, X, y, sample_weight=None):
            recorded.append((X.shape[1], None if sample_weight is None else len(sample_weight)))
            return super().fit(X, y, sample_weight=sample_weight)

    n_samples, n_features_total = X_train.shape
    weights = np.linspace(0.5, 1.5, n_samples)
    selector = GAFeatureSelectionCV(
        WeightRecordingTree(random_state=0),
        cv=2,
        scoring="accuracy",
        population_size=4,
        generations=2,
        verbose=False,
    )

    selector.fit(X_train, y_train, sample_weight=weights)

    assert recorded, "no fit calls were recorded"
    # Every fit received weights, and X never grew beyond the original features.
    assert all(w is not None for _, w in recorded)
    assert all(f <= n_features_total for f, _ in recorded)
    # Candidate CV fits get per-fold weight slices; the refit gets all samples.
    assert all(w < n_samples for _, w in recorded[:-1])
    assert recorded[-1][1] == n_samples
