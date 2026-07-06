from types import SimpleNamespace

import pytest
from sklearn.datasets import load_iris
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from ..callbacks.model_checkpoint import ModelCheckpoint
from ..genetic_search import GAFeatureSelectionCV, GASearchCV, _safe_estimator_check
from ..space import Integer, Space


def test_safe_estimator_check_returns_false_for_attribute_errors():
    def check(_estimator):
        raise AttributeError("missing estimator tag")

    assert _safe_estimator_check(check, DecisionTreeClassifier()) is False


def test_space_sample_warm_start_fills_missing_parameters():
    space = Space({"provided": Integer(1, 1), "sampled": Integer(2, 2)})

    # "provided" is given (and in range); "sampled" is missing and filled by sampling.
    sampled_params = space.sample_warm_start({"provided": 1})

    assert sampled_params == {"provided": 1, "sampled": 2}


def test_ga_search_save_and_load_round_trip(tmp_path, capsys):
    search = GASearchCV(
        estimator=DecisionTreeClassifier(),
        param_grid={"max_depth": Integer(1, 2)},
        cv=2,
        scoring="accuracy",
    )
    search.extra_state = "persisted"
    search.logbook = [{"generation": 0}]
    checkpoint_path = tmp_path / "ga_search.pkl"

    search.save(checkpoint_path)

    restored = GASearchCV(
        estimator=DecisionTreeClassifier(),
        param_grid={"max_depth": Integer(1, 2)},
        cv=2,
        scoring="accuracy",
    )
    restored.load(checkpoint_path)
    output = capsys.readouterr().out

    assert "GASearchCV model successfully saved" in output
    assert "GASearchCV model successfully loaded" in output
    assert restored.extra_state == "persisted"
    assert restored.logbook == [{"generation": 0}]


def test_fitted_ga_search_save_and_load_round_trip(tmp_path):
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.25,
        stratify=y,
        random_state=0,
    )

    search = GASearchCV(
        estimator=DecisionTreeClassifier(random_state=0),
        param_grid={"max_depth": Integer(1, 3)},
        cv=2,
        scoring="accuracy",
        population_size=3,
        generations=1,
        verbose=False,
    )
    search.fit(X_train, y_train)
    checkpoint_path = tmp_path / "fitted_ga_search.pkl"

    search.save(checkpoint_path)

    restored = GASearchCV(
        estimator=DecisionTreeClassifier(random_state=0),
        param_grid={"max_depth": Integer(1, 3)},
        cv=2,
        scoring="accuracy",
    )
    restored.load(checkpoint_path)

    assert restored.best_score_ == search.best_score_
    assert restored.best_params_ == search.best_params_
    assert restored.predict(X_test).shape[0] == X_test.shape[0]


def test_ga_search_save_and_load_errors_are_reported(tmp_path, capsys):
    search = GASearchCV(
        estimator=DecisionTreeClassifier(),
        param_grid={"max_depth": Integer(1, 2)},
        cv=2,
        scoring="accuracy",
    )

    search.save(tmp_path / "missing" / "ga_search.pkl")
    search.load(tmp_path / "missing.pkl")
    output = capsys.readouterr().out

    assert "Error saving GASearchCV" in output
    assert "Error loading GASearchCV" in output


def test_ga_feature_selection_save_and_load_round_trip(tmp_path, capsys):
    selector = GAFeatureSelectionCV(
        estimator=DecisionTreeClassifier(),
        cv=2,
        scoring="accuracy",
        population_size=4,
        generations=2,
    )
    selector.extra_state = "persisted"
    selector.logbook = [{"generation": 0}]
    checkpoint_path = tmp_path / "ga_feature_selection.pkl"

    selector.save(checkpoint_path)

    restored = GAFeatureSelectionCV(
        estimator=DecisionTreeClassifier(),
        cv=2,
        scoring="accuracy",
        population_size=4,
        generations=2,
    )
    restored.load(checkpoint_path)
    output = capsys.readouterr().out

    assert "GAFeatureSelectionCV model successfully saved" in output
    assert "GAFeatureSelectionCV model successfully loaded" in output
    assert restored.extra_state == "persisted"
    assert restored.logbook == [{"generation": 0}]


def test_ga_feature_selection_save_and_load_errors_are_reported(tmp_path, capsys):
    selector = GAFeatureSelectionCV(
        estimator=DecisionTreeClassifier(),
        cv=2,
        scoring="accuracy",
        population_size=4,
        generations=2,
    )

    selector.save(tmp_path / "missing" / "ga_feature_selection.pkl")
    selector.load(tmp_path / "missing.pkl")
    output = capsys.readouterr().out

    assert "Error saving GAFeatureSelectionCV" in output
    assert "Error loading GAFeatureSelectionCV" in output


def test_ga_feature_selection_support_mask_requires_fit():
    selector = GAFeatureSelectionCV(
        estimator=DecisionTreeClassifier(),
        cv=2,
        scoring="accuracy",
        population_size=4,
        generations=2,
    )

    with pytest.raises(NotFittedError, match="not fitted yet"):
        selector._get_support_mask()


def test_model_checkpoint_save_load_and_error_paths(tmp_path, capsys):
    estimator = SimpleNamespace(
        estimator=DecisionTreeClassifier(),
        cv=2,
        scoring="accuracy",
        population_size=4,
        generations=2,
        crossover_probability=0.8,
        mutation_probability=0.1,
        param_grid={"max_depth": Integer(1, 2)},
        algorithm="eaSimple",
    )
    checkpoint_path = tmp_path / "checkpoint.pkl"
    checkpoint = ModelCheckpoint(checkpoint_path)

    checkpoint.on_step(logbook=None, estimator=estimator)
    loaded_checkpoint = checkpoint.load()

    failing_checkpoint = ModelCheckpoint(tmp_path / "missing" / "checkpoint.pkl")
    failing_checkpoint.on_step(logbook=None, estimator=estimator)
    missing_checkpoint = ModelCheckpoint(tmp_path / "missing.pkl").load()
    output = capsys.readouterr().out

    assert "Checkpoint saved to" in output
    assert "Error saving checkpoint" in output
    assert "Error loading checkpoint" in output
    assert loaded_checkpoint["estimator_state"]["scoring"] == "accuracy"
    assert loaded_checkpoint["logbook"] is None
    assert missing_checkpoint is None


def test_model_checkpoint_estimator_state_keys(tmp_path):
    estimator = SimpleNamespace(
        estimator=DecisionTreeClassifier(),
        cv=2,
        scoring="accuracy",
        population_size=4,
        generations=2,
        crossover_probability=0.8,
        mutation_probability=0.1,
        param_grid={"max_depth": Integer(1, 2)},
        algorithm="eaSimple",
    )
    checkpoint_path = tmp_path / "checkpoint.pkl"
    checkpoint = ModelCheckpoint(checkpoint_path)

    checkpoint.on_step(logbook=None, estimator=estimator)
    loaded_checkpoint = checkpoint.load()

    estimator_state = loaded_checkpoint["estimator_state"]
    expected_keys = {
        "estimator",
        "cv",
        "scoring",
        "population_size",
        "generations",
        "param_grid",
        "algorithm",
    }
    assert expected_keys.issubset(estimator_state.keys())

    assert isinstance(estimator_state["estimator"], DecisionTreeClassifier)
    assert estimator_state["cv"] == 2
    assert estimator_state["scoring"] == "accuracy"
    assert estimator_state["population_size"] == 4
    assert estimator_state["generations"] == 2
    assert list(estimator_state["param_grid"].keys()) == ["max_depth"]
    assert estimator_state["param_grid"]["max_depth"].lower == 1
    assert estimator_state["param_grid"]["max_depth"].upper == 2
    assert estimator_state["algorithm"] == "eaSimple"


def test_feature_selection_checkpoint_resume_restores_fitness_cache(tmp_path):
    """GAFeatureSelectionCV restores its fitness cache on resume (#299).

    Covers the ``runtime_state`` restore branch in GAFeatureSelectionCV.fit,
    mirroring the GASearchCV path. The checkpoint is built directly so the test
    targets the restore branch specifically (feature-selection checkpoint
    *saving* is fixed separately).
    """
    import pickle

    X, y = load_iris(return_X_y=True)
    checkpoint_path = str(tmp_path / "fs_resume_checkpoint.pkl")

    # A minimal pre-existing checkpoint carrying a sentinel cache entry. The
    # sentinel is never generated during fit, so its presence afterwards proves
    # the resume path reloaded the cache onto the estimator.
    sentinel_key = ("__fs_resume_sentinel__",)
    checkpoint_data = {
        "estimator_state": {},
        "logbook": None,
        "runtime_state": {"fitness_cache": {sentinel_key: {"fitness": (0.5,)}}},
    }
    with open(checkpoint_path, "wb") as f:
        pickle.dump(checkpoint_data, f)

    resumed = GAFeatureSelectionCV(
        estimator=DecisionTreeClassifier(random_state=0),
        cv=2,
        scoring="accuracy",
        population_size=4,
        generations=2,
    )
    resumed.fit(X, y, callbacks=ModelCheckpoint(checkpoint_path=checkpoint_path))
    assert sentinel_key in resumed.fitness_cache
