import shutil

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


def test_ga_search_save_and_load_round_trip(tmp_path, caplog):
    import logging

    caplog.set_level(logging.INFO)
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

    assert "GASearchCV model successfully saved" in caplog.text
    assert "GASearchCV model successfully loaded" in caplog.text
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


def test_ga_search_save_and_load_errors_are_reported(tmp_path, caplog):
    search = GASearchCV(
        estimator=DecisionTreeClassifier(),
        param_grid={"max_depth": Integer(1, 2)},
        cv=2,
        scoring="accuracy",
    )

    with pytest.raises(OSError):
        search.save(tmp_path / "missing" / "ga_search.pkl")
    with pytest.raises(FileNotFoundError):
        search.load(tmp_path / "missing.pkl")

    assert "Error saving GASearchCV" in caplog.text
    assert "Error loading GASearchCV" in caplog.text


def test_ga_feature_selection_save_and_load_round_trip(tmp_path, caplog):
    import logging

    caplog.set_level(logging.INFO)
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

    assert "GAFeatureSelectionCV model successfully saved" in caplog.text
    assert "GAFeatureSelectionCV model successfully loaded" in caplog.text
    assert restored.extra_state == "persisted"
    assert restored.logbook == [{"generation": 0}]


def test_ga_feature_selection_save_and_load_errors_are_reported(tmp_path, caplog):
    selector = GAFeatureSelectionCV(
        estimator=DecisionTreeClassifier(),
        cv=2,
        scoring="accuracy",
        population_size=4,
        generations=2,
    )

    with pytest.raises(OSError):
        selector.save(tmp_path / "missing" / "ga_feature_selection.pkl")
    with pytest.raises(FileNotFoundError):
        selector.load(tmp_path / "missing.pkl")

    assert "Error saving GAFeatureSelectionCV" in caplog.text
    assert "Error loading GAFeatureSelectionCV" in caplog.text


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


def _checkpoint_estimator():
    return GASearchCV(
        estimator=DecisionTreeClassifier(),
        param_grid={"max_depth": Integer(1, 2), "min_samples_split": Integer(2, 5)},
        cv=2,
        scoring="accuracy",
        population_size=4,
        generations=2,
    )


def test_model_checkpoint_save_load_and_error_paths(tmp_path, caplog):
    import logging

    caplog.set_level(logging.INFO)
    estimator = _checkpoint_estimator()
    checkpoint_path = tmp_path / "checkpoint.pkl"
    checkpoint = ModelCheckpoint(checkpoint_path)

    checkpoint.on_step(logbook=None, estimator=estimator)
    loaded_checkpoint = checkpoint.load()

    failing_checkpoint = ModelCheckpoint(tmp_path / "missing" / "checkpoint.pkl")
    failing_checkpoint.on_step(logbook=None, estimator=estimator)

    with pytest.raises(FileNotFoundError):
        ModelCheckpoint(tmp_path / "missing.pkl").load()

    assert "Checkpoint saved to" in caplog.text
    assert "Error saving checkpoint" in caplog.text
    assert "Error loading checkpoint" in caplog.text
    assert loaded_checkpoint["estimator_state"]["scoring"] == "accuracy"
    assert loaded_checkpoint["logbook"] is None


def test_model_checkpoint_estimator_state_keys(tmp_path):
    estimator = _checkpoint_estimator()
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
    assert estimator_state["population_size"] == estimator.population_size
    assert estimator_state["generations"] == estimator.generations
    assert list(estimator_state["param_grid"].keys()) == ["max_depth", "min_samples_split"]
    assert estimator_state["param_grid"]["max_depth"].lower == 1
    assert estimator_state["param_grid"]["max_depth"].upper == 2
    assert estimator_state["algorithm"] == estimator.algorithm


@pytest.mark.parametrize(
    "estimator",
    [
        GASearchCV(
            estimator=DecisionTreeClassifier(),
            param_grid={"max_depth": Integer(1, 3), "min_samples_split": Integer(2, 5)},
        ),
        GAFeatureSelectionCV(estimator=DecisionTreeClassifier()),
    ],
    ids=["GASearchCV", "GAFeatureSelectionCV"],
)
def test_checkpoint_state_honors_serialization_contract(estimator):
    """#297: the checkpoint state is a constructor-compatible subset of save/load.

    Every key ModelCheckpoint persists must be an ``__init__`` parameter (so the
    checkpoint replays as ``ClassName(**state)``) and must also live in the full
    ``_serializable_state()`` used by save/load, so the two serialization paths
    cannot silently drift apart.
    """
    checkpoint = estimator._checkpoint_state()
    full = estimator._serializable_state()
    params = estimator.get_params(deep=False)

    # Constructor-compatible: every checkpoint key is an __init__ parameter...
    assert set(checkpoint).issubset(params), set(checkpoint) - set(params)
    # ...and the checkpoint actually replays through the constructor.
    type(estimator)(**checkpoint)

    # Consistent with save/load: the checkpoint is a subset of the full state.
    assert set(checkpoint).issubset(full), set(checkpoint) - set(full)


def test_model_checkpoint_supports_feature_selection_estimator(tmp_path, caplog):
    """Checkpointing and resuming a GAFeatureSelectionCV must work (#297).

    GAFeatureSelectionCV has no ``param_grid`` attribute; the checkpoint state
    builder used to read ``estimator.param_grid`` unconditionally, so every
    on_step call failed with AttributeError and the estimator state was never
    saved. ``param_grid`` is now only included when the estimator defines it, so
    the saved state is both correct and usable by the resume path.
    """
    X, y = load_iris(return_X_y=True)
    checkpoint_path = str(tmp_path / "fs_checkpoint.pkl")

    def build():
        return GAFeatureSelectionCV(
            estimator=DecisionTreeClassifier(random_state=0),
            cv=2,
            scoring="accuracy",
            population_size=4,
            generations=2,
        )

    build().fit(X, y, callbacks=ModelCheckpoint(checkpoint_path=checkpoint_path))
    assert "Error saving checkpoint" not in caplog.text

    loaded = ModelCheckpoint(checkpoint_path=checkpoint_path).load()
    assert "estimator_state" in loaded
    # param_grid is GASearchCV-only, so it must be absent for a feature selector.
    assert "param_grid" not in loaded["estimator_state"]
    assert loaded["estimator_state"]["scoring"] == "accuracy"

    # Round trip: the real saved checkpoint must feed the resume path without
    # error and yield a fitted selector.
    resumed = build()
    resumed.fit(X, y, callbacks=ModelCheckpoint(checkpoint_path=checkpoint_path))
    assert resumed.support_.shape[0] == X.shape[1]
    assert resumed.support_.any()


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


def test_checkpoint_resume_preserves_logbook_and_fit_stats(tmp_path):
    """#299: resuming from a checkpoint must not drop prior-run history.

    ``_register()`` unconditionally rebuilds ``self.logbook`` (it also rebuilds
    the toolbox/population/hof, which are unpicklable ``deap`` objects that
    can't be checkpointed). That used to wipe out the logbook ``fit`` had just
    restored from the checkpoint, silently dropping every pre-resume candidate
    from ``cv_results_``/``history`` and resetting ``fit_stats_`` counters back
    to zero on every resume.
    """
    X, y = load_iris(return_X_y=True)
    checkpoint_path = str(tmp_path / "resume_history_checkpoint.pkl")

    def build():
        return GASearchCV(
            estimator=DecisionTreeClassifier(random_state=0),
            param_grid={"max_depth": Integer(1, 3)},
            cv=2,
            scoring="accuracy",
            population_size=4,
            generations=1,
        )

    first = build()
    first.fit(X, y, callbacks=ModelCheckpoint(checkpoint_path=checkpoint_path))
    first_logbook_len = len(first.logbook.chapters["parameters"])
    first_evaluated = first.fit_stats_["evaluated_candidates"]
    assert first_logbook_len > 0
    assert first_evaluated > 0

    resumed = build()
    resumed.fit(X, y, callbacks=ModelCheckpoint(checkpoint_path=checkpoint_path))

    # The resumed logbook must contain the first run's records plus the new
    # ones, not just the new run's (which is what the bug produced).
    assert len(resumed.logbook.chapters["parameters"]) > first_logbook_len
    # fit_stats_ counters must accumulate across the resume, not reset to 0.
    assert resumed.fit_stats_["evaluated_candidates"] > first_evaluated


def test_checkpoint_resume_continues_generation_numbering(tmp_path):
    """#299: resumed generations must continue, not restart, the gen count.

    Every algorithm variant (``eaSimple``, ``eaMuPlusLambda``,
    ``eaMuCommaLambda``) restarted its local ``gen`` counter at 0 on every
    ``fit`` call, so a resumed run's ``history``/``logbook`` used to show
    duplicate generation indices (e.g. ``0, 1`` from the first run followed by
    another ``0, 1`` after resume) instead of one increasing sequence.
    """
    X, y = load_iris(return_X_y=True)
    checkpoint_path = str(tmp_path / "resume_gen_checkpoint.pkl")

    def build():
        return GASearchCV(
            estimator=DecisionTreeClassifier(random_state=0),
            param_grid={"max_depth": Integer(1, 3)},
            cv=2,
            scoring="accuracy",
            population_size=4,
            generations=1,
        )

    first = build()
    first.fit(X, y, callbacks=ModelCheckpoint(checkpoint_path=checkpoint_path))
    first_gens = list(first.history["gen"])

    resumed = build()
    resumed.fit(X, y, callbacks=ModelCheckpoint(checkpoint_path=checkpoint_path))
    resumed_gens = list(resumed.history["gen"])

    # First run's generation indices are preserved verbatim at the front...
    assert resumed_gens[: len(first_gens)] == first_gens
    # ...the new generations continue strictly upward from there...
    assert resumed_gens[len(first_gens) :] == [g + max(first_gens) + 1 for g in first_gens]
    # ...so no generation index repeats.
    assert len(resumed_gens) == len(set(resumed_gens))


def test_checkpoint_resume_preserves_random_state(tmp_path):
    """#299: random_state must survive checkpoint resume and actually seed it.

    ``_checkpoint_core_keys`` did not include ``random_state``, so a resumed
    estimator silently fell back to whatever its own constructor set (often
    None) instead of the value the checkpointed run was built with. Restoring
    the key alone is not enough either: seeding happened *before* the
    checkpoint's ``estimator_state`` was applied to ``self``, so it used the
    pre-resume value regardless. Both are exercised here: the restored
    attribute, and that two independent resumes from the same checkpoint (each
    built without a random_state of their own) produce identical results --
    the only source of determinism available to them is the seed restored
    from the checkpoint.
    """
    X, y = load_iris(return_X_y=True)
    base_checkpoint = tmp_path / "base_checkpoint.pkl"

    # A wide, two-parameter grid sampled by a tiny population: different random
    # trajectories are overwhelmingly likely to evaluate different candidates
    # and land on different fitness values, so this is sensitive to exactly the
    # kind of divergence an unseeded (or wrongly-seeded) resume would produce.
    def build(random_state=None):
        return GASearchCV(
            estimator=DecisionTreeClassifier(random_state=0),
            param_grid={
                "max_depth": Integer(1, 50),
                "min_samples_split": Integer(2, 50),
            },
            cv=2,
            scoring="accuracy",
            population_size=4,
            generations=1,
            random_state=random_state,
        )

    first = build(random_state=42)
    first.fit(X, y, callbacks=ModelCheckpoint(checkpoint_path=str(base_checkpoint)))

    def resume_from_base(name):
        checkpoint_path = tmp_path / f"resume_{name}.pkl"
        shutil.copy(base_checkpoint, checkpoint_path)
        resumed = build(random_state=None)
        resumed.fit(X, y, callbacks=ModelCheckpoint(checkpoint_path=str(checkpoint_path)))
        return resumed

    resumed_a = resume_from_base("a")
    resumed_b = resume_from_base("b")

    assert resumed_a.random_state == 42
    assert resumed_b.random_state == 42
    assert resumed_a.best_params_ == resumed_b.best_params_
    assert resumed_a.best_score_ == resumed_b.best_score_
    assert list(resumed_a.history["fitness"]) == list(resumed_b.history["fitness"])
