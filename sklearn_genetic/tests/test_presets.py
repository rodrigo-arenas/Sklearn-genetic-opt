import pytest

from sklearn_genetic import (
    hist_gradient_boosting_classifier_space,
    hist_gradient_boosting_regressor_space,
    logistic_regression_space,
    random_forest_classifier_space,
    random_forest_regressor_space,
    svc_space,
    xgboost_classifier_space,
    xgboost_regressor_space,
)
from sklearn_genetic.space import Categorical, Continuous, Integer
from sklearn_genetic.space.base import BaseDimension

PRESET_FUNCTIONS = [
    random_forest_classifier_space,
    random_forest_regressor_space,
    hist_gradient_boosting_classifier_space,
    hist_gradient_boosting_regressor_space,
    logistic_regression_space,
    svc_space,
    xgboost_classifier_space,
    xgboost_regressor_space,
]


def test_random_forest_classifier_preset_returns_native_space_dimensions():
    space = random_forest_classifier_space(profile="fast")

    assert isinstance(space["n_estimators"], Integer)
    assert space["n_estimators"].lower == 50
    assert space["n_estimators"].upper == 180
    assert isinstance(space["max_features"], Categorical)
    assert space["class_weight"].choices == [None, "balanced", "balanced_subsample"]


def test_random_forest_regressor_preset_uses_regression_criteria():
    space = random_forest_regressor_space()

    assert space["criterion"].choices == ["squared_error", "absolute_error", "friedman_mse"]
    assert "class_weight" not in space


def test_hist_gradient_boosting_preset_uses_log_uniform_learning_rate():
    space = hist_gradient_boosting_classifier_space(profile="wide")

    assert isinstance(space["learning_rate"], Continuous)
    assert space["learning_rate"].distribution == "log-uniform"
    assert space["max_iter"].upper == 500


def test_logistic_regression_preset_uses_valid_saga_elasticnet_path():
    space = logistic_regression_space()

    assert space["solver"].choices == ["saga"]
    assert space["penalty"].choices == ["elasticnet"]
    assert isinstance(space["l1_ratio"], Continuous)


def test_presets_support_pipeline_prefixes():
    space = svc_space(prefix="model__")

    assert "model__C" in space
    assert "C" not in space
    assert isinstance(space["model__degree"], Integer)


@pytest.mark.parametrize("preset", PRESET_FUNCTIONS)
@pytest.mark.parametrize("profile", ["fast", "balanced", "wide"])
def test_all_presets_return_native_dimensions_for_each_profile(preset, profile):
    space = preset(profile=profile)

    assert space
    assert all(isinstance(dimension, BaseDimension) for dimension in space.values())


@pytest.mark.parametrize("preset", PRESET_FUNCTIONS)
def test_all_presets_prefix_every_key_without_changing_dimension_count(preset):
    unprefixed = preset()
    prefixed = preset(prefix="model__")

    assert len(prefixed) == len(unprefixed)
    assert set(prefixed) == {f"model__{key}" for key in unprefixed}
    assert all(key.startswith("model__") for key in prefixed)


@pytest.mark.parametrize("preset", PRESET_FUNCTIONS)
def test_all_presets_treat_empty_prefix_as_unprefixed(preset):
    assert set(preset(prefix="")) == set(preset())


def test_xgboost_classifier_preset_matches_interacting_tree_space():
    space = xgboost_classifier_space(profile="balanced")

    assert isinstance(space["n_estimators"], Integer)
    assert space["n_estimators"].upper == 350
    assert isinstance(space["learning_rate"], Continuous)
    assert space["learning_rate"].distribution == "log-uniform"
    assert space["gamma"].distribution == "log-uniform"
    assert space["reg_alpha"].distribution == "log-uniform"
    assert space["reg_lambda"].distribution == "log-uniform"
    assert set(space) == {
        "n_estimators",
        "max_depth",
        "min_child_weight",
        "subsample",
        "colsample_bytree",
        "learning_rate",
        "gamma",
        "reg_alpha",
        "reg_lambda",
    }


def test_xgboost_regressor_preset_supports_wide_profile_and_prefix():
    space = xgboost_regressor_space(profile="wide", prefix="model__")

    assert space["model__n_estimators"].upper == 600
    assert space["model__max_depth"].upper == 14
    assert isinstance(space["model__subsample"], Continuous)


def test_presets_reject_unknown_profiles():
    for preset in PRESET_FUNCTIONS:
        with pytest.raises(ValueError, match="profile must be one of"):
            preset(profile="tiny")


def test_list_preset_profiles_returns_valid_profiles():
    from sklearn_genetic import list_preset_profiles

    assert list_preset_profiles() == ["balanced", "fast", "wide"]
    # Every reported profile is actually accepted by a preset.
    for profile in list_preset_profiles():
        random_forest_classifier_space(profile=profile)


def test_list_preset_spaces_matches_exported_presets():
    import sklearn_genetic
    from sklearn_genetic import list_preset_spaces

    names = list_preset_spaces()
    assert names == sorted(names)  # stable, alphabetical
    assert len(names) == len(PRESET_FUNCTIONS)
    assert "random_forest_classifier_space" in names
    # Every reported name is importable from the package and callable.
    for name in names:
        assert callable(getattr(sklearn_genetic, name))
