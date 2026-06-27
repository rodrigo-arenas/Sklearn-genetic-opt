from .space import Categorical, Continuous, Integer

__all__ = [
    "hist_gradient_boosting_classifier_space",
    "hist_gradient_boosting_regressor_space",
    "logistic_regression_space",
    "random_forest_classifier_space",
    "random_forest_regressor_space",
    "svc_space",
    "xgboost_classifier_space",
    "xgboost_regressor_space",
]

_PROFILES = {"fast", "balanced", "wide"}


def random_forest_classifier_space(profile="balanced", prefix=""):
    """Return a starter search space for ``RandomForestClassifier``."""
    _check_profile(profile)
    bounds = {
        "fast": (50, 180, 2, 16, 1, 6),
        "balanced": (80, 350, 2, 24, 1, 10),
        "wide": (100, 700, 2, 40, 1, 20),
    }[profile]
    n_low, n_high, depth_low, depth_high, leaf_low, leaf_high = bounds

    return _with_prefix(
        {
            "n_estimators": Integer(n_low, n_high),
            "max_depth": Integer(depth_low, depth_high),
            "min_samples_split": Integer(2, max(8, leaf_high * 2)),
            "min_samples_leaf": Integer(leaf_low, leaf_high),
            "max_features": Categorical(["sqrt", "log2", None]),
            "criterion": Categorical(["gini", "entropy"]),
            "class_weight": Categorical([None, "balanced", "balanced_subsample"]),
        },
        prefix,
    )


def random_forest_regressor_space(profile="balanced", prefix=""):
    """Return a starter search space for ``RandomForestRegressor``."""
    _check_profile(profile)
    bounds = {
        "fast": (50, 180, 2, 16, 1, 6),
        "balanced": (80, 350, 2, 24, 1, 10),
        "wide": (100, 700, 2, 40, 1, 20),
    }[profile]
    n_low, n_high, depth_low, depth_high, leaf_low, leaf_high = bounds

    return _with_prefix(
        {
            "n_estimators": Integer(n_low, n_high),
            "max_depth": Integer(depth_low, depth_high),
            "min_samples_split": Integer(2, max(8, leaf_high * 2)),
            "min_samples_leaf": Integer(leaf_low, leaf_high),
            "max_features": Categorical(["sqrt", "log2", None]),
            "criterion": Categorical(["squared_error", "absolute_error", "friedman_mse"]),
        },
        prefix,
    )


def hist_gradient_boosting_classifier_space(profile="balanced", prefix=""):
    """Return a starter search space for ``HistGradientBoostingClassifier``."""
    _check_profile(profile)
    bounds = {
        "fast": (40, 160, 0.02, 0.25, 7, 63, 5, 50),
        "balanced": (60, 300, 0.01, 0.3, 15, 127, 5, 100),
        "wide": (80, 500, 0.005, 0.5, 15, 255, 2, 150),
    }[profile]
    iter_low, iter_high, lr_low, lr_high, leaves_low, leaves_high, samples_low, samples_high = (
        bounds
    )

    return _with_prefix(
        {
            "max_iter": Integer(iter_low, iter_high),
            "learning_rate": Continuous(lr_low, lr_high, distribution="log-uniform"),
            "max_leaf_nodes": Integer(leaves_low, leaves_high),
            "max_depth": Categorical([None, 3, 5, 8, 12]),
            "min_samples_leaf": Integer(samples_low, samples_high),
            "l2_regularization": Continuous(1e-8, 10.0, distribution="log-uniform"),
        },
        prefix,
    )


def hist_gradient_boosting_regressor_space(profile="balanced", prefix=""):
    """Return a starter search space for ``HistGradientBoostingRegressor``."""
    return hist_gradient_boosting_classifier_space(profile=profile, prefix=prefix)


def logistic_regression_space(profile="balanced", prefix=""):
    """Return a starter search space for saga/elasticnet ``LogisticRegression``."""
    _check_profile(profile)
    c_bounds = {"fast": (1e-2, 10.0), "balanced": (1e-3, 30.0), "wide": (1e-5, 100.0)}
    c_low, c_high = c_bounds[profile]

    return _with_prefix(
        {
            "C": Continuous(c_low, c_high, distribution="log-uniform"),
            "penalty": Categorical(["elasticnet"]),
            "solver": Categorical(["saga"]),
            "l1_ratio": Continuous(0.0, 1.0),
            "class_weight": Categorical([None, "balanced"]),
            "max_iter": Integer(500, 2500),
        },
        prefix,
    )


def svc_space(profile="balanced", prefix=""):
    """Return a starter search space for ``SVC``."""
    _check_profile(profile)
    c_bounds = {"fast": (1e-2, 50.0), "balanced": (1e-3, 100.0), "wide": (1e-4, 1000.0)}
    gamma_bounds = {
        "fast": (1e-3, 1.0),
        "balanced": (1e-4, 10.0),
        "wide": (1e-6, 100.0),
    }
    c_low, c_high = c_bounds[profile]
    gamma_low, gamma_high = gamma_bounds[profile]

    return _with_prefix(
        {
            "C": Continuous(c_low, c_high, distribution="log-uniform"),
            "kernel": Categorical(["rbf", "poly", "sigmoid"]),
            "gamma": Continuous(gamma_low, gamma_high, distribution="log-uniform"),
            "degree": Integer(2, 5),
            "class_weight": Categorical([None, "balanced"]),
        },
        prefix,
    )


def xgboost_classifier_space(profile="balanced", prefix=""):
    """Return a starter search space for ``xgboost.XGBClassifier``."""
    return _xgboost_tree_space(profile=profile, prefix=prefix)


def xgboost_regressor_space(profile="balanced", prefix=""):
    """Return a starter search space for ``xgboost.XGBRegressor``."""
    return _xgboost_tree_space(profile=profile, prefix=prefix)


def _with_prefix(space, prefix):
    if not prefix:
        return space
    return {f"{prefix}{name}": dimension for name, dimension in space.items()}


def _check_profile(profile):
    if profile not in _PROFILES:
        raise ValueError(f"profile must be one of {sorted(_PROFILES)}, got {profile} instead")


def _xgboost_tree_space(profile, prefix):
    _check_profile(profile)
    bounds = {
        "fast": (50, 180, 2, 8, 1, 8, 0.03, 0.3),
        "balanced": (50, 350, 2, 10, 1, 12, 0.01, 0.3),
        "wide": (80, 600, 2, 14, 1, 20, 0.005, 0.5),
    }[profile]
    (
        n_low,
        n_high,
        depth_low,
        depth_high,
        child_low,
        child_high,
        lr_low,
        lr_high,
    ) = bounds

    return _with_prefix(
        {
            "n_estimators": Integer(n_low, n_high),
            "max_depth": Integer(depth_low, depth_high),
            "min_child_weight": Integer(child_low, child_high),
            "subsample": Continuous(0.5, 1.0),
            "colsample_bytree": Continuous(0.4, 1.0),
            "learning_rate": Continuous(lr_low, lr_high, distribution="log-uniform"),
            "gamma": Continuous(1e-4, 1.0, distribution="log-uniform"),
            "reg_alpha": Continuous(1e-5, 10.0, distribution="log-uniform"),
            "reg_lambda": Continuous(1e-5, 10.0, distribution="log-uniform"),
        },
        prefix,
    )
