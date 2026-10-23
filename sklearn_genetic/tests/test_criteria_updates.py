import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.tree import DecisionTreeRegressor

from sklearn_genetic import GASearchCV, GAFeatureSelectionCV
from sklearn_genetic.space import Integer


@pytest.mark.parametrize("search_type", [GASearchCV, GAFeatureSelectionCV])
@pytest.mark.parametrize("criteria", ["min", "max"])
@pytest.mark.parametrize("update", [False, True])
@pytest.mark.parametrize("fit_before_update", [False, True])
@pytest.mark.parametrize("refit", [False, True])
def test_fit_uses_current_optimization_criteria(
    search_type, criteria, update, fit_before_update, refit
):
    X, y = make_regression(n_samples=40, n_features=4, n_informative=2, random_state=7)
    options = {}
    if search_type is GASearchCV:
        options["param_grid"] = {
            "max_depth": Integer(1, 3),
            "min_samples_leaf": Integer(1, 2),
        }
    initial = ("max" if criteria == "min" else "min") if update else criteria
    search = search_type(
        DecisionTreeRegressor(random_state=0),
        criteria=initial,
        scoring=make_scorer(mean_squared_error),
        population_size=6,
        generations=0,
        cv=2,
        refit=refit,
        n_jobs=1,
        random_state=3,
        verbose=False,
        **options,
    )
    if fit_before_update:
        search.fit(X, y)
    if update:
        search.set_params(criteria=criteria)
    search.fit(X, y)

    scores = np.asarray(search.cv_results_["mean_test_score"])
    assert np.ptp(scores) > 0, "The search must evaluate distinct scores"
    expected_score = (np.min if criteria == "min" else np.max)(scores)
    assert search.history["fitness_best"][-1] == pytest.approx(expected_score)
    if search_type is GASearchCV:
        winner = search.hof[0]
        selected = [
            index for index, params in enumerate(search.cv_results_["params"]) if params == winner
        ]
    else:
        selected = [
            index
            for index, features in enumerate(search.cv_results_["features"])
            if np.array_equal(features, search.best_features_)
        ]
    assert selected
    assert scores[selected[0]] == pytest.approx(expected_score)


@pytest.mark.parametrize("search_type", [GASearchCV, GAFeatureSelectionCV])
def test_fit_validates_replaced_criteria(search_type):
    X, y = make_regression(n_samples=12, n_features=2, random_state=0)
    options = {}
    if search_type is GASearchCV:
        options["param_grid"] = {"max_depth": Integer(1, 2)}
    search = search_type(
        DecisionTreeRegressor(random_state=0),
        population_size=2,
        generations=0,
        cv=2,
        n_jobs=1,
        verbose=False,
        **options,
    )
    search.set_params(criteria="invalid")
    with pytest.raises(ValueError, match="Criteria must be one of"):
        search.fit(X, y)
