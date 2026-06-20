import matplotlib
import pytest
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

matplotlib.use("Agg")

from .. import GASearchCV, GAFeatureSelectionCV
from ..plots import plot_fitness_evolution, plot_history, plot_search_space
from ..space import Categorical, Continuous, Integer

data = load_diabetes()

y = data["target"]
X = data["data"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

clf = DecisionTreeRegressor()

evolved_estimator = GASearchCV(
    clf,
    cv=2,
    scoring="r2",
    population_size=4,
    generations=5,
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
    criteria="max",
    n_jobs=-1,
)

evolved_estimator.fit(X_train, y_train)


def test_plot_evolution():
    plot = plot_fitness_evolution(evolved_estimator)
    assert plot.get_xlabel() == "generations"
    assert "fitness" in plot.get_ylabel()

    plot = plot_fitness_evolution(
        evolved_estimator,
        metrics=["fitness_best", "fitness"],
        window=2,
        kind="line",
    )
    assert len(plot.lines) == 2

    with pytest.raises(Exception) as excinfo:
        plot_fitness_evolution(evolved_estimator, metric="accuracy")

    assert (
        str(excinfo.value) == "metric must be one of "
        "['fitness', 'fitness_std', 'fitness_best', 'fitness_max', 'fitness_min'], "
        "but got accuracy instead"
    )

    with pytest.raises(ValueError, match="kind must be one of"):
        plot_fitness_evolution(evolved_estimator, kind="scatter")


def test_plot_history():
    plot = plot_history(
        evolved_estimator,
        fields=["fitness_best", "unique_individual_ratio"],
        kind="line",
    )
    assert plot.get_xlabel() == "generations"

    axes = plot_history(
        evolved_estimator,
        fields=["fitness_best", "fitness_max", "fitness_min", "genotype_diversity"],
        kind="area",
        subplots=True,
    )
    assert len(axes) == 4

    logbook_plot = plot_history(
        evolved_estimator,
        fields=["score", "fit_time"],
        source="logbook",
        kind="step",
    )
    assert logbook_plot.get_xlabel() == "record index"

    with pytest.raises(ValueError, match="No plottable fields were found"):
        plot_history(evolved_estimator, fields=["missing_field"])

    with pytest.raises(ValueError, match="kind must be one of"):
        plot_history(evolved_estimator, kind="scatter")


def test_plot_space():
    plot = plot_search_space(
        evolved_estimator,
        features=["ccp_alpha", "max_depth", "min_samples_split"],
    )
    assert hasattr(plot, "fig")


def test_plot_space_heatmap():
    plot = plot_search_space(
        evolved_estimator,
        features=["ccp_alpha", "max_depth", "min_samples_split"],
        kind="heatmap",
    )
    assert plot.get_title() == "Search-space correlation heatmap"


def test_plot_space_with_hue():
    plot = plot_search_space(
        evolved_estimator,
        features=["ccp_alpha", "max_depth", "min_samples_split", "fitness_best"],
        hue="fitness_best",
    )
    assert hasattr(plot, "fig")


def test_plot_space_defaults_to_numeric_parameters():
    plot = plot_search_space(evolved_estimator)
    assert hasattr(plot, "fig")


def test_wrong_estimator_space():
    estimator = GAFeatureSelectionCV(clf, cv=3, scoring="accuracy", population_size=6)
    with pytest.raises(Exception) as excinfo:
        plot_search_space(estimator)

    assert (
        str(excinfo.value)
        == "Estimator must be a GASearchCV instance, not a GAFeatureSelectionCV instance"
    )
