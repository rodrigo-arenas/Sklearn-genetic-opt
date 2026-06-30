import matplotlib
import pytest
import re
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

matplotlib.use("Agg")

from .. import GASearchCV, GAFeatureSelectionCV
from ..plots import (
    SearchPlotter,
    plot_candidate_rankings,
    plot_candidate_scores,
    plot_convergence,
    plot_cv_scores,
    plot_diversity,
    plot_feature_selection,
    plot_fitness_evolution,
    plot_history,
    plot_optimizer_events,
    plot_parameter_evolution,
    plot_search_decisions,
    plot_search_overview,
    plot_search_space,
    plot_score_landscape,
)
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

    with pytest.raises(ValueError, match="fields not found in history") as excinfo:
        plot_history(evolved_estimator, fields=["missing_field"])
    message = str(excinfo.value)
    assert "Available fields include" in message
    assert "gen" in message



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
        features=["ccp_alpha", "max_depth", "min_samples_split", "criterion"],
        hue="criterion",
    )
    assert hasattr(plot, "fig")


def test_plot_space_defaults_to_numeric_parameters():
    plot = plot_search_space(evolved_estimator)
    assert hasattr(plot, "fig")


def test_plot_parameter_evolution():
    axes = plot_parameter_evolution(
        evolved_estimator,
        parameters=["ccp_alpha", "max_depth"],
    )
    assert len(axes) == 2
    assert axes[0].get_title() == "Parameter exploration over time"

    plot = plot_parameter_evolution(evolved_estimator, parameters="max_depth")
    assert plot.get_xlabel() == "evaluation index"


def test_plot_search_decisions():
    axes = plot_search_decisions(evolved_estimator)
    assert len(axes) >= 1
    assert axes[0].figure._suptitle.get_text() == "Optimizer decisions"


def test_plot_candidate_scores():
    plot = plot_candidate_scores(evolved_estimator, top_k=3)
    assert plot.get_title() == "Top 3 candidate scores"
    assert plot.get_xlabel() == "mean_test_score"
    labels = [label.get_text() for label in plot.get_yticklabels()]
    assert any("+2 more" in label for label in labels)
    ccp_alpha_values = [
        match.group(1)
        for label in labels
        for match in [re.search(r"ccp_alpha=([^,]+)", label)]
        if match
    ]
    assert ccp_alpha_values
    assert all(len(value) <= 8 for value in ccp_alpha_values)
    assert all("max_depth=" in label for label in labels)

    selected_labels = plot_candidate_scores(
        evolved_estimator,
        top_k=2,
        label_params=["max_depth", "min_samples_split"],
    )
    assert all(
        "max_depth=" in label.get_text() and "min_samples_split=" in label.get_text()
        for label in selected_labels.get_yticklabels()
    )


def test_plot_convergence_and_diversity():
    convergence = plot_convergence(evolved_estimator)
    assert convergence.get_title() == "Convergence"
    assert convergence.get_xlabel() == "generations"

    diversity = plot_diversity(evolved_estimator)
    assert diversity.get_title() == "Diversity"
    assert diversity.get_ylabel() == "diversity ratio"


def test_plot_optimizer_events():
    plot = plot_optimizer_events(evolved_estimator)
    assert plot.get_title() == "Optimizer events"
    assert plot.get_xlabel() == "generations"

    with pytest.raises(ValueError, match="fields not found in history"):
        plot_optimizer_events(evolved_estimator, fields=["missing_event"])


def test_plot_score_landscape():
    plot = plot_score_landscape(
        evolved_estimator,
        x="ccp_alpha",
        y="max_depth",
        kind="scatter",
    )
    assert plot.get_title() == "Score landscape: ccp_alpha vs max_depth"

    hexbin = plot_score_landscape(
        evolved_estimator,
        x="ccp_alpha",
        y="max_depth",
        kind="hexbin",
    )
    assert hexbin.get_xlabel() == "ccp_alpha"

    with pytest.raises(ValueError, match="kind must be one of"):
        plot_score_landscape(evolved_estimator, x="ccp_alpha", y="max_depth", kind="contour")


def test_plot_cv_scores_and_candidate_rankings():
    cv_plot = plot_cv_scores(evolved_estimator, top_k=3)
    assert cv_plot.get_title() == "Cross-validation scores for top 3 candidates"
    assert any("+2 more" in label.get_text() for label in cv_plot.get_xticklabels())

    strip_plot = plot_cv_scores(evolved_estimator, top_k=2, kind="strip")
    assert strip_plot.get_xlabel() == "candidate"

    ranking = plot_candidate_rankings(evolved_estimator, top_k=3)
    assert ranking.get_title() == "Candidate ranking (top 3)"
    assert ranking.get_xlabel() == "mean_test_score"
    assert any("+2 more" in label.get_text() for label in ranking.get_yticklabels())

    selected_ranking = plot_candidate_rankings(
        evolved_estimator,
        top_k=2,
        label_params="max_depth",
    )
    assert all("max_depth=" in label.get_text() for label in selected_ranking.get_yticklabels())

    with pytest.raises(ValueError, match="kind must be one of"):
        plot_cv_scores(evolved_estimator, kind="hist")


def test_plot_search_overview():
    axes = plot_search_overview(evolved_estimator, top_k=3)
    assert axes.shape == (2, 2)
    assert axes[0, 0].get_title() == "Convergence"
    assert axes[1, 1].get_title() == "Best evaluated candidates"


def test_search_plotter_facade():
    plotter = SearchPlotter(evolved_estimator)
    assert plotter.convergence().get_title() == "Convergence"
    assert plotter.diversity().get_title() == "Diversity"
    assert plotter.optimizer_events().get_title() == "Optimizer events"
    assert plotter.parameter_evolution("max_depth").get_xlabel() == "evaluation index"
    assert plotter.score_landscape("ccp_alpha", "max_depth").get_xlabel() == "ccp_alpha"
    assert plotter.candidate_rankings(top_k=2).get_title() == "Candidate ranking (top 2)"
    assert plotter.cv_scores(top_k=2).get_xlabel() == "candidate"
    assert plotter.overview(top_k=2).shape == (2, 2)


def test_wrong_estimator_space():
    estimator = GAFeatureSelectionCV(clf, cv=3, scoring="accuracy", population_size=6)
    with pytest.raises(Exception) as excinfo:
        plot_search_space(estimator)

    assert (
        str(excinfo.value)
        == "Estimator must be a GASearchCV instance, not a GAFeatureSelectionCV instance"
    )


def test_feature_selection_plot():
    estimator = GAFeatureSelectionCV(
        clf,
        cv=2,
        scoring="r2",
        population_size=4,
        generations=2,
        max_features=4,
        n_jobs=1,
    )
    estimator.fit(X_train, y_train)

    plot = plot_feature_selection(estimator)
    assert plot.get_xlabel() == "selection"
    assert "Selected features" in plot.get_title()

    axes = plot_search_overview(estimator, top_k=3)
    assert axes[1, 1].get_xlabel() == "selection"

    plotter = SearchPlotter(estimator)
    assert plotter.feature_selection().get_xlabel() == "selection"

    with pytest.raises(TypeError, match="supports GASearchCV"):
        plot_parameter_evolution(estimator)

    with pytest.raises(TypeError, match="supports GASearchCV only"):
        plot_score_landscape(estimator, x="feature_0", y="feature_1")

    expected_count = len(estimator.best_features_)
    received_count = 1
    with pytest.raises(ValueError) as excinfo:
        plot_feature_selection(estimator, feature_names=["one"])

    message = str(excinfo.value)
    assert "feature_names" in message
    assert "expected" in message
    assert "got" in message
    assert str(expected_count) in message
    assert str(received_count) in message


def test_plot_on_unfitted_estimator_gives_actionable_error():
    """Plotting before fit should tell the user to call .fit(X, y) (issue #222)."""
    unfitted = GASearchCV(
        DecisionTreeRegressor(),
        cv=2,
        scoring="r2",
        param_grid={"max_depth": Integer(2, 8)},
    )
    with pytest.raises(ValueError) as excinfo:
        plot_candidate_scores(unfitted)
    message = str(excinfo.value)
    assert "fitted GASearchCV" in message
    assert "estimator.fit(X, y)" in message
    assert "estimator.cv_results_" in message


def test_plot_feature_selection_on_unfitted_estimator_names_the_plot():
    """plot_feature_selection should name itself and best_features_ (issue #222)."""
    unfitted = GAFeatureSelectionCV(DecisionTreeRegressor(), cv=2, scoring="r2")
    with pytest.raises(ValueError) as excinfo:
        plot_feature_selection(unfitted)
    message = str(excinfo.value)
    assert "plot_feature_selection requires" in message
    assert "estimator.fit(X, y)" in message
    assert "estimator.best_features_" in message
