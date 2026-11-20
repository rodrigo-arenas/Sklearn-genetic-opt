import matplotlib
import pytest
import numpy as np
import re
from types import SimpleNamespace
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

matplotlib.use("Agg")

from .. import GASearchCV, GAFeatureSelectionCV
from ..plots import (
    _as_list,
    _candidate_label,
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
    assert plot.get_xlabel() == "Generation"
    assert plot.get_ylabel() == "Fitness (r2)"

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
        "['fitness', 'fitness_std', 'fitness_best', 'fitness_max', 'fitness_min', 'fitness_best_observed'], "
        "but got accuracy instead"
    )

    with pytest.raises(ValueError, match="kind must be one of"):
        plot_fitness_evolution(evolved_estimator, kind="scatter")


def test_live_plot_tracks_generations_without_tensorflow(tmp_path, monkeypatch):
    from deap.tools import Logbook
    from matplotlib import pyplot as plt
    from ..callbacks.validations import check_callback
    from ..plots import LivePlot, plot_generation_metrics

    monkeypatch.setattr(plt, "pause", lambda interval: None)
    logbook = Logbook()
    logbook.record(gen=0, fitness_best=0.4, fitness=0.2)
    path = tmp_path / "progress.png"
    callback = LivePlot(save_path=path)
    assert check_callback(callback) == [callback]
    assert callback.on_step(logbook=logbook) is False
    first_image = path.read_bytes()

    logbook.record(gen=1, fitness_best=0.7, fitness=0.5)
    assert callback.on_step(logbook=logbook) is False
    assert path.read_bytes() != first_image
    assert len(callback.ax.lines) == 2
    assert list(callback.ax.lines[0].get_xdata()) == [0, 1]
    assert list(callback.ax.lines[0].get_ydata()) == [0.4, 0.7]

    axis = plot_generation_metrics(logbook, metrics=["fitness_best"])
    assert list(axis.lines[0].get_ydata()) == [0.4, 0.7]
    plt.close("all")


@pytest.mark.parametrize(
    "rows, metrics, message",
    [
        ([], ["fitness_best"], "no generations"),
        ([{"gen": 0, "fitness_best": 0.4}], [], "at least one field"),
        ([{"gen": 0, "fitness_best": 0.4}], ["missing"], "missing"),
    ],
)
def test_generation_plot_rejects_unavailable_metrics(rows, metrics, message):
    from deap.tools import Logbook
    from ..plots import plot_generation_metrics

    logbook = Logbook()
    for row in rows:
        logbook.record(**row)
    with pytest.raises(ValueError, match=message):
        plot_generation_metrics(logbook, metrics=metrics)


@pytest.mark.parametrize(
    "plot, field_group",
    [
        (plot_convergence, "convergence"),
        (plot_diversity, "diversity"),
        (plot_search_decisions, "optimizer-control"),
        (plot_optimizer_events, "optimizer event"),
    ],
)
def test_diagnostic_plot_reports_missing_telemetry(plot, field_group):
    legacy = SimpleNamespace(history={"gen": [0, 1]})
    with pytest.raises(ValueError, match=f"No {field_group} fields"):
        plot(legacy)


def test_candidate_rankings_reject_all_failed_candidates():
    estimator = SimpleNamespace(cv_results_={"mean_test_score": [np.nan, -np.inf]})
    with pytest.raises(ValueError, match="No candidate scores"):
        plot_candidate_rankings(estimator)


@pytest.mark.parametrize("kind", ["hpo", "features"])
def test_live_plot_receives_both_searches_generation_history(kind, tmp_path, monkeypatch):
    from matplotlib import pyplot as plt
    from ..plots import LivePlot

    monkeypatch.setattr(plt, "pause", lambda interval: None)
    callback = LivePlot(save_path=tmp_path / "training.png")
    common = dict(
        cv=2,
        scoring="neg_mean_squared_error",
        population_size=4,
        generations=1,
        random_state=7,
        n_jobs=1,
        verbose=False,
    )
    search = (
        GASearchCV(DecisionTreeRegressor(), param_grid={"max_depth": Integer(1, 3)}, **common)
        if kind == "hpo"
        else GAFeatureSelectionCV(DecisionTreeRegressor(), **common)
    ).fit(X[:40], y[:40], callbacks=callback)

    assert (tmp_path / "training.png").is_file()
    assert list(callback.ax.lines[0].get_xdata()) == search.history["gen"]
    assert list(callback.ax.lines[0].get_ydata()) == search.history["fitness_best"]
    plt.close("all")


def test_live_plot_preserves_axes_and_refreshes_final_generation():
    from matplotlib import pyplot as plt
    from ..plots import LivePlot

    fig, ax = plt.subplots()
    guide = ax.axhline(0)
    rows = [{"gen": 3, "fitness_best": -6, "fitness": -3}]
    live = LivePlot(ax=ax)
    try:
        live.on_step(logbook=rows)
        best = ax.lines[-2]
        rows.append({"gen": 5, "fitness_best": -6, "fitness": -2})
        live.on_step(logbook=rows)
        assert guide in ax.lines and best is ax.lines[-2]
        rows[-1].update(fitness_best=-8, local_refinements=1)
        live.on_end(logbook=rows)
        assert list(best.get_xdata()) == [3, 5]
        assert list(best.get_ydata()) == [-6, -8]
    finally:
        plt.close(fig)


@pytest.mark.parametrize("direction", ["min", "max"])
def test_live_observed_includes_reevaluations_and_final_refinement(direction):
    from itertools import count
    from matplotlib import pyplot as plt
    from ..callbacks import BaseCallback
    from ..plots import LivePlot

    # Reevaluating the same genotype can yield a better CV score.
    sequence = count()
    sign = 1 if direction == "max" else -1

    def scorer(estimator, X, y):
        return sign * next(sequence)

    search = GASearchCV(
        DecisionTreeRegressor(random_state=7),
        cv=2,
        scoring=scorer,
        criteria=direction,
        param_grid={"max_depth": Integer(1, 2)},
        random_state=7,
        population_size=6,
        generations=2,
        keep_top_k=2,
        use_cache=False,
        n_jobs=1,
        verbose=False,
        algorithm="eaMuPlusLambda",
        local_search=True,
        local_search_steps=1,
    )
    live = LivePlot(metrics=("fitness_best_observed", "fitness"))
    before_refinement = []

    class CheckFrame(BaseCallback):
        def on_step(self, record, logbook, estimator):
            candidates = estimator.logbook.chapters["parameters"]
            expected = (max if direction == "max" else min)(row["score"] for row in candidates)
            np.testing.assert_equal(live.ax.lines[0].get_ydata()[-1], expected)
            np.testing.assert_equal(live.ax.lines[1].get_ydata()[-1], record["fitness"])
            before_refinement.append(expected)

    try:
        search.fit(X[:40], y[:40], callbacks=[live, CheckFrame()])
        assert len(before_refinement) >= 2
        assert sign * live.ax.lines[0].get_ydata()[-1] > sign * before_refinement[-1]
        assert search.history["local_refinements"][-1] > 0
        for line, field in zip(live.ax.lines, live.metrics):
            np.testing.assert_array_equal(line.get_ydata(), search.history[field])
        scores = search.cv_results_["mean_test_score"]
        assert live.ax.lines[0].get_ydata()[-1] == (max if direction == "max" else min)(scores)
        assert search.history["fitness_best"][-1] != live.ax.lines[0].get_ydata()[-1]
    finally:
        plt.close("all")


def test_plus_lambda_live_final_frame_matches_improved_r2():
    from matplotlib import pyplot as plt
    from ..callbacks import BaseCallback
    from ..plots import LivePlot

    live = LivePlot(metrics=("fitness_best_observed", "fitness"))
    step_scores = []

    class Capture(BaseCallback):
        def on_step(self, record, logbook, estimator):
            step_scores.append(live.ax.lines[0].get_ydata()[-1])

    search = GASearchCV(
        DecisionTreeRegressor(random_state=3),
        cv=2,
        scoring="r2",
        param_grid={"max_depth": Integer(1, 18), "min_samples_leaf": Integer(1, 15)},
        population_size=4,
        generations=1,
        random_state=0,
        n_jobs=1,
        verbose=False,
        algorithm="eaMuPlusLambda",
        local_search=True,
        local_search_steps=3,
        local_search_top_k=2,
    )
    try:
        search.fit(X[:100], y[:100], callbacks=[live, Capture()])
        assert len(step_scores) == 2
        assert live.ax.lines[0].get_ydata()[-1] > step_scores[-1]
        for line, field in zip(live.ax.lines, live.metrics):
            np.testing.assert_array_equal(line.get_ydata(), search.history[field])
    finally:
        plt.close("all")


@pytest.mark.parametrize("kind", ["hpo", "features"])
def test_live_resume_early_stop_and_multimetric(kind, tmp_path):
    from matplotlib import pyplot as plt
    from ..callbacks import ModelCheckpoint, ThresholdStopping
    from ..plots import LivePlot

    def build():
        common = dict(
            cv=2,
            scoring={"loss": "neg_mean_absolute_error", "r2": "r2"},
            refit="loss",
            population_size=4,
            generations=2,
            n_jobs=1,
            verbose=False,
            random_state=18,
        )
        tree = DecisionTreeRegressor(random_state=3)
        if kind == "hpo":
            return GASearchCV(tree, param_grid={"max_depth": Integer(1, 12)}, **common)
        return GAFeatureSelectionCV(tree, **common)

    live = LivePlot(metrics=("fitness_best_observed", "fitness"))
    checkpoint = ModelCheckpoint(str(tmp_path / "checkpoint.pkl"))
    try:
        first = build().fit(
            X[:60],
            y[:60],
            callbacks=[checkpoint, ThresholdStopping(-np.inf, metric="fitness_best"), live],
        )
        assert first.history["gen"] == [0]
        first_score = live.ax.lines[0].get_ydata()[0]
        assert first_score == max(first.cv_results_["mean_test_loss"]) < 0
        resumed = build().fit(X[:60], y[:60], callbacks=[checkpoint, live])
        assert resumed.history["gen"][0] == 0 and resumed.history["gen"][-1] > 0
        assert len(live.ax.lines) == 2
        assert live.ax.get_ylabel() == "Fitness (loss)"
        np.testing.assert_array_equal(live.ax.lines[0].get_xdata(), resumed.history["gen"])
        best = live.ax.lines[0].get_ydata()
        assert best[0] == first_score and np.all(np.diff(best) >= 0)
        assert best[-1] == max(resumed.cv_results_["mean_test_loss"])
    finally:
        plt.close("all")


def test_generation_plot_preserves_style_and_legend_on_standard_axes():
    from matplotlib import pyplot as plt
    from ..plots import plot_generation_metrics

    fig, ax = plt.subplots()
    try:
        with plt.rc_context({"axes.facecolor": "ivory", "axes.grid": False}):
            before = plt.rcParams.copy()
            plot_fitness_evolution(evolved_estimator, metrics=["fitness_best", "fitness"], ax=ax)
            assert plt.rcParams == before
        fig.canvas.draw()
        legend = ax.get_legend().get_window_extent()
        assert fig.bbox.contains(*legend.min) and fig.bbox.contains(*legend.max)
        ax.clear()
        plot_generation_metrics([{"gen": 0, "fitness": -3}], "fitness", ax=ax)
        assert all(float(value).is_integer() for value in ax.get_xticks())
    finally:
        plt.close(fig)


def test_plot_history():
    plot = plot_history(
        evolved_estimator,
        fields=["fitness_best", "unique_individual_ratio"],
        kind="line",
    )
    assert plot.get_xlabel() == "Generation"

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
    assert plot.get_xlabel() == "Mean CV score (r2)"
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


@pytest.mark.parametrize(
    "plot_function",
    [
        plot_candidate_scores,
        plot_cv_scores,
        plot_candidate_rankings,
    ],
)
@pytest.mark.parametrize("top_k", [0, -1, 1.5])
def test_candidate_plots_reject_invalid_top_k(plot_function, top_k):
    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        plot_function(evolved_estimator, top_k=top_k)


@pytest.mark.parametrize(
    "plot_function",
    [
        plot_candidate_scores,
        plot_cv_scores,
        plot_candidate_rankings,
    ],
)
@pytest.mark.parametrize("criteria", ["min", "max"])
@pytest.mark.parametrize("top_k", [1, None])
def test_candidate_plots_show_best_finite_scores_first(plot_function, criteria, top_k):
    from matplotlib import pyplot as plt

    scores = [np.inf, np.nan, -np.inf] + [float(index % 3) for index in range(30)]
    expected = sorted(range(3, len(scores)), key=scores.__getitem__, reverse=criteria == "max")
    deviations = np.arange(len(scores)) / 10
    estimator = SimpleNamespace(
        criteria=criteria,
        cv_results_={
            "mean_test_loss": scores,
            "std_test_loss": deviations,
            "split0_test_loss": scores,
        },
    )
    fig, ax = plt.subplots()
    try:
        plot_function(estimator, metric="loss", top_k=top_k, ax=ax)
        if plot_function is plot_cv_scores:
            labels = ax.get_xticklabels()
        else:
            labels = sorted(ax.get_yticklabels(), key=lambda t: -t.get_position()[1])
        assert [int(label.get_text().split()[-1]) for label in labels] == expected[:top_k]
        if plot_function is plot_candidate_rankings:
            line, _, (error_bars,) = ax.containers[0].lines
            indices = expected[:top_k]
            np.testing.assert_array_equal(line.get_xdata(), np.asarray(scores)[indices])
            for index, label, segment in zip(
                indices, labels, error_bars.get_segments(), strict=True
            ):
                np.testing.assert_allclose(
                    segment[:, 0], scores[index] + deviations[index] * np.array([-1, 1])
                )
                np.testing.assert_array_equal(segment[:, 1], [label.get_position()[1]] * 2)
    finally:
        plt.close(fig)


def test_candidate_scores_keep_distinct_rows_with_identical_display_labels():
    from matplotlib import pyplot as plt

    estimator = SimpleNamespace(
        cv_results_={
            "params": [{"alpha": 1.0001}, {"alpha": 1.0002}],
            "mean_test_score": [0.6, 0.8],
        }
    )
    fig, ax = plt.subplots()
    try:
        plot_candidate_scores(estimator, sort=False, ax=ax)
        bars = sorted(ax.patches, key=lambda bar: -bar.get_y())
        assert bars[0].get_y() > bars[1].get_y()
        assert [bar.get_width() for bar in bars] == [0.6, 0.8]
    finally:
        plt.close(fig)


@pytest.mark.parametrize("diagnostic", [plot_convergence, plot_diversity, plot_optimizer_events])
def test_diagnostics_use_recorded_generations(diagnostic):
    from matplotlib import pyplot as plt

    generations = [4, 7, 12]
    estimator = SimpleNamespace(
        history={
            "gen": generations,
            "fitness_best": [0.6, 0.7, 0.7],
            "genotype_diversity": [0.8, 0.5, 0.2],
            "stagnation_generations": [0, 0, 1],
            "random_immigrants": [0, 2, 0],
        }
    )
    fig, ax = plt.subplots()
    try:
        diagnostic(estimator, ax=ax)
        if diagnostic is plot_optimizer_events:
            np.testing.assert_array_equal(ax.collections[0].get_offsets()[:, 0], [7])
        else:
            assert len(fig.axes) == (2 if diagnostic is plot_diversity else 1)
            for axis in fig.axes:
                np.testing.assert_array_equal(axis.lines[0].get_xdata(), generations)
    finally:
        plt.close(fig)


def test_plot_convergence_and_diversity():
    convergence = plot_convergence(evolved_estimator)
    assert convergence.get_title() == "Convergence"
    assert convergence.get_xlabel() == "Generation"

    diversity = plot_diversity(evolved_estimator)
    assert diversity.get_title() == "Diversity"
    assert diversity.get_ylabel() == "diversity ratio"


def test_plot_optimizer_events():
    plot = plot_optimizer_events(evolved_estimator)
    assert plot.get_title() == "Optimizer events"
    assert plot.get_xlabel() == "Generation"

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
    for points in strip_plot.collections:
        offsets = points.get_offsets()
        assert len(set(offsets[:, 0])) == evolved_estimator.n_splits_
    assert len(strip_plot.collections) == 2

    ranking = plot_candidate_rankings(evolved_estimator, top_k=3)
    assert ranking.get_title() == "Candidate ranking (top 3)"
    assert ranking.get_xlabel() == "Mean CV score (r2) ± 1 fold SD"
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


def test_as_list_normalizes_plot_inputs():
    """_as_list normalizes the field/parameter-name inputs used across the plots."""
    # None means "nothing selected" -> empty list.
    assert _as_list(None) == []

    # A single string becomes a one-element list, not a list of characters.
    assert _as_list("score") == ["score"]
    assert _as_list("score") != ["s", "c", "o", "r", "e"]

    # An iterable of strings is materialized into a list.
    assert _as_list(("a", "b")) == ["a", "b"]
    assert _as_list(["a", "b"]) == ["a", "b"]

    # A lazy iterable (generator) is consumed into a concrete list.
    assert _as_list(name for name in ("x", "y")) == ["x", "y"]

    # A non-iterable scalar is wrapped as a single-element list.
    assert _as_list(5) == [5]


def test_candidate_label_shows_hidden_count():
    params = {
        "n_estimators": 100,
        "max_depth": 5,
        "learning_rate": 0.1,
        "criterion": "gini",
    }
    label = _candidate_label(params, fallback=0, max_label_params=2)

    assert "n_estimators=100" in label
    assert "max_depth=5" in label
    assert "+2 more" in label


def test_candidate_label_respects_label_params():
    params = {"n_estimators": 100, "max_depth": 5, "criterion": "gini"}
    label = _candidate_label(params, fallback=0, label_params=["max_depth"])

    assert label == "max_depth=5"
    assert "n_estimators" not in label
    assert "criterion" not in label
    assert "more" not in label


def test_candidate_label_truncates_long_label():
    params = {"kernel": "rbf_with_a_very_long_descriptive_kernel_name"}
    label = _candidate_label(params, fallback=0, label_params=["kernel"], max_length=20)

    assert len(label) == 20
    assert label.endswith("...")


def test_candidate_label_float_formatting():
    params = {"learning_rate": 0.123456789}
    label = _candidate_label(params, fallback=0, label_params=["learning_rate"], float_precision=3)

    assert label == "learning_rate=0.123"


def test_metric_column_lists_available_metrics_on_unknown():
    """An unknown metric error lists the available metric names (#259)."""
    from types import SimpleNamespace
    from ..plots import _metric_column

    estimator = SimpleNamespace(
        cv_results_={
            "mean_test_accuracy": [0.9],
            "mean_test_f1": [0.8],
            "std_test_accuracy": [0.0],  # different prefix, must be ignored
        },
        refit_metric="accuracy",
    )

    # A valid metric still resolves to its column.
    assert _metric_column(estimator, metric="accuracy") == "mean_test_accuracy"

    # An unknown metric names the missing column and the available metrics.
    with pytest.raises(ValueError) as excinfo:
        _metric_column(estimator, metric="roc_auc")
    message = str(excinfo.value)
    assert "mean_test_roc_auc" in message
    assert "Available metrics" in message
    assert "accuracy" in message and "f1" in message


# _as_list – extra edge cases not covered by #270


def test_as_list_with_numpy_array():
    # sklearn pipelines pass arrays here, worth making sure it doesn't blow up
    arr = np.array(["feature_0", "feature_1"])
    assert _as_list(arr) == ["feature_0", "feature_1"]


def test_as_list_with_set():
    # sets are unordered so just check the contents, not the order
    result = _as_list({"a", "b"})
    assert sorted(result) == ["a", "b"]


def test_as_list_empty_string():
    # empty string is not None — should return [""] not []
    assert _as_list("") == [""]


def test_as_list_falsy_values():
    # 0 and False are falsy but they're not None — easy bug to introduce
    assert _as_list(0) == [0]
    assert _as_list(False) == [False]
