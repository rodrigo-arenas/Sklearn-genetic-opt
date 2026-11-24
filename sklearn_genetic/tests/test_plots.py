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
    plot_cv_fold_scores,
    plot_cv_scores,
    plot_diversity,
    plot_feature_selection,
    plot_finalist_scores,
    plot_fit_work,
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


@pytest.mark.parametrize("task", ["parameters", "features"])
def test_plot_fit_work_reports_observed_fits_and_reused_scores(task):
    from matplotlib import pyplot as plt
    from sklearn.model_selection import KFold
    from ..callbacks import CVBudgetStopping

    training_sizes = []

    class CountingTree(DecisionTreeRegressor):
        def fit(self, X, y, **kwargs):
            training_sizes.append(len(X))
            return super().fit(X, y, **kwargs)

    X_fit, y_fit = X_train[:80], y_train[:80]
    folds = list(KFold(3, shuffle=True, random_state=17).split(X_fit))
    alternate = list(KFold(3, shuffle=True, random_state=29).split(X_fit))
    options = dict(
        cv=folds,
        final_selection="reuse_folds",
        final_selection_top_k=2,
        final_selection_cv=[folds[0], alternate[1], folds[2]],
        population_size=6,
        generations=0,
        error_score="raise",
        use_cache=False,
        n_jobs=1,
        verbose=False,
        random_state=17,
    )
    tree = CountingTree(random_state=17)
    if task == "features":
        search = GAFeatureSelectionCV(tree, max_features=4, **options)
    else:
        search = GASearchCV(
            tree,
            param_grid={"max_depth": Integer(1, 8), "min_samples_leaf": Integer(1, 6)},
            **options,
        )
    search.fit(X_fit, y_fit, callbacks=CVBudgetStopping(4))
    ax = plot_fit_work(search)
    try:
        widths = [bar.get_width() for bar in ax.patches]
        assert widths == [12, 2, 1, 4]
        assert sum(widths[:3]) == len(training_sizes)
        assert training_sizes.count(len(X_fit)) == 1
        assert [tick.get_text() for tick in ax.get_yticklabels()] == [
            "Search fold fits",
            "Finalist fold fits",
            "Full-data refit",
            "Reused finalist scores",
        ]
        assert ax.yaxis_inverted()
        assert all(bar.get_hatch() is None for bar in ax.patches[:3])
        assert ax.patches[-1].get_hatch()
        ax.figure.canvas.draw()
    finally:
        plt.close(ax.figure)

    elapsed_ax = plot_fit_work(search, unit="seconds")
    try:
        assert [bar.get_width() for bar in elapsed_ax.patches] == [
            search.search_time_,
            search.final_selection_results_["time_seconds"],
            search.refit_time_,
        ]
        assert [tick.get_text() for tick in elapsed_ax.get_yticklabels()] == [
            "Search",
            "Finalist validation",
            "Full-data refit",
        ]
        assert elapsed_ax.get_xlabel() == "Elapsed seconds"
        assert all(bar.get_hatch() is None for bar in elapsed_ax.patches)
        elapsed_ax.figure.canvas.draw()
    finally:
        plt.close(elapsed_ax.figure)


@pytest.mark.parametrize("refit", [True, False])
def test_plot_fit_work_respects_checkpoint_and_reuses_supplied_axes(refit, tmp_path):
    from matplotlib import pyplot as plt
    from ..callbacks import ModelCheckpoint

    pytest.importorskip("pymoo")
    search = GAFeatureSelectionCV(
        DecisionTreeRegressor(random_state=17),
        cv=2,
        generations=0,
        population_size=6,
        optimizer_backend="pymoo",
        use_cache=False,
        refit=refit,
        n_jobs=1,
        verbose=False,
        random_state=17,
    )
    checkpoint = ModelCheckpoint(tmp_path / "fit-work.pkl")
    search.fit(X_train[:80], y_train[:80], callbacks=checkpoint)
    assert search.fit_counts_["search_fold_fits"] == 12
    search.fit(X_train[:80], y_train[:80], callbacks=checkpoint)
    assert search.fit_stats_["cross_validate_calls"] == 6

    figure, (ax, elapsed_ax) = plt.subplots(2, 1)
    try:
        assert SearchPlotter(search).fit_work(ax=ax) is ax
        assert [bar.get_width() for bar in ax.patches] == [0, 0, int(refit), 0]
        assert ax.get_xlim()[1] > 0
        assert SearchPlotter(search).fit_work(unit="seconds", ax=elapsed_ax) is elapsed_ax
        assert [bar.get_width() for bar in elapsed_ax.patches] == [
            search.search_time_,
            0.0,
            getattr(search, "refit_time_", 0.0),
        ]
        assert elapsed_ax.texts[1].get_text() == "No timing"
        assert (elapsed_ax.texts[2].get_text() == "No timing") is (not refit)
        figure.canvas.draw()
    finally:
        plt.close(figure)


def test_plot_fit_work_explains_missing_fit_counts():
    with pytest.raises(ValueError, match=r"plot_fit_work.*estimator\.fit_counts_"):
        plot_fit_work(SimpleNamespace())


@pytest.mark.parametrize("has_search_time", [True, False])
def test_plot_fit_work_distinguishes_missing_timing_from_zero(has_search_time):
    from matplotlib import pyplot as plt

    model = SimpleNamespace(
        fit_counts_={}, final_selection_results_={"enabled": False, "time_seconds": 0.0}
    )
    if has_search_time:
        model.search_time_ = 0.0
    figure, ax = plt.subplots()
    try:
        assert SearchPlotter(model).fit_work(unit="seconds", ax=ax) is ax
        assert [text.get_text() for text in ax.texts] == [
            "0 s" if has_search_time else "No timing",
            "No timing",
            "No timing",
        ]
        assert ax.get_xlim()[1] > 0
        figure.canvas.draw()
    finally:
        plt.close(figure)


def test_plot_fit_work_rejects_unknown_unit():
    with pytest.raises(ValueError, match="unit must be 'count' or 'seconds'"):
        plot_fit_work(SimpleNamespace(), unit="minutes")


def test_plot_fit_work_can_plot_search_time_after_optimizer_error():
    from matplotlib import pyplot as plt

    model = SimpleNamespace(search_time_=0.25)
    ax = plot_fit_work(model, unit="seconds")
    try:
        assert [bar.get_width() for bar in ax.patches] == [0.25, 0.0, 0.0]
        assert [text.get_text() for text in ax.texts] == ["0.25 s", "No timing", "No timing"]
        ax.figure.canvas.draw()
    finally:
        plt.close(ax.figure)


@pytest.mark.parametrize("task", ["parameters", "features"])
@pytest.mark.parametrize("unavailable", ["none", "first", "both", "raises"])
def test_plot_finalist_scores_shows_results_and_winner_when_available(task, unavailable):
    from matplotlib import pyplot as plt

    X_fit, y_fit = X_train[:40, :2], y_train[:40]

    def score(estimator, X, y):
        simple = estimator.max_depth == 1 if task == "parameters" else X.shape[1] == 1
        if len(X) == 20:
            return 0.9 if simple else 0.8
        if unavailable == "both":
            return np.nan
        if simple:
            return np.nan if unavailable == "first" else 0.6
        if unavailable == "raises":
            raise RuntimeError("fixture interrupts finalist validation")
        return 0.95

    def optimize(problem, evaluate, report):
        proposals = [(1, "squared_error"), (2, "squared_error")]
        if task == "features":
            proposals = [(1, 0), (1, 1)]
        report([handle for _, handle in evaluate(proposals)])

    options = dict(
        cv=2,
        scoring=score,
        final_selection=True,
        final_selection_top_k=2,
        final_selection_cv=4,
        optimizer_backend=optimize,
        population_size=2,
        generations=0,
        n_jobs=1,
        verbose=False,
        error_score="raise",
        random_state=19,
    )
    tree = DecisionTreeRegressor(random_state=19)
    if task == "parameters":
        search = GASearchCV(
            tree,
            param_grid={"max_depth": Integer(1, 2), "criterion": Categorical(["squared_error"])},
            **options,
        )
    else:
        search = GAFeatureSelectionCV(tree, max_features=2, **options)
    if unavailable == "both":
        with pytest.raises(ValueError, match="All final-selection candidates"):
            search.fit(X_fit, y_fit)
    elif unavailable == "raises":
        with pytest.raises(RuntimeError, match="fixture interrupts finalist validation"):
            search.fit(X_fit, y_fit)
    else:
        search.fit(X_fit, y_fit)
    results = search.final_selection_results_
    assert results["original_best_index"] == 0

    figure, ax = plt.subplots(layout="constrained")
    try:
        assert SearchPlotter(search).finalist_scores(ax=ax) is ax
        if unavailable in {"both", "raises"}:
            assert results["selected_index"] is None
            assert not hasattr(search, "best_estimator_")
            assert search.fit_counts_["refit_fits"] == 0
            completed = 2 if unavailable == "both" else 1
            assert search.fit_counts_["finalist_cv_calls"] == completed
            assert search.fit_counts_["finalist_fold_fits"] == 4 * completed
            labels = [tick.get_text() for tick in ax.get_yticklabels()]
            expected = (
                ["Row 0\nsearch winner, finalist CV unavailable", "Row 1\nfinalist CV unavailable"]
                if unavailable == "both"
                else ["Row 0\nsearch winner"]
            )
            assert labels == expected
            assert "Selected finalist" not in ax.get_legend_handles_labels()[1]
            figure.canvas.draw()
            return

        assert results["selected_index"] == search.best_index_ == 1
        sweep, finalists, selected = ax.collections
        np.testing.assert_allclose(sweep.get_offsets()[:, 0], [0.9, 0.8])
        np.testing.assert_allclose(
            finalists.get_offsets()[:, 0], [] if unavailable == "first" else [0.6]
        )
        np.testing.assert_allclose(selected.get_offsets(), [[0.95, 1.13]])
        labels = [tick.get_text() for tick in ax.get_yticklabels()]
        note = ", finalist CV unavailable" if unavailable == "first" else ""
        assert labels == [f"Row 0\nsearch winner{note}", "Row 1"]
        assert len(ax.lines) == (1 if unavailable == "first" else 2)
        assert ax.yaxis_inverted()
        figure.canvas.draw()
    finally:
        plt.close(figure)


def test_plot_finalist_scores_explains_missing_finalist_validation():
    with pytest.raises(ValueError, match=r"plot_finalist_scores.*estimator\.cv_results_"):
        plot_finalist_scores(SimpleNamespace())
    with pytest.raises(ValueError, match="requires at least one completed finalist evaluation"):
        plot_finalist_scores(evolved_estimator)


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


def test_fitness_evolution_uses_cv_calls_and_keeps_repeated_budgets():
    from matplotlib import pyplot as plt

    estimator = SimpleNamespace(
        history={
            "gen": [0, 1, 2, 3],
            "cross_validate_calls": [None, 4, 4, 8],
            "fitness_best_observed": [0.3, 0.5, 0.5, 0.7],
        }
    )
    ax = plot_fitness_evolution(
        estimator, metric="fitness_best_observed", x="cv_calls", kind="step"
    )
    try:
        np.testing.assert_array_equal(ax.lines[0].get_xdata(), [4, 4, 8])
        np.testing.assert_array_equal(ax.lines[0].get_ydata(), [0.5, 0.5, 0.7])
        assert ax.get_xlabel() == "Candidate CV calls"
    finally:
        plt.close(ax.figure)


@pytest.mark.parametrize("kind", ["line", "area", "step", "bar"])
def test_fitness_cv_axis_breaks_between_generations_with_unknown_cost(kind):
    from matplotlib import pyplot as plt
    from matplotlib.path import Path

    estimator = SimpleNamespace(
        history={
            "gen": [0, 1, 2, 3, 4, 5],
            "cross_validate_calls": [None, 4, None, 8, 8, None],
            "fitness": [0.05, 0.3, 0.95, 0.6, 0.8, 0.99],
        }
    )
    ax = plot_fitness_evolution(estimator, metric="fitness", x="cv_calls", kind=kind, window=2)
    try:
        if kind == "bar":
            np.testing.assert_allclose(
                [bar.get_height() for bar in ax.containers[0]], [0.3, 0.45, 0.7]
            )
        else:
            (line,) = ax.lines
            np.testing.assert_array_equal(line.get_xdata(), [4, np.nan, 8, 8])
            np.testing.assert_allclose(line.get_ydata(), [0.3, np.nan, 0.45, 0.7])
            moves = [
                vertices
                for vertices, code in line.get_path().iter_segments()
                if code == Path.MOVETO
            ]
            np.testing.assert_allclose(moves, [[4, 0.3], [8, 0.45]])
            if kind == "area":
                assert len(ax.collections[0].get_paths()) == 2
    finally:
        plt.close(ax.figure)


@pytest.mark.parametrize("kind", ["line", "area", "step", "bar"])
def test_fitness_plot_marks_partial_rows_at_their_plot_positions(kind):
    from matplotlib import pyplot as plt
    from matplotlib.collections import PathCollection

    estimator = SimpleNamespace(
        history={
            "gen": [0, 1, 2, 3, 4],
            "cross_validate_calls": [None, 4, 4, 8, 12],
            "fitness_best_observed": [0.4, 0.5, 0.7, np.nan, 0.8],
            "fitness": [0.3, 0.4, 0.6, np.nan, 0.65],
            "partial": [True, False, True, True, False],
        }
    )
    metrics = ["fitness_best_observed", "fitness"]
    ax = plot_fitness_evolution(estimator, metrics=metrics, x="cv_calls", kind=kind, window=2)
    try:
        markers = [artist for artist in ax.collections if isinstance(artist, PathCollection)]
        assert len(markers) == 2
        for i, marker in enumerate(markers):
            if kind == "bar":
                bar = ax.containers[i][1]
                expected = [bar.get_x() + bar.get_width() / 2, bar.get_height()]
            else:
                expected = [4, ax.lines[i].get_ydata()[1]]
            np.testing.assert_allclose(marker.get_offsets(), [expected])
        assert ax.get_legend_handles_labels()[1].count("Partial generation") == 1
    finally:
        plt.close(ax.figure)


@pytest.mark.parametrize("metrics", [["fitness"], ["fitness", "fitness_min"]])
@pytest.mark.parametrize("costs", [[4, 4, 4, 8], [4, 4, 4, 4]])
def test_fitness_bars_group_generations_at_the_same_cv_budget(metrics, costs):
    from matplotlib import pyplot as plt

    estimator = SimpleNamespace(
        history={
            "gen": [0, 1, 2, 3],
            "cross_validate_calls": costs,
            "fitness": [0.5, 0.8, np.nan, 0.9],
            "fitness_min": [0.4, 0.6, 0.5, 0.8],
        }
    )
    ax = plot_fitness_evolution(estimator, metrics=metrics, x="cv_calls", kind="bar")
    try:
        first_group = [
            bar for generation in zip(*ax.containers, strict=True) for bar in generation
        ][: costs.count(4) * len(metrics)]
        previous_right = -np.inf
        for bar in first_group:
            left, width = bar.get_x(), bar.get_width()
            assert left >= previous_right - 1e-12
            previous_right = left + width
        assert (first_group[0].get_x() + previous_right) / 2 == pytest.approx(4)
        for field, bars in zip(metrics, ax.containers, strict=True):
            np.testing.assert_array_equal(
                [bar.get_height() for bar in bars], estimator.history[field]
            )
    finally:
        plt.close(ax.figure)


@pytest.mark.parametrize("window", [None, 2])
def test_fitness_evolution_preserves_gaps_at_nonfinite_generations(window):
    from matplotlib import pyplot as plt

    estimator = SimpleNamespace(
        history={
            "gen": [0, 1, 2, 3],
            "cross_validate_calls": [2, 4, 6, 8],
            "fitness": [-np.inf, 0.5, np.inf, 0.7],
        }
    )
    ax = plot_fitness_evolution(estimator, metric="fitness", x="cv_calls", window=window)
    try:
        np.testing.assert_array_equal(ax.lines[0].get_xdata(), [2, 4, 6, 8])
        np.testing.assert_array_equal(ax.lines[0].get_ydata(), [np.nan, 0.5, np.nan, 0.7])
        left, right = ax.get_xlim()
        assert left <= 2 and right >= 8
    finally:
        plt.close(ax.figure)


@pytest.mark.parametrize("counts", [None, [None, None]])
def test_fitness_evolution_requires_recorded_cv_counts(counts):
    history = {"gen": [0, 1], "fitness_best": [0.5, 0.7]}
    if counts is not None:
        history["cross_validate_calls"] = counts
    estimator = SimpleNamespace(history=history)
    with pytest.raises(ValueError, match="cumulative candidate CV calls"):
        plot_fitness_evolution(estimator, x="cv_calls")
    with pytest.raises(ValueError, match="x must be one of"):
        plot_fitness_evolution(estimator, x="unsupported")


def test_generation_plot_shows_gap_for_older_rows_without_finite_counts():
    from matplotlib import pyplot as plt
    from ..plots import plot_generation_metrics

    rows = [{"gen": 0}, {"gen": 1, "finite_fitness_count": 3}]
    ax = plot_generation_metrics(rows, metrics=["finite_fitness_count"])
    try:
        np.testing.assert_array_equal(ax.lines[0].get_xdata(), [0, 1])
        np.testing.assert_array_equal(ax.lines[0].get_ydata(), [np.nan, 3])
    finally:
        plt.close(ax.figure)


def test_generation_plot_keeps_gaps_when_the_latest_row_omits_a_metric():
    from matplotlib import pyplot as plt
    from ..plots import plot_generation_metrics

    rows = [
        {"gen": 0, "cross_validate_calls": 4},
        {"gen": 1},
        {"gen": 2, "cross_validate_calls": 10},
        {"gen": 3},
    ]
    ax = plot_generation_metrics(rows, metrics=["cross_validate_calls"])
    try:
        np.testing.assert_array_equal(ax.lines[0].get_xdata(), [0, 1, 2, 3])
        np.testing.assert_array_equal(ax.lines[0].get_ydata(), [4, np.nan, 10, np.nan])
        left, right = ax.get_xlim()
        assert left <= 0 and right >= 3
    finally:
        plt.close(ax.figure)


@pytest.mark.parametrize("backend", ["deap", "pymoo"])
@pytest.mark.parametrize("task", ["parameters", "features"])
def test_live_plot_resumes_a_checkpoint_with_missing_cost_and_diversity(backend, task, tmp_path):
    import pickle

    from matplotlib import pyplot as plt
    from sklearn.base import clone
    from ..callbacks import CVBudgetStopping, ModelCheckpoint
    from ..plots import LivePlot

    if backend == "pymoo":
        pytest.importorskip("pymoo")
    options = dict(
        estimator=DecisionTreeRegressor(random_state=4721),
        optimizer_backend=backend,
        cv=2,
        population_size=4,
        generations=2,
        use_cache=False,
        random_state=4721,
        n_jobs=1,
        verbose=False,
    )
    if task == "parameters":
        search = GASearchCV(
            param_grid={"max_depth": Integer(2, 9), "min_samples_leaf": Integer(1, 5)},
            **options,
        )
    else:
        search = GAFeatureSelectionCV(max_features=4, **options)

    checkpoint = ModelCheckpoint(tmp_path / "older.pkl")
    train = X_train[:80], y_train[:80]
    clone(search).fit(*train, callbacks=[CVBudgetStopping(4), checkpoint])
    saved = checkpoint.load()
    fields = ("cross_validate_calls", "genotype_diversity")
    for field in fields:
        del saved["logbook"][0][field]
    checkpoint.checkpoint_path.write_bytes(pickle.dumps(saved))

    figure, ax = plt.subplots()
    image = tmp_path / "resumed-progress.png"
    try:
        live = LivePlot(metrics=fields, ax=ax, save_path=image)
        search.fit(*train, callbacks=[CVBudgetStopping(8), live, checkpoint])
        assert search.fit_counts_["search_cv_calls"] == 8
        assert search.history["cross_validate_calls"] == [None, 8, 12]
        for line, field in zip(ax.lines, fields, strict=True):
            np.testing.assert_array_equal(line.get_xdata(), [0, 1, 2])
            np.testing.assert_array_equal(line.get_ydata(), [np.nan, *search.history[field][1:]])
        left, right = ax.get_xlim()
        assert left <= 0 and right >= 2
        assert image.stat().st_size > 0
    finally:
        plt.close(figure)


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


def test_plot_space_uses_candidate_parameters_alongside_cv_diagnostics():
    from matplotlib import pyplot as plt
    from sklearn.datasets import load_iris
    from .test_catalog import _TreeWithDiagnosticParameterNames

    model = GASearchCV(
        _TreeWithDiagnosticParameterNames(),
        param_grid={"fit_time": Integer(1, 5), "test_accuracy": Integer(1, 4)},
        scoring={"accuracy": "accuracy", "balanced": "balanced_accuracy"},
        refit="accuracy",
        population_size=6,
        generations=0,
        cv=2,
        random_state=19,
        n_jobs=1,
        verbose=False,
    ).fit(*load_iris(return_X_y=True))
    default = plot_search_space(model)
    requested = plot_search_space(
        model,
        features=["param_fit_time", "mean_fit_time", "mean_test_balanced"],
        hue="param_test_accuracy",
    )
    try:
        assert default.x_vars == ["param_fit_time", "param_test_accuracy", "score"]
        np.testing.assert_array_equal(
            default.data["param_fit_time"], model.cv_results_["param_fit_time"]
        )
        np.testing.assert_array_equal(
            default.data["score"], model.cv_results_["mean_test_accuracy"]
        )
        assert requested.x_vars == ["param_fit_time", "mean_fit_time", "mean_test_balanced"]
        for field in ("param_test_accuracy", "mean_fit_time", "mean_test_balanced"):
            np.testing.assert_array_equal(requested.data[field], model.cv_results_[field])
    finally:
        for plot in (default, requested):
            plt.close(plot.fig)


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


def test_fold_score_plot_matches_metric_and_retains_missing_values():
    from matplotlib import pyplot as plt

    search = SimpleNamespace(
        refit_metric="accuracy",
        cv_results_={
            "mean_test_accuracy": [0.8, np.nan, 0.9],
            "rank_test_accuracy": [2, 3, 1],
            "split10_test_accuracy": [0.7, np.inf, 0.95],
            "split2_test_accuracy": [0.9, 0.4, 0.85],
            "split2_test_aux_test_accuracy": [-7.0, -7.0, -7.0],
        },
    )
    previous = plot_cv_scores(search, kind="strip", top_k=1)
    ranked = plot_cv_fold_scores(search, top_k=2)
    picked = SearchPlotter(search).cv_fold_scores(candidate_indices=[1, 0])
    try:
        assert len(previous.collections[0].get_offsets()) == 2
        assert [tick.get_text() for tick in ranked.get_xticklabels()] == ["2", "10"]
        assert [tick.get_text() for tick in ranked.get_yticklabels()] == ["2", "0"]
        assert [tick.get_text() for tick in picked.get_yticklabels()] == ["1", "0"]
        values = picked.images[0].get_array()
        np.testing.assert_array_equal(values.mask, [[False, True], [False, False]])
        np.testing.assert_allclose(values.compressed(), [0.4, 0.9, 0.7])
        assert picked.get_legend().get_texts()[0].get_text() == "Non-finite score"
    finally:
        for ax in (previous, ranked, picked):
            plt.close(ax.figure)


def test_fold_score_plot_uses_stable_prefix_for_a_long_rank_history(monkeypatch):
    from matplotlib import pyplot as plt

    ranks = np.full(4096, 12)
    ranks[1::11] = 4
    ranks[[31, 817, 2065]] = 1
    failed = np.zeros(len(ranks), dtype=bool)
    failed[2065] = True
    scores = np.zeros(len(ranks))
    scores[817] = np.nan
    search = SimpleNamespace(
        cv_results_={
            "rank_test_score": ranks,
            "split0_test_score": scores,
            "all_fits_failed": failed,
        }
    )
    argsort = np.argsort

    def bounded_sort(values, *args, **kwargs):
        assert values is not ranks
        return argsort(values, *args, **kwargs)

    monkeypatch.setattr(np, "argsort", bounded_sort)
    ax = plot_cv_fold_scores(search, top_k=6)
    try:
        assert [tick.get_text() for tick in ax.get_yticklabels()] == [
            "31",
            "817",
            "2065",
            "1",
            "12",
            "23",
        ]
        marks = {item.get_label(): item.get_offsets() for item in ax.collections}
        np.testing.assert_array_equal(marks["Non-finite score"], [[0, 1]])
        np.testing.assert_array_equal(marks["All fits failed"], [[0, 2]])
    finally:
        plt.close(ax.figure)


def test_fold_score_plot_uses_existing_axis_and_can_show_a_fully_missing_row():
    from matplotlib import pyplot as plt

    search = SimpleNamespace(
        cv_results_={
            "rank_test_score": [1, 2],
            "split0_test_score": [0.7, np.nan],
            "split1_test_score": [0.8, np.nan],
        }
    )
    fig, ax = plt.subplots()
    try:
        assert plot_cv_fold_scores(search, candidate_indices=np.int64(1), ax=ax) is ax
        assert fig.axes == [ax]
        assert ax.images[0].get_array().mask.all()
        for indices in ([], [-1], [True], [2], [1.0]):
            with pytest.raises(ValueError, match="valid row positions"):
                plot_cv_fold_scores(search, candidate_indices=indices, ax=ax)
        with pytest.raises(ValueError, match="positive integer"):
            plot_cv_fold_scores(search, top_k=0, ax=ax)
    finally:
        plt.close(fig)


@pytest.fixture
def search_with_missing_fold_scores():
    from sklearn.exceptions import FitFailedWarning

    first, second = np.arange(20), np.arange(20, 40)
    samples = np.column_stack((np.repeat([-1, 1], 20), np.arange(40) % 5))
    targets = samples[:, 1]

    def conditional_score(estimator, X, y):
        if estimator.max_depth == 4 or (estimator.max_depth == 2 and X[0, 0] < 0):
            return np.nan
        return estimator.score(X, y)

    search = GASearchCV(
        DecisionTreeRegressor(random_state=42),
        cv=[(second, first), (first, second)],
        param_grid={
            "max_depth": Categorical([0, 2, 3, 4]),
            "min_samples_leaf": Categorical([1]),
        },
        warm_start_configs=[{"max_depth": depth, "min_samples_leaf": 1} for depth in (0, 2, 3, 4)],
        scoring=conditional_score,
        refit=False,
        population_size=4,
        generations=0,
        n_jobs=1,
        random_state=42,
        verbose=False,
    )
    with pytest.warns(FitFailedWarning, match="All the 2 fits failed"):
        search.fit(samples, targets)
    return search


def test_fold_score_plot_marks_all_fit_failures_separately_from_missing_scores(
    search_with_missing_fold_scores,
):
    from matplotlib import pyplot as plt

    search = search_with_missing_fold_scores
    depths = list(search.cv_results_["param_max_depth"])
    indices = [depths.index(depth) for depth in (2, 3, 0)]
    ax = plot_cv_fold_scores(search, candidate_indices=indices)
    try:
        values = ax.images[0].get_array()
        np.testing.assert_array_equal(values.mask, [[True, False], [False, False], [True, True]])
        marks = {item.get_label(): item.get_offsets() for item in ax.collections}
        np.testing.assert_array_equal(marks["Non-finite score"], [[0, 0]])
        np.testing.assert_array_equal(marks["All fits failed"], [[0, 2], [1, 2]])
        assert {item.get_text() for item in ax.get_legend().get_texts()} == set(marks)
        ax.figure.canvas.draw()
    finally:
        plt.close(ax.figure)


@pytest.mark.parametrize("kind", ["box", "violin", "strip"])
@pytest.mark.parametrize("legacy", [False, True])
def test_cv_score_plot_can_show_explicit_rows_with_incomplete_folds(
    search_with_missing_fold_scores, kind, legacy
):
    from matplotlib import pyplot as plt

    search = search_with_missing_fold_scores
    depths = list(search.cv_results_["param_max_depth"])
    partial, failed, undefined, complete = [depths.index(depth) for depth in (2, 0, 4, 3)]
    assert search.cv_results_["all_fits_failed"][[failed, undefined]].tolist() == [True, False]
    if legacy:
        del search.cv_results_["all_fits_failed"]
    fig, ax = plt.subplots()
    try:
        assert (
            SearchPlotter(search).cv_scores(
                candidate_indices=[partial, failed, undefined, complete, partial],
                top_k=1,
                kind=kind,
                label_params=[],
                ax=ax,
            )
            is ax
        )
        failed_label = "0/2 finite folds" if legacy else "All fits failed"
        assert [tick.get_text() for tick in ax.get_xticklabels()] == [
            f"row {partial}\n1/2 finite folds",
            f"row {failed}\n{failed_label}",
            f"row {undefined}\n0/2 finite folds",
            f"row {complete}",
            f"row {partial}\n1/2 finite folds",
        ]
        if kind == "strip":
            assert [len(item.get_offsets()) for item in ax.collections] == [1, 2, 1]
            np.testing.assert_allclose(
                ax.collections[0].get_offsets()[:, 1],
                [search.cv_results_["split1_test_score"][partial]],
            )
        fig.canvas.draw()
    finally:
        plt.close(fig)


def test_cv_score_plot_checks_explicit_rows_with_no_finite_scores(
    search_with_missing_fold_scores,
):
    search = search_with_missing_fold_scores
    failed = list(search.cv_results_["param_max_depth"]).index(0)
    with pytest.raises(ValueError, match="No finite fold scores.*plot_cv_fold_scores"):
        plot_cv_scores(search, candidate_indices=failed)
    for indices in ([], [True], [-1], [len(search.cv_results_["params"])], [1.0]):
        with pytest.raises(ValueError, match="valid row positions"):
            plot_cv_scores(search, candidate_indices=indices)


def test_cv_score_plot_marks_a_feature_mask_that_failed_to_fit():
    from matplotlib import pyplot as plt
    from sklearn.exceptions import FitFailedWarning

    class TwoFeatureTree(DecisionTreeRegressor):
        def fit(self, X, y, **kwargs):
            if X.shape[1] < 2:
                raise ValueError("two features are required")
            return super().fit(X, y, **kwargs)

    def optimize(problem, evaluate, report):
        report([handle for _, handle in evaluate([(1, 0, 0), (1, 1, 0)])])

    search = GAFeatureSelectionCV(
        TwoFeatureTree(random_state=17),
        optimizer_backend=optimize,
        max_features=2,
        cv=2,
        population_size=2,
        generations=0,
        refit=False,
        random_state=17,
        n_jobs=1,
        verbose=False,
    )
    with pytest.warns(FitFailedWarning, match="All the 2 fits failed"):
        search.fit(X_train[:40, :3], y_train[:40])
    ax = plot_cv_scores(search, candidate_indices=[0, 1], kind="strip")
    try:
        labels = [tick.get_text() for tick in ax.get_xticklabels()]
        assert labels[0].endswith("\nAll fits failed")
        assert "All fits failed" not in labels[1]
        assert len(ax.collections[0].get_offsets()) == 2
    finally:
        plt.close(ax.figure)


def test_plot_search_overview():
    axes = plot_search_overview(evolved_estimator, top_k=3)
    assert axes.shape == (2, 2)
    assert axes[0, 0].get_title() == "Convergence"
    assert axes[1, 1].get_title() == "Best evaluated candidates"


@pytest.mark.parametrize("keep_folds", [True, False])
@pytest.mark.parametrize("metric", [None, "unavailable"])
def test_overview_without_finite_means_preserves_available_diagnostics(keep_folds, metric):
    from matplotlib import pyplot as plt

    X_small = np.random.default_rng(4783).normal(size=(40, 4))
    X_small[:20, 0] = -1
    X_small[20:, 0] = 1
    y_small = X_small[:, 1]
    first, second = np.arange(20), np.arange(20, 40)

    def unavailable_score(estimator, X, y):
        return np.nan if X[0, 0] < 0 else estimator.score(X, y)

    scoring = (
        unavailable_score if metric is None else {"r2": "r2", "unavailable": unavailable_score}
    )
    search = GASearchCV(
        DecisionTreeRegressor(random_state=4783),
        cv=[(second, first), (first, second)],
        param_grid={"max_depth": Integer(2, 9), "min_samples_leaf": Integer(1, 5)},
        scoring=scoring,
        objective_metric=None if metric is None else "r2",
        refit=False,
        use_cache=False,
        population_size=4,
        generations=1,
        n_jobs=1,
        random_state=4783,
        verbose=False,
    )
    if metric is None:
        with pytest.warns(UserWarning, match="No candidate has a finite mean CV score"):
            search.fit(X_small, y_small)
    else:
        search.fit(X_small, y_small)
    if not keep_folds:
        search.cv_results_ = {
            name: values
            for name, values in search.cv_results_.items()
            if not name.startswith("split")
        }
    axes = plot_search_overview(search, metric=metric, top_k=2)
    try:
        assert axes.shape == (2, 2)
        assert axes[0, 1].lines
        panel = axes[1, 1]
        if keep_folds:
            assert "No finite mean CV scores" in panel.get_title()
            values = panel.images[0].get_array()
            np.testing.assert_array_equal(values.mask, [[True, False], [True, False]])
            assert [tick.get_text() for tick in panel.get_yticklabels()] == ["0", "1"]
        else:
            assert [label.get_text() for label in panel.texts] == ["No finite mean CV scores"]
        axes[0, 0].figure.canvas.draw()
    finally:
        plt.close(axes[0, 0].figure)


@pytest.mark.parametrize("task", ["parameters", "features"])
@pytest.mark.parametrize("legacy", ["omitted", "empty", "partial", "no_history"])
def test_overview_from_saved_search_keeps_available_diagnostics(task, legacy, tmp_path):
    from matplotlib import pyplot as plt
    from sklearn.base import clone

    options = dict(
        estimator=DecisionTreeRegressor(random_state=4751),
        cv=2,
        population_size=4,
        generations=1,
        random_state=4751,
        n_jobs=1,
        refit=False,
        verbose=False,
    )
    if task == "parameters":
        search = GASearchCV(
            param_grid={"max_depth": Integer(2, 9), "min_samples_leaf": Integer(1, 5)},
            **options,
        )
    else:
        search = GAFeatureSelectionCV(max_features=4, **options)
    search.fit(X_train[:80], y_train[:80])
    fields = {
        "genotype_diversity",
        "unique_individual_ratio",
        "stagnation_generations",
        "diversity_control_triggered",
        "random_immigrants",
        "duplicate_replacements",
        "local_refinements",
        "fitness_sharing_applied",
    }
    if legacy == "no_history":
        del search.history
    elif legacy == "omitted":
        for field in fields:
            search.history.pop(field, None)
    else:
        for field in fields:
            search.history[field] = [None, None]
        if legacy == "partial":
            search.history["random_immigrants"][-1] = 0
            search.history["genotype_diversity"][-1] = 0.5

    path = tmp_path / "older.pkl"
    search.save(path)
    restored = clone(search)
    restored.load(path)
    axes = plot_search_overview(restored, top_k=3)
    try:
        assert axes.shape == (2, 2)
        convergence = axes[0, 0]
        if legacy == "no_history":
            assert [label.get_text() for label in convergence.texts] == ["No data recorded"]
        else:
            assert convergence.lines
        assert len(axes[0, 0].figure.axes) == 4
        expected = "Selected features" if task == "features" else "Best evaluated candidates"
        assert expected in axes[1, 1].get_title()
        diversity, events = axes[0, 1], axes[1, 0]
        if legacy == "partial":
            np.testing.assert_array_equal(diversity.lines[0].get_ydata(), [np.nan, 0.5])
            assert [label.get_text() for label in events.texts] == ["No optimizer events recorded"]
        else:
            for panel in (diversity, events):
                assert [label.get_text() for label in panel.texts] == ["No data recorded"]
                assert not panel.lines
        axes[0, 0].figure.canvas.draw()
    finally:
        plt.close(axes[0, 0].figure)


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


@pytest.mark.parametrize("kind", ["parameters", "features"])
@pytest.mark.parametrize("engine", ["plugin", "scipy"])
def test_liveplot_and_early_stop_on_portable_engines(kind, engine, tmp_path):
    from matplotlib import pyplot as plt
    from scipy import __version__ as scipy_version
    from ..plots import LivePlot
    from ..callbacks import DeltaThreshold

    def plugin(problem, evaluate, report):
        for _ in range(problem.generations + 1):
            report([handle for _, handle in evaluate(problem.initial_population)])

    if engine == "scipy" and np.lib.NumpyVersion(scipy_version) < "1.15.0":
        pytest.skip("requires the SciPy optimizer extra")
    opts = dict(
        cv=2,
        population_size=6,
        generations=5,
        n_jobs=1,
        random_state=6,
        optimizer_backend=plugin if engine == "plugin" else engine,
        verbose=False,
    )
    reg = DecisionTreeRegressor(max_depth=2, random_state=6)
    model = (
        GASearchCV(
            reg, param_grid={"max_depth": Integer(1, 5), "min_samples_leaf": Integer(1, 3)}, **opts
        )
        if kind == "parameters"
        else GAFeatureSelectionCV(reg, **opts)
    )
    callback = LivePlot(metrics=("fitness_best_observed",), save_path=tmp_path / "live.png")
    try:
        model.fit(X_train[:40], y_train[:40], callbacks=[DeltaThreshold(1e10, 2), callback])
        assert model.history["gen"] == [0, 1]
        line = callback.ax.lines[0]
        np.testing.assert_equal(line.get_xdata(), model.history["gen"])
        np.testing.assert_equal(line.get_ydata(), model.history["fitness_best_observed"])
        assert callback.save_path.stat().st_size > 0
    finally:
        if callback.ax is not None:
            plt.close(callback.ax.figure)
