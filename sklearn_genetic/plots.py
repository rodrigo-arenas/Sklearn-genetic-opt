import logging
from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)  # noqa

# Check if seaborn is installed as an extra requirement
try:
    import seaborn as sns
except ModuleNotFoundError:  # noqa
    sns = None
    logger.error("seaborn not found, pip install seaborn to use plots functions")  # noqa

from .genetic_search import GAFeatureSelectionCV
from .parameters import Metrics
from .utils import logbook_to_pandas

"""
This module contains useful plotting helpers to explore optimization results.
"""

_SEARCH_SPACE_KINDS = {"pair", "heatmap"}
_HISTORY_KINDS = {"line", "bar", "area", "step"}
_LANDSCAPE_KINDS = {"scatter", "hexbin"}
_CV_SCORE_KINDS = {"box", "violin", "strip"}
_CONTROL_FIELDS = [
    "mutation_probability",
    "selection_pressure",
    "diversity_control_triggered",
    "random_immigrants",
    "duplicate_replacements",
    "local_refinements",
    "fitness_sharing_applied",
]
_CONVERGENCE_FIELDS = ["fitness_best", "fitness", "fitness_max", "fitness_min"]
_DIVERSITY_FIELDS = ["unique_individual_ratio", "genotype_diversity"]
_STAGNATION_FIELD = "stagnation_generations"
_EVENT_FIELDS = [
    "diversity_control_triggered",
    "random_immigrants",
    "duplicate_replacements",
    "local_refinements",
    "fitness_sharing_applied",
]


def _require_seaborn():
    if sns is None:  # pragma: no cover
        raise ImportError("seaborn is required to use sklearn_genetic.plots")


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable):
        return list(value)
    return [value]


def _history_frame(estimator, source="history", fields=None):
    if source == "history":
        frame = pd.DataFrame(estimator.history)
    elif source == "logbook":
        frame = logbook_to_pandas(estimator.logbook)
    else:
        raise ValueError("source must be one of ['history', 'logbook']")

    if fields is not None:
        missing = [field for field in fields if field not in frame.columns]
        if missing:
            raise ValueError(f"fields not found in {source}: {missing}")
        frame = frame.loc[:, list(fields)]

    return frame


def _select_numeric_columns(frame, excluded_columns=None):
    excluded_columns = set(excluded_columns or [])
    numeric = frame.select_dtypes(include=["number", "bool"]).copy()
    if excluded_columns:
        numeric = numeric[[column for column in numeric.columns if column not in excluded_columns]]

    for column in numeric.columns:
        if numeric[column].dtype == bool:
            numeric[column] = numeric[column].astype(float)

    return numeric


def _available_fields(estimator, fields):
    return [field for field in fields if field in getattr(estimator, "history", {})]


def _require_fitted(estimator, attribute, plot_name=None):
    """Raise an actionable error if ``estimator`` lacks a fit-time ``attribute``.

    Plotting is often the first thing new users try after creating an
    estimator, so the message tells them to call ``estimator.fit(X, y)`` first
    and names the attribute the plot relies on (and, when known, the plotting
    function).
    """
    if not hasattr(estimator, attribute):
        who = f"{plot_name} requires" if plot_name else "This plot requires"
        raise ValueError(
            f"{who} a fitted {type(estimator).__name__} estimator. "
            f"Call estimator.fit(X, y) before plotting because this plot reads "
            f"estimator.{attribute}."
        )


def _cv_results_frame(estimator):
    _require_fitted(estimator, "cv_results_")

    return pd.DataFrame(estimator.cv_results_)


def _metric_column(estimator, metric=None, prefix="mean_test"):
    metric = metric or getattr(estimator, "refit_metric", "score")
    column = f"{prefix}_{metric}"
    if column not in getattr(estimator, "cv_results_", {}):
        raise ValueError(f"metric column not found in estimator.cv_results_: {column}")
    return column


def _std_metric_column(estimator, metric=None):
    metric = metric or getattr(estimator, "refit_metric", "score")
    column = f"std_test_{metric}"
    return column if column in getattr(estimator, "cv_results_", {}) else None


def _split_metric_columns(estimator, metric=None):
    metric = metric or getattr(estimator, "refit_metric", "score")
    columns = [
        column
        for column in getattr(estimator, "cv_results_", {})
        if column.startswith("split") and column.endswith(f"_test_{metric}")
    ]
    return sorted(columns, key=lambda column: int(column.split("_", 1)[0].replace("split", "")))


def _format_candidate_value(value, float_precision=3):
    if isinstance(value, (np.floating, float)):
        return f"{float(value):.{float_precision}g}"

    if isinstance(value, (np.integer, int)):
        return str(int(value))

    return str(value)


def _candidate_label(
    params,
    fallback,
    max_length=70,
    label_params=None,
    max_label_params=2,
    float_precision=3,
):
    if not isinstance(params, dict):
        return str(fallback)

    if label_params is None:
        numeric_items = [
            (key, value)
            for key, value in params.items()
            if isinstance(value, (np.floating, float, np.integer, int))
        ]
        remaining_items = [
            (key, value) for key, value in params.items() if key not in dict(numeric_items)
        ]
        shown_items = (numeric_items + remaining_items)[:max_label_params]
    else:
        shown_items = [(key, params[key]) for key in _as_list(label_params) if key in params]

    label_parts = [
        f"{key}={_format_candidate_value(value, float_precision=float_precision)}"
        for key, value in shown_items
    ]

    hidden_count = max(0, len(params) - len(shown_items)) if label_params is None else 0
    if hidden_count:
        label_parts.append(f"+{hidden_count} more")

    label = ", ".join(label_parts) or str(fallback)
    if len(label) > max_length:
        label = f"{label[: max_length - 3]}..."
    return label


def _parameter_columns(estimator, parameters=None):
    if isinstance(estimator, GAFeatureSelectionCV):
        return []

    if parameters is None:
        parameters = getattr(estimator.space, "parameters", [])
    else:
        parameters = _as_list(parameters)

    df = _cv_results_frame(estimator)
    missing = [parameter for parameter in parameters if f"param_{parameter}" not in df.columns]
    if missing:
        raise ValueError(f"parameters not found in estimator.cv_results_: {missing}")

    return [f"param_{parameter}" for parameter in parameters]


def _score_values(estimator, metric=None):
    frame = _cv_results_frame(estimator)
    score_column = _metric_column(estimator, metric=metric)
    return frame, score_column, pd.to_numeric(frame[score_column], errors="coerce")


def _candidate_frame(estimator, metric=None, top_k=None, sort=True):
    frame, score_column, scores = _score_values(estimator, metric=metric)
    candidates = frame.copy()
    candidates[score_column] = scores
    candidates = candidates.dropna(subset=[score_column])

    if candidates.empty:
        raise ValueError("No candidate scores were found in estimator.cv_results_")

    if sort:
        candidates = candidates.sort_values(score_column, ascending=False)

    if top_k is not None:
        candidates = candidates.head(top_k)

    return candidates, score_column


def _candidate_labels(
    candidates,
    ranked=False,
    label_params=None,
    max_label_params=2,
    float_precision=3,
    max_length=70,
):
    if "params" in candidates.columns:
        labels = [
            _candidate_label(
                params,
                fallback=index,
                label_params=label_params,
                max_label_params=max_label_params,
                float_precision=float_precision,
                max_length=max_length,
            )
            for index, params in zip(candidates.index, candidates["params"])
        ]
    elif "n_features" in candidates.columns:
        labels = [
            f"candidate {index} ({n} features)" for index, n in candidates["n_features"].items()
        ]
    else:
        labels = [f"candidate {index}" for index in candidates.index]

    if ranked:
        return [f"#{rank} {label}" for rank, label in enumerate(labels, start=1)]

    return labels


def _plottable_values(values):
    normalized = []
    for value in values:
        if isinstance(value, (list, tuple, np.ndarray)):
            array = np.asarray(value, dtype=float)
            normalized.append(float(np.nanmean(array)) if array.size else np.nan)
        else:
            normalized.append(value)

    return normalized


def _plot_single_series(ax, series, kind="line", label=None, alpha=0.9, color=None):
    if kind == "bar":
        ax.bar(series.index, series.values, label=label, alpha=alpha, color=color)
    elif kind == "area":
        ax.fill_between(series.index, series.values, alpha=alpha, label=label, color=color)
        ax.plot(
            series.index,
            series.values,
            alpha=0.95,
            linewidth=1.5,
            label=label,
            color=color,
        )
    elif kind == "step":
        ax.step(series.index, series.values, where="post", label=label, alpha=alpha, color=color)
    else:
        ax.plot(series.index, series.values, label=label, alpha=alpha, color=color)


def plot_fitness_evolution(
    estimator,
    metric="fitness_best",
    metrics=None,
    *,
    kind="line",
    window=None,
    ax=None,
    title=None,
    palette=None,
):
    """
    Plot one or more evolution metrics stored in ``estimator.history``.

    Parameters
    ----------
    estimator: estimator object
        A fitted estimator from :class:`~sklearn_genetic.GASearchCV` or
        :class:`~sklearn_genetic.GAFeatureSelectionCV`.
    metric: str, default="fitness_best"
        Backward-compatible name for a single metric to plot.
    metrics: list[str] | tuple[str] | None, default=None
        Optional collection of history fields to plot together.
    kind: {"line", "bar", "area", "step"}, default="line"
        Plot style.
    window: int | None, default=None
        Optional rolling window applied before plotting.
    ax: matplotlib.axes.Axes | None, default=None
        Axis to draw on. A new axis is created if omitted.
    title: str | None, default=None
        Optional plot title.
    palette: str | None, default=None
        Optional seaborn palette name used for multiple series.

    Returns
    -------
    matplotlib.axes.Axes
        The axis used for the plot.
    """

    _require_seaborn()

    if kind not in _HISTORY_KINDS:
        raise ValueError(f"kind must be one of {sorted(_HISTORY_KINDS)}")

    if metrics is None:
        metrics = [metric]
    else:
        metrics = _as_list(metrics)

    if metrics == [metric] and metric not in Metrics.list():
        raise ValueError(f"metric must be one of {Metrics.list()}, but got {metric} instead")

    missing = [name for name in metrics if name not in estimator.history]
    if missing:
        raise ValueError(f"metrics not found in estimator.history: {missing}")

    frame = pd.DataFrame({"gen": estimator.history["gen"]})
    for name in metrics:
        frame[name] = estimator.history[name]

    if window is not None:
        frame.loc[:, metrics] = frame.loc[:, metrics].rolling(window=window, min_periods=1).mean()

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    sns.set_style("white")
    colors = sns.color_palette(palette or "rocket", n_colors=len(metrics))

    for color, name in zip(colors, metrics):
        series = frame.set_index("gen")[name]
        _plot_single_series(ax, series, kind=kind, label=name, alpha=0.9, color=color)

    title = title or ("Best fitness so far" if metrics == ["fitness_best"] else "Fitness evolution")
    if window is not None:
        title = f"{title} (rolling window={window})"

    ax.set_title(title)
    ax.set(xlabel="generations", ylabel=f"fitness ({estimator.refit_metric})")
    if len(metrics) > 1:
        ax.legend(title="metric")
    return ax


def plot_history(
    estimator,
    fields=None,
    *,
    source="history",
    kind="line",
    rolling=None,
    subplots=None,
    figsize=None,
    title=None,
    palette=None,
):
    """
    Plot arbitrary history or logbook fields in an easier-to-read layout.

    Parameters
    ----------
    estimator: estimator object
        A fitted estimator with ``history`` or ``logbook`` data.
    fields: list[str] | str | None, default=None
        Explicit fields to plot. If omitted, numeric fields are selected
        automatically from the chosen source.
    source: {"history", "logbook"}, default="history"
        Data source to plot from.
    kind: {"line", "bar", "area", "step"}, default="line"
        Plot style for each field.
    rolling: int | None, default=None
        Optional rolling window applied to the plotted values.
    subplots: bool | None, default=None
        If True, plot one subplot per field. If False, overlay everything on
        one axis. If None, a readable default is chosen automatically.
    figsize: tuple[float, float] | None, default=None
        Optional figure size.
    title: str | None, default=None
        Optional figure title.
    palette: str | None, default=None
        Optional seaborn palette name.

    Returns
    -------
    matplotlib.axes.Axes | numpy.ndarray[matplotlib.axes.Axes]
        The created axis or axes.
    """

    _require_seaborn()

    if kind not in _HISTORY_KINDS:
        raise ValueError(f"kind must be one of {sorted(_HISTORY_KINDS)}")

    frame = _history_frame(estimator, source=source, fields=None)
    if fields is None:
        fields = _select_numeric_columns(frame, excluded_columns={"gen", "index"}).columns.tolist()
    else:
        fields = _as_list(fields)

    if not fields:
        raise ValueError("No plottable fields were found")

    missing = [field for field in fields if field not in frame.columns]
    if missing:
        raise ValueError(f"fields not found in {source}: {missing}")

    plotted = frame.loc[:, fields].copy()
    if rolling is not None:
        plotted = plotted.rolling(window=rolling, min_periods=1).mean()

    x_values = frame["gen"] if "gen" in frame.columns else frame.index
    x_label = "generations" if "gen" in frame.columns else "record index"

    if subplots is None:
        subplots = len(fields) > 3

    sns.set_style("white")
    colors = sns.color_palette(palette or "crest", n_colors=len(fields))

    if subplots:
        fig, axes = plt.subplots(
            len(fields),
            1,
            sharex=True,
            figsize=figsize or (10, max(3, 2.75 * len(fields))),
        )
        axes = np.atleast_1d(axes)
        for axis, color, field in zip(axes, colors, fields):
            series = pd.Series(_plottable_values(plotted[field]), index=x_values)
            _plot_single_series(axis, series, kind=kind, label=field, alpha=0.9, color=color)
            axis.set_ylabel(field)
        axes[-1].set_xlabel(x_label)
        if title:
            fig.suptitle(title)
        else:
            fig.suptitle(f"{source.capitalize()} overview")
        return axes

    if figsize is None:
        figsize = (10, 6)
    _, ax = plt.subplots(figsize=figsize)
    for color, field in zip(colors, fields):
        series = pd.Series(_plottable_values(plotted[field]), index=x_values)
        _plot_single_series(ax, series, kind=kind, label=field, alpha=0.9, color=color)

    ax.set_title(title or f"{source.capitalize()} fields")
    ax.set(xlabel=x_label, ylabel="value")
    if len(fields) > 1:
        ax.legend(title="field")
    return ax


def plot_search_space(
    estimator,
    height=2,
    s=25,
    features=None,
    *,
    kind="pair",
    hue=None,
):
    """
    Plot the sampled search space used during the optimization.

    Parameters
    ----------
    estimator: estimator object
        A fitted estimator from :class:`~sklearn_genetic.GASearchCV`.
    height: float, default=2
        Height of each facet for pair plots.
    s: float, default=25
        Marker size for scatter-based plots.
    features: list[str] | None, default=None
        Subset of fields to plot. If omitted, numeric parameter fields are used.
    kind: {"pair", "heatmap"}, default="pair"
        Plot style. ``pair`` shows pairwise relationships, while ``heatmap``
        shows a correlation matrix.
    hue: str | None, default=None
        Optional column used to color the pair plot.

    Returns
    -------
    seaborn.axisgrid.PairGrid | matplotlib.axes.Axes
        Pair grid or heatmap axis depending on ``kind``.
    """

    _require_seaborn()

    if isinstance(estimator, GAFeatureSelectionCV):
        raise TypeError(
            "Estimator must be a GASearchCV instance, not a GAFeatureSelectionCV instance"
        )

    if kind not in _SEARCH_SPACE_KINDS:
        raise ValueError(f"kind must be one of {sorted(_SEARCH_SPACE_KINDS)}")

    sns.set_style("white")

    df = logbook_to_pandas(estimator.logbook)
    if features:
        available_features = [feature for feature in _as_list(features) if feature in df.columns]
        missing = [feature for feature in _as_list(features) if feature not in df.columns]
        if missing:
            raise ValueError(f"features not found in estimator.logbook: {missing}")
        stats = df[available_features].copy()
    else:
        base_columns = [*estimator.space.parameters, estimator.refit_metric]
        if hue and hue in df.columns and hue not in base_columns:
            base_columns.append(hue)
        stats = df[base_columns].copy()

    if kind == "heatmap":
        heatmap_frame = _select_numeric_columns(stats)
        if heatmap_frame.empty:
            raise ValueError("No numeric columns available to plot the heatmap")
        corr = heatmap_frame.corr(numeric_only=True)
        _, ax = plt.subplots(figsize=(max(6, 1.2 * len(corr.columns)), max(5, 0.8 * len(corr))))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="crest", ax=ax, vmin=-1, vmax=1)
        ax.set_title("Search-space correlation heatmap")
        return ax

    numeric_stats = _select_numeric_columns(stats, excluded_columns={hue} if hue else None)
    if numeric_stats.empty:
        raise ValueError("No numeric columns available to plot the search space")

    plot_kwargs = {
        "data": numeric_stats,
        "vars": numeric_stats.columns.tolist(),
        "height": height,
        "diag_kind": "hist",
        "corner": False,
        "plot_kws": {"s": s, "alpha": 0.25, "edgecolor": "none"},
        "diag_kws": {"alpha": 0.35, "bins": 15},
    }
    if hue and hue in df.columns:
        plot_kwargs["data"] = df[numeric_stats.columns.tolist() + [hue]].copy()
        plot_kwargs["hue"] = hue

    grid = sns.pairplot(**plot_kwargs)
    grid.fig.suptitle("Search-space relationships", y=1.02)
    return grid


def plot_parameter_evolution(
    estimator,
    parameters=None,
    *,
    metric=None,
    ax=None,
    palette="viridis",
):
    """
    Plot how candidate parameter values were explored through evaluations.

    Parameters
    ----------
    estimator: fitted GASearchCV
        A fitted hyperparameter search estimator.
    parameters: list[str] | str | None, default=None
        Parameters to plot. If omitted, all search-space parameters are used.
    metric: str | None, default=None
        Metric used to color each evaluated candidate. Defaults to the refit
        metric.
    ax: matplotlib.axes.Axes | None, default=None
        Axis to draw on when only one parameter is plotted.
    palette: str, default="viridis"
        Matplotlib colormap name used for candidate scores.

    Returns
    -------
    matplotlib.axes.Axes | numpy.ndarray[matplotlib.axes.Axes]
        The axis or axes used for the plot.
    """

    _require_seaborn()

    if isinstance(estimator, GAFeatureSelectionCV):
        raise TypeError(
            "plot_parameter_evolution supports GASearchCV. "
            "Use plot_feature_selection for GAFeatureSelectionCV."
        )

    frame, score_column, scores = _score_values(estimator, metric=metric)
    parameter_columns = _parameter_columns(estimator, parameters=parameters)
    if not parameter_columns:
        raise ValueError("No parameters were found to plot")

    sns.set_style("white")
    if len(parameter_columns) == 1:
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 4))
        axes = np.asarray([ax])
    else:
        _, axes = plt.subplots(
            len(parameter_columns),
            1,
            sharex=True,
            figsize=(10, max(3, 2.5 * len(parameter_columns))),
        )
        axes = np.atleast_1d(axes)

    x_values = np.arange(len(frame))
    mappable = None
    for axis, column in zip(axes, parameter_columns):
        values = frame[column]
        mappable = axis.scatter(
            x_values,
            values,
            c=scores,
            cmap=palette,
            alpha=0.75,
            edgecolors="none",
        )
        axis.set_ylabel(column.replace("param_", ""))
        axis.grid(True, axis="x", alpha=0.15)

    axes[-1].set_xlabel("evaluation index")
    axes[0].set_title("Parameter exploration over time")
    if mappable is not None:
        axes[0].figure.colorbar(mappable, ax=axes.tolist(), label=score_column)

    return axes[0] if len(axes) == 1 else axes


def plot_search_decisions(
    estimator,
    fields=None,
    *,
    rolling=None,
    figsize=None,
    palette=None,
):
    """
    Plot optimizer-control decisions made during the genetic search.

    Parameters
    ----------
    estimator: fitted GASearchCV or GAFeatureSelectionCV
        Fitted estimator with ``history`` telemetry.
    fields: list[str] | str | None, default=None
        Control fields to plot. By default, available optimizer-control
        fields are selected automatically.
    rolling: int | None, default=None
        Optional rolling window applied to plotted values.
    figsize: tuple[float, float] | None, default=None
        Optional figure size.
    palette: str | None, default=None
        Optional seaborn palette name.

    Returns
    -------
    numpy.ndarray[matplotlib.axes.Axes]
        One axis per plotted decision field.
    """

    if fields is None:
        fields = _available_fields(estimator, _CONTROL_FIELDS)

    if not fields:
        raise ValueError("No optimizer-control fields were found in estimator.history")

    return plot_history(
        estimator,
        fields=fields,
        kind="step",
        rolling=rolling,
        subplots=True,
        figsize=figsize,
        title="Optimizer decisions",
        palette=palette,
    )


def plot_candidate_scores(
    estimator,
    top_k=10,
    *,
    metric=None,
    sort=True,
    ax=None,
    title=None,
    color=None,
    label_params=None,
    max_label_params=2,
    float_precision=3,
):
    """
    Plot the best evaluated candidates from ``cv_results_``.

    Parameters
    ----------
    estimator: fitted GASearchCV or GAFeatureSelectionCV
        Fitted estimator with ``cv_results_``.
    top_k: int, default=10
        Number of candidates to show.
    metric: str | None, default=None
        Metric to rank. Defaults to the refit metric.
    sort: bool, default=True
        If True, sort candidates by score before plotting.
    ax: matplotlib.axes.Axes | None, default=None
        Axis to draw on.
    title: str | None, default=None
        Optional plot title.
    color: matplotlib color | None, default=None
        Optional bar color.
    label_params: list[str] | str | None, default=None
        Parameters to show in candidate labels. If omitted, the first
        ``max_label_params`` parameters are shown.
    max_label_params: int, default=2
        Maximum number of parameter values shown in each label when
        ``label_params`` is omitted.
    float_precision: int, default=3
        Significant digits used for float parameter values.

    Returns
    -------
    matplotlib.axes.Axes
        The axis used for the plot.
    """

    _require_seaborn()

    frame, score_column, scores = _score_values(estimator, metric=metric)
    candidates = frame.copy()
    candidates[score_column] = scores
    candidates = candidates.dropna(subset=[score_column])

    if candidates.empty:
        raise ValueError("No candidate scores were found in estimator.cv_results_")

    if sort:
        candidates = candidates.sort_values(score_column, ascending=False)

    candidates = candidates.head(top_k)
    labels = _candidate_labels(
        candidates,
        label_params=label_params,
        max_label_params=max_label_params,
        float_precision=float_precision,
    )

    if ax is None:
        _, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(candidates))))

    sns.set_style("white")
    bar_color = color or sns.color_palette("crest", n_colors=1)[0]
    ax.barh(labels[::-1], candidates[score_column].to_numpy()[::-1], color=bar_color, alpha=0.9)
    ax.set_xlabel(score_column)
    ax.set_ylabel("candidate")
    ax.set_title(title or f"Top {len(candidates)} candidate scores")
    return ax


def plot_feature_selection(
    estimator,
    feature_names=None,
    *,
    ax=None,
    selected_color=None,
    rejected_color="0.82",
):
    """
    Plot the feature mask selected by ``GAFeatureSelectionCV``.

    Parameters
    ----------
    estimator: fitted GAFeatureSelectionCV
        A fitted feature-selection estimator.
    feature_names: list[str] | None, default=None
        Names to display for each feature. If omitted, ``feature_0``,
        ``feature_1``, ... are used.
    ax: matplotlib.axes.Axes | None, default=None
        Axis to draw on.
    selected_color: matplotlib color | None, default=None
        Bar color for selected features.
    rejected_color: matplotlib color, default="0.82"
        Bar color for rejected features.

    Returns
    -------
    matplotlib.axes.Axes
        The axis used for the plot.
    """

    _require_seaborn()

    if not isinstance(estimator, GAFeatureSelectionCV):
        raise TypeError("plot_feature_selection requires a GAFeatureSelectionCV estimator")

    _require_fitted(estimator, "best_features_", plot_name="plot_feature_selection")

    mask = np.asarray(estimator.best_features_, dtype=bool)
    if feature_names is None:
        feature_names = [f"feature_{index}" for index in range(mask.size)]
    else:
        feature_names = _as_list(feature_names)
        if len(feature_names) != mask.size:
            raise ValueError("feature_names must have the same length as estimator.best_features_")

    if ax is None:
        _, ax = plt.subplots(figsize=(10, max(4, 0.28 * mask.size)))

    sns.set_style("white")
    selected_color = selected_color or sns.color_palette("crest", n_colors=1)[0]
    colors = [selected_color if selected else rejected_color for selected in mask]
    ax.barh(feature_names[::-1], mask.astype(int)[::-1], color=colors[::-1], alpha=0.95)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 1], labels=["rejected", "selected"])
    ax.set_xlabel("selection")
    ax.set_title(f"Selected features ({int(mask.sum())}/{mask.size})")
    return ax


def plot_convergence(
    estimator,
    fields=None,
    *,
    ax=None,
    palette=None,
    title="Convergence",
):
    """
    Plot convergence signals across generations.

    Parameters
    ----------
    estimator: fitted GASearchCV or GAFeatureSelectionCV
        Fitted estimator with ``history`` telemetry.
    fields: list[str] | str | None, default=None
        Convergence fields to plot. Defaults to available fitness summary
        fields.
    ax: matplotlib.axes.Axes | None, default=None
        Axis to draw on.
    palette: str | None, default=None
        Optional seaborn palette name.
    title: str, default="Convergence"
        Plot title.

    Returns
    -------
    matplotlib.axes.Axes
        The axis used for the plot.
    """

    _require_seaborn()

    fields = (
        _available_fields(estimator, _CONVERGENCE_FIELDS) if fields is None else _as_list(fields)
    )
    if not fields:
        raise ValueError("No convergence fields were found in estimator.history")

    frame = _history_frame(estimator, fields=fields)
    x_values = frame["gen"] if "gen" in frame.columns else frame.index

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    colors = sns.color_palette(palette or "rocket", n_colors=len(fields))
    for color, field in zip(colors, fields):
        series = pd.Series(_plottable_values(frame[field]), index=x_values)
        _plot_single_series(ax, series, kind="line", label=field, color=color)

    ax.set_title(title)
    ax.set_xlabel("generations")
    ax.set_ylabel(f"fitness ({getattr(estimator, 'refit_metric', 'score')})")
    if len(fields) > 1:
        ax.legend(title="metric")
    return ax


def plot_diversity(
    estimator,
    fields=None,
    *,
    ax=None,
    stagnation=True,
    palette=None,
    title="Diversity",
):
    """
    Plot population diversity without mixing optimizer events.

    ``stagnation_generations`` is drawn on a secondary y-axis by default
    because its scale is different from diversity ratios.

    Parameters
    ----------
    estimator: fitted GASearchCV or GAFeatureSelectionCV
        Fitted estimator with ``history`` telemetry.
    fields: list[str] | str | None, default=None
        Diversity fields to plot. Defaults to available diversity ratios.
    ax: matplotlib.axes.Axes | None, default=None
        Axis to draw on.
    stagnation: bool, default=True
        If True, draw ``stagnation_generations`` on a secondary axis when
        available.
    palette: str | None, default=None
        Optional seaborn palette name.
    title: str, default="Diversity"
        Plot title.

    Returns
    -------
    matplotlib.axes.Axes
        The main axis used for the plot.
    """

    _require_seaborn()

    fields = _available_fields(estimator, _DIVERSITY_FIELDS) if fields is None else _as_list(fields)
    if not fields:
        raise ValueError("No diversity fields were found in estimator.history")

    frame = _history_frame(estimator, fields=fields)
    x_values = frame["gen"] if "gen" in frame.columns else frame.index

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    colors = sns.color_palette(palette or "crest", n_colors=len(fields))
    for color, field in zip(colors, fields):
        series = pd.Series(_plottable_values(frame[field]), index=x_values)
        _plot_single_series(ax, series, kind="line", label=field, color=color)

    ax.set_title(title)
    ax.set_xlabel("generations")
    ax.set_ylabel("diversity ratio")
    ax.set_ylim(bottom=0)
    ax.legend(title="field", loc="upper left")

    if stagnation and _STAGNATION_FIELD in getattr(estimator, "history", {}):
        stagnation_frame = _history_frame(estimator, fields=[_STAGNATION_FIELD])
        twin = ax.twinx()
        twin.plot(
            x_values,
            stagnation_frame[_STAGNATION_FIELD],
            color="0.25",
            linestyle="--",
            label=_STAGNATION_FIELD,
        )
        twin.set_ylabel("stagnant generations")
        twin.legend(loc="upper right")

    return ax


def plot_optimizer_events(
    estimator,
    fields=None,
    *,
    ax=None,
    min_event_value=0,
    marker_size=90,
    palette=None,
    title="Optimizer events",
):
    """
    Plot optimizer interventions as a generation timeline.

    Parameters
    ----------
    estimator: fitted GASearchCV or GAFeatureSelectionCV
        Fitted estimator with ``history`` telemetry.
    fields: list[str] | str | None, default=None
        Event fields to show. Defaults to available optimizer event fields.
    ax: matplotlib.axes.Axes | None, default=None
        Axis to draw on.
    min_event_value: float, default=0
        Minimum numeric value treated as an event.
    marker_size: float, default=90
        Base marker size.
    palette: str | None, default=None
        Optional seaborn palette name.
    title: str, default="Optimizer events"
        Plot title.

    Returns
    -------
    matplotlib.axes.Axes
        The axis used for the event timeline.
    """

    _require_seaborn()

    fields = _available_fields(estimator, _EVENT_FIELDS) if fields is None else _as_list(fields)
    if not fields:
        raise ValueError("No optimizer event fields were found in estimator.history")

    frame = _history_frame(estimator, fields=fields)
    generations = pd.Series(
        frame["gen"] if "gen" in frame.columns else frame.index, index=frame.index
    )

    if ax is None:
        _, ax = plt.subplots(figsize=(10, max(3, 0.45 * len(fields))))

    colors = sns.color_palette(palette or "tab10", n_colors=len(fields))
    any_event = False
    for row, (field, color) in enumerate(zip(fields, colors)):
        values = pd.to_numeric(frame[field], errors="coerce").fillna(0)
        event_mask = values > min_event_value
        if event_mask.any():
            any_event = True
            sizes = marker_size + (values[event_mask].astype(float) * marker_size * 0.35)
            ax.scatter(
                generations[event_mask],
                np.full(int(event_mask.sum()), row),
                s=sizes,
                color=color,
                alpha=0.85,
                edgecolors="white",
                linewidth=0.7,
            )
        ax.hlines(row, generations.iloc[0], generations.iloc[-1], color="0.9", linewidth=1)

    if not any_event:
        ax.text(
            0.5,
            0.5,
            "No optimizer events recorded",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    ax.set_yticks(range(len(fields)), labels=fields)
    ax.set_xlabel("generations")
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.15)
    return ax


def plot_score_landscape(
    estimator,
    x,
    y,
    *,
    metric=None,
    kind="scatter",
    ax=None,
    gridsize=25,
    palette="viridis",
    alpha=0.75,
):
    """
    Plot where candidate scores were strong in a two-parameter slice.

    Parameters
    ----------
    estimator: fitted GASearchCV
        Fitted hyperparameter search estimator.
    x: str
        Parameter name for the x-axis.
    y: str
        Parameter name for the y-axis.
    metric: str | None, default=None
        Metric used for coloring. Defaults to the refit metric.
    kind: {"scatter", "hexbin"}, default="scatter"
        Landscape visualization. ``hexbin`` aggregates dense numeric spaces.
    ax: matplotlib.axes.Axes | None, default=None
        Axis to draw on.
    gridsize: int, default=25
        Hexbin grid size.
    palette: str, default="viridis"
        Matplotlib colormap name.
    alpha: float, default=0.75
        Marker transparency for scatter plots.

    Returns
    -------
    matplotlib.axes.Axes
        The axis used for the landscape.
    """

    _require_seaborn()

    if isinstance(estimator, GAFeatureSelectionCV):
        raise TypeError("plot_score_landscape supports GASearchCV only")

    if kind not in _LANDSCAPE_KINDS:
        raise ValueError(f"kind must be one of {sorted(_LANDSCAPE_KINDS)}")

    frame, score_column, scores = _score_values(estimator, metric=metric)
    x_column, y_column = _parameter_columns(estimator, [x, y])
    landscape = pd.DataFrame(
        {
            x: pd.to_numeric(frame[x_column], errors="coerce"),
            y: pd.to_numeric(frame[y_column], errors="coerce"),
            score_column: scores,
        }
    ).dropna()

    if landscape.empty:
        raise ValueError("No numeric candidate values were found for the requested landscape")

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    if kind == "hexbin":
        mappable = ax.hexbin(
            landscape[x],
            landscape[y],
            C=landscape[score_column],
            reduce_C_function=np.nanmean,
            gridsize=gridsize,
            cmap=palette,
            mincnt=1,
        )
    else:
        std_column = _std_metric_column(estimator, metric=metric)
        sizes = 55
        if std_column is not None and std_column in frame:
            std_values = pd.to_numeric(
                frame.loc[landscape.index, std_column], errors="coerce"
            ).fillna(0)
            sizes = 50 + (std_values.to_numpy() * 500)
        mappable = ax.scatter(
            landscape[x],
            landscape[y],
            c=landscape[score_column],
            s=sizes,
            cmap=palette,
            alpha=alpha,
            edgecolors="none",
        )

    ax.figure.colorbar(mappable, ax=ax, label=score_column)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"Score landscape: {x} vs {y}")
    return ax


def plot_cv_scores(
    estimator,
    top_k=5,
    *,
    metric=None,
    kind="box",
    ax=None,
    palette=None,
    label_params=None,
    max_label_params=2,
    float_precision=3,
):
    """
    Plot fold-level CV scores for the strongest candidates.

    Parameters
    ----------
    estimator: fitted GASearchCV or GAFeatureSelectionCV
        Fitted estimator with split-level ``cv_results_``.
    top_k: int, default=5
        Number of ranked candidates to include.
    metric: str | None, default=None
        Metric to plot. Defaults to the refit metric.
    kind: {"box", "violin", "strip"}, default="box"
        Distribution plot type.
    ax: matplotlib.axes.Axes | None, default=None
        Axis to draw on.
    palette: str | None, default=None
        Optional seaborn palette name.
    label_params: list[str] | str | None, default=None
        Parameters to show in candidate labels. If omitted, the first
        ``max_label_params`` parameters are shown.
    max_label_params: int, default=2
        Maximum number of parameter values shown in each label when
        ``label_params`` is omitted.
    float_precision: int, default=3
        Significant digits used for float parameter values.

    Returns
    -------
    matplotlib.axes.Axes
        The axis used for the plot.
    """

    _require_seaborn()

    if kind not in _CV_SCORE_KINDS:
        raise ValueError(f"kind must be one of {sorted(_CV_SCORE_KINDS)}")

    split_columns = _split_metric_columns(estimator, metric=metric)
    if not split_columns:
        raise ValueError("No split-level test scores were found in estimator.cv_results_")

    candidates, score_column = _candidate_frame(estimator, metric=metric, top_k=top_k)
    labels = _candidate_labels(
        candidates,
        ranked=True,
        label_params=label_params,
        max_label_params=max_label_params,
        float_precision=float_precision,
    )
    records = []
    for label, (_, row) in zip(labels, candidates.iterrows()):
        for split_column in split_columns:
            records.append(
                {
                    "candidate": label,
                    "split": split_column.split("_", 1)[0],
                    "score": row[split_column],
                }
            )

    long_frame = pd.DataFrame(records)
    if ax is None:
        _, ax = plt.subplots(figsize=(max(8, 1.2 * len(labels)), 5))

    colors = sns.color_palette(palette or "crest", n_colors=len(labels))
    if kind == "violin":
        sns.violinplot(
            data=long_frame,
            x="candidate",
            y="score",
            hue="candidate",
            palette=colors,
            legend=False,
            ax=ax,
        )
    elif kind == "strip":
        sns.stripplot(
            data=long_frame,
            x="candidate",
            y="score",
            hue="candidate",
            palette=colors,
            legend=False,
            ax=ax,
            size=7,
        )
    else:
        sns.boxplot(
            data=long_frame,
            x="candidate",
            y="score",
            hue="candidate",
            palette=colors,
            legend=False,
            ax=ax,
        )
        sns.stripplot(data=long_frame, x="candidate", y="score", color="0.25", ax=ax, size=4)

    ax.set_title(f"Cross-validation scores for top {len(labels)} candidates")
    ax.set_xlabel("candidate")
    ax.set_ylabel(score_column.replace("mean_", "split_"))
    ax.tick_params(axis="x", rotation=30)
    return ax


def plot_candidate_rankings(
    estimator,
    top_k=15,
    *,
    metric=None,
    ax=None,
    color=None,
    title=None,
    label_params=None,
    max_label_params=2,
    float_precision=3,
):
    """
    Plot ranked candidates with mean score and CV standard deviation.

    Parameters
    ----------
    estimator: fitted GASearchCV or GAFeatureSelectionCV
        Fitted estimator with ``cv_results_``.
    top_k: int, default=15
        Number of candidates to show.
    metric: str | None, default=None
        Metric to rank. Defaults to the refit metric.
    ax: matplotlib.axes.Axes | None, default=None
        Axis to draw on.
    color: matplotlib color | None, default=None
        Optional marker color.
    title: str | None, default=None
        Optional plot title.
    label_params: list[str] | str | None, default=None
        Parameters to show in candidate labels. If omitted, the first
        ``max_label_params`` parameters are shown.
    max_label_params: int, default=2
        Maximum number of parameter values shown in each label when
        ``label_params`` is omitted.
    float_precision: int, default=3
        Significant digits used for float parameter values.

    Returns
    -------
    matplotlib.axes.Axes
        The axis used for the ranking plot.
    """

    _require_seaborn()

    candidates, score_column = _candidate_frame(estimator, metric=metric, top_k=top_k)
    labels = _candidate_labels(
        candidates,
        ranked=True,
        label_params=label_params,
        max_label_params=max_label_params,
        float_precision=float_precision,
    )
    scores = candidates[score_column].to_numpy()
    std_column = _std_metric_column(estimator, metric=metric)
    errors = candidates[std_column].to_numpy() if std_column in candidates else None

    if ax is None:
        _, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(candidates))))

    marker_color = color or sns.color_palette("crest", n_colors=1)[0]
    y_positions = np.arange(len(candidates))[::-1]
    ax.errorbar(
        scores[::-1],
        y_positions,
        xerr=errors[::-1] if errors is not None else None,
        fmt="o",
        color=marker_color,
        ecolor="0.55",
        capsize=3,
        markersize=7,
    )
    ax.set_yticks(y_positions, labels=labels[::-1])
    ax.set_xlabel(score_column)
    ax.set_ylabel("candidate")
    ax.set_title(title or f"Candidate ranking (top {len(candidates)})")
    ax.grid(True, axis="x", alpha=0.2)
    return ax


def plot_search_overview(
    estimator,
    *,
    metric=None,
    top_k=8,
    figsize=(14, 10),
    palette=None,
):
    """
    Create a compact diagnostic dashboard for a fitted genetic search.

    The overview answers the most common post-fit questions: convergence,
    diversity, optimizer-control decisions, and whether strong candidates were
    found. For feature-selection searches, the last panel shows the selected
    feature mask.

    Parameters
    ----------
    estimator: fitted GASearchCV or GAFeatureSelectionCV
        Fitted estimator to summarize.
    metric: str | None, default=None
        Metric used for candidate scores. Defaults to the refit metric.
    top_k: int, default=8
        Number of candidates to show in the score panel.
    figsize: tuple[float, float], default=(14, 10)
        Figure size.
    palette: str | None, default=None
        Optional seaborn palette name for line plots.

    Returns
    -------
    numpy.ndarray[matplotlib.axes.Axes]
        The 2x2 axes array used for the dashboard.
    """

    _require_seaborn()

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    plot_convergence(estimator, ax=axes[0, 0], palette=palette)
    plot_diversity(estimator, ax=axes[0, 1])
    plot_optimizer_events(estimator, ax=axes[1, 0])

    if isinstance(estimator, GAFeatureSelectionCV):
        plot_feature_selection(estimator, ax=axes[1, 1])
    else:
        plot_candidate_rankings(
            estimator,
            top_k=top_k,
            metric=metric,
            ax=axes[1, 1],
            title="Best evaluated candidates",
        )

    fig.tight_layout()
    return axes


class SearchPlotter:
    """
    Object-oriented plotting facade for fitted genetic search estimators.

    Parameters
    ----------
    estimator: fitted GASearchCV or GAFeatureSelectionCV
        Fitted estimator whose post-fit metadata should be visualized.

    Examples
    --------
    >>> plotter = SearchPlotter(search)
    >>> plotter.overview()
    >>> plotter.score_landscape("max_depth", "min_samples_split")
    """

    def __init__(self, estimator):
        self.estimator = estimator

    def overview(self, **kwargs):
        return plot_search_overview(self.estimator, **kwargs)

    def convergence(self, **kwargs):
        return plot_convergence(self.estimator, **kwargs)

    def diversity(self, **kwargs):
        return plot_diversity(self.estimator, **kwargs)

    def optimizer_events(self, **kwargs):
        return plot_optimizer_events(self.estimator, **kwargs)

    def parameter_evolution(self, parameters=None, **kwargs):
        return plot_parameter_evolution(self.estimator, parameters=parameters, **kwargs)

    def score_landscape(self, x, y, **kwargs):
        return plot_score_landscape(self.estimator, x=x, y=y, **kwargs)

    def candidate_scores(self, **kwargs):
        return plot_candidate_scores(self.estimator, **kwargs)

    def candidate_rankings(self, **kwargs):
        return plot_candidate_rankings(self.estimator, **kwargs)

    def cv_scores(self, **kwargs):
        return plot_cv_scores(self.estimator, **kwargs)

    def feature_selection(self, **kwargs):
        return plot_feature_selection(self.estimator, **kwargs)


__all__ = [
    "SearchPlotter",
    "plot_candidate_scores",
    "plot_candidate_rankings",
    "plot_convergence",
    "plot_cv_scores",
    "plot_diversity",
    "plot_feature_selection",
    "plot_fitness_evolution",
    "plot_history",
    "plot_optimizer_events",
    "plot_parameter_evolution",
    "plot_search_decisions",
    "plot_search_overview",
    "plot_search_space",
    "plot_score_landscape",
]
