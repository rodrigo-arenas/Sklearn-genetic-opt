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
            series = pd.Series(plotted[field].to_numpy(), index=x_values)
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
        series = pd.Series(plotted[field].to_numpy(), index=x_values)
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


__all__ = ["plot_fitness_evolution", "plot_history", "plot_search_space"]
