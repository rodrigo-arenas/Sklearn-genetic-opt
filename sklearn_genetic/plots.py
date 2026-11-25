import logging
from collections.abc import Iterable

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
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
from .callbacks.base import BaseCallback
from .parameters import FITNESS_HISTORY_METRICS as _FITNESS_METRICS
from .evaluation import candidate_parameters, candidate_records
from ._base import _short_cv_positions
from .utils._score_statistics import score_mean

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
_FITNESS_AXES = {
    "generation": ("gen", "Generation"),
    "cv_calls": ("cross_validate_calls", "Candidate CV calls"),
    "fold_fits": ("search_fold_fits", "Search CV fold fits"),
}
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
        frame = pd.DataFrame(candidate_records(estimator))
    else:
        raise ValueError("source must be one of ['history', 'logbook']")

    if fields is not None:
        missing = [field for field in fields if field not in frame.columns]
        if missing:
            raise ValueError(
                f"fields not found in {source}: {missing}. Available fields include: {list(frame.columns)}"
            )
        columns = list(fields)
        if source == "history" and "gen" in frame and "gen" not in columns:
            columns.append("gen")
        frame = frame.loc[:, columns]

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
    cv_results = getattr(estimator, "cv_results_", {})
    if column not in cv_results:
        available = sorted(
            key[len(prefix) + 1 :] for key in cv_results if key.startswith(f"{prefix}_")
        )
        raise ValueError(
            f"metric column not found in estimator.cv_results_: {column}. "
            f"Available metrics: {available}"
        )
    return column


def _std_metric_column(estimator, metric=None):
    metric = metric or getattr(estimator, "refit_metric", "score")
    column = f"std_test_{metric}"
    return column if column in getattr(estimator, "cv_results_", {}) else None


def _split_metric_columns(estimator, metric=None):
    metric = metric or getattr(estimator, "refit_metric", "score")
    suffix = f"_test_{metric}"
    columns = []
    for column in getattr(estimator, "cv_results_", {}):
        if column.startswith("split") and column.endswith(suffix):
            fold = column[5 : -len(suffix)]
            if fold.isdecimal():
                columns.append((int(fold), column))
    return [column for _, column in sorted(columns)]


def _cv_fold_score_matrix(results, columns, indices):
    return np.array(
        [[results[column][i] for column in columns] for i in indices], dtype=np.longdouble
    )


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


def _validate_top_k(top_k):
    if top_k is None:
        return

    if isinstance(top_k, bool) or not isinstance(top_k, (int, np.integer)) or top_k < 1:
        raise ValueError("top_k must be a positive integer")


def _candidate_row_positions(candidate_indices, n_rows):
    indices = _as_list(candidate_indices)
    if not indices or any(
        isinstance(i, (bool, np.bool_))
        or not isinstance(i, (int, np.integer))
        or not 0 <= i < n_rows
        for i in indices
    ):
        raise ValueError("candidate_indices must contain valid row positions in cv_results_")
    return indices


def _candidate_frame(estimator, metric=None, top_k=None, sort=True, *, extra_columns=()):
    _validate_top_k(top_k)

    _require_fitted(estimator, "cv_results_")
    score_column = _metric_column(estimator, metric=metric)
    fields = (score_column, "params", "n_features", *extra_columns)
    data = estimator.cv_results_
    values = {name: data[name] for name in fields if name in data}
    scores = values[score_column]
    if isinstance(scores, (list, tuple)):
        # Pandas can narrow extended-precision floats when inferring the dtype.
        values[score_column] = np.asarray(scores)
    frame = pd.DataFrame(values)
    scores = pd.to_numeric(frame[score_column], errors="coerce")
    frame[score_column] = scores
    candidates = frame.loc[np.isfinite(scores)]

    if candidates.empty:
        raise ValueError("No candidate scores were found in estimator.cv_results_")

    if sort:
        candidates = candidates.sort_values(
            score_column, ascending=getattr(estimator, "criteria", "max") == "min", kind="stable"
        )

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


def _plottable_values(values, *, preserve_range=False):
    normalized = []
    for value in values:
        if isinstance(value, (list, tuple, np.ndarray)):
            array = np.asarray(value, dtype=np.longdouble if preserve_range else float)
            finite = array[np.isfinite(array)]
            if not finite.size:
                normalized.append(np.nan)
            else:
                normalized.append(score_mean(finite) if preserve_range else float(finite.mean()))
        elif isinstance(value, (float, np.floating)) and not np.isfinite(value):
            normalized.append(np.nan)
        else:
            normalized.append(value)

    return normalized


def _scaled_plot_scores(scores, *, headroom=4, variance=False):
    largest = np.max(np.abs(scores[np.isfinite(scores)]), initial=0)
    limits = np.finfo(float)
    lowest = np.sqrt(limits.tiny) if variance else limits.tiny
    highest = np.sqrt(limits.max) if variance else limits.max
    exponent = 0
    with np.errstate(under="ignore"):
        # Leave room for axis spans, margins and a possible rolling sum.
        # Narrow only after scaling so finite extremes remain visible.
        if largest > highest / headroom or 0 < largest < lowest:
            exponent = int(np.frexp(largest)[1])
            scores = np.ldexp(scores, -exponent)
        return scores.astype(float), exponent


def _scaled_axis_label(label, exponent):
    return f"{label} / $2^{{{exponent}}}$" if exponent else label


_COLORS = ("#185b75", "#a94c17", "#416d51", "#685884")
_DASHES = ("-", "--", ":", "-.")
_MARKERS = ("o", "s", "^", "D")
_LABELS = {
    "fitness_best_observed": "Best observed so far",
    "fitness_best": "Best archived",
    "fitness": "Population mean",
    "fitness_max": "Generation maximum",
    "fitness_min": "Generation minimum",
    "fitness_std": "Population SD",
    "finite_fitness_count": "Finite population scores",
    "unique_individual_ratio": "Unique individuals",
    "genotype_diversity": "Genotype diversity",
    "fit_time": "Fit time (s)",
    "score_time": "Scoring time (s)",
}


def _field_label(field):
    return _LABELS.get(field, field.replace("_", " ").capitalize())


def _metric_name(estimator, metric=None):
    name = metric or getattr(estimator, "refit_metric", "score")
    scoring = getattr(estimator, "scoring", None)
    if name == "score" and isinstance(scoring, str):
        name = scoring
    return name.replace("_", " ")


def _style_axis(ax, grid="y"):
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_alpha(0.35)
    ax.tick_params(length=3, labelsize=10)
    ax.title.set(fontsize=14, fontweight="bold")
    if grid:
        ax.grid(axis=grid, alpha=0.18, linewidth=0.7)


def _legend(ax, **kwargs):
    # An unmanaged caller-owned figure may have no margin below its axes.
    position = (
        dict(loc="upper left", bbox_to_anchor=(0, -0.23), borderaxespad=0)
        if ax.figure.get_layout_engine()
        else dict(loc="best")
    )
    ax.legend(frameon=False, ncol=2, fontsize=9, **position, **kwargs)


def _colors(palette, fields):
    if palette is not None:
        return sns.color_palette(palette, n_colors=len(fields))
    return [_COLORS[i % len(_COLORS)] for i in range(len(fields))]


def _line_style(field, position, color, n):
    i = position
    return dict(
        color=color,
        linestyle=_DASHES[i % len(_DASHES)],
        marker=_MARKERS[(i + i // len(_DASHES)) % len(_MARKERS)],
        markersize=4,
        markevery=max(1, n // 12),
        linewidth=2.6 if field in ("fitness_best", "fitness_best_observed") else 1.7,
    )


def _plot_single_series(
    ax, series, kind="line", label=None, alpha=0.9, color=None, index=0, count=1
):
    name = _field_label(label)
    if kind == "bar":
        positions, groups, sizes = np.unique(series.index, return_inverse=True, return_counts=True)
        spacing = np.diff(positions)
        # The available width belongs to the entire x group. Allocate slots
        # in row order, and keep metrics for each generation together.
        rank = pd.Series(groups).groupby(groups, sort=False).cumcount().to_numpy()
        group_sizes = sizes[groups]
        slots = group_sizes * count
        slot_width = 0.8 * (spacing.min() if spacing.size else 1) / slots
        centers = series.index + (rank * count + index - (slots - 1) / 2) * slot_width
        return ax.bar(
            centers,
            series.values,
            width=np.where(group_sizes > 1, 0.9 * slot_width, slot_width),
            label=name,
            alpha=alpha,
            color=color,
            hatch=("", "//", "..", "xx")[index % 4],
        )
    else:
        if kind == "area":
            ax.fill_between(series.index, series.values, alpha=0.16, color=color)
        ax.plot(
            series.index,
            series.values,
            label=name,
            alpha=alpha,
            drawstyle="steps-post" if kind == "step" else "default",
            **_line_style(label, index, color, len(series)),
        )


def _generation_data(logbook, metrics):
    if not logbook:
        raise ValueError("logbook has no generations to plot")
    if not metrics:
        raise ValueError("metrics must contain at least one field")
    # Newer generations can add metrics that older checkpoints did not record.
    missing = [metric for metric in metrics if not any(metric in row for row in reversed(logbook))]
    if missing:
        raise ValueError(f"metrics missing from logbook: {missing}")
    fitness_only = all(metric in _FITNESS_METRICS for metric in metrics)
    values = [
        _plottable_values((row.get(metric, np.nan) for row in logbook), preserve_range=fitness_only)
        for metric in metrics
    ]
    exponent = 0
    if fitness_only:
        values, exponent = _scaled_plot_scores(np.asarray(values, dtype=np.longdouble))
    return [row["gen"] for row in logbook], values, exponent


def _generation_ylabel(metrics, estimator, exponent):
    if all(m in _FITNESS_METRICS for m in metrics):
        return _scaled_axis_label(f"Fitness ({_metric_name(estimator)})", exponent)
    if all(m in _DIVERSITY_FIELDS for m in metrics):
        return "Population diversity (ratio)"
    return _field_label(metrics[0]) if len(metrics) == 1 else "Recorded value"


def _generation_axis(metrics, ax, estimator=None, *, exponent=0):
    if ax is None:
        _, ax = plt.subplots(figsize=(7.6, 4.5), layout="constrained")
    ylabel = _generation_ylabel(metrics, estimator, exponent)
    ax.set(xlabel="Generation", ylabel=ylabel, title="Training progress")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=1))
    _style_axis(ax)
    return ax


def _generation_lines(ax, metrics, generations, values):
    lines = []
    for i, (field, ys, color) in enumerate(zip(metrics, values, _colors(None, metrics))):
        (line,) = ax.plot(
            generations,
            ys,
            label=_field_label(field),
            **_line_style(field, i, color, len(generations)),
        )
        lines.append(line)
    _autoscale_generations(ax, generations)
    _legend(ax)
    return lines


def _autoscale_generations(ax, generations):
    # Matplotlib otherwise drops generation positions where every series has a gap.
    ax.update_datalim([(min(generations), 0), (max(generations), 0)], updatey=False)
    ax.autoscale_view()


def plot_generation_metrics(logbook, metrics=("fitness_best", "fitness"), *, ax=None):
    """Plot recorded generation metrics, leaving gaps where values are unavailable.

    Use fitness_best_observed to include discarded candidates.
    When all selected fields are fitness metrics, extreme scores share a
    labeled scale.
    """
    metrics = _as_list(metrics)
    generations, values, exponent = _generation_data(logbook, metrics)
    ax = _generation_axis(metrics, ax, exponent=exponent)
    _generation_lines(ax, metrics, generations, values)
    return ax


class LivePlot(BaseCallback):
    """Update the same chart at each generation boundary and after final refinement.

    Pass this callback to fit to enable it. The default series are the archived
    best fitness and the current population mean. Pass metrics=("fitness_best_observed",
    "fitness") for the best finite candidate score and the population mean.
    Missing values in individual generations appear as gaps. Use save_path to
    write the current figure on a non-interactive backend.
    Extreme fitness scores use a shared, labeled scale that can change as
    new generations arrive.
    """

    def __init__(self, metrics=("fitness_best", "fitness"), *, ax=None, save_path=None):
        self.metrics = metrics
        self.ax = ax
        self.save_path = save_path
        self._display_handle = None
        self._lines = []
        self._axis_exponent = 0

    def on_start(self, estimator=None):
        self._display_handle = None
        for line in self._lines:
            line.remove()
        self._lines = []
        self._axis_exponent = 0

    def on_step(self, record=None, logbook=None, estimator=None):
        metrics = _as_list(self.metrics)
        generations, values, exponent = _generation_data(logbook, metrics)
        if not self._lines:
            self.ax = _generation_axis(metrics, self.ax, estimator, exponent=exponent)
            self._lines = _generation_lines(self.ax, metrics, generations, values)
            canvas = self.ax.figure.canvas
            if canvas.required_interactive_framework and canvas.manager is not None:
                canvas.manager.show()
        else:
            if exponent != getattr(self, "_axis_exponent", 0):
                self.ax.set_ylabel(_generation_ylabel(metrics, estimator, exponent))
            for line, ys in zip(self._lines, values):
                line.set_data(generations, ys)
                line.set_markevery(max(1, len(generations) // 12))
            self.ax.relim()
            _autoscale_generations(self.ax, generations)
        self._axis_exponent = exponent
        canvas = self.ax.figure.canvas
        if "inline" in plt.get_backend().lower():
            # Inline figures are static. Jupyter can replace a displayed snapshot.
            from IPython.display import display

            if self._display_handle is None:
                self._display_handle = display(self.ax.figure, display_id=True)
            else:
                self._display_handle.update(self.ax.figure)
        else:
            canvas.draw_idle()
            canvas.flush_events()
        if self.save_path is not None:
            self.ax.figure.savefig(self.save_path)
        return False

    def on_end(self, logbook=None, estimator=None):
        # Local refinement can replace the final row after the last on_step.
        if logbook and (not self._lines or logbook[-1].get("local_refinements")):
            self.on_step(logbook=logbook, estimator=estimator)


def plot_fitness_evolution(
    estimator,
    metric="fitness_best",
    metrics=None,
    *,
    x="generation",
    kind="line",
    window=None,
    ax=None,
    title=None,
    palette=None,
):
    """
    Plot one or more evolution metrics stored in ``estimator.history``.

    Finite points in a partial generation are marked with hollow diamonds.
    Extreme fitness scores share a labeled scale applied before smoothing.

    Parameters
    ----------
    estimator: estimator object
        A fitted estimator from :class:`~sklearn_genetic.GASearchCV` or
        :class:`~sklearn_genetic.GAFeatureSelectionCV`.
    metric: str, default="fitness_best"
        Backward-compatible name for a single metric to plot.
    metrics: list[str] | tuple[str] | None, default=None
        Optional collection of history fields to plot together.
    x: {"generation", "cv_calls", "fold_fits"}, default="generation"
        Plot against generations, cumulative candidate CV calls, or search
        fold fits. The fold-fit axis multiplies recorded calls by the fitted
        search split count. Missing call counts leave gaps on either cost
        axis. Bar charts omit those generations.
    kind: {"line", "bar", "area", "step"}, default="line"
        Plot style. Bars for repeated CV counts sit side by side around
        their shared count, in generation order.
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
    if x not in _FITNESS_AXES:
        raise ValueError(f"x must be one of {sorted(_FITNESS_AXES)}")
    _require_fitted(estimator, "history", "plot_fitness_evolution")

    if metrics is None:
        metrics = [metric]
    else:
        metrics = _as_list(metrics)

    if metrics == [metric] and metric not in _FITNESS_METRICS:
        raise ValueError(f"metric must be one of {_FITNESS_METRICS}, but got {metric} instead")

    missing = [name for name in metrics if name not in estimator.history]
    if missing:
        raise ValueError(f"metrics not found in estimator.history: {missing}")

    axis_field, axis_label = _FITNESS_AXES[x]
    cost_axis = x != "generation"
    if cost_axis:
        coordinates = estimator.history.get("cross_validate_calls", [])
        if not pd.notna(coordinates).any():
            raise ValueError(
                "No cumulative candidate CV calls were recorded. Use x='generation' for this history."
            )
        if x == "fold_fits":
            n_splits = getattr(estimator, "n_splits_", None)
            if (
                not isinstance(n_splits, (int, np.integer))
                or isinstance(n_splits, (bool, np.bool_))
                or n_splits < 1
            ):
                raise ValueError(
                    "The fold-fit axis needs a positive n_splits_ from the fitted search. "
                    "Use x='cv_calls' for this history."
                )
            coordinates = [
                value * int(n_splits) if pd.notna(value) else value for value in coordinates
            ]
    else:
        coordinates = estimator.history[axis_field]
    frame = pd.DataFrame({axis_field: coordinates})
    fitness_only = all(name in _FITNESS_METRICS for name in metrics)
    for name in metrics:
        values = _plottable_values(estimator.history[name], preserve_range=fitness_only)
        frame[name] = np.asarray(values, dtype=np.longdouble) if fitness_only else values
    if cost_axis:
        frame = frame.dropna(subset=[axis_field]).copy()
    axis_exponent = 0
    if fitness_only:
        headroom = max(4, int(window)) if isinstance(window, (int, np.integer)) else 4
        scaled, axis_exponent = _scaled_plot_scores(frame[metrics].to_numpy().T, headroom=headroom)
        for name, values in zip(metrics, scaled, strict=True):
            frame[name] = values
    if window is not None:
        values = frame.drop(columns=axis_field)
        frame[values.columns] = (
            values.rolling(window=window, min_periods=1).mean().where(values.notna())
        )
    if cost_axis and kind != "bar":
        start, end = frame.index[0], frame.index[-1]
        # Restore interior gaps after smoothing the visible observations.
        if len(frame) != end - start + 1:
            frame = frame.reindex(range(start, end + 1))

    partial = estimator.history.get("partial")
    if partial is not None:
        partial = (
            pd.Series(partial, dtype="boolean")
            .reindex(frame.index, fill_value=False)
            .fillna(False)
            .to_numpy(dtype=bool)
        )
    frame = frame.set_index(axis_field)

    if ax is None:
        _, ax = plt.subplots(figsize=(7.6, 4.5), layout="constrained")

    colors = _colors(palette, metrics)
    marked_partial = False

    for i, (color, name) in enumerate(zip(colors, metrics)):
        series = frame[name]
        bars = _plot_single_series(
            ax, series, kind=kind, label=name, alpha=0.9, color=color, index=i, count=len(metrics)
        )
        if partial is not None:
            visible = partial & series.notna().to_numpy()
            if visible.any():
                positions = series.index[visible]
                if kind == "bar":
                    selected_bars = (bars[index] for index in np.flatnonzero(visible))
                    positions = [bar.get_x() + bar.get_width() / 2 for bar in selected_bars]
                ax.scatter(
                    positions,
                    series.to_numpy()[visible],
                    marker="D",
                    s=55,
                    facecolors="none",
                    edgecolors=color,
                    linewidths=1.4,
                    zorder=4,
                    label=None if marked_partial else "Partial generation",
                )
                marked_partial = True
    if cost_axis:
        # Matplotlib's line limits otherwise omit costs whose scores are missing.
        ax.update_datalim([(frame.index.min(), 0), (frame.index.max(), 0)], updatey=False)
        ax.autoscale_view(scaley=False)

    title = title or (_field_label(metrics[0]) if len(metrics) == 1 else "Fitness evolution")
    if window is not None:
        title = f"{title} (rolling window={window})"

    ax.set_title(title)
    _style_axis(ax)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=1))
    ylabel = f"Fitness ({_metric_name(estimator)})"
    if len(metrics) == 1 and metrics[0] not in ("fitness_best_observed", "fitness_best"):
        ylabel = f"{_field_label(metrics[0])} ({_metric_name(estimator)})"
    ax.set(xlabel=axis_label, ylabel=_scaled_axis_label(ylabel, axis_exponent))
    if len(metrics) > 1 or marked_partial:
        _legend(ax)
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
        raise ValueError(
            f"fields not found in {source}: {missing}. Available fields include: {list(frame.columns)}"
        )

    plotted = frame.loc[:, fields].copy()
    if rolling is not None:
        plotted = plotted.rolling(window=rolling, min_periods=1).mean()

    x_values = frame["gen"] if "gen" in frame.columns else frame.index
    x_label = "Generation" if "gen" in frame.columns else "record index"

    if subplots is None:
        subplots = len(fields) > 3

    colors = _colors(palette, fields)

    if subplots:
        fig, axes = plt.subplots(
            len(fields),
            1,
            sharex=True,
            figsize=figsize or (7.6, max(3, 1.65 * len(fields))),
            layout="constrained",
        )
        axes = np.atleast_1d(axes)
        for i, (axis, color, field) in enumerate(zip(axes, colors, fields)):
            series = pd.Series(_plottable_values(plotted[field]), index=x_values)
            _plot_single_series(axis, series, kind=kind, label=field, alpha=0.9, color=color)
            axis.set_title(_field_label(field), loc="left", fontsize=11)
            _style_axis(axis)
            axis.xaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=1))
        axes[-1].set_xlabel(x_label)
        if title:
            fig.suptitle(title)
        else:
            fig.suptitle(f"{source.capitalize()} overview")
        return axes

    if figsize is None:
        figsize = (7.6, 4.5)
    _, ax = plt.subplots(figsize=figsize, layout="constrained")
    for i, (color, field) in enumerate(zip(colors, fields)):
        series = pd.Series(_plottable_values(plotted[field]), index=x_values)
        _plot_single_series(
            ax, series, kind=kind, label=field, alpha=0.9, color=color, index=i, count=len(fields)
        )

    _style_axis(ax)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=1))
    ax.set_title(title or f"{source.capitalize()} fields")
    ax.set(xlabel=x_label, ylabel=_field_label(fields[0]) if len(fields) == 1 else "Value")
    if len(fields) > 1:
        _legend(ax)
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
        Fields from `cv_results_` or the candidate logbook. Use `param_*` to
        select parameter values whose names also appear in the logbook.
        Defaults to numeric parameters and the optimization score.
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

    records = candidate_records(estimator)
    df = pd.DataFrame(records)
    columns = _as_list(features)
    use_defaults = not columns
    aliases = columns + _as_list(hue)
    if use_defaults:
        columns = list(estimator.space.parameters)
        if any(candidate_parameters(row) is not row for row in records):
            columns = [f"param_{name}" for name in columns]
            aliases.extend(columns)
        columns.append("score")
    results = getattr(estimator, "cv_results_", {})
    for name in dict.fromkeys(aliases):
        if name in results:
            df[name] = results[name]
    if use_defaults and hue in df.columns and hue not in columns:
        columns.append(hue)
    missing = [name for name in columns if name not in df.columns]
    if missing:
        raise ValueError(f"features not found in estimator.logbook or cv_results_: {missing}")
    stats = df[list(dict.fromkeys(columns))].copy()

    if kind == "heatmap":
        heatmap_frame = _select_numeric_columns(stats)
        if heatmap_frame.empty:
            raise ValueError("No numeric columns available to plot the heatmap")
        corr = heatmap_frame.corr(numeric_only=True)
        _, ax = plt.subplots(
            figsize=(max(6, 1.2 * len(corr.columns)), max(5, 0.8 * len(corr))), layout="constrained"
        )
        sns.heatmap(
            corr,
            annot=True,
            fmt=".2f",
            cmap="RdBu_r",
            center=0,
            ax=ax,
            vmin=-1,
            vmax=1,
            cbar_kws={"label": "Pearson correlation"},
            linewidths=0.6,
        )
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
        "plot_kws": {"s": s, "alpha": 0.55, "edgecolor": "white", "linewidth": 0.35},
        "diag_kws": {"alpha": 0.35, "bins": 15},
    }
    if hue and hue in df.columns:
        plot_kwargs["data"] = df[numeric_stats.columns.tolist() + [hue]].copy()
        plot_kwargs["hue"] = hue
        levels = pd.Categorical(df[hue]).categories
        plot_kwargs["markers"] = [_MARKERS[i % len(_MARKERS)] for i in range(len(levels))]

    grid = sns.pairplot(**plot_kwargs)
    for axis in grid.axes.flat:
        _style_axis(axis, grid=None)
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

    if len(parameter_columns) == 1:
        if ax is None:
            _, ax = plt.subplots(figsize=(7.6, 4), layout="constrained")
        axes = np.asarray([ax])
    else:
        _, axes = plt.subplots(
            len(parameter_columns),
            1,
            sharex=True,
            figsize=(7.6, max(3, 2.2 * len(parameter_columns))),
            layout="constrained",
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
        _style_axis(axis)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=1))

    axes[-1].set_xlabel("evaluation index")
    axes[0].set_title("Parameter exploration over time")
    if mappable is not None:
        axes[0].figure.colorbar(
            mappable, ax=axes.tolist(), label=f"Mean CV score ({_metric_name(estimator, metric)})"
        )

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

    Extreme finite scores share a scale shown on the score axis.

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

    candidates, score_column = _candidate_frame(
        estimator,
        metric=metric,
        top_k=top_k,
        sort=sort,
    )
    labels = _candidate_labels(
        candidates,
        label_params=label_params,
        max_label_params=max_label_params,
        float_precision=float_precision,
    )

    if ax is None:
        _, ax = plt.subplots(
            figsize=(8.4, max(3.4, 1.4 + 0.40 * len(candidates))), layout="constrained"
        )

    bar_color = color or _COLORS[0]
    positions = np.arange(len(candidates))[::-1]
    scores, exponent = _scaled_plot_scores(candidates[score_column].to_numpy())
    ax.barh(positions, scores, color=bar_color, alpha=0.9)
    ax.set_yticks(positions, labels=labels)
    ax.set_xlabel(
        _scaled_axis_label(f"Mean CV score ({_metric_name(estimator, metric)})", exponent)
    )
    _style_axis(ax, grid="x")
    ax.set_ylabel("candidate")
    ax.set_title(title or f"Top {len(candidates)} candidate scores")
    return ax


def plot_feature_selection(
    estimator,
    feature_names=None,
    *,
    ax=None,
    selected_color=None,
    rejected_color="0.42",
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
        Marker color for selected features.
    rejected_color: matplotlib color, default="0.42"
        Marker color for rejected features.

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
            raise ValueError(
                "feature_names must have the same length as estimator.best_features_: "
                f"expected {mask.size}, got {len(feature_names)}"
            )

    if ax is None:
        _, ax = plt.subplots(figsize=(7.6, max(3.5, 1.3 + 0.30 * mask.size)), layout="constrained")

    selected_color = selected_color or _COLORS[0]
    positions = np.arange(mask.size)[::-1]
    ax.set_yticks(positions, labels=feature_names)
    for selected, marker in ((False, "x"), (True, "o")):
        subset = mask == selected
        ax.scatter(
            mask[subset].astype(int),
            positions[subset],
            marker=marker,
            color=selected_color if selected else rejected_color,
            s=50,
            zorder=3,
        )
    ax.set_xlim(-0.15, 1.15)
    ax.set_xticks([0, 1], labels=["rejected", "selected"])
    _style_axis(ax, grid="x")
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
        _, ax = plt.subplots(figsize=(7.6, 4.5), layout="constrained")

    colors = _colors(palette, fields)
    for i, (color, field) in enumerate(zip(colors, fields)):
        series = pd.Series(_plottable_values(frame[field]), index=x_values)
        _plot_single_series(ax, series, kind="line", label=field, color=color, index=i)

    ax.set_title(title)
    _style_axis(ax)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=1))
    ax.set_xlabel("Generation")
    ax.set_ylabel(f"Fitness ({_metric_name(estimator)})")
    if len(fields) > 1:
        _legend(ax)
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
        _, ax = plt.subplots(figsize=(7.6, 4.5), layout="constrained")

    colors = _colors(palette, fields)
    for i, (color, field) in enumerate(zip(colors, fields)):
        series = pd.Series(_plottable_values(frame[field]), index=x_values)
        _plot_single_series(ax, series, kind="line", label=field, color=color, index=i)

    ax.set_title(title)
    _style_axis(ax)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=1))
    ax.set_xlabel("Generation")
    ax.set_ylabel("diversity ratio")
    ax.set_ylim(bottom=0)
    handles, labels = ax.get_legend_handles_labels()

    if stagnation and _STAGNATION_FIELD in getattr(estimator, "history", {}):
        stagnation_frame = _history_frame(estimator, fields=[_STAGNATION_FIELD])
        twin = ax.twinx()
        twin.plot(
            x_values,
            stagnation_frame[_STAGNATION_FIELD],
            color="0.25",
            linestyle=":",
            marker="x",
            label="Stagnation (right axis)",
        )
        twin.set_ylabel("stagnant generations")
        twin.yaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=1))
        twin.spines["top"].set_visible(False)
        more_handles, more_labels = twin.get_legend_handles_labels()
        handles += more_handles
        labels += more_labels
    _legend(ax, handles=handles, labels=labels)

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
        _, ax = plt.subplots(figsize=(7.6, max(3, 1.3 + 0.48 * len(fields))), layout="constrained")

    colors = _colors(palette, fields)
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

    ax.set_yticks(range(len(fields)), labels=[_field_label(f) for f in fields])
    ax.set_xlabel("Generation")
    ax.set_title(title)
    _style_axis(ax)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=1))
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
        _, ax = plt.subplots(figsize=(7.6, 5), layout="constrained")

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

    ax.figure.colorbar(mappable, ax=ax, label=f"Mean CV score ({_metric_name(estimator, metric)})")
    _style_axis(ax, grid=None)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"Score landscape: {x} vs {y}")
    return ax


def _cv_row_flag(results, name, n_rows, rows=slice(None)):
    flag = results.get(name)
    if flag is None:
        return np.zeros(n_rows, dtype=bool)
    return np.asarray(flag, dtype=bool)[rows]


def plot_cv_scores(
    estimator,
    top_k=5,
    *,
    metric=None,
    candidate_indices=None,
    kind="box",
    ax=None,
    palette=None,
    label_params=None,
    max_label_params=2,
    float_precision=3,
):
    """
    Plot finite fold scores for ranked or explicitly selected candidates.

    Extreme finite scores share a scale shown on the score axis.

    Parameters
    ----------
    estimator: fitted GASearchCV or GAFeatureSelectionCV
        Fitted estimator with split-level ``cv_results_``.
    top_k: int or None, default=5
        Number of ranked candidates to include. None uses all ranked rows.
        Ignored when candidate_indices is supplied.
    metric: str | None, default=None
        Metric to plot. Defaults to the refit metric.
    candidate_indices: int or iterable of int, optional
        Zero-based row positions in `cv_results_`, in display order. This
        can include candidates with a non-finite mean. Incomplete rows
        show the number of finite folds. Candidates flagged as
        `all_fits_failed` display "All fits failed". Skipped feature masks
        display "Skipped before CV".
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

    _require_fitted(estimator, "cv_results_", "plot_cv_scores")
    split_columns = _split_metric_columns(estimator, metric=metric)
    if not split_columns:
        raise ValueError("No split-level test scores were found in estimator.cv_results_")

    diagnostics = ("all_fits_failed", "skipped_invalid_mask")
    results = estimator.cv_results_
    if candidate_indices is None:
        candidates, _ = _candidate_frame(
            estimator, metric=metric, top_k=top_k, extra_columns=diagnostics
        )
    else:
        indices = _candidate_row_positions(candidate_indices, len(results[split_columns[0]]))
        columns = (*diagnostics, "params", "n_features")
        candidates = pd.DataFrame(
            {name: [results[name][i] for i in indices] for name in columns if name in results},
            index=indices,
        )
    scores = _cv_fold_score_matrix(results, split_columns, candidates.index)
    skipped_rows = _cv_row_flag(candidates, "skipped_invalid_mask", len(candidates))
    failed_rows = _cv_row_flag(candidates, "all_fits_failed", len(candidates))
    scores[skipped_rows | failed_rows] = np.nan
    finite = np.isfinite(scores)
    if not finite.any():
        raise ValueError(
            "No finite fold scores for the selected candidates. "
            "Use plot_cv_fold_scores to inspect missing values."
        )
    # The violin density estimator uses squared deviations from the mean.
    scores, exponent = _scaled_plot_scores(scores, variance=kind == "violin")

    labels = _candidate_labels(
        candidates,
        ranked=candidate_indices is None,
        label_params=label_params,
        max_label_params=max_label_params,
        float_precision=float_precision,
    )
    if candidate_indices is None:
        labels = [label.replace(" ", "\n", 1) for label in labels]
    elif "params" in candidates:
        labels = [
            f"row {index}" if label == str(index) else f"row {index}\n{label}"
            for index, label in zip(candidates.index, labels, strict=True)
        ]
    n_splits = len(split_columns)
    tick_labels = [label.replace(", ", "\n") for label in labels]
    for index, count in enumerate(finite.sum(axis=1)):
        if skipped_rows[index]:
            tick_labels[index] += "\nSkipped before CV"
        elif failed_rows[index]:
            tick_labels[index] += "\nAll fits failed"
        elif count < n_splits:
            tick_labels[index] += f"\n{count}/{n_splits} finite folds"

    if ax is None:
        _, ax = plt.subplots(figsize=(max(7.6, 1.55 * len(labels)), 4.6), layout="constrained")

    colors = (
        sns.color_palette(palette, n_colors=len(labels)) if palette else ["#9dbfca"] * len(labels)
    )
    positions = np.arange(len(labels))
    if kind != "strip":
        long_frame = pd.DataFrame(
            {
                "candidate": np.repeat(positions, n_splits),
                "score": np.where(finite, scores, np.nan).ravel(),
            }
        )
        options = dict(
            data=long_frame,
            x="candidate",
            y="score",
            hue="candidate",
            order=positions,
            hue_order=positions,
            palette=colors,
            legend=False,
            ax=ax,
        )
        if kind == "violin":
            sns.violinplot(cut=0, **options)
        else:
            sns.boxplot(**options)
    if kind in ("box", "strip"):
        offsets = np.linspace(-0.16, 0.16, n_splits) if n_splits > 1 else np.zeros(1)
        for index, available in enumerate(finite):
            if not available.any():
                continue
            ax.scatter(
                index + offsets[available],
                scores[index, available],
                s=28 if kind == "strip" else 16,
                zorder=3,
                color=colors[index] if kind == "strip" and palette else _COLORS[0],
            )

    ax.set_xlim(-0.5, len(labels) - 0.5)
    title = (
        f"Cross-validation scores for top {len(labels)} candidates"
        if candidate_indices is None
        else "Cross-validation scores for selected candidates"
    )
    ax.set_title(title)
    ax.set_xlabel("candidate")
    ax.set_ylabel(_scaled_axis_label(f"Fold score ({_metric_name(estimator, metric)})", exponent))
    _style_axis(ax)
    ax.set_xticks(positions, tick_labels)
    return ax


def plot_cv_fold_scores(
    estimator, top_k=8, *, metric=None, candidate_indices=None, ax=None, cmap="viridis"
):
    """Show recorded test scores with candidates as rows and CV folds as columns.

    Extreme finite scores share a scale shown on the colorbar.

    Parameters
    ----------
    estimator : fitted GASearchCV or GAFeatureSelectionCV
        Fitted search with split-level scores in `cv_results_`.
    top_k : int or None, default=8
        Maximum number of candidates to show in test-rank order. Ties keep
        evaluation order. None uses all rows. Ignored with candidate_indices.
    metric : str or None, default=None
        Test metric to display. Defaults to the search objective.
    candidate_indices : int or iterable of int, optional
        Zero-based positions in `cv_results_`, shown in the requested order.
        For example, pass `estimator.best_index_` to inspect the selected row.
    ax : matplotlib.axes.Axes or None, default=None
        Axis to draw on. Creates a new figure when omitted.
    cmap : str, default="viridis"
        Matplotlib colormap for finite scores. Crosses mark non-finite scores.
        Outlined squares mark rows flagged as `all_fits_failed`. Outlined
        diamonds mark feature masks skipped before CV.

    Returns
    -------
    matplotlib.axes.Axes
        The axis used for the heatmap.
    """
    _require_fitted(estimator, "cv_results_", "plot_cv_fold_scores")
    columns = _split_metric_columns(estimator, metric=metric)
    if not columns:
        _metric_column(estimator, metric=metric)
        raise ValueError("No split-level test scores were found in estimator.cv_results_")
    results = estimator.cv_results_
    n_rows = len(results[columns[0]])
    if candidate_indices is None:
        _validate_top_k(top_k)
        rank_column = _metric_column(estimator, metric=metric, prefix="rank_test")
        ranks = results[rank_column]
        indices = _short_cv_positions(ranks, top_k) if type(ranks) is np.ndarray else None
        if indices is None:
            indices = np.argsort(ranks, kind="stable")[:top_k]
    else:
        indices = _candidate_row_positions(candidate_indices, n_rows)
    if len(indices) == 0:
        raise ValueError("No candidate scores were found in estimator.cv_results_")

    scores = _cv_fold_score_matrix(results, columns, indices)
    skipped_rows = _cv_row_flag(results, "skipped_invalid_mask", len(indices), indices)
    failed_rows = _cv_row_flag(results, "all_fits_failed", len(indices), indices) & ~skipped_rows
    unavailable_rows = skipped_rows | failed_rows
    scores[unavailable_rows] = np.nan
    scaled, exponent = _scaled_plot_scores(scores)
    scores = np.ma.masked_invalid(scaled)
    if ax is None:
        _, ax = plt.subplots(
            figsize=(max(6.4, min(12, len(columns) * 0.5)), max(3.0, min(8, len(indices) * 0.5))),
            layout="constrained",
        )
    colors = plt.get_cmap(cmap).with_extremes(bad="#e6e8e9")
    image = ax.imshow(scores, cmap=colors, aspect="auto", interpolation="nearest", origin="upper")
    if scores.count():
        colorbar = ax.figure.colorbar(image, ax=ax)
        if exponent:
            colorbar.set_label(
                _scaled_axis_label(f"Fold score ({_metric_name(estimator, metric)})", exponent)
            )
    missing_rows, missing_folds = np.nonzero(
        np.ma.getmaskarray(scores) & ~unavailable_rows[:, None]
    )
    if missing_rows.size:
        ax.scatter(
            missing_folds, missing_rows, marker="x", color="#384047", label="Non-finite score"
        )
    for row_mask, marker, color, label in (
        (failed_rows, "s", _COLORS[1], "All fits failed"),
        (skipped_rows, "D", _COLORS[2], "Skipped before CV"),
    ):
        if row_mask.any():
            rows, folds = np.nonzero(np.broadcast_to(row_mask[:, None], scores.shape))
            ax.scatter(folds, rows, marker=marker, facecolors="none", edgecolors=color, label=label)
    if missing_rows.size or unavailable_rows.any():
        ax.legend(loc="upper left", bbox_to_anchor=(0, -0.22), frameon=False)
    folds = np.linspace(0, len(columns) - 1, min(len(columns), 12), dtype=int)
    ax.set_xticks(folds, [columns[i].split("_", 1)[0][5:] for i in folds])
    ax.set_yticks(np.arange(len(indices)), indices)
    ax.set_xlabel("CV fold")
    ax.set_ylabel("Candidate row")
    ax.set_title(f"Cross-validation scores by fold ({_metric_name(estimator, metric)})")
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

    Extreme finite means and standard deviations share a scale shown on the score axis.

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

    std_column = _std_metric_column(estimator, metric=metric)
    candidates, score_column = _candidate_frame(estimator, metric=metric, top_k=top_k)
    labels = _candidate_labels(
        candidates,
        ranked=True,
        label_params=label_params,
        max_label_params=max_label_params,
        float_precision=float_precision,
    )
    scores = candidates[score_column].to_numpy()
    errors = None
    if std_column is None:
        scores, exponent = _scaled_plot_scores(scores)
    else:
        deviations = estimator.cv_results_[std_column]
        errors = np.array([deviations[i] for i in candidates.index], dtype=np.longdouble)
        (scores, errors), exponent = _scaled_plot_scores(np.stack((scores, errors)))

    if ax is None:
        _, ax = plt.subplots(
            figsize=(8.4, max(3.4, 1.4 + 0.40 * len(candidates))), layout="constrained"
        )

    marker_color = color or _COLORS[0]
    y_positions = np.arange(len(candidates))[::-1]
    ax.errorbar(
        scores,
        y_positions,
        xerr=errors,
        fmt="o",
        color=marker_color,
        ecolor="0.55",
        capsize=3,
        markersize=7,
    )
    ax.set_yticks(y_positions, labels=labels)
    label = f"Mean CV score ({_metric_name(estimator, metric)})"
    if errors is not None:
        label += " ± 1 fold SD"
        if exponent:
            label = f"({label})"
    ax.set_xlabel(_scaled_axis_label(label, exponent))
    _style_axis(ax, grid="x")
    ax.set_ylabel("candidate")
    ax.set_title(title or f"Candidate ranking (top {len(candidates)})")
    ax.grid(True, axis="x", alpha=0.2)
    return ax


def plot_fit_work(estimator, *, unit="count", ax=None):
    """Plot fit counts or recorded phase times from the latest fit.

    In count mode, each search or finalist fold fit counts once. The full-data
    refit has a separate bar. Reused finalist scores appear in a hatched bar.
    Cached or restored search scores add no fresh work.
    If search or finalist validation raises, recorded fold fits in that
    phase may be shown as a lower bound. A failed refit is labeled in both
    modes. Seconds mode can show how long the failed refit ran.

    Parameters
    ----------
    estimator : GASearchCV or GAFeatureSelectionCV
        Search with ``fit_counts_`` or, in seconds mode, a recorded
        ``search_time_`` from its most recent call to ``fit``.
    unit : {"count", "seconds"}, default="count"
        Show fit and reuse counts or elapsed seconds for search, finalist
        validation, and refit. In seconds mode, phases without recorded
        durations are marked ``No timing``.
    ax : matplotlib.axes.Axes or None, default=None
        Axis to draw on. Creates a new figure when omitted.

    Returns
    -------
    matplotlib.axes.Axes
        The axis used for the fit-work plot.
    """
    if unit not in ("count", "seconds"):
        raise ValueError("unit must be 'count' or 'seconds'")
    selection = getattr(estimator, "final_selection_results_", {})
    failed_refit_time = getattr(estimator, "refit_failure_time_", None)
    refit_label = "Full-data refit (failed)" if failed_refit_time is not None else "Full-data refit"
    show_counts = unit == "count"
    if show_counts:
        _require_fitted(estimator, "fit_counts_", "plot_fit_work")
        counts = estimator.fit_counts_
        labels = [
            "Search fold fits",
            "Finalist fold fits",
            refit_label,
            "Reused finalist scores",
        ]
        values = [
            counts["search_fold_fits"],
            counts["finalist_fold_fits"],
            counts["refit_fits"],
            counts["finalist_reused_folds"],
        ]
        annotations = [f"{value:,}" for value in values]
        if getattr(estimator, "search_interrupted_", False):
            annotations[0] = f"≥{annotations[0]}"
        failed = selection.get("failed_candidate")
        if failed and failed["cv_calls"]:
            annotations[1] = f"≥{annotations[1]}"
        xlabel, title = "Count", "Work in the latest fit"
    else:
        if not hasattr(estimator, "fit_counts_"):
            _require_fitted(estimator, "search_time_", "plot_fit_work")
        durations = [
            getattr(estimator, "search_time_", None),
            selection.get("time_seconds") if selection.get("enabled") else None,
            getattr(estimator, "refit_time_", failed_refit_time),
        ]
        finalist_failed = selection.get("failed") or "failed_candidate" in selection
        finalist_label = (
            "Finalist validation (failed)" if finalist_failed else "Finalist validation"
        )
        labels = ["Search", finalist_label, refit_label]
        values = [0.0 if duration is None else duration for duration in durations]
        annotations = [
            "No timing" if duration is None else f"{duration:.3g} s" for duration in durations
        ]
        xlabel, title = "Elapsed seconds", "Elapsed time by fit phase"
    if ax is None:
        _, ax = plt.subplots(figsize=(7.2, 3.4), layout="constrained")

    positions = range(len(labels))
    bars = ax.barh(positions, values, height=0.6, color=_COLORS[0])
    if show_counts:
        bars[-1].set(facecolor="#e6eef3", edgecolor=_COLORS[0], hatch="///")
    ax.bar_label(bars, labels=annotations, padding=5)
    ax.set_yticks(positions, labels=labels)
    ax.set(xlabel=xlabel, title=title, ylim=(len(labels) - 0.5, -0.5))
    ax.set_xlim(0, (max(values) or 1) * 1.18)
    ax.xaxis.set_major_locator(MaxNLocator(integer=show_counts))
    _style_axis(ax, grid="x")
    return ax


def plot_finalist_scores(estimator, *, ax=None):
    """Compare each finalist's search and final validation scores.

    Rows follow the recorded shortlist. A star marks a selected finalist when available.
    Non-finite scores leave gaps and their row labels identify the missing
    stage. If validation was interrupted, the last row shows the interrupted
    candidate and its available search score, even if no finalist finished.
    Extreme finite scores use a shared power-of-two divisor shown on the
    score axis.

    Parameters
    ----------
    estimator : GASearchCV or GAFeatureSelectionCV
        Search with recorded finalist validation results. The star marks the
        candidate chosen for refit, even if the refit later raised an error.
    ax : matplotlib.axes.Axes or None, default=None
        Axis to draw on. Creates a new figure when omitted.

    Returns
    -------
    matplotlib.axes.Axes
        The axis used for the finalist comparison.
    """
    _require_fitted(estimator, "cv_results_", "plot_finalist_scores")
    results = getattr(estimator, "final_selection_results_", {})
    finalists = results.get("candidates") or ()
    interrupted = results.get("failed_candidate")
    if interrupted:
        index = interrupted["index"]
        column = _metric_column(estimator)
        interrupted = {
            "index": index,
            "original_score": estimator.cv_results_[column][index],
            "score": np.nan,
        }
        finalists = [*finalists, interrupted]
    if not finalists:
        raise ValueError(
            "plot_finalist_scores requires at least one completed finalist evaluation "
            "or a recorded interruption. Fit with refitting and final_selection enabled."
        )

    positions = np.arange(len(finalists))
    scores = np.asarray(
        [[item["original_score"] for item in finalists], [item["score"] for item in finalists]],
        dtype=np.longdouble,
    )
    has_search, has_final = np.isfinite(scores)
    (search, final), axis_exponent = _scaled_plot_scores(scores)
    selected = np.asarray([item["index"] == results["selected_index"] for item in finalists])
    labels = []
    for row, position in zip(finalists, positions, strict=True):
        notes = []
        if row["index"] == results["original_best_index"]:
            notes.append("search winner")
        if not has_search[position]:
            notes.append("search CV unavailable")
        if row is interrupted:
            notes.append("finalist CV interrupted")
        elif not has_final[position]:
            notes.append("finalist CV unavailable")
        label = f"Row {row['index']}"
        labels.append(label + ("\n" + ", ".join(notes) if notes else ""))

    if ax is None:
        _, ax = plt.subplots(
            figsize=(7.6, max(3.2, 2.0 + 0.52 * len(finalists))), layout="constrained"
        )

    pairs = has_search & has_final
    for y, first, last in zip(positions[pairs], search[pairs], final[pairs], strict=True):
        ax.plot([first, last], [y - 0.13, y + 0.13], color="0.68", linewidth=1.3, zorder=2)
    ax.scatter(
        search[has_search],
        positions[has_search] - 0.13,
        facecolors="white",
        edgecolors=_COLORS[0],
        label="Search CV",
        zorder=3,
    )
    remaining = has_final & ~selected
    ax.scatter(
        final[remaining],
        positions[remaining] + 0.13,
        color=_COLORS[1],
        marker="s",
        label="Finalist CV",
        zorder=3,
    )
    winner = has_final & selected
    if winner.any():
        ax.scatter(
            final[winner],
            positions[winner] + 0.13,
            color=_COLORS[1],
            marker="*",
            s=130,
            label="Selected finalist",
            zorder=4,
        )
    ax.set_yticks(positions, labels)
    ax.set_ylim(len(finalists) - 0.55, -0.45)
    xlabel = f"Mean CV score ({_metric_name(estimator)})"
    ax.set_xlabel(_scaled_axis_label(xlabel, axis_exponent))
    ax.set_title("Finalist validation", loc="left", pad=30, fontsize=14, fontweight="bold")
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0, 1.01),
        ncols=3,
        borderaxespad=0,
        frameon=False,
        fontsize=9,
    )
    _style_axis(ax, grid="x")
    return ax


def _recorded_history_fields(estimator, fields):
    history = getattr(estimator, "history", {})
    return [field for field in fields if pd.notna(_plottable_values(history.get(field, ()))).any()]


def _empty_overview_panel(ax, title, message):
    ax.set_title(title)
    ax.set_axis_off()
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)


def _plot_overview_history(estimator, ax, plot, fields, title, **kwargs):
    recorded = _recorded_history_fields(estimator, fields)
    if recorded:
        plot(estimator, fields=recorded, ax=ax, **kwargs)
    else:
        _empty_overview_panel(ax, title, "No data recorded")


def _plot_overview_candidates(estimator, ax, top_k, metric):
    _validate_top_k(top_k)
    column = _metric_column(estimator, metric=metric)
    scores = pd.to_numeric(pd.Series(estimator.cv_results_[column]), errors="coerce")
    if np.isfinite(scores).any():
        plot_candidate_rankings(
            estimator,
            top_k=top_k,
            metric=metric,
            ax=ax,
            title="Best evaluated candidates",
            label_params=[],
        )
        ax.set_ylabel("Rank / candidate row")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        return

    message = "No finite mean CV scores"
    if (
        len(scores)
        and _cv_row_flag(estimator.cv_results_, "skipped_invalid_mask", len(scores)).all()
    ):
        message = "All feature masks skipped"
    if len(scores) and _split_metric_columns(estimator, metric=metric):
        plot_cv_fold_scores(
            estimator, metric=metric, candidate_indices=range(len(scores))[:top_k], ax=ax
        )
        ax.set_title(f"{message}\n{_metric_name(estimator, metric)}")
    else:
        _empty_overview_panel(ax, "Candidate scores", message)


def plot_search_overview(
    estimator,
    *,
    metric=None,
    top_k=8,
    figsize=(9.6, 8),
    palette=None,
):
    """
    Create a compact diagnostic dashboard for a fitted genetic search.

    The overview answers the most common post-fit questions: convergence,
    diversity, optimizer-control decisions, and whether strong candidates were
    found. The last panel shows the selected feature mask when available.
    Otherwise it shows candidate scores, including recorded folds in
    evaluation order when every mean CV score is non-finite. This also
    supports feature searches that recorded CV diagnostics before raising.
    Panels with unavailable history metrics indicate that no data was recorded.

    Parameters
    ----------
    estimator: fitted GASearchCV or GAFeatureSelectionCV
        Fitted estimator to summarize.
    metric: str | None, default=None
        Metric used for candidate scores. Defaults to the refit metric.
    top_k: int, default=8
        Number of candidates to show in the score panel.
    figsize: tuple[float, float], default=(9.6, 8)
        Figure size.
    palette: str | None, default=None
        Optional seaborn palette name for line plots.

    Returns
    -------
    numpy.ndarray[matplotlib.axes.Axes]
        The 2x2 axes array used for the dashboard.
    """

    _require_seaborn()

    has_selection = isinstance(estimator, GAFeatureSelectionCV) and hasattr(
        estimator, "best_features_"
    )
    if not has_selection:
        _require_fitted(estimator, "cv_results_", "plot_search_overview")
    _, axes = plt.subplots(2, 2, figsize=figsize, layout="constrained")
    _plot_overview_history(
        estimator, axes[0, 0], plot_convergence, _CONVERGENCE_FIELDS, "Convergence", palette=palette
    )
    _plot_overview_history(
        estimator,
        axes[0, 1],
        plot_diversity,
        _DIVERSITY_FIELDS,
        "Diversity",
        stagnation=bool(_recorded_history_fields(estimator, [_STAGNATION_FIELD])),
    )
    _plot_overview_history(
        estimator, axes[1, 0], plot_optimizer_events, _EVENT_FIELDS, "Optimizer events"
    )

    if has_selection:
        plot_feature_selection(estimator, ax=axes[1, 1])
    else:
        _plot_overview_candidates(estimator, axes[1, 1], top_k, metric)

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

    def fit_work(self, **kwargs):
        return plot_fit_work(self.estimator, **kwargs)

    def finalist_scores(self, **kwargs):
        return plot_finalist_scores(self.estimator, **kwargs)

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

    def cv_fold_scores(self, **kwargs):
        return plot_cv_fold_scores(self.estimator, **kwargs)

    def feature_selection(self, **kwargs):
        return plot_feature_selection(self.estimator, **kwargs)


__all__ = [
    "LivePlot",
    "SearchPlotter",
    "plot_candidate_scores",
    "plot_candidate_rankings",
    "plot_convergence",
    "plot_cv_fold_scores",
    "plot_cv_scores",
    "plot_diversity",
    "plot_feature_selection",
    "plot_finalist_scores",
    "plot_fit_work",
    "plot_fitness_evolution",
    "plot_generation_metrics",
    "plot_history",
    "plot_optimizer_events",
    "plot_parameter_evolution",
    "plot_search_decisions",
    "plot_search_overview",
    "plot_search_space",
    "plot_score_landscape",
]
