"""Render compact homepage figures from the plotting gallery's fitted search."""

from io import StringIO

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator

# The image inherits the embedding page's color-scheme, including the manual
# VitePress theme switch. Keep a single lazy-loaded SVG per chart.
INK = "#384038"
MUTED = "#596257"
GRID = "#cdd1c7"
CORAL = "#a53624"
TEAL = "#247169"
NIGHT = {
    INK: "#edf0e8",
    MUTED: "#bdc4b8",
    GRID: "#464c44",
    CORAL: "#ff9a7e",
    TEAL: "#81d1c4",
}


def save(fig, path):
    stream = StringIO()
    fig.savefig(stream, format="svg", transparent=True, metadata={"Date": None})
    plt.close(fig)
    svg = stream.getvalue()
    properties = []
    for color, dark in NIGHT.items():
        name = f"--c{color[1:]}"
        svg = svg.replace(color, f"var({name}, {color})")
        properties.append(f"{name}:{dark}")
    style = "@media (prefers-color-scheme:dark){:root{" + ";".join(properties) + "}}"
    svg = svg.replace("</style>", style + "</style>", 1)
    path.write_text("\n".join(line.rstrip() for line in svg.splitlines()) + "\n", encoding="utf-8")


def panel(*, bottom=0.24, top=0.77):
    fig = plt.figure(figsize=(4.5, 2.5))
    ax = fig.add_subplot()
    fig.subplots_adjust(left=0.16, right=0.95, bottom=bottom, top=top)
    ax.grid(axis="y", linewidth=0.65, color=GRID)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, pad=4, colors=MUTED)
    ax.spines[:].set_visible(False)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 5, 10]))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    ax.ticklabel_format(useOffset=False)
    return fig, ax


def generations(ax, gen):
    ax.set_xlim(gen[0] - 0.5, gen[-1] + 0.5)
    ax.set_xticks([gen[0], gen[len(gen) // 2], gen[-1]])
    ax.set_xlabel("Generation")


def lines(ax, gen, series, legend_height=1.05):
    for values, label, color, style in series:
        ax.plot(gen, values, style, label=label, color=color, linewidth=2, markersize=4)
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(-0.02, legend_height),
        ncols=2,
        frameon=False,
        handlelength=1.65,
        columnspacing=1.2,
        borderaxespad=0,
    )
    generations(ax, gen)


def score_key(fig, points):
    # A vector color ramp (the default colorbar may rasterize itself).
    key = fig.add_axes([0.53, 0.88, 0.20, 0.041])
    key.axis("off")
    for i in range(32):
        key.add_patch(
            Rectangle((i / 32, 0), 1 / 32 + 0.002, 1, facecolor=points.cmap(i / 31), linewidth=0)
        )
    lo, hi = points.get_clim()
    fig.text(0.05, 0.90, "CV ROC AUC", va="center", weight="bold")
    fig.text(0.51, 0.90, f"{lo:.3f}", va="center", ha="right")
    fig.text(0.75, 0.90, f"{hi:.3f}", va="center")


def write_home_gallery(search, directory):
    """Called by build_docs.py plotting-gallery after its seeded fit."""
    directory.mkdir(parents=True, exist_ok=True)
    h = search.history
    gen = h["gen"]
    rows = pd.DataFrame(search.cv_results_)
    scores = rows["mean_test_score"]
    style = {
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.labelpad": 5,
        "legend.fontsize": 11,
        "svg.fonttype": "none",
        "svg.hashsalt": "home-gallery",
    }
    with plt.style.context("default"), mpl.rc_context(style):
        fig, ax = panel()
        lines(
            ax,
            gen,
            [
                (h["fitness_best_observed"], "Best observed", CORAL, "-o"),
                (h["fitness"], "Mean", TEAL, "--"),
            ],
            legend_height=1.23,
        )
        ax.set_title("CV ROC AUC", loc="left", fontsize=10, pad=2, color=MUTED)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 5, 10]))
        save(fig, directory / "fitness.svg")

        fig, ax = panel(top=0.54)
        for x, label, value in [
            (0.06, "Best CV ROC AUC", f"{scores.max():.4f}"),
            (0.58, "Evaluations", str(len(rows))),
        ]:
            fig.text(x, 0.89, label, color=MUTED)
            fig.text(x, 0.68, value, size=25, weight="bold")
        ax.bar(gen, h["duplicate_replacements"], color=CORAL, width=0.60)
        ax.set_title("Duplicate replacements", loc="left", fontsize=10, pad=4, color=MUTED)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3, integer=True, steps=[1, 2, 5, 10]))
        generations(ax, gen)
        save(fig, directory / "overview.svg")

        fig, ax = panel(bottom=0.09, top=0.74)
        fig.subplots_adjust(left=0.24)
        names = ["n_estimators", "max_depth", "min_samples_split", "max_features"]
        corr = rows[[f"param_{name}" for name in names]].astype(float).corr().to_numpy()
        labels = ["Trees", "Depth", "Split", "Features"]
        ax.grid(False)
        ax.set(xlim=(-0.5, 3.5), ylim=(3.5, -0.5))
        ax.set_xticks(range(4), labels)
        ax.set_yticks(range(4), labels)
        ax.xaxis.tick_top()
        for row, col in np.ndindex(corr.shape):
            r = corr[row, col]
            ax.add_patch(
                Rectangle(
                    (col - 0.48, row - 0.47),
                    0.96,
                    0.94,
                    color=CORAL if r >= 0 else TEAL,
                    alpha=0.10 + 0.38 * abs(r),
                    linewidth=0,
                )
            )
            ax.text(col, row, f"{r:.2f}", ha="center", va="center")
        fig.text(0.06, 0.92, "Pearson correlation", weight="bold")
        save(fig, directory / "correlation.svg")

        fig, ax = panel(top=0.71)
        palette = mpl.colors.LinearSegmentedColormap.from_list(
            "cv_score", ["#533c62", "#975362", "#d57a61", "#f8cb86"]
        )
        points = ax.scatter(
            np.arange(len(rows)),
            rows["param_n_estimators"],
            c=scores,
            cmap=palette,
            s=18,
            linewidths=0.4,
            edgecolors=INK,
        )
        ax.set_title("Number of trees", loc="left", fontsize=10, pad=6, color=MUTED)
        ax.set_xlabel("Evaluation index")
        score_key(fig, points)
        save(fig, directory / "parameters.svg")

        fig, ax = panel()
        lines(
            ax,
            gen,
            [
                (h["unique_individual_ratio"], "Unique ratio", CORAL, "-o"),
                (h["genotype_diversity"], "Genotype diversity", TEAL, "--"),
            ],
        )
        ax.set_ylim(-0.04, 1.04)
        ax.set_yticks([0, 0.5, 1], ["0", "0.5", "1"])
        save(fig, directory / "diversity.svg")

        fig, ax = panel(top=0.71)
        points = ax.scatter(
            rows["param_max_depth"],
            rows["param_min_samples_split"],
            c=scores,
            cmap=palette,
            s=34,
            linewidths=0.5,
            edgecolors=INK,
        )
        ax.set_title("Min. samples to split", loc="left", fontsize=10, pad=6, color=MUTED)
        ax.set_xlabel("Maximum depth")
        score_key(fig, points)
        save(fig, directory / "landscape.svg")
