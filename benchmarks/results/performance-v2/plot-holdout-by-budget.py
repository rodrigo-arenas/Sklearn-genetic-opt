"""Render the mean external score at candidate budgets shared by all runs."""

import json
from io import BytesIO
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter

directory = Path(__file__).resolve().parent
rows = json.loads((directory / "holdout-by-budget.json").read_text(encoding="utf-8"))["rows"]
budgets = [6, 12, 18, 24]
tasks = [
    ("classification_lr", "Breast Cancer · logistic regression", "ROC AUC"),
    ("regression_ridge", "Diabetes · Ridge", "R²"),
]

with plt.rc_context({"svg.fonttype": "none", "svg.hashsalt": "holdout-by-budget", "font.size": 10}):
    fig, axes = plt.subplots(2, 2, figsize=(9, 5.7), layout="constrained", sharex=True)
    for column, (task, title, metric) in enumerate(tasks):
        for row, (search, label) in enumerate(
            [("hpo", "Hyperparameters"), ("features", "Features")]
        ):
            ax = axes[row, column]
            if row == 0:
                ax.set_title(title, fontsize=12, pad=10)
            pairs = [
                record for record in rows if record["task"] == task and record["search"] == search
            ]
            for method, name, color, marker, style in [
                ("genetic", "Genetic", "#005e75", "o", "-"),
                ("random", "Random continuation", "#964616", "s", "--"),
            ]:
                by_budget = [{p["unique"]: p["holdout"] for p in pair[method]} for pair in pairs]
                scores = [mean(split[budget] for split in by_budget) for budget in budgets]
                ax.plot(
                    budgets,
                    scores,
                    color=color,
                    marker=marker,
                    linestyle=style,
                    markerfacecolor="white" if method == "random" else color,
                    label=name,
                )
            ax.set_ylabel(f"{label} · {metric}")
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.yaxis.set_major_formatter(
                FormatStrFormatter("%.4f" if metric == "ROC AUC" else "%.3f")
            )
            ax.set_xticks(budgets)
            ax.margins(y=0.20)
            ax.grid(axis="y", alpha=0.18)
            ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("Distinct candidates evaluated")
    fig.legend(
        *axes[0, 0].get_legend_handles_labels(), loc="outside lower center", ncols=2, frameon=False
    )
    svg = BytesIO()
    fig.savefig(
        svg,
        format="svg",
        metadata={"Date": None, "Title": "Mean held-out winner score by candidate budget"},
    )
    (directory / "holdout-by-budget.svg").write_bytes(
        b"\n".join(line.rstrip() for line in svg.getvalue().splitlines()) + b"\n"
    )
