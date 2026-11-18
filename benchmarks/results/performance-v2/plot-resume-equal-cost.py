"""Render paired changes in outer-holdout scores from the equal-budget experiment."""

import json
from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

directory = Path(__file__).resolve().parent
records = json.loads((directory / "resume-equal-cost.json").read_text(encoding="utf-8"))["records"]
seeds = sorted({record["seed"] for record in records})
budgets = sorted({record["budget"] for record in records})
tasks = [
    ("wine", "hyperparameters", "Wine · hyperparameters"),
    ("wine", "features", "Wine · features"),
    ("diabetes", "hyperparameters", "Diabetes · hyperparameters"),
    ("diabetes", "features", "Diabetes · features"),
]
differences = {
    (r["dataset"], r["kind"], r["budget"], r["seed"]): r["resumed"]["holdout"]
    - r["continuous"]["holdout"]
    for r in records
}
assert len(differences) == len(records) == len(tasks) * len(budgets) * len(seeds)

with plt.rc_context({"svg.fonttype": "none", "svg.hashsalt": "resume-equal-cost", "font.size": 10}):
    fig, axes = plt.subplots(
        1, len(budgets), figsize=(8.5, 3.15), layout="constrained", sharey=True
    )
    colors = ListedColormap(["#f6d8bd", "#f1f3f4", "#c4e7df"])
    for ax, budget in zip(axes, budgets, strict=True):
        values = np.array(
            [
                [differences[(dataset, kind, budget, seed)] for seed in seeds]
                for dataset, kind, _ in tasks
            ]
        )
        ax.imshow(np.sign(values), cmap=colors, vmin=-1, vmax=1, aspect="auto")
        for (row, column), delta in np.ndenumerate(values):
            ax.text(
                column,
                row,
                f"{delta:+.4f}" if delta else "0",
                color="#172b34",
                ha="center",
                va="center",
            )
        ax.set_title(f"{budget} candidates", pad=9)
        ax.set_xticks(range(len(seeds)), [str(seed) for seed in seeds])
        ax.set_xlabel("Outer split seed")
        ax.set_yticks(range(len(tasks)), [label for _, _, label in tasks])
        ax.set_xticks(np.arange(len(seeds) + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(tasks) + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="white", linewidth=2)
        ax.tick_params(which="both", length=0)
        ax.spines[:].set_visible(False)
    fig.suptitle("Change in held-out score after resuming", fontsize=14)
    svg = BytesIO()
    fig.savefig(
        svg,
        format="svg",
        metadata={"Date": None, "Title": "Resumed minus continuous held-out score"},
    )
    (directory / "resume-equal-cost.svg").write_bytes(
        b"\n".join(line.rstrip() for line in svg.getvalue().splitlines()) + b"\n"
    )
