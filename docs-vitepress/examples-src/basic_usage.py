from __future__ import annotations

import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits, load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from sklearn_genetic import EvolutionConfig, GAFeatureSelectionCV, GASearchCV
from sklearn_genetic import PopulationConfig, RuntimeConfig
from sklearn_genetic.plots import (
    plot_feature_selection,
    plot_parameter_evolution,
    plot_search_overview,
)
from sklearn_genetic.space import Categorical, Continuous, Integer

ROOT = Path(__file__).resolve().parents[2]
IMAGE_TARGETS = [ROOT / "docs-vitepress" / "public" / "images", ROOT / "docs" / "images"]
RANDOM_STATE = 42

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


def save_figure(name: str) -> None:
    for target in IMAGE_TARGETS:
        target.mkdir(parents=True, exist_ok=True)
        plt.savefig(target / name, dpi=150, bbox_inches="tight")
    plt.close()


def markdown_table(rows: list[dict[str, object]], columns: list[tuple[str, str]]) -> str:
    header = "| " + " | ".join(label for _, label in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row[key]) for key, _ in columns) + " |")
    return "\n".join([header, separator, *body])


def fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, dict):
        items = [f"{key}: {fmt(val)}" for key, val in sorted(value.items())]
        return "{ " + ", ".join(items) + " }"
    return str(value)


data = load_digits()
n_samples = len(data.images)
X = data.images.reshape((n_samples, -1))
y = data["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.33,
    stratify=y,
    random_state=RANDOM_STATE,
)
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

param_grid = {
    "tol": Continuous(1e-4, 1e-1, distribution="log-uniform", random_state=RANDOM_STATE),
    "alpha": Continuous(1e-5, 1e-3, distribution="log-uniform", random_state=RANDOM_STATE + 1),
    "activation": Categorical(["logistic", "tanh"]),
    "batch_size": Integer(128, 256, random_state=RANDOM_STATE + 2),
}

clf = MLPClassifier(
    hidden_layer_sizes=(30,),
    max_iter=120,
    early_stopping=True,
    random_state=RANDOM_STATE,
)

search = GASearchCV(
    estimator=clf,
    cv=cv,
    scoring="accuracy",
    param_grid=param_grid,
    evolution_config=EvolutionConfig(population_size=6, generations=5),
    population_config=PopulationConfig(initializer="smart"),
    runtime_config=RuntimeConfig(n_jobs=1, verbose=False),
)
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
search.fit(X_train, y_train)

history = pd.DataFrame(search.history)
y_predict_ga = search.predict(X_test)
accuracy = accuracy_score(y_test, y_predict_ga)

history_rows = []
for _, row in history.tail(5).iterrows():
    history_rows.append(
        {
            "gen": int(row["gen"]),
            "fitness_best": fmt(row["fitness_best"]),
            "genotype_diversity": fmt(row.get("genotype_diversity", np.nan)),
            "unique_individual_ratio": fmt(row.get("unique_individual_ratio", np.nan)),
            "stagnation_generations": int(row.get("stagnation_generations", 0)),
        }
    )

fit_stats = {
    key: search.fit_stats_.get(key)
    for key in [
        "evaluated_candidates",
        "unique_candidates",
        "cache_hits",
        "random_immigrants",
        "skipped_invalid_candidates",
    ]
    if key in search.fit_stats_
}

plot_search_overview(search, top_k=6)
save_figure("basic_usage_search_overview.png")

plot_parameter_evolution(search, parameters=["tol", "batch_size", "alpha"])
save_figure("basic_usage_parameter_evolution.png")

iris = load_iris()
X_iris, y_iris = iris["data"], iris["target"]
rng = np.random.default_rng(RANDOM_STATE)
noise = rng.uniform(0, 10, size=(X_iris.shape[0], 10))
X_iris = np.hstack((X_iris, noise))
X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(
    X_iris,
    y_iris,
    test_size=0.33,
    stratify=y_iris,
    random_state=RANDOM_STATE,
)

selector = GAFeatureSelectionCV(
    estimator=SVC(gamma="auto"),
    cv=3,
    scoring="accuracy",
    evolution_config=EvolutionConfig(
        population_size=10,
        generations=6,
        keep_top_k=2,
        elitism=True,
    ),
    population_config=PopulationConfig(initializer="smart"),
    runtime_config=RuntimeConfig(n_jobs=1, verbose=False),
)
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
selector.fit(X_train_iris, y_train_iris)
selector_accuracy = accuracy_score(y_test_iris, selector.predict(X_test_iris))
feature_names = list(iris.feature_names) + [f"noise_{i}" for i in range(noise.shape[1])]
selected_names = [name for name, selected in zip(feature_names, selector.support_) if selected]
selected_noise_count = sum(name.startswith("noise_") for name in selected_names)

plot_feature_selection(selector, feature_names=feature_names)
save_figure("basic_usage_feature_selection.png")

GENERATED_SNIPPETS = {
    "basic-usage-hyperparameter-output": "\n".join(
        [
            "```text",
            f"Best CV accuracy: {search.best_score_:.4f}",
            f"Holdout accuracy: {accuracy:.4f}",
            f"Best parameters: {fmt(search.best_params_)}",
            "```",
        ]
    ),
    "basic-usage-history-output": markdown_table(
        history_rows,
        [
            ("gen", "Generation"),
            ("fitness_best", "Best CV accuracy"),
            ("genotype_diversity", "Diversity"),
            ("unique_individual_ratio", "Unique ratio"),
            ("stagnation_generations", "Stagnation"),
        ],
    ),
    "basic-usage-fit-stats-output": "\n".join(
        [
            "```text",
            *(f"{key}: {value}" for key, value in fit_stats.items()),
            "```",
        ]
    ),
    "basic-usage-feature-output": "\n".join(
        [
            "```text",
            f"Holdout accuracy: {selector_accuracy:.4f}",
            f"Selected features: {len(selected_names)} of {len(feature_names)}",
            f"Selected noise features: {selected_noise_count}",
            "Selected feature names:",
            *(f"- {name}" for name in selected_names),
            "```",
        ]
    ),
}
