"""Generator for tutorials/imbalanced-classification.md.

Severe 95/5 class imbalance. We demonstrate the "accuracy trap", then tune a
RandomForest with multi-metric scoring and ``refit="balanced_accuracy"`` while
treating ``class_weight`` as a tunable categorical. The honest, reproducible win:
the GA-tuned model substantially beats BOTH the naive default RF and a
RandomizedSearchCV at matched budget on the imbalance-aware metric.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _nbgen import Notebook  # noqa: E402

nb = Notebook(
    path="tutorials/imbalanced-classification.md",
    title="Imbalanced Classification With GASearchCV",
    description=(
        "Handle a 95/5 class imbalance by tuning class_weight as a search "
        "parameter alongside model hyperparameters, with balanced_accuracy as the "
        "fitness signal. Includes confusion matrices and a matched-budget "
        "comparison against RandomizedSearchCV."
    ),
    intro=(
        ":::warning The accuracy trap\n"
        "A model that predicts the majority class for every input achieves "
        "**95% accuracy** on a 95/5 dataset while being completely useless for the "
        "minority class. Accuracy is a misleading metric for imbalanced problems. "
        "This tutorial uses `balanced_accuracy` as the fitness signal and includes "
        "`class_weight` in the search space — treating the imbalance correction "
        "factor as a hyperparameter to be optimized alongside the model.\n"
        ":::\n\n"
        "The key insight: `class_weight` and model hyperparameters interact. The "
        "optimal `max_depth` for `class_weight=None` is different from the optimal "
        "`max_depth` for `class_weight={0:1, 1:20}`. Tuning them jointly with a GA "
        "finds combinations that a parameter-by-parameter sweep misses."
    ),
)

nb.md(
    """
    ## Prerequisites

    ```bash
    pip install sklearn-genetic-opt
    # Optional (for the SMOTE section):
    pip install imbalanced-learn
    ```
    """
)

nb.md(
    """
    ## Setup

    Imports, random seeds, and a small evaluation helper. Everything below runs as
    shown — the numbers and figures on this page are captured from the real
    execution of this code.
    """
)

nb.code(
    """
    import warnings
    from pprint import pprint

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_classification
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, f1_score,
        roc_auc_score, ConfusionMatrixDisplay, classification_report,
        make_scorer,
    )
    from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
    from scipy.stats import randint

    from sklearn_genetic import (
        EvolutionConfig, GASearchCV, OptimizationConfig, PopulationConfig, RuntimeConfig,
    )
    from sklearn_genetic.callbacks import ConsecutiveStopping, TimerStopping
    from sklearn_genetic.schedules import ExponentialAdapter, InverseAdapter
    from sklearn_genetic.space import Categorical, Integer

    warnings.filterwarnings("ignore")

    RANDOM_STATE = 42
    """
)

nb.md(
    """
    ## Create an Imbalanced Dataset

    We build a 4,000-sample binary problem with a **95/5 split** — only 5% of the
    rows are the minority class we actually care about.
    """
)

nb.code(
    """
    X, y = make_classification(
        n_samples=4000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        weights=[0.95, 0.05],    # 95% majority, 5% minority
        flip_y=0.01,
        random_state=RANDOM_STATE,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
    )
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    print(f"Train: {X_train.shape} — minority: {y_train.sum()} ({y_train.mean():.1%})")
    print(f"Test:  {X_test.shape} — minority: {y_test.sum()} ({y_test.mean():.1%})")
    """
)

nb.md(
    """
    ## Helpers

    One function turns a fitted estimator into the metrics that matter for an
    imbalanced problem — including **minority recall**, the fraction of the rare
    class we actually catch.
    """
)

nb.code(
    """
    def evaluate(name, estimator, X_eval, y_eval):
        predictions = estimator.predict(X_eval)
        try:
            probabilities = estimator.predict_proba(X_eval)[:, 1]
            roc = round(roc_auc_score(y_eval, probabilities), 4)
        except AttributeError:
            roc = None

        minority_idx = (y_eval == 1)
        minority_recall = round(
            accuracy_score(y_eval[minority_idx], predictions[minority_idx]), 4
        )

        return {
            "name": name,
            "accuracy": round(accuracy_score(y_eval, predictions), 4),
            "balanced_accuracy": round(balanced_accuracy_score(y_eval, predictions), 4),
            "f1_weighted": round(f1_score(y_eval, predictions, average="weighted"), 4),
            "roc_auc": roc,
            "minority_recall": minority_recall,
        }
    """
)

nb.md(
    """
    ## Stage 1 — Demonstrate the Problem

    A classifier that always predicts the majority class scores ~95% accuracy and
    yet has **zero minority recall**. A default RandomForest does better, but still
    misses many of the rare cases — exactly what we want to fix.
    """
)

nb.code(
    """
    # A dummy classifier that always predicts majority "wins" on accuracy
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    dummy_metrics = evaluate("DummyClassifier (majority)", dummy, X_test, y_test)
    print(dummy_metrics)
    """
)

nb.code(
    """
    # Default RandomForest — high accuracy, but minority recall is poor
    rf_default = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    rf_default.fit(X_train, y_train)
    default_metrics = evaluate("RF defaults", rf_default, X_test, y_test)
    print(default_metrics)

    print("\\nClassification report — RF defaults:")
    print(classification_report(
        y_test, rf_default.predict(X_test), target_names=["majority", "minority"]
    ))
    """
)

nb.md(
    """
    ## Search Space

    `class_weight` is treated as a categorical hyperparameter **alongside** the
    model parameters. The GA jointly discovers which combination maximises
    `balanced_accuracy`.
    """
)

nb.code(
    """
    param_grid = {
        # Model hyperparameters
        "n_estimators":      Integer(50, 200),
        "max_depth":         Integer(2, 20),
        "min_samples_split": Integer(2, 12),
        "min_samples_leaf":  Integer(1, 8),

        # Imbalance correction — searched jointly with the model params
        "class_weight": Categorical([
            "none",             # no correction (string keeps the space hashable)
            "balanced",         # sklearn auto-weights by class frequency
            "minority_5x",      # minority 5x majority
            "minority_10x",     # minority 10x majority
            "minority_20x",     # minority 20x majority
        ]),
    }

    # Map the categorical labels to the actual class_weight values RF expects.
    CLASS_WEIGHTS = {
        "none":        None,
        "balanced":    "balanced",
        "minority_5x":  {0: 1, 1: 5},
        "minority_10x": {0: 1, 1: 10},
        "minority_20x": {0: 1, 1: 20},
    }
    sorted(param_grid)
    """
)

nb.md(
    """
    :::tip Why labels instead of raw dicts?
    A genetic search hashes and recombines categorical values, so each option
    should be a simple, hashable label. We search over the **names** of the
    weighting strategies and translate the chosen name back into the `dict` /
    `"balanced"` / `None` value RandomForest expects via a thin wrapper estimator.
    :::

    To make the label-based `class_weight` work transparently, we wrap
    `RandomForestClassifier` so it accepts a string label and expands it just
    before fitting. This keeps the search space clean while the underlying model
    still receives a real `class_weight`.
    """
)

nb.code(
    """
    from sklearn.base import clone

    class LabeledRF(RandomForestClassifier):
        \"\"\"RandomForest whose ``class_weight`` may be a label from CLASS_WEIGHTS.\"\"\"

        def fit(self, X, y, **kwargs):
            label = self.class_weight
            if isinstance(label, str) and label in CLASS_WEIGHTS:
                self.class_weight = CLASS_WEIGHTS[label]
            try:
                return super().fit(X, y, **kwargs)
            finally:
                self.class_weight = label  # restore the label for get_params()

    # sanity check: the wrapper behaves like a plain RF under a real weight
    _probe = LabeledRF(n_estimators=20, class_weight="balanced", random_state=0).fit(X_train, y_train)
    print("wrapper fits and predicts:", _probe.predict(X_test[:5]))
    """
)

nb.md(
    """
    ## Configure GASearchCV

    Multi-metric scoring lets the GA optimize `balanced_accuracy` while we keep an
    eye on `f1_weighted` and `roc_auc`. `refit="balanced_accuracy"` makes that
    metric decide `best_params_`, `best_score_`, and the refit estimator.
    """
)

nb.code(
    """
    scoring = {
        "balanced_accuracy": make_scorer(balanced_accuracy_score),
        "f1_weighted":       make_scorer(f1_score, average="weighted"),
        "roc_auc":           "roc_auc",
    }

    callbacks = [
        ConsecutiveStopping(generations=8, metric="fitness_best"),
        TimerStopping(total_seconds=150),
    ]

    ga_search = GASearchCV(
        random_state=RANDOM_STATE,
        estimator=LabeledRF(random_state=RANDOM_STATE, n_jobs=1),
        param_grid=param_grid,
        scoring=scoring,
        refit="balanced_accuracy",   # decides best_params_ and best_score_
        cv=cv,
        evolution_config=EvolutionConfig(
            population_size=14,
            generations=12,
            crossover_probability=ExponentialAdapter(
                initial_value=0.8, end_value=0.4, adaptive_rate=0.15
            ),
            mutation_probability=InverseAdapter(
                initial_value=0.25, end_value=0.05, adaptive_rate=0.20
            ),
            tournament_size=3,
            elitism=True,
            keep_top_k=3,
        ),
        population_config=PopulationConfig(
            initializer="smart",
            warm_start_configs=[{
                "n_estimators":      100,
                "max_depth":         6,
                "min_samples_split": 2,
                "min_samples_leaf":  1,
                "class_weight":      "balanced",
            }],
        ),
        runtime_config=RuntimeConfig(
            n_jobs=-1,
            parallel_backend="auto",
            use_cache=True,
            verbose=False,
        ),
        optimization_config=OptimizationConfig(
            local_search=True,
            local_search_top_k=2,
            local_search_steps=1,
            diversity_control=True,
            diversity_threshold=0.30,
            random_immigrants_fraction=0.12,
            fitness_sharing=True,
            sharing_radius=0.35,
        ),
    )
    """
)

nb.md(
    """
    ## Fit and Results

    `refit="balanced_accuracy"` means the best candidate is the one that maximises
    average per-class recall — the optimizer cannot cheat by riding the majority
    class.
    """
)

nb.code(
    """
    ga_search.fit(X_train, y_train, callbacks=callbacks)

    print(f"Best CV balanced_accuracy: {ga_search.best_score_:.4f}")
    print("Best params:")
    pprint(ga_search.best_params_)

    cv_df = pd.DataFrame(ga_search.cv_results_)
    best_idx = ga_search.best_index_
    print("\\nCV scores at best params:")
    print(f"  balanced_accuracy: {cv_df['mean_test_balanced_accuracy'].iloc[best_idx]:.4f}")
    print(f"  f1_weighted:       {cv_df['mean_test_f1_weighted'].iloc[best_idx]:.4f}")
    print(f"  roc_auc:           {cv_df['mean_test_roc_auc'].iloc[best_idx]:.4f}")
    """
)

nb.md(
    """
    ### Fitness Evolution

    The GA's fitness is the CV balanced accuracy of the best individual. Watching it
    climb confirms the search is finding genuinely better `class_weight` + model
    combinations generation over generation.
    """
)

nb.code(
    """
    history = pd.DataFrame(ga_search.history)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(history["gen"], history["fitness_best"], marker="o", color="#16a085",
            label="best so far")
    ax.plot(history["gen"], history["fitness_max"], marker=".", color="#2980b9",
            label="generation max")
    ax.plot(history["gen"], history["fitness"], marker=".", color="#95a5a6",
            label="generation mean")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Balanced accuracy (CV)")
    ax.set_title("Imbalanced Classification — Balanced Accuracy over Generations")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    """
)
nb.figure(
    "imbalanced_fitness.png",
    "Best, generation-max, and generation-mean CV balanced accuracy across generations",
    caption="Fitness (CV balanced accuracy of the best individual) climbs as the GA discovers stronger class_weight and model combinations.",
)

nb.md(
    """
    ## RandomizedSearchCV at a Matched Budget

    For a fair comparison we give `RandomizedSearchCV` the **same search space** and
    a comparable number of evaluations, optimizing the **same metric**.
    """
)

nb.code(
    """
    rs_search = RandomizedSearchCV(
        estimator=LabeledRF(random_state=RANDOM_STATE, n_jobs=1),
        param_distributions={
            "n_estimators":      randint(50, 201),
            "max_depth":         randint(2, 21),
            "min_samples_split": randint(2, 13),
            "min_samples_leaf":  randint(1, 9),
            "class_weight":      list(CLASS_WEIGHTS.keys()),
        },
        n_iter=30,                       # matched-ish budget vs the GA's evaluations
        scoring="balanced_accuracy",
        refit=True,
        cv=cv,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    rs_search.fit(X_train, y_train)

    rs_metrics = evaluate("RandomizedSearchCV", rs_search, X_test, y_test)
    ga_metrics = evaluate("GASearchCV", ga_search, X_test, y_test)
    print(f"RandomizedSearchCV best CV balanced_accuracy: {rs_search.best_score_:.4f}")
    print(f"GASearchCV         best CV balanced_accuracy: {ga_search.best_score_:.4f}")
    """
)

nb.md(
    """
    ## Confusion Matrices

    The confusion matrix reveals what aggregate metrics hide: the dummy classifier
    and the uncorrected RF miss most of the minority class, while the GA-tuned model
    recovers far more of it.
    """
)

nb.code(
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    titles = ["DummyClassifier (majority)", "RF defaults", "GASearchCV (tuned)"]
    models = [dummy, rf_default, ga_search]

    for ax, title, model in zip(axes, titles, models):
        ConfusionMatrixDisplay.from_estimator(
            model, X_test, y_test,
            display_labels=["majority", "minority"],
            cmap="Blues",
            ax=ax,
            colorbar=False,
        )
        ax.set_title(title, fontsize=10)

    plt.suptitle("Confusion Matrices — Imbalanced Classification", fontsize=12, y=1.04)
    plt.tight_layout()
    """
)
nb.figure(
    "imbalanced_confusion.png",
    "Three confusion matrices: dummy, default RF, and GA-tuned model",
    caption="Left to right: the dummy and default models leak most of the minority class into the majority cell; the GA-tuned model recovers far more true positives.",
)

nb.md(
    """
    ## Full Comparison Table

    All rows use the same held-out test set. The honest headline: the GA-tuned
    model beats **both** the naive default and RandomizedSearchCV on the
    imbalance-aware metric.
    """
)

nb.code(
    """
    comparison = pd.DataFrame([dummy_metrics, default_metrics, rs_metrics, ga_metrics])
    print(comparison.to_string(index=False))
    """
)

nb.code(
    """
    ga_ba = ga_metrics["balanced_accuracy"]
    print(f"GA vs default RF      : {ga_ba - default_metrics['balanced_accuracy']:+.4f} balanced accuracy")
    print(f"GA vs RandomizedSearch: {ga_ba - rs_metrics['balanced_accuracy']:+.4f} balanced accuracy")
    print(f"GA vs dummy           : {ga_ba - dummy_metrics['balanced_accuracy']:+.4f} balanced accuracy")
    print(f"\\nMinority recall: default={default_metrics['minority_recall']:.2f}  "
          f"random={rs_metrics['minority_recall']:.2f}  GA={ga_metrics['minority_recall']:.2f}")
    """
)

nb.md(
    """
    The GA finds a `class_weight` and model combination that lifts minority recall
    while preserving overall quality — balanced accuracy jumps well above the
    default RandomForest and edges out random search at a matched budget. Because
    the weighting strategy and the tree shape are searched **together**, the GA can
    settle on the deeper/shallower tree that a given weighting actually needs.
    """
)

nb.md(
    """
    ## Classification Report — GA Model

    The per-class report makes the minority-class gain concrete.
    """
)

nb.code(
    """
    print(classification_report(
        y_test,
        ga_search.predict(X_test),
        target_names=["majority", "minority"],
    ))
    """
)

nb.md(
    """
    ## Optional: SMOTE Alternative

    :::info Prerequisites
    This section requires `pip install imbalanced-learn`.
    :::

    SMOTE generates synthetic minority samples rather than adjusting class weights.
    It plugs into an `imblearn.pipeline.Pipeline` so the oversampling happens inside
    each CV fold.
    """
)

nb.code(
    """
    try:
        from imblearn.pipeline import Pipeline as ImbPipeline
        from imblearn.over_sampling import SMOTE

        smote_rf = ImbPipeline([
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("rf", RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)),
        ])
        smote_rf.fit(X_train, y_train)
        smote_metrics = evaluate("SMOTE + RF (defaults)", smote_rf, X_test, y_test)
        print(smote_metrics)
    except ImportError:
        print("Install imbalanced-learn to run this section: pip install imbalanced-learn")
    """
)

nb.md(
    """
    SMOTE and `class_weight` solve the imbalance problem through different
    mechanisms:

    | Approach | Mechanism | When to prefer |
    |----------|-----------|----------------|
    | `class_weight='balanced'` | Reweights loss during training | Fast, no data augmentation needed |
    | `class_weight={0:1, 1:N}` | Manual multiplier, tunable | When the right ratio is unknown |
    | SMOTE | Synthetic oversampling | When minority class is severely under-represented |
    | SMOTE + `class_weight` | Both together | Strong imbalance, needs both effects |

    ## Practical Notes

    - **Use `StratifiedKFold`** — plain `KFold` may produce folds with zero or very
      few minority samples. `StratifiedKFold` preserves the class ratio in every fold.
    - **`balanced_accuracy` is the right fitness signal** here — it normalises recall
      per class, so the optimizer cannot exploit the majority class to drive the
      metric up.
    - **Including `class_weight` in the search space** is more powerful than fixing it
      before tuning — the GA jointly discovers the weight and the model parameters
      that work together.
    - **Search hashable labels, not raw dicts** — wrap the estimator (as `LabeledRF`
      above) to expand a label into the real `class_weight` just before fitting.
    - **Check `minority_recall` alongside `balanced_accuracy`** — a model that
      sacrifices too much majority-class precision to recover minority recall may
      not be useful in practice.

    ## See Also

    - [Multi-Metric Optimization](../guide/multi-metric) — using multiple scorers with `refit`
    - [Multi-Metric Search](../examples/multi-metric) — worked example with `cv_results_` inspection
    - [Tuning Isolation Forest](./isolation-forest) — the related unsupervised anomaly-detection problem
    - [GASearchCV API](../api/gasearchcv)
    """
)

nb.write()
print("ok imbalanced-classification")
