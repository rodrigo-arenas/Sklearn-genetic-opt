import logging

logger = logging.getLogger(__name__)  # noqa

# Check if seaborn is installed as an extra requirement
try:
    import seaborn as sns
except ModuleNotFoundError:  # noqa
    logger.error("seaborn not found, pip install seaborn to use plots functions")  # noqa

import numpy as np

from .utils import logbook_to_pandas
from .parameters import Metrics

from .genetic_search import GAFeatureSelectionCV

"""
This module contains some useful function to explore the results of the optimization routines
"""


def plot_fitness_evolution(estimator, metric="fitness"):
    """
    Parameters
    ----------
    estimator: estimator object
        A fitted estimator from :class:`~sklearn_genetic.GASearchCV`
    metric: {"fitness", "fitness_std", "fitness_max", "fitness_min"}, default="fitness"
        Logged metric into the estimator history to plot

    Returns
    -------
    Lines plot with the fitness value in each generation

    """

    if metric not in Metrics.list():
        raise ValueError(f"metric must be one of {Metrics.list()}, but got {metric} instead")

    sns.set_style("white")

    fitness_history = estimator.history[metric]

    palette = sns.color_palette("rocket")
    sns.set(rc={"figure.figsize": (10, 10)})

    ax = sns.lineplot(x=range(len(estimator)), y=fitness_history, markers=True, palette=palette)
    ax.set_title(f"{metric.capitalize()} average evolution over generations")

    ax.set(xlabel="generations", ylabel=f"fitness ({estimator.refit_metric})")
    return ax


def plot_search_space(estimator, height=2, s=25, features: list = None):
    """
    Parameters
    ----------
    estimator: estimator object
        A fitted estimator from :class:`~sklearn_genetic.GASearchCV`
    height: float, default=2
        Height of each facet
    s: float, default=5
        Size of the markers in scatter plot
    features: list, default=None
        Subset of features to plot, if ``None`` it plots all the features by default

    Returns
    -------
    Pair plot of the used hyperparameters during the search

    """

    if isinstance(estimator, GAFeatureSelectionCV):
        raise TypeError(
            "Estimator must be a GASearchCV instance, not a GAFeatureSelectionCV instance"
        )

    sns.set_style("white")

    df = logbook_to_pandas(estimator.logbook)
    if features:
        stats = df[features].astype(np.float64)
    else:
        variables = [*estimator.space.parameters, estimator.refit_metric]
        stats = df[variables].select_dtypes(["number", "bool"]).astype(np.float64)

    g = sns.PairGrid(stats, diag_sharey=False, height=height)
    g = g.map_upper(sns.scatterplot, s=s, color="r", alpha=0.2)
    g = g.map_lower(
        sns.kdeplot,
        shade=True,
        cmap=sns.color_palette("ch:s=.25,rot=-.25", as_cmap=True),
    )
    g = g.map_diag(sns.kdeplot, shade=True, palette="crest", alpha=0.2, color="red")
    return g
