import seaborn as sns

from .utils import logbook_to_pandas


"""
This module contains some useful function to explore the results of the optimization routines
"""


def plot_fitness_evolution(estimator):
    """
    Parameters
    ----------
    estimator: estimator object
        A fitted estimator from :class:`~sklearn_genetic.GASearchCV`

    Returns
    -------
    Lines plot with the fitness value in each generation

    """
    sns.set_style("white")

    fitness_history = estimator.history["fitness"]

    palette = sns.color_palette("rocket")
    sns.set(rc={"figure.figsize": (10, 10)})

    ax = sns.lineplot(
        x=range(len(estimator)), y=fitness_history, markers=True, palette=palette
    )
    ax.set_title("Fitness average evolution over generations")

    ax.set(xlabel="generations", ylabel=f"fitness ({estimator.scoring})")
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
    sns.set_style("white")

    df = logbook_to_pandas(estimator.logbook)
    if features:
        stats = df[features]
    else:
        variables = [*estimator.space.parameters, "score"]
        stats = df[variables]

    g = sns.PairGrid(stats, diag_sharey=False, height=height)
    g = g.map_upper(sns.scatterplot, s=s, color="r", alpha=0.2)
    g = g.map_lower(
        sns.kdeplot,
        shade=True,
        cmap=sns.color_palette("ch:s=.25,rot=-.25", as_cmap=True),
    )
    g = g.map_diag(sns.kdeplot, shade=True, palette="crest", alpha=0.2, color="red")
    return g
