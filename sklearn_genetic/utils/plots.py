import seaborn as sns

sns.set_style("darkgrid")


def plot_fitness_evolution(estimator):
    """
    Parameters
    ----------
    estimator: Fitted GASearchCV

    Returns
    plot with the fitness value in each generation
    -------

    """
    fitness_history = estimator.history["fitness"]

    ax = sns.lineplot(x=range(len(estimator)), y=fitness_history)
    ax.set(xlabel='generations', ylabel=f'fitness ({estimator.scoring})')
    return ax

