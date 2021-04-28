import seaborn as sns
from sklearn_genetic import GASearchCV

sns.set_style("darkgrid")


def plot_fitness_evolution(estimator: GASearchCV):
    """
    Parameters
    ----------
    estimator: Fitted GASearchCV

    Returns
    plot with the fitness value in each generation
    -------

    """
    fitness_history = []
    for generation in estimator:
        fitness_history.append(generation['fitness'])

    ax = sns.lineplot(x=range(len(estimator)), y=fitness_history)
    ax.set(xlabel='generations', ylabel=f'fitness ({estimator.scoring})')
    return ax

