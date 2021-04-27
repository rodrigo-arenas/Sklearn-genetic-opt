import seaborn as sns
import numpy as np
from sklearn_genetic import GASearchCV


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

    return sns.lineplot(x=range(len(estimator)), y=fitness_history)


