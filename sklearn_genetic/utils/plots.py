import seaborn as sns

sns.set_style("whitegrid")


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

    palette = sns.color_palette("rocket")
    sns.set(rc={'figure.figsize': (12, 15)})

    ax = sns.lineplot(x=range(len(estimator)), y=fitness_history,
                      markers=True,
                      palette=palette)
    ax.set_title('Fitness average evolution over generations')

    ax.set(xlabel='generations', ylabel=f'fitness ({estimator.scoring})')
    return ax
