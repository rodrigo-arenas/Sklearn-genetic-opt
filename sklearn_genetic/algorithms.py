from typing import Union
from collections.abc import Callable

from deap import tools
from deap.algorithms import varAnd, varOr

from .callbacks import eval_callbacks


def eaSimple(
    population,
    toolbox,
    cxpb,
    mutpb,
    ngen,
    stats=None,
    halloffame=None,
    callbacks=None,
    verbose=True,
):
    """
    The base implementation is directly taken from: https://github.com/DEAP/deap/blob/master/deap/algorithms.py

    This algorithm reproduce the simplest evolutionary algorithm as
    presented in chapter 7 of Back2000.

    population: A list of individuals.
        Population resulting of the iteration process.

    toolbox: A :class:`~deap.base.Toolbox`
        Contains the evolution operators.

    cxpb: float, default=None
        The probability of mating two individuals.

    mutpb: float, default=None
        The probability of mutating an individual.

    ngen: int, default=None
        The number of generation.

    stats: A :class:`~deap.tools.Statistics`
        Object that is updated inplace, optional.

    halloffame: A :class:`~deap.tools.HallOfFame`
        Object that will contain the best individuals, optional.

    callbacks: list or callable
        One or a list of the :class:`~sklearn_genetic.callbacks` methods available in the package.

    verbose: bool, default=True
        Whether or not to log the statistics.

    Returns
    -------

    pop: list
        The final population.

    log: Logbook
        Statistics of the evolution.

    n_gen: int
        Number of generations used.

    """
    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))

        # Vary the pool of individuals
        offspring = varAnd(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        # Check if any of the callbacks conditions are True to stop the iteration
        if eval_callbacks(callbacks, record, logbook):
            print("Process stopped earlier due a callback")
            break

    n_gen = gen + 1

    return population, logbook, n_gen


def eaMuPlusLambda(
    population,
    toolbox,
    mu,
    lambda_,
    cxpb,
    mutpb,
    ngen,
    stats=None,
    halloffame=None,
    callbacks: Union[list, Callable] = None,
    verbose=True,
):
    """
    The base implementation is directly taken from: https://github.com/DEAP/deap/blob/master/deap/algorithms.py

    This is the :math:`(\mu + \lambda)` evolutionary algorithm.

    population: A list of individuals.
        Population resulting of the iteration process.

    toolbox: A :class:`~deap.base.Toolbox`
        Contains the evolution operators.

    mu: int, default=None
        The number of individuals to select for the next generation.

    lambda\_: int, default=None
        The number of children to produce at each generation.

    cxpb: float, default=None
        The probability that an offspring is produced by crossover.

    mutpb: float, default=None
        The probability that an offspring is produced by mutation.

    ngen: int, default=None
        The number of generation.
    stats: A :class:`~deap.tools.Statistics`
        Object that is updated inplace, optional.

    halloffame: A :class:`~deap.tools.HallOfFame`
        Object that will contain the best individuals, optional.

    callbacks: list or Callable
        One or a list of the :class:`~sklearn_genetic.callbacks` methods available in the package.

    verbose: bool, default=True
        Whether or not to log the statistics.

    Returns
    -------

    pop: list
        The final population.

    log: Logbook
        Statistics of the evolution.

    n_gen: int
        Number of generations used.

    """
    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats is not None else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Vary the population
        offspring = varOr(population, toolbox, lambda_, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Select the next generation population
        population[:] = toolbox.select(population + offspring, mu)

        # Update the statistics with the new population
        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        if eval_callbacks(callbacks, record, logbook):
            print("Process stopped earlier due a callback")
            break

    n_gen = gen + 1
    return population, logbook, n_gen


def eaMuCommaLambda(
    population,
    toolbox,
    mu,
    lambda_,
    cxpb,
    mutpb,
    ngen,
    stats=None,
    halloffame=None,
    callbacks: Union[list, Callable] = None,
    verbose=True,
):
    """
    The base implementation is directly taken from: https://github.com/DEAP/deap/blob/master/deap/algorithms.py

    This is the :math:`(\mu~,~\lambda)` evolutionary algorithm.

    population: A list of individuals.
        Population resulting of the iteration process.

    toolbox: A :class:`~deap.base.Toolbox`
        Contains the evolution operators.

    mu: int, default=None,
        The number of individuals to select for the next generation.

    lambda\_: int, default=None
        The number of children to produce at each generation.

    cxpb: float, default=None
        The probability that an offspring is produced by crossover.

    mutpb: float, default=None
        The probability that an offspring is produced by mutation.

    ngen: int, default=None
        The number of generation.

    stats: A :class:`~deap.tools.Statistics`
        Object that is updated inplace, optional.

    halloffame: A :class:`~deap.tools.HallOfFame`
        Object that will contain the best individuals, optional.

    callbacks: list or Callable
        One or a list of the :class:`~sklearn_genetic.callbacks` methods available in the package.

    verbose: bool, default=True
        Whether or not to log the statistics.

    Returns
    -------

    pop: list
        The final population.

    log: Logbook
        Statistics of the evolution.

    n_gen: int
        Number of generations used.

    """
    assert lambda_ >= mu, "lambda must be greater or equal to mu."

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])

    record = stats.compile(population) if stats is not None else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Vary the population
        offspring = varOr(population, toolbox, lambda_, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Select the next generation population
        population[:] = toolbox.select(offspring, mu)

        # Update the statistics with the new population
        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        # Check if any of the callbacks conditions are True to stop the iteration
        if eval_callbacks(callbacks, record, logbook):
            print("Process stopped earlier due a callback")
            break

    n_gen = gen + 1
    return population, logbook, n_gen
