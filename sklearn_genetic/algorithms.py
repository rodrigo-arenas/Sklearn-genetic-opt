import numpy as np
from deap import tools
from deap.algorithms import varAnd, varOr

from .callbacks.validations import eval_callbacks


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
    estimator=None,
    **kwargs,
):
    """
    The base implementation is directly taken from: https://github.com/DEAP/deap/blob/master/deap/algorithms.py

    This algorithm reproduce the simplest evolutionary algorithm as
    presented in chapter 7 of Back2000.

    population: A list of individuals.
        Population resulting of the iteration process.

    toolbox: A :class:`~deap.base.Toolbox`
        Contains the evolution operators.

    cxpb: Scheduler, default=None
        An adaptive scheduler representing the probability of mating two individuals.

    mutpb: Scheduler, default=None
        An adaptive scheduler representing the probability that an offspring is produced by mutation.

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

    estimator: :class:`~sklearn_genetic.GASearchCV`, default = None
        Estimator that is being optimized

    Returns
    -------

    pop: list
        The final population.

    log: Logbook
        Statistics of the evolution.

    n_gen: int
        Number of generations used.

    """
    stored_exception = None
    callbacks_start_args = {
        "callbacks": callbacks,
        "record": None,
        "logbook": None,
        "estimator": estimator,
        "method": "on_start",
    }
    eval_callbacks(**callbacks_start_args)

    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)
    hof_size = len(halloffame.items) if (halloffame.items and estimator.elitism) else 0

    record = stats.compile(population) if stats else {}
    if isinstance(record["fitness"], np.ndarray):
        record = {key: value[0] for key, value in record.items()}

    n_gen = gen = 0
    logbook.record(gen=n_gen, nevals=len(invalid_ind), **record)

    if verbose:
        print(logbook.stream)

    # Check if any of the callbacks conditions are True to stop the iteration

    callbacks_step_args = {
        "callbacks": callbacks,
        "record": record,
        "logbook": logbook,
        "estimator": estimator,
        "method": "on_step",
    }

    if eval_callbacks(**callbacks_step_args):
        callbacks_end_args = {
            "callbacks": callbacks,
            "record": None,
            "logbook": logbook,
            "estimator": estimator,
            "method": "on_end",
        }

        # Call ending callback
        eval_callbacks(**callbacks_end_args)
        print("INFO: Stopping the algorithm")
        return population, logbook, n_gen

    for gen in range(1, ngen + 1):
        try:
            # Select the next generation individuals
            offspring = toolbox.select(population, len(population) - hof_size)

            # Vary the pool of individuals
            offspring = varAnd(offspring, toolbox, cxpb.step(), mutpb.step())

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            if estimator.elitism:
                offspring.extend(halloffame.items)

            # Update the hall of fame with the generated individuals
            if halloffame is not None:
                halloffame.update(offspring)

            # Replace the current population by the offspring
            population[:] = offspring

            # Append the current generation statistics to the logbook
            record = stats.compile(population) if stats else {}
            if isinstance(record["fitness"], np.ndarray):
                record = {key: value[0] for key, value in record.items()}

            logbook.record(gen=gen, nevals=len(invalid_ind), **record)

            if verbose:
                print(logbook.stream)

            callbacks_step_args = {
                "callbacks": callbacks,
                "record": record,
                "logbook": logbook,
                "estimator": estimator,
                "method": "on_step",
            }

            # Check if any of the callbacks conditions are True to stop the iteration
            if eval_callbacks(**callbacks_step_args) or stored_exception:
                if stored_exception:
                    print(
                        f"{stored_exception}\nsklearn-genetic-opt closed prematurely. Will use the current best model."
                    )
                print("INFO: Stopping the algorithm")
                break
        except (KeyboardInterrupt, SystemExit, StopIteration) as e:
            stored_exception = e

    n_gen = gen + 1

    callbacks_end_args = {
        "callbacks": callbacks,
        "record": None,
        "logbook": logbook,
        "estimator": estimator,
        "method": "on_end",
    }

    # Call ending callback
    eval_callbacks(**callbacks_end_args)

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
    callbacks=None,
    verbose=True,
    estimator=None,
    **kwargs,
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

    cxpb: Scheduler, default=None
        The probability that an offspring is produced by crossover.

    mutpb: Scheduler, default=None
        An adaptive scheduler representing the probability that an offspring is produced by mutation.

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

    estimator: :class:`~sklearn_genetic.GASearchCV`, default = None
        Estimator that is being optimized

    Returns
    -------

    pop: list
        The final population.

    log: Logbook
        Statistics of the evolution.

    n_gen: int
        Number of generations used.

    """
    stored_exception = None
    callbacks_start_args = {
        "callbacks": callbacks,
        "record": None,
        "logbook": None,
        "estimator": estimator,
        "method": "on_start",
    }
    eval_callbacks(**callbacks_start_args)

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
    if isinstance(record["fitness"], np.ndarray):
        record = {key: value[0] for key, value in record.items()}

    n_gen = gen = 0
    logbook.record(gen=n_gen, nevals=len(invalid_ind), **record)

    if verbose:
        print(logbook.stream)

    # Check if any of the callbacks conditions are True to stop the iteration
    callbacks_step_args = {
        "callbacks": callbacks,
        "record": record,
        "logbook": logbook,
        "estimator": estimator,
        "method": "on_step",
    }

    if eval_callbacks(**callbacks_step_args):
        # Call ending callback
        callbacks_end_args = {
            "callbacks": callbacks,
            "record": None,
            "logbook": None,
            "estimator": estimator,
            "method": "on_end",
        }

        eval_callbacks(**callbacks_end_args)
        print("INFO: Stopping the algorithm")
        return population, logbook, n_gen

    for gen in range(1, ngen + 1):
        try:
            # Vary the population
            offspring = varOr(population, toolbox, lambda_, cxpb.step(), mutpb.step())

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
            if isinstance(record["fitness"], np.ndarray):
                record = {key: value[0] for key, value in record.items()}

            logbook.record(gen=gen, nevals=len(invalid_ind), **record)

            if verbose:
                print(logbook.stream)

            callbacks_step_args = {
                "callbacks": callbacks,
                "record": record,
                "logbook": logbook,
                "estimator": estimator,
                "method": "on_step",
            }

            if eval_callbacks(**callbacks_step_args) or stored_exception:
                if stored_exception:
                    print(
                        f"{stored_exception}\nsklearn-genetic-opt closed prematurely. Will use the current best model."
                    )
                print("INFO: Stopping the algorithm")
                break

        except (KeyboardInterrupt, SystemExit, StopIteration) as e:
            stored_exception = e

    n_gen = gen + 1

    callbacks_end_args = {
        "callbacks": callbacks,
        "record": None,
        "logbook": None,
        "estimator": estimator,
        "method": "on_end",
    }

    eval_callbacks(**callbacks_end_args)

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
    callbacks=None,
    verbose=True,
    estimator=None,
    **kwargs,
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

    cxpb: Scheduler, default=None
        The probability that an offspring is produced by crossover.

    mutpb: Scheduler, default=None
        An adaptive scheduler representing the probability that an offspring is produced by mutation.

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

    estimator: :class:`~sklearn_genetic.GASearchCV`, default = None
        Estimator that is being optimized

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

    stored_exception = None
    callbacks_start_args = {
        "callbacks": callbacks,
        "record": None,
        "logbook": None,
        "estimator": estimator,
        "method": "on_start",
    }

    eval_callbacks(**callbacks_start_args)

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
    if isinstance(record["fitness"], np.ndarray):
        record = {key: value[0] for key, value in record.items()}

    n_gen = gen = 0
    logbook.record(gen=n_gen, nevals=len(invalid_ind), **record)

    if verbose:
        print(logbook.stream)

    callbacks_step_args = {
        "callbacks": callbacks,
        "record": record,
        "logbook": logbook,
        "estimator": estimator,
        "method": "on_step",
    }

    # Check if any of the callbacks conditions are True to stop the iteration
    if eval_callbacks(**callbacks_step_args):
        callbacks_end_args = {
            "callbacks": callbacks,
            "record": None,
            "logbook": logbook,
            "estimator": estimator,
            "method": "on_end",
        }

        eval_callbacks(**callbacks_end_args)
        print("INFO: Stopping the algorithm")
        return population, logbook, n_gen

    for gen in range(1, ngen + 1):
        try:
            # Vary the population
            offspring = varOr(population, toolbox, lambda_, cxpb.step(), mutpb.step())

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
            if isinstance(record["fitness"], np.ndarray):
                record = {key: value[0] for key, value in record.items()}

            logbook.record(gen=gen, nevals=len(invalid_ind), **record)

            if verbose:
                print(logbook.stream)

            callbacks_step_args = {
                "callbacks": callbacks,
                "record": record,
                "logbook": logbook,
                "estimator": estimator,
                "method": "on_step",
            }

            # Check if any of the callbacks conditions are True to stop the iteration
            if eval_callbacks(**callbacks_step_args) or stored_exception:
                if stored_exception:
                    print(
                        f"{stored_exception}\nsklearn-genetic-opt closed prematurely. Will use the current best model."
                    )
                print("INFO: Stopping the algorithm")
                break

        except (KeyboardInterrupt, SystemExit, StopIteration) as e:
            stored_exception = e

    n_gen = gen + 1

    callbacks_end_args = {
        "callbacks": callbacks,
        "record": None,
        "logbook": logbook,
        "estimator": estimator,
        "method": "on_end",
    }

    eval_callbacks(**callbacks_end_args)

    return population, logbook, n_gen


algorithms_factory = {
    "eaSimple": eaSimple,
    "eaMuPlusLambda": eaMuPlusLambda,
    "eaMuCommaLambda": eaMuCommaLambda,
}
