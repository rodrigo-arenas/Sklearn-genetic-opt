import numpy as np
from deap import tools
from deap.algorithms import varAnd, varOr

from .callbacks.validations import eval_callbacks

TELEMETRY_FIELDS = [
    "population_size",
    "unique_individuals",
    "unique_individual_ratio",
    "genotype_diversity",
    "fitness_improvement",
    "fitness_improved",
    "stagnation_generations",
    "best_generation",
]


def _evaluate_invalid_individuals(toolbox, invalid_individuals):
    if hasattr(toolbox, "evaluate_population"):
        return toolbox.evaluate_population(invalid_individuals)

    return toolbox.map(toolbox.evaluate, invalid_individuals)


def _fitness_scalar(value):
    if isinstance(value, np.ndarray):
        return value.flat[0]

    if isinstance(value, (list, tuple)):
        return value[0]

    return value


def _flatten_record(record):
    return {key: _fitness_scalar(value) for key, value in record.items()}


def _individual_key(individual):
    return tuple(individual)


def _population_diversity(population):
    population_size = len(population)
    if population_size == 0:
        return {
            "population_size": 0,
            "unique_individuals": 0,
            "unique_individual_ratio": 0.0,
            "genotype_diversity": 0.0,
        }

    unique_individuals = len({_individual_key(individual) for individual in population})
    unique_individual_ratio = unique_individuals / population_size

    if population_size == 1:
        genotype_diversity = 0.0
    else:
        gene_diversities = []
        for gene_values in zip(*population):
            unique_gene_values = len(set(gene_values))
            gene_diversities.append((unique_gene_values - 1) / (population_size - 1))
        genotype_diversity = float(np.mean(gene_diversities)) if gene_diversities else 0.0

    return {
        "population_size": population_size,
        "unique_individuals": unique_individuals,
        "unique_individual_ratio": unique_individual_ratio,
        "genotype_diversity": genotype_diversity,
    }


def _compile_generation_record(stats, population, state, gen):
    record = stats.compile(population) if stats else {}
    record = _flatten_record(record)

    record.update(_population_diversity(population))

    current_best = record.get("fitness_max")
    if current_best is None:
        fitness_improvement = 0.0
        fitness_improved = False
    elif state["best_fitness"] is None or current_best > state["best_fitness"]:
        fitness_improvement = (
            0.0 if state["best_fitness"] is None else current_best - state["best_fitness"]
        )
        fitness_improved = True
        state["best_fitness"] = current_best
        state["best_generation"] = gen
        state["stagnation_generations"] = 0
    else:
        fitness_improvement = current_best - state["best_fitness"]
        fitness_improved = False
        state["stagnation_generations"] += 1

    record.update(
        {
            "fitness_improvement": fitness_improvement,
            "fitness_improved": fitness_improved,
            "stagnation_generations": state["stagnation_generations"],
            "best_generation": state["best_generation"],
        }
    )

    return record


def _new_telemetry_state():
    return {
        "best_fitness": None,
        "best_generation": 0,
        "stagnation_generations": 0,
    }


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

    telemetry_state = _new_telemetry_state()
    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + (stats.fields if stats else []) + TELEMETRY_FIELDS

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = _evaluate_invalid_individuals(toolbox, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)
    hof_size = len(halloffame.items) if (halloffame.items and estimator.elitism) else 0

    n_gen = gen = 0
    record = _compile_generation_record(stats, population, telemetry_state, gen)
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
            fitnesses = _evaluate_invalid_individuals(toolbox, invalid_ind)
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
            record = _compile_generation_record(stats, population, telemetry_state, gen)
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
    r"""
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

    telemetry_state = _new_telemetry_state()
    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + (stats.fields if stats else []) + TELEMETRY_FIELDS

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = _evaluate_invalid_individuals(toolbox, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    n_gen = gen = 0
    record = _compile_generation_record(stats, population, telemetry_state, gen)
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
            fitnesses = _evaluate_invalid_individuals(toolbox, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Update the hall of fame with the generated individuals
            if halloffame is not None:
                halloffame.update(offspring)

            # Select the next generation population
            population[:] = toolbox.select(population + offspring, mu)

            # Update the statistics with the new population
            record = _compile_generation_record(stats, population, telemetry_state, gen)
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
    r"""
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
    fitnesses = _evaluate_invalid_individuals(toolbox, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    telemetry_state = _new_telemetry_state()
    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + (stats.fields if stats else []) + TELEMETRY_FIELDS

    n_gen = gen = 0
    record = _compile_generation_record(stats, population, telemetry_state, gen)
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
            fitnesses = _evaluate_invalid_individuals(toolbox, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Update the hall of fame with the generated individuals
            if halloffame is not None:
                halloffame.update(offspring)

            # Select the next generation population
            population[:] = toolbox.select(offspring, mu)

            # Update the statistics with the new population
            record = _compile_generation_record(stats, population, telemetry_state, gen)
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
