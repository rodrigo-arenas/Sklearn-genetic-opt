import numpy as np
from deap import tools
from deap.algorithms import varAnd, varOr

from .callbacks.validations import eval_callbacks
from .optimizer_control import (
    inject_random_immigrants,
    local_neighbor,
    local_search_enabled,
    mutation_probability,
    replace_duplicate_candidates,
    shared_fitness,
)

TELEMETRY_FIELDS = [
    "fitness_best",
    "population_size",
    "unique_individuals",
    "unique_individual_ratio",
    "genotype_diversity",
    "fitness_improvement",
    "fitness_improved",
    "stagnation_generations",
    "best_generation",
    "mutation_probability",
    "selection_pressure",
    "diversity_control_triggered",
    "random_immigrants",
    "duplicate_replacements",
    "local_refinements",
    "fitness_sharing_applied",
    "mean_niche_count",
    "max_niche_count",
]


VERBOSE_COLUMNS = [
    ("gen", "gen", 4, "int"),
    ("nevals", "evals", 5, "int"),
    ("fitness", "avg", 13, "score"),
    ("fitness_best", "best", 13, "score"),
    ("genotype_diversity", "div", 7, "ratio"),
    ("unique_individual_ratio", "unique", 7, "ratio"),
    ("stagnation_generations", "stag", 5, "int"),
    ("mutation_probability", "mut", 7, "ratio"),
    ("selection_pressure", "sel", 5, "int"),
    ("events", "events", 18, "text"),
]


def _format_verbose_value(value, value_type, width):
    if value is None:
        return "-".rjust(width)

    if isinstance(value, (bool, np.bool_)):
        return ("yes" if value else "no").rjust(width)

    if value_type == "text":
        return str(value)[:width].ljust(width)

    if value_type == "int":
        return f"{int(value):>{width}d}"

    if value_type == "score":
        return f"{float(value):>{width}.5f}"

    if value_type == "ratio":
        return f"{float(value):>{width}.3f}"

    return str(value)[:width].rjust(width)


def _verbose_events(record):
    events = []

    if record.get("diversity_control_triggered"):
        events.append("div")

    random_immigrants = record.get("random_immigrants", 0)
    if random_immigrants:
        events.append(f"imm={random_immigrants}")

    duplicate_replacements = record.get("duplicate_replacements", 0)
    if duplicate_replacements:
        events.append(f"dup={duplicate_replacements}")

    local_refinements = record.get("local_refinements", 0)
    if local_refinements:
        events.append(f"local={local_refinements}")

    if record.get("fitness_sharing_applied"):
        events.append("share")

    return ",".join(events) if events else "-"


def _print_verbose_record(gen, nevals, record, print_header=False):
    display_record = {
        "gen": gen,
        "nevals": nevals,
        "events": _verbose_events(record),
        **record,
    }

    if print_header:
        header = " ".join(label.rjust(width) for _, label, width, _ in VERBOSE_COLUMNS)
        separator = " ".join("-" * width for _, _, width, _ in VERBOSE_COLUMNS)
        print(header)
        print(separator)

    row = " ".join(
        _format_verbose_value(display_record.get(key), value_type, width)
        for key, _, width, value_type in VERBOSE_COLUMNS
    )
    print(row)


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


def _primary_fitness_weight(population):
    if not population:
        return 1.0

    return population[0].fitness.weights[0]


def _is_fitness_improvement(candidate, best, primary_weight):
    if best is None:
        return True

    if primary_weight >= 0:
        return candidate > best

    return candidate < best


def _fitness_delta(candidate, best, primary_weight):
    if best is None:
        return 0.0

    delta = candidate - best
    return delta if primary_weight >= 0 else -delta


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


def _compile_generation_record(stats, population, state, gen, control_record=None):
    record = stats.compile(population) if stats else {}
    record = _flatten_record(record)

    record.update(_population_diversity(population))

    primary_weight = _primary_fitness_weight(population)
    current_best_key = "fitness_max" if primary_weight >= 0 else "fitness_min"
    current_best = record.get(current_best_key)
    if current_best is None:
        fitness_improvement = 0.0
        fitness_improved = False
    elif _is_fitness_improvement(current_best, state["best_fitness"], primary_weight):
        fitness_improvement = _fitness_delta(current_best, state["best_fitness"], primary_weight)
        fitness_improved = True
        state["best_fitness"] = current_best
        state["best_generation"] = gen
        state["stagnation_generations"] = 0
    else:
        fitness_improvement = _fitness_delta(current_best, state["best_fitness"], primary_weight)
        fitness_improved = False
        state["stagnation_generations"] += 1

    record.update(
        {
            "fitness_best": state["best_fitness"],
            "fitness_improvement": fitness_improvement,
            "fitness_improved": fitness_improved,
            "stagnation_generations": state["stagnation_generations"],
            "best_generation": state["best_generation"],
        }
    )
    record.update(
        {
            "mutation_probability": None,
            "selection_pressure": None,
            "diversity_control_triggered": False,
            "random_immigrants": 0,
            "duplicate_replacements": 0,
            "local_refinements": 0,
            "fitness_sharing_applied": False,
            "mean_niche_count": 0.0,
            "max_niche_count": 0.0,
        }
    )

    if control_record is not None:
        record.update(control_record)

    return record


def _refresh_last_record_after_local_refinement(
    logbook, stats, population, state, local_refinements
):
    if not local_refinements or len(logbook) == 0:
        return

    previous_record = dict(logbook[-1])
    gen = previous_record.get("gen", len(logbook) - 1)
    control_record = {
        field: previous_record.get(field)
        for field in TELEMETRY_FIELDS
        if field
        not in {
            "fitness_best",
            "population_size",
            "unique_individuals",
            "unique_individual_ratio",
            "genotype_diversity",
            "fitness_improvement",
            "fitness_improved",
            "stagnation_generations",
            "best_generation",
            "local_refinements",
        }
    }
    control_record["local_refinements"] = local_refinements

    refreshed_record = _compile_generation_record(
        stats, population, state.copy(), gen, control_record
    )
    logbook[-1].clear()
    logbook[-1].update(
        {
            "gen": previous_record.get("gen"),
            "nevals": previous_record.get("nevals"),
            **refreshed_record,
        }
    )


def _seed_logbook(resume_log, stats):
    """Continue generation numbering across a checkpoint resume (see #299).

    Without this, every resumed run restarts its local ``gen`` counter at 0,
    so the logbook/history end up with duplicate generation indices (e.g.
    ``0..N`` from the first run followed by another ``0..M`` after resume)
    instead of a single, monotonically increasing sequence.
    """
    if resume_log is not None and len(resume_log) > 0:
        logbook = resume_log
        gen_offset = max(logbook.select("gen")) + 1
    else:
        logbook = tools.Logbook()
        gen_offset = 0
    logbook.header = ["gen", "nevals"] + (stats.fields if stats else []) + TELEMETRY_FIELDS
    return logbook, gen_offset


def _new_telemetry_state():
    return {
        "best_fitness": None,
        "best_generation": 0,
        "stagnation_generations": 0,
    }


def _record_optimizer_control_stats(estimator, random_immigrants=0, local_refinements=0):
    if estimator is None or not hasattr(estimator, "fit_stats_"):
        return

    estimator.fit_stats_["random_immigrants"] += random_immigrants
    estimator.fit_stats_["local_refinement_candidates"] += local_refinements


def _run_local_refinement(population, toolbox, halloffame, estimator):
    if not local_search_enabled(estimator) or halloffame is None or len(halloffame.items) == 0:
        return 0

    top_k = min(estimator.local_search_top_k, len(halloffame.items))
    neighbors = []

    for parent in halloffame.items[:top_k]:
        for _ in range(estimator.local_search_steps):
            neighbor = local_neighbor(estimator, parent, parent.__class__)
            neighbors.append(neighbor)

    fitnesses = _evaluate_invalid_individuals(toolbox, neighbors)
    for neighbor, fitness in zip(neighbors, fitnesses):
        neighbor.fitness.values = fitness

    if neighbors:
        halloffame.update(neighbors)
        population.extend(neighbors)
        population[:] = tools.selBest(population, len(population) - len(neighbors))

    _record_optimizer_control_stats(estimator, local_refinements=len(neighbors))
    return len(neighbors)


def _control_record(
    mutation_prob,
    selection_pressure,
    diversity_triggered,
    random_immigrants,
    duplicate_replacements,
    sharing_record=None,
):
    sharing_record = sharing_record or {}
    return {
        "mutation_probability": mutation_prob,
        "selection_pressure": selection_pressure,
        "diversity_control_triggered": diversity_triggered or random_immigrants > 0,
        "random_immigrants": random_immigrants,
        "duplicate_replacements": duplicate_replacements,
        "local_refinements": 0,
        "fitness_sharing_applied": sharing_record.get("fitness_sharing_applied", False),
        "mean_niche_count": sharing_record.get("mean_niche_count", 0.0),
        "max_niche_count": sharing_record.get("max_niche_count", 0.0),
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
    resume_log=None,
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
    logbook, gen_offset = _seed_logbook(resume_log, stats)

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
    logbook.record(gen=n_gen + gen_offset, nevals=len(invalid_ind), **record)

    if verbose:
        _print_verbose_record(n_gen + gen_offset, len(invalid_ind), record, print_header=True)

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
        return population, logbook, n_gen + gen_offset

    for gen in range(1, ngen + 1):
        try:
            # Select the next generation individuals
            if estimator is not None:
                estimator._last_generation_record = record
            with shared_fitness(population, estimator) as sharing_record:
                offspring = toolbox.select(population, len(population) - hof_size)
            selection_pressure = getattr(estimator, "_selection_pressure_", None)

            mutation_prob, diversity_triggered = mutation_probability(mutpb, estimator, record)

            crossover_prob = cxpb.step()

            # Vary the pool of individuals
            offspring = varAnd(offspring, toolbox, crossover_prob, mutation_prob)
            duplicate_replacements = replace_duplicate_candidates(
                offspring, toolbox, estimator, reference_population=population
            )
            random_immigrants = inject_random_immigrants(offspring, toolbox, estimator, record)
            _record_optimizer_control_stats(estimator, random_immigrants=random_immigrants)

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
            record = _compile_generation_record(
                stats,
                population,
                telemetry_state,
                gen,
                _control_record(
                    mutation_prob,
                    selection_pressure,
                    diversity_triggered,
                    random_immigrants,
                    duplicate_replacements,
                    sharing_record,
                ),
            )
            logbook.record(gen=gen + gen_offset, nevals=len(invalid_ind), **record)

            if verbose:
                _print_verbose_record(gen + gen_offset, len(invalid_ind), record)

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

    n_gen = gen + 1 + gen_offset
    local_refinements = _run_local_refinement(population, toolbox, halloffame, estimator)
    _refresh_last_record_after_local_refinement(
        logbook, stats, population, telemetry_state, local_refinements
    )

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
    resume_log=None,
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
    logbook, gen_offset = _seed_logbook(resume_log, stats)

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = _evaluate_invalid_individuals(toolbox, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    n_gen = gen = 0
    record = _compile_generation_record(stats, population, telemetry_state, gen)
    logbook.record(gen=n_gen + gen_offset, nevals=len(invalid_ind), **record)

    if verbose:
        _print_verbose_record(n_gen + gen_offset, len(invalid_ind), record, print_header=True)

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
        return population, logbook, n_gen + gen_offset

    for gen in range(1, ngen + 1):
        try:
            mutation_prob, diversity_triggered = mutation_probability(mutpb, estimator, record)

            crossover_prob = cxpb.step()
            mutation_prob = min(mutation_prob, max(0.0, 1.0 - crossover_prob))

            # Vary the population
            offspring = varOr(population, toolbox, lambda_, crossover_prob, mutation_prob)
            duplicate_replacements = replace_duplicate_candidates(
                offspring, toolbox, estimator, reference_population=population
            )
            random_immigrants = inject_random_immigrants(offspring, toolbox, estimator, record)
            _record_optimizer_control_stats(estimator, random_immigrants=random_immigrants)

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = _evaluate_invalid_individuals(toolbox, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Update the hall of fame with the generated individuals
            if halloffame is not None:
                halloffame.update(offspring)

            # Select the next generation population
            selection_pool = population + offspring
            if estimator is not None:
                estimator._last_generation_record = record
            with shared_fitness(selection_pool, estimator) as sharing_record:
                population[:] = toolbox.select(selection_pool, mu)
            selection_pressure = getattr(estimator, "_selection_pressure_", None)

            # Update the statistics with the new population
            record = _compile_generation_record(
                stats,
                population,
                telemetry_state,
                gen,
                _control_record(
                    mutation_prob,
                    selection_pressure,
                    diversity_triggered,
                    random_immigrants,
                    duplicate_replacements,
                    sharing_record,
                ),
            )
            logbook.record(gen=gen + gen_offset, nevals=len(invalid_ind), **record)

            if verbose:
                _print_verbose_record(gen + gen_offset, len(invalid_ind), record)

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

    n_gen = gen + 1 + gen_offset
    local_refinements = _run_local_refinement(population, toolbox, halloffame, estimator)
    _refresh_last_record_after_local_refinement(
        logbook, stats, population, telemetry_state, local_refinements
    )

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
    resume_log=None,
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
    logbook, gen_offset = _seed_logbook(resume_log, stats)

    n_gen = gen = 0
    record = _compile_generation_record(stats, population, telemetry_state, gen)
    logbook.record(gen=n_gen + gen_offset, nevals=len(invalid_ind), **record)

    if verbose:
        _print_verbose_record(n_gen + gen_offset, len(invalid_ind), record, print_header=True)

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
        return population, logbook, n_gen + gen_offset

    for gen in range(1, ngen + 1):
        try:
            mutation_prob, diversity_triggered = mutation_probability(mutpb, estimator, record)

            crossover_prob = cxpb.step()
            mutation_prob = min(mutation_prob, max(0.0, 1.0 - crossover_prob))

            # Vary the population
            offspring = varOr(population, toolbox, lambda_, crossover_prob, mutation_prob)
            duplicate_replacements = replace_duplicate_candidates(
                offspring, toolbox, estimator, reference_population=population
            )
            random_immigrants = inject_random_immigrants(offspring, toolbox, estimator, record)
            _record_optimizer_control_stats(estimator, random_immigrants=random_immigrants)

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = _evaluate_invalid_individuals(toolbox, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Update the hall of fame with the generated individuals
            if halloffame is not None:
                halloffame.update(offspring)

            # Select the next generation population
            if estimator is not None:
                estimator._last_generation_record = record
            with shared_fitness(offspring, estimator) as sharing_record:
                population[:] = toolbox.select(offspring, mu)
            selection_pressure = getattr(estimator, "_selection_pressure_", None)

            # Update the statistics with the new population
            record = _compile_generation_record(
                stats,
                population,
                telemetry_state,
                gen,
                _control_record(
                    mutation_prob,
                    selection_pressure,
                    diversity_triggered,
                    random_immigrants,
                    duplicate_replacements,
                    sharing_record,
                ),
            )
            logbook.record(gen=gen + gen_offset, nevals=len(invalid_ind), **record)

            if verbose:
                _print_verbose_record(gen + gen_offset, len(invalid_ind), record)

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

    n_gen = gen + 1 + gen_offset
    local_refinements = _run_local_refinement(population, toolbox, halloffame, estimator)
    _refresh_last_record_after_local_refinement(
        logbook, stats, population, telemetry_state, local_refinements
    )

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
