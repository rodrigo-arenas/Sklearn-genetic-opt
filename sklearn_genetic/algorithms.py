import numpy as np
from deap import tools
from deap.algorithms import varAnd, varOr

from .callbacks.validations import eval_callbacks
from .evaluation import _evaluation_limit_reached, candidate_records, start_evaluation_callbacks
from ._telemetry import (
    TELEMETRY_FIELDS,
    _population_diversity,
    feature_population_diversity as feature_population_diversity,
    _new_telemetry_state,
    _print_verbose_record,
    complete_generation,
)
from .optimizer_control import (
    differential_offspring,
    finite_search_space_size,
    inject_random_immigrants,
    local_neighbor,
    local_search_enabled,
    mutation_probability,
    replace_duplicate_candidates,
    shared_fitness,
)


def _evaluate_invalid_individuals(toolbox, invalid_individuals):
    if hasattr(toolbox, "evaluate_population"):
        return toolbox.evaluate_population(invalid_individuals)

    return toolbox.map(toolbox.evaluate, invalid_individuals)


def _evaluate_generation(toolbox, population, *, evaluate_batch=None):
    invalid = [individual for individual in population if not individual.fitness.valid]
    requested = len(invalid)
    if evaluate_batch is None:
        fitnesses = _evaluate_invalid_individuals(toolbox, invalid)
    else:
        fitnesses = evaluate_batch(invalid)
    for individual, fitness in zip(invalid, fitnesses):
        individual.fitness.values = fitness
    nevals = len(invalid)
    partial = nevals < requested
    # Leave an unstarted generation untouched when the budget cuts the whole batch.
    if nevals or not partial:
        population[:] = [individual for individual in population if individual.fitness.valid]
    return nevals, partial


def _fitness_scalar(value):
    if isinstance(value, np.ndarray):
        return value.flat[0]

    if isinstance(value, (list, tuple)):
        return value[0]

    return value


def _flatten_record(record):
    return {key: _fitness_scalar(value) for key, value in record.items()}


def _primary_fitness_weight(population):
    if not population:
        return 1.0

    return population[0].fitness.weights[0]


def _compile_generation_record(
    stats, population, state, gen, control_record=None, halloffame=None, *, partial=False
):
    record = stats.compile(population) if stats else {}
    record = _flatten_record(record)
    if partial:
        record["partial"] = True

    diversity = getattr(stats, "population_diversity", _population_diversity)
    record.update(diversity(population))

    primary_weight = _primary_fitness_weight(population)
    current_best_key = "fitness_max" if primary_weight >= 0 else "fitness_min"
    current_best = (
        halloffame[0].fitness.values[0]
        if halloffame is not None and len(halloffame)
        else record.get(current_best_key)
    )
    return complete_generation(record, state, gen, primary_weight, current_best, control_record)


def _refresh_last_record_after_local_refinement(
    logbook, stats, population, state, local_refinements, halloffame
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
        stats,
        population,
        state.copy(),
        gen,
        control_record,
        halloffame=halloffame,
        partial=previous_record.get("partial", False),
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


def _initialize_population(population, toolbox, callbacks, estimator, resume_log, halloffame):
    start_evaluation_callbacks(callbacks, estimator)
    if resume_log and halloffame:
        timed = bool(getattr(estimator, "_evaluation_timers", ()))
        verified = getattr(estimator, "_deap_checkpoint_data_hash", None) is not None
        check_coverage = (
            verified
            and estimator.use_cache
            and not callable(estimator.refit)
            and not (estimator.refit and estimator.final_selection)
        )
        if (timed or check_coverage) and any(
            row.get("fit_time") is not None for row in candidate_records(estimator)
        ):
            if check_coverage and _cached_space_is_exhausted(estimator):
                _finish_resume_without_search(callbacks, estimator, resume_log, exhausted=True)
                return None
            # Usable checkpoint results already satisfy the initial-batch allowance.
            if timed:
                estimator._evaluation_started = True
                if _evaluation_limit_reached(estimator):
                    _finish_resume_without_search(callbacks, estimator, resume_log)
                    return None
    nevals, partial = _evaluate_generation(
        toolbox, population, evaluate_batch=getattr(toolbox, "_evaluate_initial_masks", None)
    )
    if resume_log and partial and not nevals:
        # The deadline may have expired between the preflight and evaluation.
        _finish_resume_without_search(callbacks, estimator, resume_log)
        return None
    return nevals, partial


def _cached_space_is_exhausted(estimator):
    cached = len(getattr(estimator, "fitness_cache", {}))
    return finite_search_space_size(estimator, cached) == cached


def _finish_resume_without_search(callbacks, estimator, resume_log, *, exhausted=None):
    if estimator is not None:
        # No new generation will report coverage. The restored cache can still
        # establish that every candidate in the finite space was evaluated.
        estimator.search_space_exhausted_ = (
            _cached_space_is_exhausted(estimator) if exhausted is None else exhausted
        )
    eval_callbacks(callbacks, None, resume_log, estimator, "on_end")


def _record_optimizer_control_stats(estimator, random_immigrants=0, local_refinements=0):
    if estimator is None or not hasattr(estimator, "fit_stats_"):
        return

    estimator.fit_stats_["random_immigrants"] += random_immigrants
    estimator.fit_stats_["local_refinement_candidates"] += local_refinements


def _report_interruption(error):
    print(f"{error}\nsklearn-genetic-opt closed prematurely. Will use the current best model.")


def _finite_search_size(estimator, halloffame, proposal_budget):
    if halloffame is None:
        return None
    if local_search_enabled(estimator):
        proposal_budget += estimator.local_search_top_k * estimator.local_search_steps
    return finite_search_space_size(estimator, proposal_budget)


def _run_local_refinement(population, toolbox, halloffame, estimator, space_size, seen_candidates):
    if not local_search_enabled(estimator) or halloffame is None or len(halloffame.items) == 0:
        return 0

    top_k = min(estimator.local_search_top_k, len(halloffame.items))
    search_only = not getattr(estimator, "refit", True)
    count = 0
    for _ in range(estimator.local_search_steps):
        if _evaluation_limit_reached(estimator):
            break
        # Preserve the candidate history and RNG stream used by finalist selection
        # and refit. A search-only run cannot add observations after coverage.
        if search_only and len(seen_candidates) == space_size:
            break
        # Each step starts from the current elites so accepted improvements
        # guide the next neighborhood. Evaluate independent elites as a batch.
        neighbors = [
            local_neighbor(estimator, parent, parent.__class__)
            for parent in halloffame.items[:top_k]
        ]
        fitnesses = _evaluate_invalid_individuals(toolbox, neighbors)
        for neighbor, fitness in zip(neighbors, fitnesses):
            neighbor.fitness.values = fitness
        halloffame.update(neighbors)
        _update_search_coverage(estimator, neighbors, space_size, seen_candidates)
        population.extend(neighbors)
        population[:] = tools.selBest(population, len(population) - len(neighbors))
        count += len(neighbors)

    _record_optimizer_control_stats(estimator, local_refinements=count)
    return count


def _select_survivors(candidates, toolbox, size, halloffame, estimator):
    elites = (
        halloffame.items[:size]
        if getattr(estimator, "elitism", False) and halloffame is not None
        else []
    )
    with shared_fitness(candidates, estimator) as sharing_record:
        survivors = toolbox.select(candidates, size - len(elites))
    survivors.extend(toolbox.clone(elite) for elite in elites)
    return survivors, sharing_record


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


def _update_search_coverage(estimator, candidates, size, seen):
    # Only proposals observed by this run count. A restored cache can include
    # configurations that the current hall of fame has never encountered.
    if size is not None:
        seen.update(estimator._individual_key(individual) for individual in candidates)
    exhausted = size is not None and len(seen) == size
    if estimator is not None:
        estimator.search_space_exhausted_ = exhausted
    return exhausted


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
    proposal_budget = len(population) * (ngen + 1)
    initial = _initialize_population(
        population, toolbox, callbacks, estimator, resume_log, halloffame
    )
    if initial is None:
        return population, resume_log, len(resume_log)
    nevals, partial = initial
    space_size = _finite_search_size(estimator, halloffame, proposal_budget)
    seen_candidates = set()

    telemetry_state = _new_telemetry_state()
    logbook, gen_offset = _seed_logbook(resume_log, stats)

    if halloffame is not None:
        halloffame.update(population)

    exhausted = _update_search_coverage(estimator, population, space_size, seen_candidates)
    n_gen = gen = 0
    record = _compile_generation_record(
        stats,
        population,
        telemetry_state,
        gen,
        halloffame=halloffame,
        partial=partial,
    )
    logbook.record(gen=n_gen + gen_offset, nevals=nevals, **record)

    if verbose:
        _print_verbose_record(n_gen + gen_offset, nevals, record, print_header=True)

    # Check if any of the callbacks conditions are True to stop the iteration

    callbacks_step_args = {
        "callbacks": callbacks,
        "record": record,
        "logbook": logbook,
        "estimator": estimator,
        "method": "on_step",
    }

    if eval_callbacks(**callbacks_step_args) or exhausted or _evaluation_limit_reached(estimator):
        callbacks_end_args = {
            "callbacks": callbacks,
            "record": None,
            "logbook": logbook,
            "estimator": estimator,
            "method": "on_end",
        }

        # Call ending callback
        eval_callbacks(**callbacks_end_args)
        if not exhausted or verbose:
            print("INFO: Stopping the algorithm")
        return population, logbook, n_gen + gen_offset

    stopped_by_callback = False
    try:
        for gen in range(1, ngen + 1):
            # Select the next generation individuals
            if estimator is not None:
                estimator._last_generation_record = record
            elites = (
                halloffame.items[: len(population)]
                if halloffame is not None and getattr(estimator, "elitism", False)
                else []
            )
            with shared_fitness(population, estimator) as sharing_record:
                offspring = toolbox.select(population, len(population) - len(elites))
            selection_pressure = getattr(estimator, "_selection_pressure_", None)

            mutation_prob, diversity_triggered = mutation_probability(mutpb, estimator, record)

            crossover_prob = cxpb.step()

            # Vary the pool of individuals
            offspring = varAnd(offspring, toolbox, crossover_prob, mutation_prob)
            duplicate_replacements = replace_duplicate_candidates(
                offspring, toolbox, estimator, reference_population=population
            )
            random_immigrants = inject_random_immigrants(offspring, toolbox, estimator, record)

            nevals, partial = _evaluate_generation(toolbox, offspring)
            if partial and not nevals:
                break
            _record_optimizer_control_stats(estimator, random_immigrants=random_immigrants)

            offspring.extend(elites)

            # Update the hall of fame with the generated individuals
            if halloffame is not None:
                halloffame.update(offspring)

            # Replace the current population by the offspring
            exhausted = _update_search_coverage(estimator, offspring, space_size, seen_candidates)
            # A fully evaluated generation still updates survivors at the deadline.
            if not partial:
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
                halloffame=halloffame,
                partial=partial,
            )
            logbook.record(gen=gen + gen_offset, nevals=nevals, **record)

            if verbose:
                _print_verbose_record(gen + gen_offset, nevals, record)

            callbacks_step_args = {
                "callbacks": callbacks,
                "record": record,
                "logbook": logbook,
                "estimator": estimator,
                "method": "on_step",
            }

            # Check if any of the callbacks conditions are True to stop the iteration
            stopped_by_callback = eval_callbacks(**callbacks_step_args)
            if stopped_by_callback or exhausted or _evaluation_limit_reached(estimator):
                if not exhausted or verbose:
                    print("INFO: Stopping the algorithm")
                break
    except (KeyboardInterrupt, SystemExit, StopIteration) as error:
        _report_interruption(error)
    else:
        if not stopped_by_callback:
            local_refinements = _run_local_refinement(
                population, toolbox, halloffame, estimator, space_size, seen_candidates
            )
            _refresh_last_record_after_local_refinement(
                logbook, stats, population, telemetry_state, local_refinements, halloffame
            )

    n_gen = len(logbook)

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
    differential = kwargs.get("_differential_evolution", False)
    if differential and (estimator is None or len(population) < 4):
        raise ValueError("Differential evolution requires an estimator and population_size >= 4")
    proposal_budget = len(population) + ngen * lambda_
    initial = _initialize_population(
        population, toolbox, callbacks, estimator, resume_log, halloffame
    )
    if initial is None:
        return population, resume_log, len(resume_log)
    nevals, partial = initial
    space_size = _finite_search_size(estimator, halloffame, proposal_budget)
    seen_candidates = set()

    telemetry_state = _new_telemetry_state()
    logbook, gen_offset = _seed_logbook(resume_log, stats)

    if halloffame is not None:
        halloffame.update(
            [ind for ind in population if not np.isnan(ind.fitness.values[0])]
            if differential
            else population
        )

    exhausted = _update_search_coverage(estimator, population, space_size, seen_candidates)
    n_gen = gen = 0
    record = _compile_generation_record(
        stats,
        population,
        telemetry_state,
        gen,
        halloffame=halloffame,
        partial=partial,
    )
    logbook.record(gen=n_gen + gen_offset, nevals=nevals, **record)

    if verbose:
        _print_verbose_record(n_gen + gen_offset, nevals, record, print_header=True)

    # Check if any of the callbacks conditions are True to stop the iteration
    callbacks_step_args = {
        "callbacks": callbacks,
        "record": record,
        "logbook": logbook,
        "estimator": estimator,
        "method": "on_step",
    }

    if eval_callbacks(**callbacks_step_args) or exhausted or _evaluation_limit_reached(estimator):
        # Call ending callback
        callbacks_end_args = {
            "callbacks": callbacks,
            "record": None,
            "logbook": logbook,
            "estimator": estimator,
            "method": "on_end",
        }

        eval_callbacks(**callbacks_end_args)
        if not exhausted or verbose:
            print("INFO: Stopping the algorithm")
        return population, logbook, n_gen + gen_offset

    stopped_by_callback = False
    try:
        for gen in range(1, ngen + 1):
            mutation_prob, diversity_triggered = mutation_probability(mutpb, estimator, record)

            crossover_prob = cxpb.step()
            if differential:
                offspring = differential_offspring(
                    population, toolbox, estimator, crossover_prob, mutation_prob
                )
                duplicate_replacements = 0
            else:
                mutation_prob = min(mutation_prob, max(0.0, 1.0 - crossover_prob))
                offspring = varOr(population, toolbox, lambda_, crossover_prob, mutation_prob)
                duplicate_replacements = replace_duplicate_candidates(
                    offspring, toolbox, estimator, reference_population=population
                )
            random_immigrants = inject_random_immigrants(offspring, toolbox, estimator, record)

            nevals, partial = _evaluate_generation(toolbox, offspring)
            if partial and not nevals:
                break
            _record_optimizer_control_stats(estimator, random_immigrants=random_immigrants)

            # Update the hall of fame with the generated individuals
            if halloffame is not None:
                halloffame.update(
                    [ind for ind in offspring if not np.isnan(ind.fitness.values[0])]
                    if differential
                    else offspring
                )

            exhausted = _update_search_coverage(estimator, offspring, space_size, seen_candidates)

            # Select the next generation population
            if estimator is not None:
                estimator._last_generation_record = record
            if partial:
                sharing_record, selection_pressure = {}, None
            elif differential:
                population[:] = [
                    (
                        trial
                        if not np.isnan(trial.fitness.values[0])
                        and (np.isnan(target.fitness.values[0]) or trial.fitness > target.fitness)
                        else target
                    )
                    for target, trial in zip(population, offspring)
                ]
                sharing_record, selection_pressure = {}, None
            else:
                population[:], sharing_record = _select_survivors(
                    population + offspring, toolbox, mu, halloffame, estimator
                )
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
                halloffame=halloffame,
                partial=partial,
            )
            logbook.record(gen=gen + gen_offset, nevals=nevals, **record)

            if verbose:
                _print_verbose_record(gen + gen_offset, nevals, record)

            callbacks_step_args = {
                "callbacks": callbacks,
                "record": record,
                "logbook": logbook,
                "estimator": estimator,
                "method": "on_step",
            }

            stopped_by_callback = eval_callbacks(**callbacks_step_args)
            if stopped_by_callback or exhausted or _evaluation_limit_reached(estimator):
                if not exhausted or verbose:
                    print("INFO: Stopping the algorithm")
                break

    except (KeyboardInterrupt, SystemExit, StopIteration) as error:
        _report_interruption(error)
    else:
        if not stopped_by_callback:
            local_refinements = _run_local_refinement(
                population, toolbox, halloffame, estimator, space_size, seen_candidates
            )
            _refresh_last_record_after_local_refinement(
                logbook, stats, population, telemetry_state, local_refinements, halloffame
            )

    n_gen = len(logbook)

    callbacks_end_args = {
        "callbacks": callbacks,
        "record": None,
        "logbook": logbook,
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

    proposal_budget = len(population) + ngen * lambda_
    initial = _initialize_population(
        population, toolbox, callbacks, estimator, resume_log, halloffame
    )
    if initial is None:
        return population, resume_log, len(resume_log)
    nevals, partial = initial
    space_size = _finite_search_size(estimator, halloffame, proposal_budget)
    seen_candidates = set()

    if halloffame is not None:
        halloffame.update(population)

    telemetry_state = _new_telemetry_state()
    logbook, gen_offset = _seed_logbook(resume_log, stats)

    exhausted = _update_search_coverage(estimator, population, space_size, seen_candidates)
    n_gen = gen = 0
    record = _compile_generation_record(
        stats,
        population,
        telemetry_state,
        gen,
        halloffame=halloffame,
        partial=partial,
    )
    logbook.record(gen=n_gen + gen_offset, nevals=nevals, **record)

    if verbose:
        _print_verbose_record(n_gen + gen_offset, nevals, record, print_header=True)

    callbacks_step_args = {
        "callbacks": callbacks,
        "record": record,
        "logbook": logbook,
        "estimator": estimator,
        "method": "on_step",
    }

    # Check if any of the callbacks conditions are True to stop the iteration
    if eval_callbacks(**callbacks_step_args) or exhausted or _evaluation_limit_reached(estimator):
        callbacks_end_args = {
            "callbacks": callbacks,
            "record": None,
            "logbook": logbook,
            "estimator": estimator,
            "method": "on_end",
        }

        eval_callbacks(**callbacks_end_args)
        if not exhausted or verbose:
            print("INFO: Stopping the algorithm")
        return population, logbook, n_gen + gen_offset

    stopped_by_callback = False
    try:
        for gen in range(1, ngen + 1):
            mutation_prob, diversity_triggered = mutation_probability(mutpb, estimator, record)

            crossover_prob = cxpb.step()
            mutation_prob = min(mutation_prob, max(0.0, 1.0 - crossover_prob))

            # Vary the population
            offspring = varOr(population, toolbox, lambda_, crossover_prob, mutation_prob)
            duplicate_replacements = replace_duplicate_candidates(
                offspring, toolbox, estimator, reference_population=population
            )
            random_immigrants = inject_random_immigrants(offspring, toolbox, estimator, record)

            nevals, partial = _evaluate_generation(toolbox, offspring)
            if partial and not nevals:
                break
            _record_optimizer_control_stats(estimator, random_immigrants=random_immigrants)

            # Update the hall of fame with the generated individuals
            if halloffame is not None:
                halloffame.update(offspring)

            exhausted = _update_search_coverage(estimator, offspring, space_size, seen_candidates)

            # Select the next generation population
            if estimator is not None:
                estimator._last_generation_record = record
            if partial:
                sharing_record, selection_pressure = {}, None
            else:
                population[:], sharing_record = _select_survivors(
                    offspring, toolbox, mu, halloffame, estimator
                )
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
                halloffame=halloffame,
                partial=partial,
            )
            logbook.record(gen=gen + gen_offset, nevals=nevals, **record)

            if verbose:
                _print_verbose_record(gen + gen_offset, nevals, record)

            callbacks_step_args = {
                "callbacks": callbacks,
                "record": record,
                "logbook": logbook,
                "estimator": estimator,
                "method": "on_step",
            }

            # Check if any of the callbacks conditions are True to stop the iteration
            stopped_by_callback = eval_callbacks(**callbacks_step_args)
            if stopped_by_callback or exhausted or _evaluation_limit_reached(estimator):
                if not exhausted or verbose:
                    print("INFO: Stopping the algorithm")
                break

    except (KeyboardInterrupt, SystemExit, StopIteration) as error:
        _report_interruption(error)
    else:
        if not stopped_by_callback:
            local_refinements = _run_local_refinement(
                population, toolbox, halloffame, estimator, space_size, seen_candidates
            )
            _refresh_last_record_after_local_refinement(
                logbook, stats, population, telemetry_state, local_refinements, halloffame
            )

    n_gen = len(logbook)

    callbacks_end_args = {
        "callbacks": callbacks,
        "record": None,
        "logbook": logbook,
        "estimator": estimator,
        "method": "on_end",
    }

    eval_callbacks(**callbacks_end_args)

    return population, logbook, n_gen


def eaDifferentialEvolution(population, toolbox, **kwargs):
    """Mixed-domain rand/1/bin with F=0.8 and deferred greedy replacement.

    One trial per target is evaluated per generation. Crossover uses ``cxpb``;
    ``mutpb`` controls additional standard mutation. Numeric coordinates use
    their linear or logarithmic domain, categories inherit a donor value, and
    feature coordinates are decoded as binary masks. Tournament selection and
    fitness sharing do not apply to pairwise replacement. Other lifecycle,
    evaluation, callback and local-refinement behavior uses the shared engine.
    """
    kwargs.update(mu=len(population), lambda_=len(population), _differential_evolution=True)
    return eaMuPlusLambda(population, toolbox, **kwargs)


algorithms_factory = {
    "eaSimple": eaSimple,
    "eaMuPlusLambda": eaMuPlusLambda,
    "eaMuCommaLambda": eaMuCommaLambda,
    "eaDifferentialEvolution": eaDifferentialEvolution,
}
