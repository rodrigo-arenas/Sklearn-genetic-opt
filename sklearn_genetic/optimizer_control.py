import random

import numpy as np

from .space import Categorical, Continuous, Integer


def validate_optimizer_control(
    local_search_top_k,
    local_search_steps,
    local_search_radius,
    diversity_threshold,
    diversity_stagnation_generations,
    diversity_mutation_boost,
    random_immigrants_fraction,
):
    if local_search_top_k < 1:
        raise ValueError("local_search_top_k must be greater than or equal to 1")

    if local_search_steps < 1:
        raise ValueError("local_search_steps must be greater than or equal to 1")

    if not 0 < local_search_radius <= 1:
        raise ValueError("local_search_radius must be in the interval (0, 1]")

    if not 0 <= diversity_threshold <= 1:
        raise ValueError("diversity_threshold must be in the interval [0, 1]")

    if diversity_stagnation_generations < 1:
        raise ValueError("diversity_stagnation_generations must be greater than or equal to 1")

    if diversity_mutation_boost < 1:
        raise ValueError("diversity_mutation_boost must be greater than or equal to 1")

    if not 0 <= random_immigrants_fraction <= 1:
        raise ValueError("random_immigrants_fraction must be in the interval [0, 1]")


def diversity_control_triggered(estimator, record):
    if not getattr(estimator, "diversity_control", False) or record is None:
        return False

    low_unique_ratio = record.get("unique_individual_ratio", 1.0) < estimator.diversity_threshold
    low_genotype_diversity = record.get("genotype_diversity", 1.0) < estimator.diversity_threshold
    stagnant = record.get("stagnation_generations", 0) >= estimator.diversity_stagnation_generations

    return low_unique_ratio or low_genotype_diversity or stagnant


def mutation_probability(mutpb, estimator, record):
    probability = mutpb.step()
    triggered = diversity_control_triggered(estimator, record)

    if triggered:
        probability = min(1.0, probability * estimator.diversity_mutation_boost)

    return probability, triggered


def random_immigrant_count(estimator, population_size):
    fraction = getattr(estimator, "random_immigrants_fraction", 0.0)
    if fraction <= 0 or population_size <= 0:
        return 0

    return max(1, int(np.ceil(population_size * fraction)))


def inject_random_immigrants(offspring, toolbox, estimator, record):
    if not diversity_control_triggered(estimator, record):
        return 0

    count = min(len(offspring), random_immigrant_count(estimator, len(offspring)))
    if count == 0:
        return 0

    for index in range(count):
        offspring[-(index + 1)] = toolbox.individual()

    return count


def replace_duplicate_candidates(population, toolbox, estimator):
    if not getattr(estimator, "diversity_control", False):
        return 0

    seen = set()
    replacements = 0

    for index, individual in enumerate(population):
        key = tuple(individual)
        if key not in seen:
            seen.add(key)
            continue

        population[index] = toolbox.individual()
        replacements += 1

    return replacements


def local_search_enabled(estimator):
    return getattr(estimator, "local_search", False)


def _bounded_integer_neighbor(dimension, value, radius):
    step = max(1, int(np.ceil((dimension.upper - dimension.lower) * radius)))
    candidates = [
        candidate
        for candidate in range(
            max(dimension.lower, value - step), min(dimension.upper, value + step) + 1
        )
        if candidate != value
    ]
    return random.choice(candidates) if candidates else value


def _bounded_continuous_neighbor(dimension, value, radius):
    if dimension.distribution == "log-uniform" and dimension.lower > 0 and value > 0:
        log_value = np.log(value)
        log_span = (np.log(dimension.upper) - np.log(dimension.lower)) * radius
        sampled = np.exp(random.uniform(log_value - log_span, log_value + log_span))
    else:
        span = (dimension.upper - dimension.lower) * radius
        sampled = random.uniform(value - span, value + span)

    return float(np.clip(sampled, dimension.lower, dimension.upper))


def _categorical_neighbor(dimension, value):
    choices = [choice for choice in dimension.choices if choice != value]
    return random.choice(choices) if choices else value


def _search_space_neighbor(estimator, parent):
    values = list(parent)
    index = random.randrange(0, len(estimator.space))
    parameter_name = estimator.space.parameters[index]
    dimension = estimator.space[parameter_name]
    value = values[index]

    if isinstance(dimension, Integer):
        values[index] = _bounded_integer_neighbor(dimension, value, estimator.local_search_radius)
    elif isinstance(dimension, Continuous):
        values[index] = _bounded_continuous_neighbor(
            dimension, value, estimator.local_search_radius
        )
    elif isinstance(dimension, Categorical):
        values[index] = _categorical_neighbor(dimension, value)
    else:  # pragma: no cover
        values[index] = dimension.sample()

    return values


def _feature_neighbor(estimator, parent):
    values = list(parent)
    flips = max(1, int(np.ceil(len(values) * estimator.local_search_radius)))

    for index in random.sample(range(len(values)), k=min(flips, len(values))):
        values[index] = 0 if values[index] else 1

    if estimator.max_features and sum(values) > estimator.max_features:
        selected = [index for index, value in enumerate(values) if value]
        random.shuffle(selected)
        for index in selected[estimator.max_features :]:
            values[index] = 0

    if sum(values) == 0:
        values[random.randrange(0, len(values))] = 1

    return values


def local_neighbor(estimator, parent, individual_cls):
    if hasattr(estimator, "space"):
        values = _search_space_neighbor(estimator, parent)
    else:
        values = _feature_neighbor(estimator, parent)

    return individual_cls(values)
