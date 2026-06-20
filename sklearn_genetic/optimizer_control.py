import random
from contextlib import contextmanager

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
    sharing_radius,
    sharing_alpha,
    selection_pressure_min,
    selection_pressure_max,
    offspring_diversity_retries,
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

    if not 0 < sharing_radius <= 1:
        raise ValueError("sharing_radius must be in the interval (0, 1]")

    if sharing_alpha <= 0:
        raise ValueError("sharing_alpha must be greater than 0")

    if selection_pressure_min < 1:
        raise ValueError("selection_pressure_min must be greater than or equal to 1")

    if selection_pressure_max is not None and selection_pressure_max < selection_pressure_min:
        raise ValueError(
            "selection_pressure_max must be greater than or equal to selection_pressure_min"
        )

    if offspring_diversity_retries < 0:
        raise ValueError("offspring_diversity_retries must be greater than or equal to 0")


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


def adaptive_tournament_size(estimator, record, population_size):
    base_size = int(getattr(estimator, "tournament_size", 3))
    if not getattr(estimator, "adaptive_selection", False):
        return min(base_size, population_size)

    minimum = int(getattr(estimator, "selection_pressure_min", 2))
    maximum = getattr(estimator, "selection_pressure_max", None)
    if maximum is None:
        maximum = max(base_size + 1, minimum)

    minimum = min(max(1, minimum), population_size)
    maximum = min(max(minimum, int(maximum)), population_size)
    base_size = min(max(base_size, minimum), maximum)

    if diversity_control_triggered(estimator, record):
        return minimum

    if record is None:
        return base_size

    if record.get("fitness_improved", False):
        diversity = min(
            record.get("unique_individual_ratio", 1.0),
            record.get("genotype_diversity", 1.0),
        )
        if diversity >= max(getattr(estimator, "diversity_threshold", 0.0), 0.2):
            return min(maximum, base_size + 1)

    if record.get("stagnation_generations", 0) > 0:
        return max(minimum, base_size - 1)

    return base_size


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


def _candidate_key(estimator, individual):
    if hasattr(estimator, "_individual_key"):
        return estimator._individual_key(individual)

    return tuple(individual)


def _new_unique_individual(toolbox, estimator, seen):
    retries = getattr(estimator, "offspring_diversity_retries", 0)
    candidate = None
    candidate_key = None

    for _ in range(retries + 1):
        candidate = toolbox.individual()
        candidate_key = _candidate_key(estimator, candidate)
        if candidate_key not in seen:
            break

    return candidate, candidate_key


def replace_duplicate_candidates(population, toolbox, estimator, reference_population=None):
    if not getattr(estimator, "diversity_control", False) and not getattr(
        estimator, "offspring_diversity_retries", 0
    ):
        return 0

    seen = set()
    replacements = 0
    if reference_population is not None:
        seen.update(_candidate_key(estimator, individual) for individual in reference_population)

    for index, individual in enumerate(population):
        key = _candidate_key(estimator, individual)
        if key not in seen:
            seen.add(key)
            continue

        population[index], replacement_key = _new_unique_individual(toolbox, estimator, seen)
        seen.add(replacement_key)
        replacements += 1

    return replacements


def local_search_enabled(estimator):
    return getattr(estimator, "local_search", False)


def _bounded_integer_neighbor(dimension, value, radius):
    value = int(round(float(value)))
    value = int(np.clip(value, dimension.lower, dimension.upper))
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


def fitness_sharing_enabled(estimator):
    return getattr(estimator, "fitness_sharing", False)


def _search_space_distance(estimator, left, right):
    distances = []

    for index, parameter_name in enumerate(estimator.space.parameters):
        dimension = estimator.space[parameter_name]
        left_value = left[index]
        right_value = right[index]

        if isinstance(dimension, (Continuous, Integer)):
            span = dimension.upper - dimension.lower
            distance = 0.0 if span == 0 else abs(left_value - right_value) / span
        elif isinstance(dimension, Categorical):
            distance = 0.0 if left_value == right_value else 1.0
        else:  # pragma: no cover
            distance = 0.0 if left_value == right_value else 1.0

        distances.append(distance)

    return float(np.mean(distances)) if distances else 0.0


def individual_distance(estimator, left, right):
    if hasattr(estimator, "space"):
        return _search_space_distance(estimator, left, right)

    if len(left) == 0:
        return 0.0

    differences = sum(left_value != right_value for left_value, right_value in zip(left, right))
    return differences / len(left)


def sharing_value(distance, radius, alpha):
    if distance >= radius:
        return 0.0

    return 1.0 - (distance / radius) ** alpha


def niche_counts(population, estimator):
    radius = estimator.sharing_radius
    alpha = estimator.sharing_alpha
    counts = []

    for individual in population:
        count = 0.0
        for other in population:
            distance = individual_distance(estimator, individual, other)
            count += sharing_value(distance, radius, alpha)
        counts.append(max(count, 1.0))

    return counts


def _shared_primary_fitness(population, counts):
    primary_values = [individual.fitness.values[0] for individual in population]
    if not primary_values:
        return []

    primary_weight = population[0].fitness.weights[0]
    sign = 1.0 if primary_weight >= 0 else -1.0
    utilities = [value * sign for value in primary_values]
    minimum_utility = min(utilities)
    shift = -minimum_utility + 1e-12 if minimum_utility <= 0 else 0.0

    shared_values = []
    for individual, utility, count in zip(population, utilities, counts):
        shared_utility = (utility + shift) / count
        shared_primary = (shared_utility - shift) * sign
        shared_values.append((shared_primary, *individual.fitness.values[1:]))

    return shared_values


@contextmanager
def shared_fitness(population, estimator):
    if not fitness_sharing_enabled(estimator) or not population:
        yield {
            "fitness_sharing_applied": False,
            "mean_niche_count": 0.0,
            "max_niche_count": 0.0,
        }
        return

    counts = niche_counts(population, estimator)
    original_fitness = [individual.fitness.values for individual in population]
    shared_values = _shared_primary_fitness(population, counts)

    try:
        for individual, shared_value in zip(population, shared_values):
            individual.fitness.values = shared_value

        yield {
            "fitness_sharing_applied": True,
            "mean_niche_count": float(np.mean(counts)),
            "max_niche_count": float(np.max(counts)),
        }
    finally:
        for individual, original_value in zip(population, original_fitness):
            individual.fitness.values = original_value
