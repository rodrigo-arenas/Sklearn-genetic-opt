import random
from contextlib import contextmanager
from math import comb, isfinite

import numpy as np

from ._mixed_space import decode_real, encode_real, real_coordinate_interval
from .space import Categorical, Continuous, Integer
from .space.space import _category_types, _definite_equality, _nan_category_key, _same_config
from .utils.tools import _hashable_key


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


def finite_search_space_size(estimator, limit):
    """Count reachable cached configurations only when coverage is possible."""
    if not getattr(estimator, "use_cache", False):
        return None
    if hasattr(estimator, "space"):
        size = 1
        for name, dimension in estimator.space.param_grid.items():
            if isinstance(dimension, Integer):
                size *= dimension.upper - dimension.lower + 1
            elif isinstance(dimension, Categorical):
                choices = {
                    _nan_category_key(value) or _hashable_key(value) for value in dimension.choices
                }
                if any(not _definite_equality(value, value) for value in choices):
                    return None
                if name in estimator.space._typed_categories:
                    choices = {
                        estimator.space._category_key(name, value) for value in dimension.choices
                    }
                size *= len(choices)
            elif isinstance(dimension, Continuous):
                if dimension.lower != dimension.upper:
                    return None
            else:
                return None
            if size > limit:
                return None
        return size
    if hasattr(estimator, "n_features"):
        maximum = int(estimator.max_features or estimator.n_features)
        size = 0
        for selected in range(1, maximum + 1):
            size += comb(estimator.n_features, selected)
            if size > limit:
                return None
        return size
    return None


def mutable_dimensions(dimensions):
    """Return positions and dimensions eligible for HPO variation."""
    mutable = []
    for index, dimension in enumerate(dimensions):
        if isinstance(dimension, (Integer, Continuous)) and dimension.lower == dimension.upper:
            continue
        if isinstance(dimension, Categorical) and len(dimension.choices) == 1:
            continue
        mutable.append((index, dimension))
    return mutable


def differential_offspring(population, toolbox, estimator, crossover, mutation):
    """Deferred rand/1/bin trials, with mixed-domain and binary decoding."""
    dimensions = tuple(estimator.space.param_grid.values()) if hasattr(estimator, "space") else ()
    mutable = mutable_dimensions(dimensions)
    variable = [index for index, _ in mutable]
    real_intervals = {
        index: real_coordinate_interval(dimension)
        for index, dimension in mutable
        if type(dimension) is Continuous
    }
    offspring = []
    for index, target in enumerate(population):
        donors = random.sample(range(len(population) - 1), 3)
        a, b, c = [population[j + (j >= index)] for j in donors]
        trial = toolbox.clone(target)
        forced = random.choice(variable) if variable else random.randrange(len(target))
        for gene in range(len(target)):
            if gene != forced and random.random() >= crossover:
                continue
            dimension = dimensions[gene] if dimensions else None
            if dimension is None:
                trial[gene] = int(a[gene] + 0.8 * (b[gene] - c[gene]) >= 0.5)
            elif isinstance(dimension, Categorical):
                trial[gene] = a[gene]
            elif dimension.lower == dimension.upper:
                trial[gene] = dimension.lower
            else:
                lower, upper = dimension.lower, dimension.upper
                continuous = isinstance(dimension, Continuous)
                if continuous:
                    interval = real_intervals.get(gene)
                    units = [
                        encode_real(dimension, parent[gene], interval=interval)
                        for parent in (a, b, c)
                    ]
                else:
                    units = [(parent[gene] - lower) / (upper - lower) for parent in (a, b, c)]
                unit = float(np.clip(units[0] + 0.8 * (units[1] - units[2]), 0.0, 1.0))
                if continuous:
                    trial[gene] = decode_real(dimension, unit, interval=interval)
                else:
                    trial[gene] = lower + int(round(unit * (upper - lower)))
        if random.random() < mutation:
            (trial,) = toolbox.mutate(trial)
        estimator._repair_individual(trial)
        del trial.fitness.values
        offspring.append(trial)
    return offspring


def _bounded_integer_neighbor(dimension, value, radius):
    value = dimension._round_and_clip(value)
    step = max(1, int(np.ceil((dimension.upper - dimension.lower) * radius)))
    lower = max(dimension.lower, value - step)
    upper = min(dimension.upper, value + step)
    if lower == upper:
        return value
    # Draw uniformly from all neighbors without allocating the integer interval.
    sampled = random.randint(lower, upper - 1)
    return sampled + (sampled >= value)


def _bounded_continuous_neighbor(dimension, value, radius):
    lower, upper = dimension._sampling_bounds()
    value, radius = float(value), float(radius)
    if dimension.distribution == "log-uniform" and value > 0:
        log_lower, log_upper = np.log(lower), np.log(upper)
        if log_lower != log_upper:
            log_value = np.log(value)
            log_span = (log_upper - log_lower) * radius
            sampled = random.uniform(log_value - log_span, log_value + log_span)
            if sampled < log_lower:
                return lower
            if sampled > log_upper:
                return upper
            return dimension._clip_scalar(np.exp(sampled))

    width = upper - lower
    span = width * radius
    left, right = value - span, value + span
    if isfinite(right - left):
        sampled = random.uniform(left, right)
    else:
        unit = (value - lower) / width
        draw = random.uniform(unit - radius, unit + radius)
        sampled = lower + float(np.clip(draw, 0.0, 1.0)) * width

    return dimension._clip_scalar(sampled)


def _same_category(left, right):
    if type(left) is not type(right):
        return False
    if isinstance(left, (dict, list, tuple, set, frozenset, np.ndarray)):
        return _same_config(left, right, strict_types=True) and (
            _category_types(left) == _category_types(right)
        )
    if left is right:
        return True
    nan_key = _nan_category_key(left)
    return (
        nan_key == _nan_category_key(right)
        if nan_key is not None
        else _definite_equality(left, right)
    )


def _categorical_neighbor(dimension, value):
    choices = [choice for choice in dimension.choices if not _same_category(choice, value)]
    return random.choice(choices) if choices else value


def _search_space_neighbor(estimator, parent):
    values = list(parent)
    mutable = mutable_dimensions(estimator.space.param_grid.values())
    if not mutable:
        return values
    index, dimension = mutable[random.randrange(len(mutable))]
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


def _distance_number(value):
    return int(value) if isinstance(value, np.integer) else value


def _search_space_distance(estimator, left, right):
    distances = []

    for index, parameter_name in enumerate(estimator.space.parameters):
        dimension = estimator.space[parameter_name]
        left_value = left[index]
        right_value = right[index]

        if isinstance(dimension, (Continuous, Integer)):
            span = dimension.upper - dimension.lower
            distance = (
                abs(_distance_number(left_value) - _distance_number(right_value)) / span
                if span
                else 0.0
            )
        elif isinstance(dimension, Categorical):
            distance = 0.0 if _same_category(left_value, right_value) else 1.0
        else:  # pragma: no cover
            distance = 0.0 if _same_category(left_value, right_value) else 1.0

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


def _packed_niche_counts(values, radius, alpha):
    size, width = values.shape
    masks = [int.from_bytes(row) for row in np.packbits(values, axis=1)]
    counts = [0.0] * size
    for index, mask in enumerate(masks):
        # Earlier rows already added their contributions in ascending order.
        # Including this row and then the later rows preserves the same sums.
        count = counts[index] + 1.0
        for other in range(index + 1, size):
            distance = (mask ^ masks[other]).bit_count() / width
            if distance < radius:
                contribution = float(1.0 - (distance / radius) ** alpha)
                count += contribution
                counts[other] += contribution
        counts[index] = count
    return counts


def niche_counts(population, estimator):
    radius = estimator.sharing_radius
    alpha = estimator.sharing_alpha
    size = len(population)
    if size == 0:
        return []

    # Convert columns once and vectorize distances one row at a time. Keep
    # contribution accumulation ordered: reassociation can break selection ties.
    feature_values = np.asarray(population) if not hasattr(estimator, "space") else None
    if (
        feature_values is not None
        and size >= 16
        and feature_values.shape[1] >= 1024
        and feature_values.dtype.kind in "biu"
        and radius > 0
        and alpha > 0
        and feature_values.min() >= 0
        and feature_values.max() <= 1
    ):
        return _packed_niche_counts(feature_values, radius, alpha)

    columns = []
    for index, values in enumerate(zip(*population)) if feature_values is None else []:
        dimension = estimator.space[estimator.space.parameters[index]]
        if isinstance(dimension, (Continuous, Integer)):
            # Preserve subtraction precision for Python integers, including
            # integer values supplied in a Continuous dimension.
            numeric_values = np.fromiter(
                (_distance_number(value) for value in values), dtype=object, count=size
            )
            columns.append((numeric_values, dimension.upper - dimension.lower))
        else:
            columns.append((np.fromiter(values, dtype=object, count=size), None))

    counts = []
    for index in range(size):
        distances = np.zeros(size)
        if feature_values is not None:
            if feature_values.shape[1]:
                distances = (
                    np.count_nonzero(feature_values != feature_values[index], axis=1)
                    / feature_values.shape[1]
                )
        else:
            components = np.zeros((size, len(columns)))
            for column, (values, span) in enumerate(columns):
                if span is None:
                    components[:, column] = np.fromiter(
                        (0.0 if _same_category(values[index], value) else 1.0 for value in values),
                        dtype=float,
                    )
                elif span:
                    components[:, column] = np.asarray(
                        np.abs(values[index] - values) / span, dtype=float
                    )
            if columns:
                distances = components.mean(axis=1)
        contributions = np.fromiter(
            # Python floats preserve the scalar sharing formula's rounding;
            # inlining avoids a function call for every population pair.
            (
                0.0 if distance >= radius else 1.0 - (distance / radius) ** alpha
                for distance in distances.tolist()
            ),
            dtype=float,
        )
        counts.append(max(float(np.cumsum(contributions)[-1]), 1.0))

    return counts


def _shared_primary_fitness(population, counts):
    primary_values = [individual.fitness.values[0] for individual in population]
    if not primary_values:
        return []

    primary_weight = population[0].fitness.weights[0]
    sign = 1.0 if primary_weight >= 0 else -1.0
    utilities = [value * sign for value in primary_values]
    finite_utilities = [utility for utility in utilities if np.isfinite(utility)]
    if not finite_utilities:
        return [individual.fitness.values for individual in population]
    minimum_utility = min(finite_utilities)
    shift = -minimum_utility + 1e-12 if minimum_utility <= 0 else 0.0
    half_shift = (
        float(shift) / 2
        if shift and not isfinite(float(max(finite_utilities)) + float(shift))
        else 0.0
    )

    shared_values = []
    for individual, utility, count in zip(population, utilities, counts):
        if np.isfinite(utility):
            if half_shift:
                # An isolated candidate keeps its score. Halve both terms
                # before adding to keep the crowded candidate's shift finite.
                shared_utility = (
                    utility
                    if count == 1
                    else (((float(utility) / 2 + half_shift) / count) - half_shift) * 2
                )
            else:
                shared_utility = (utility + shift) / count - shift
            shared_primary = shared_utility * sign
        else:
            # Preserve the worst sentinel; including it in the shift would
            # turn every otherwise valid candidate's shared fitness into NaN.
            shared_primary = individual.fitness.values[0]
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
