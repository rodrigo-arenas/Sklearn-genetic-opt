import inspect
from math import floor
import random

import numpy as np
from scipy.stats import qmc
from sklearn.utils import check_random_state

from ._mixed_space import decode_real, real_coordinate_interval
from .optimizer_control import finite_search_space_size
from .space import Categorical, Continuous, Integer, Space
from .utils.tools import _hashable_key


def validate_population_initializer(population_initializer, *, feature_selection=False):
    valid_initializers = {"random", "smart"}
    if feature_selection:
        valid_initializers.add("spread")
    if population_initializer not in valid_initializers:
        raise ValueError(
            f"population_initializer must be one of {sorted(valid_initializers)}, "
            f"got {population_initializer} instead"
        )


def _is_dimension_value_valid(dimension, value):
    # Single source of truth in Space.value_in_dimension so warm-start
    # validation and population initialization can't drift.
    return Space.value_in_dimension(dimension, value)


def _default_estimator_params(estimator, space):
    estimator_params = estimator.get_params(deep=True)
    defaults = {}

    for parameter, dimension in space.param_grid.items():
        if parameter not in estimator_params:
            return None

        value = estimator_params[parameter]
        if not _is_dimension_value_valid(dimension, value):
            return None

        defaults[parameter] = value

    return defaults


def _sample_dimension_from_unit(dimension, value, interval=None):
    if isinstance(dimension, Integer):
        lower, upper = dimension.lower, dimension.upper
        if lower == upper:
            return int(lower)
        sampled = lower + floor(value * (upper - lower + 1))
        return int(min(upper, max(lower, sampled)))

    if isinstance(dimension, Continuous):
        return decode_real(dimension, value, interval=interval)

    raise TypeError("Latin hypercube sampling only supports numeric dimensions")


def _append_unique_individual(
    population, seen_individuals, individual_values, individual_cls, key=None
):
    # Feature masks are already immutable binary keys. HPO values can be mixed types.
    individual_key = (
        key(individual_values)
        if key is not None
        else (
            individual_values
            if isinstance(individual_values, bytes)
            else _hashable_key(tuple(individual_values))
        )
    )
    if individual_key in seen_individuals:
        return False

    population.append(individual_cls(individual_values))
    seen_individuals.add(individual_key)
    return True


def _feature_mask(n_features, n_selected):
    # Sampling the excluded columns gives the same uniform mask distribution.
    dense = n_selected > n_features // 2
    count = n_features - n_selected if dense else n_selected
    values = bytearray([dense]) * n_features
    for index in random.sample(range(n_features), k=count):
        values[index] = not dense
    return bytes(values)


def random_search_population(estimator, toolbox, individual_cls):
    population = []
    population_size = int(estimator.population_size)
    warm_start_configs = getattr(estimator, "_warm_start_configs", estimator.warm_start_configs)
    warm_start_configs = warm_start_configs or []
    num_warm_start = min(len(warm_start_configs), population_size)

    for config in warm_start_configs[:num_warm_start]:
        individual_values = estimator.space.sample_warm_start(config)
        population.append(individual_cls(list(individual_values.values())))

    num_random = population_size - num_warm_start
    population.extend(toolbox.population(n=num_random))

    return population


def _normalized_search_seed(space, parameters):
    return [
        space._repair_value(dimension, parameters[name])
        for name, dimension in space.param_grid.items()
    ]


def smart_search_population(estimator, toolbox, individual_cls):
    population = []
    population_size = int(estimator.population_size)
    seen_individuals = set()
    key = estimator.space._population_key
    warm_start_configs = getattr(estimator, "_warm_start_configs", estimator.warm_start_configs)
    warm_start_configs = warm_start_configs or []

    for config in warm_start_configs:
        individual_values = estimator.space.sample_warm_start(config)
        _append_unique_individual(
            population,
            seen_individuals,
            _normalized_search_seed(estimator.space, individual_values),
            individual_cls,
            key,
        )

        if len(population) == population_size:
            return population

    default_params = _default_estimator_params(estimator.estimator, estimator.space)
    if default_params is not None:
        _append_unique_individual(
            population,
            seen_individuals,
            _normalized_search_seed(estimator.space, default_params),
            individual_cls,
            key,
        )

    remaining = population_size - len(population)
    categorical_indices = {}
    for parameter, dimension in estimator.space.param_grid.items():
        if isinstance(dimension, Categorical):
            if not remaining:
                continue
            eligible = range(len(dimension.choices))
            weights = None
            if dimension.priors is not None:
                priors = np.asarray(dimension.priors, dtype=float)
                eligible = np.flatnonzero(priors).tolist()
                weights = priors[eligible]
            if weights is None or np.all(weights == weights[0]):
                # Sample equal-weight choices uniformly without replacement per cycle.
                choices = random.sample(eligible, k=min(remaining, len(eligible)))
                indices = [choices[index % len(choices)] for index in range(remaining)]
            else:
                # Systematic sampling gives each stratum the same random offset.
                cumulative = np.cumsum(weights)
                boundaries = cumulative / cumulative[-1] * remaining
                ends = np.floor(boundaries).astype(int)
                fractions = boundaries - ends
                if np.any(fractions):
                    ends += random.random() < fractions
                indices = np.repeat(eligible, np.diff(ends, prepend=0)).tolist()
            # Independent strata prevent unrelated categorical genes from
            # being locked into the same repeating combinations.
            random.shuffle(indices)
            categorical_indices[parameter] = indices
    numeric_parameters = [
        parameter
        for parameter, dimension in estimator.space.param_grid.items()
        if isinstance(dimension, (Continuous, Integer))
    ]

    lhs_samples = None
    if numeric_parameters and remaining > 0:
        # The quasi-random sampler must be seeded explicitly: ``qmc.LatinHypercube``
        # otherwise draws from its own fresh entropy and ignores any seeding,
        # making numeric-parameter searches non-reproducible. Derive its seed from
        # the estimator's ``random_state`` when set (threaded from the public API),
        # and otherwise fall back to the global ``random`` RNG so the documented
        # ``random.seed`` convention still controls it.
        random_state = getattr(estimator, "random_state", None)
        # Resumed searches seed new strata from the continuing proposal stream.
        if random_state is None or getattr(estimator, "_resume_generation_log", None):
            lhs_seed = random.randint(0, 2**31 - 1)
        else:
            lhs_seed = int(check_random_state(random_state).randint(0, 2**31 - 1))
        # SciPy < 1.15 calls this argument seed; retain rng on newer releases
        # because switching keywords can change their seeded sample sequence.
        seed_keyword = (
            "rng" if "rng" in inspect.signature(qmc.LatinHypercube).parameters else "seed"
        )
        sampler = qmc.LatinHypercube(d=len(numeric_parameters), **{seed_keyword: lhs_seed})
        lhs_samples = sampler.random(n=remaining)

    coordinate_intervals = {
        parameter: real_coordinate_interval(dimension)
        for parameter, dimension in estimator.space.param_grid.items()
        if remaining > 1 and type(dimension) is Continuous
    }
    for sample_index in range(remaining):
        individual_values = []
        numeric_index = 0
        for parameter, dimension in estimator.space.param_grid.items():
            if isinstance(dimension, (Continuous, Integer)):
                value = _sample_dimension_from_unit(
                    dimension,
                    lhs_samples[sample_index, numeric_index],
                    coordinate_intervals.get(parameter),
                )
                numeric_index += 1
            elif isinstance(dimension, Categorical):
                value = dimension.choices[categorical_indices[parameter][sample_index]]
            else:  # pragma: no cover
                value = dimension.sample()

            individual_values.append(value)

        _append_unique_individual(
            population, seen_individuals, individual_values, individual_cls, key
        )

    space_size = finite_search_space_size(estimator, population_size)
    attempts = 0
    max_attempts = max(100, population_size * 20)
    while (
        len(population) < population_size
        and attempts < max_attempts
        and (space_size is None or len(seen_individuals) < space_size)
    ):
        individual = toolbox.individual()
        _append_unique_individual(
            population, seen_individuals, list(individual), individual_cls, key
        )
        attempts += 1

    while len(population) < population_size:
        population.append(toolbox.individual())

    return population


def initialize_search_population(estimator, toolbox, individual_cls):
    if estimator.population_initializer == "random":
        return random_search_population(estimator, toolbox, individual_cls)

    return smart_search_population(estimator, toolbox, individual_cls)


def _warm_feature_masks(estimator):
    masks = getattr(getattr(estimator, "population_config", None), "warm_start_masks", None)
    if masks is None:
        return ()
    if not isinstance(masks, (list, tuple, np.ndarray)) or (
        isinstance(masks, np.ndarray) and masks.ndim != 2
    ):
        raise ValueError("warm_start_masks must be a list, tuple, or 2D array of feature masks")

    unique = {}
    population_size = int(estimator.population_size)
    for index, mask in enumerate(masks):
        label = f"warm_start_masks mask {index}"
        shape_error = f"{label} must be one-dimensional with length {estimator.n_features}"
        try:
            values = np.asarray(mask)
        except ValueError as exc:
            raise ValueError(shape_error) from exc
        if values.shape != (estimator.n_features,):
            raise ValueError(shape_error)
        if values.dtype.kind not in "biu" or not np.all((values == 0) | (values == 1)):
            raise ValueError(f"{label} must contain Boolean or integer 0/1 values")
        selected = np.count_nonzero(values)
        if selected == 0:
            raise ValueError(f"{label} must select at least one feature")
        if estimator.max_features is not None and selected > estimator.max_features:
            raise ValueError(f"{label} selects {selected} features, exceeding max_features")
        unique[values.astype(np.uint8, copy=False).tobytes()] = None
        if len(unique) == population_size:
            break
    return tuple(unique)


def smart_feature_population(estimator, toolbox, individual_cls, warm_masks=()):
    population = [individual_cls(mask) for mask in warm_masks]
    population_size = int(estimator.population_size)
    if len(population) == population_size:
        return population
    seen_individuals = set(warm_masks)
    max_selected = int(estimator.max_features or estimator.n_features)
    max_selected = max(1, min(max_selected, estimator.n_features))

    selected_counts = np.linspace(1, max_selected, num=population_size - len(population))
    selected_counts = np.rint(selected_counts).astype(int)

    space_size = finite_search_space_size(estimator, population_size)
    padding_needed = population_size - space_size if space_size is not None else 0
    padding = []
    for n_selected in selected_counts:
        values = _feature_mask(estimator.n_features, int(n_selected))
        added = _append_unique_individual(population, seen_individuals, values, individual_cls)
        # A saturated space needs repeated masks to fill the population. Keep
        # the first such draws while reserving room for every distinct mask.
        if not added and len(padding) < padding_needed:
            padding.append(values)

    attempts = 0
    max_attempts = max(100, population_size * 20)
    while (
        len(population) < population_size
        and attempts < max_attempts
        and (space_size is None or len(seen_individuals) < space_size)
    ):
        values = _feature_mask(estimator.n_features, random.randint(1, max_selected))
        _append_unique_individual(population, seen_individuals, values, individual_cls)
        attempts += 1

    population.extend(individual_cls(values) for values in padding)
    while len(population) < population_size:
        values = _feature_mask(estimator.n_features, random.randint(1, max_selected))
        population.append(individual_cls(values))

    return population


def _spread_feature_masks(population):
    """Visit a middle mask size, then choose sizes far from those visited."""
    if len(population) < 2:
        return population

    counts = np.asarray([sum(mask) for mask in population])
    midpoint = (int(counts.min()) + int(counts.max())) / 2
    first = int(np.argmin(np.abs(counts - midpoint)))
    ordered = [population[first]]
    distances = np.abs(counts - counts[first])
    distances[first] = -1
    while len(ordered) < len(population):
        index = int(np.argmax(distances))
        ordered.append(population[index])
        # Keep the distance to the closest selected size, excluding used masks.
        np.minimum(distances, np.abs(counts - counts[index]), out=distances)
        distances[index] = -1
    return ordered


def initialize_feature_population(estimator, toolbox, individual_cls):
    warm_masks = _warm_feature_masks(estimator)
    if estimator.population_initializer == "random":
        population_size = int(estimator.population_size)
        if not warm_masks:
            return toolbox.population(n=population_size)
        population = [individual_cls(mask) for mask in warm_masks]
        population.extend(toolbox.population(n=population_size - len(population)))
        return population

    population = smart_feature_population(estimator, toolbox, individual_cls, warm_masks)
    if estimator.population_initializer == "spread":
        if not warm_masks:
            return _spread_feature_masks(population)
        count = len(warm_masks)
        return population[:count] + _spread_feature_masks(population[count:])
    return population
