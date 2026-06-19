import random

import numpy as np
from scipy.stats import qmc

from .space import Categorical, Continuous, Integer


def validate_population_initializer(population_initializer):
    valid_initializers = {"random", "smart"}
    if population_initializer not in valid_initializers:
        raise ValueError(
            f"population_initializer must be one of {sorted(valid_initializers)}, "
            f"got {population_initializer} instead"
        )


def _is_dimension_value_valid(dimension, value):
    if isinstance(dimension, Integer):
        return isinstance(value, int) and dimension.lower <= value <= dimension.upper

    if isinstance(dimension, Continuous):
        return isinstance(value, (int, float)) and dimension.lower <= value <= dimension.upper

    if isinstance(dimension, Categorical):
        return value in dimension.choices

    return False


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


def _sample_dimension_from_unit(dimension, value):
    if isinstance(dimension, Integer):
        n_values = dimension.upper - dimension.lower + 1
        sampled = dimension.lower + int(np.floor(value * n_values))
        return int(np.clip(sampled, dimension.lower, dimension.upper))

    if isinstance(dimension, Continuous):
        if dimension.distribution == "log-uniform" and dimension.lower > 0:
            log_lower = np.log(dimension.lower)
            log_upper = np.log(dimension.upper)
            return float(np.exp(log_lower + value * (log_upper - log_lower)))

        return float(dimension.lower + value * (dimension.upper - dimension.lower))

    raise TypeError("Latin hypercube sampling only supports numeric dimensions")


def _stratified_categorical_value(dimension, index, population_size):
    if dimension.priors is not None:
        midpoint = (index + 0.5) / population_size
        cumulative_priors = np.cumsum(dimension.priors)
        choice_index = int(np.searchsorted(cumulative_priors, midpoint, side="right"))
        choice_index = min(choice_index, len(dimension.choices) - 1)
        return dimension.choices[choice_index]

    return dimension.choices[index % len(dimension.choices)]


def _append_unique_individual(population, seen_individuals, individual_values, individual_cls):
    individual_key = tuple(individual_values)
    if individual_key in seen_individuals:
        return False

    population.append(individual_cls(individual_values))
    seen_individuals.add(individual_key)
    return True


def random_search_population(estimator, toolbox, individual_cls):
    population = []
    num_warm_start = min(len(estimator.warm_start_configs), estimator.population_size)

    for config in estimator.warm_start_configs[:num_warm_start]:
        individual_values = estimator.space.sample_warm_start(config)
        population.append(individual_cls(list(individual_values.values())))

    num_random = estimator.population_size - num_warm_start
    population.extend(toolbox.population(n=num_random))

    return population


def smart_search_population(estimator, toolbox, individual_cls):
    population = []
    seen_individuals = set()

    for config in estimator.warm_start_configs[: estimator.population_size]:
        individual_values = estimator.space.sample_warm_start(config)
        _append_unique_individual(
            population,
            seen_individuals,
            list(individual_values.values()),
            individual_cls,
        )

        if len(population) == estimator.population_size:
            return population

    default_params = _default_estimator_params(estimator.estimator, estimator.space)
    if default_params is not None:
        _append_unique_individual(
            population,
            seen_individuals,
            [default_params[parameter] for parameter in estimator.space.parameters],
            individual_cls,
        )

    remaining = estimator.population_size - len(population)
    numeric_parameters = [
        parameter
        for parameter, dimension in estimator.space.param_grid.items()
        if isinstance(dimension, (Continuous, Integer))
    ]

    lhs_samples = None
    if numeric_parameters and remaining > 0:
        sampler = qmc.LatinHypercube(d=len(numeric_parameters))
        lhs_samples = sampler.random(n=remaining)

    for sample_index in range(remaining):
        individual_values = []
        numeric_index = 0
        for parameter, dimension in estimator.space.param_grid.items():
            if isinstance(dimension, (Continuous, Integer)):
                value = _sample_dimension_from_unit(
                    dimension, lhs_samples[sample_index, numeric_index]
                )
                numeric_index += 1
            elif isinstance(dimension, Categorical):
                value = _stratified_categorical_value(dimension, sample_index, remaining)
            else:  # pragma: no cover
                value = dimension.sample()

            individual_values.append(value)

        _append_unique_individual(population, seen_individuals, individual_values, individual_cls)

    attempts = 0
    max_attempts = max(100, estimator.population_size * 20)
    while len(population) < estimator.population_size and attempts < max_attempts:
        individual = toolbox.individual()
        _append_unique_individual(population, seen_individuals, list(individual), individual_cls)
        attempts += 1

    while len(population) < estimator.population_size:
        population.append(toolbox.individual())

    return population


def initialize_search_population(estimator, toolbox, individual_cls):
    if estimator.population_initializer == "random":
        return random_search_population(estimator, toolbox, individual_cls)

    return smart_search_population(estimator, toolbox, individual_cls)


def smart_feature_population(estimator, toolbox, individual_cls):
    population = []
    seen_individuals = set()
    max_selected = estimator.max_features or estimator.n_features
    max_selected = max(1, min(max_selected, estimator.n_features))

    selected_counts = np.linspace(1, max_selected, num=estimator.population_size)
    selected_counts = np.rint(selected_counts).astype(int)

    for sample_index, n_selected in enumerate(selected_counts):
        offset = sample_index % estimator.n_features
        feature_order = list(range(estimator.n_features))
        random.shuffle(feature_order)
        feature_order = feature_order[offset:] + feature_order[:offset]

        values = [0] * estimator.n_features
        for feature_index in feature_order[:n_selected]:
            values[feature_index] = 1

        _append_unique_individual(population, seen_individuals, values, individual_cls)

    attempts = 0
    max_attempts = max(100, estimator.population_size * 20)
    while len(population) < estimator.population_size and attempts < max_attempts:
        individual = toolbox.individual()
        _append_unique_individual(population, seen_individuals, list(individual), individual_cls)
        attempts += 1

    while len(population) < estimator.population_size:
        population.append(toolbox.individual())

    return population


def initialize_feature_population(estimator, toolbox, individual_cls):
    if estimator.population_initializer == "random":
        return toolbox.population(n=estimator.population_size)

    return smart_feature_population(estimator, toolbox, individual_cls)
