import random
from dataclasses import dataclass

import numpy as np
from joblib import hash as joblib_hash


@dataclass(frozen=True)
class _UnhashableCandidate:
    kind: type
    value: object


def _hashable_key(value):
    """Retain hashable values without requiring their siblings to be pickled."""
    try:
        hash(value)
    except TypeError:
        if isinstance(value, dict):
            key = tuple((_hashable_key(k), _hashable_key(v)) for k, v in value.items())
        elif isinstance(value, (list, tuple)):
            key = tuple(_hashable_key(item) for item in value)
        elif isinstance(value, set):
            key = frozenset(_hashable_key(item) for item in value)
        elif isinstance(value, np.ndarray) and value.dtype.hasobject:
            key = (value.dtype, value.shape, _hashable_key(value.tolist()))
        else:
            key = joblib_hash(value)
        return _UnhashableCandidate(type(value), key)
    return value


def mutFlipBit(individual, indpb):
    """Apply DEAP bit mutation and keep the mask nonempty."""
    from deap import tools

    (individual,) = tools.mutFlipBit(individual, indpb)
    return (check_bool_individual(individual),)


def cxUniform(ind1, ind2, indpb):
    """Swap feature genes independently and keep both masks nonempty."""
    # Resolve the random generator once per crossover for wide masks. Keep
    # one draw per shared coordinate, including when the two genes match.
    draw = random.random
    for index in range(min(len(ind1), len(ind2))):
        if draw() < indpb:
            ind1[index], ind2[index] = ind2[index], ind1[index]
    return check_bool_individual(ind1), check_bool_individual(ind2)


def check_bool_individual(individual):
    """
    Makes sures there is no individual with all its values at zero
    """
    if sum(individual) == 0:
        index = random.choice(range(len(individual)))
        individual[index] = 1

    return individual


def novelty_scorer(individual, population, k=15):
    """
    Calculate novelty score for an individual based on its distance from other individuals in the population.

    Parameters
    ----------
    individual: Individual object
        The individual (set of hyperparameters) that is being evaluated.

    population: List[Individual]
        The current population of individuals.

    k: int, default=15
        The number of nearest neighbors to consider for the novelty calculation.

    Returns
    -------
    novelty_score: float
        The novelty score for the individual.
    """
    distances = []

    # Calculate distances between the individual and every other individual in the population
    for other in population:
        if other != individual:
            # Here we use Hamming distance to measure difference
            distance = sum(i != o for i, o in zip(individual, other))
            distances.append(distance)

    # Sort the distances and take the average of the k nearest neighbors
    distances = sorted(distances)
    k_min = min(k, len(population))
    nearest_distances = distances[:k_min]

    # Novelty score is the average distance to the k-nearest neighbors
    novelty_score = np.mean(nearest_distances) if nearest_distances else 0
    return novelty_score
