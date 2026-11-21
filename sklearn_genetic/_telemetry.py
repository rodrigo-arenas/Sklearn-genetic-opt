"""Generation statistics shared by optimization engines."""

from collections import Counter
from itertools import repeat
from operator import is_

import numpy as np

from .utils.tools import _hashable_key

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


def _individual_key(individual):
    return _hashable_key(tuple(individual))


def _is_fitness_improvement(candidate, best, primary_weight):
    # Keep initial unavailable scores numeric for stopping callbacks, while
    # allowing a later valid generation to recover from that NaN sentinel.
    if best is None:
        return True
    if np.isnan(candidate):
        return False
    if np.isnan(best):
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
            counts = Counter(_hashable_key(value) for value in gene_values)
            # Probability that two distinct population members disagree at this
            # gene. Unlike distinct-allele counts, this stays meaningful for
            # binary/categorical genes as the population grows.
            equal_pairs = sum(count * (count - 1) for count in counts.values())
            gene_diversities.append(1 - equal_pairs / (population_size * (population_size - 1)))
        genotype_diversity = float(np.mean(gene_diversities)) if gene_diversities else 0.0

    return {
        "population_size": population_size,
        "unique_individuals": unique_individuals,
        "unique_individual_ratio": unique_individual_ratio,
        "genotype_diversity": genotype_diversity,
    }


def feature_population_diversity(population):
    size = len(population)
    if size < 2 or len(population[0]) <= 16:
        return _population_diversity(population)

    width = len(population[0])
    counts = np.zeros(width, dtype=np.int64)
    distinct = set()
    for mask in population:
        if (
            len(mask) != width
            or not all(map(is_, map(type, mask), repeat(int)))
            or not set(mask) <= {0, 1}
        ):
            return _population_diversity(population)
        row = np.frombuffer(bytes(iter(mask)), dtype=np.uint8)
        counts += row
        distinct.add(np.packbits(row).tobytes())

    pairs = size * (size - 1)
    # Preserve the generic calculation's floating-point subtraction order.
    equal_pairs = pairs - 2 * counts * (size - counts)
    return {
        "population_size": size,
        "unique_individuals": len(distinct),
        "unique_individual_ratio": len(distinct) / size,
        "genotype_diversity": float(np.mean(1 - equal_pairs / pairs)),
    }


def complete_generation(record, state, gen, primary_weight, current_best, control_record=None):
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


def _new_telemetry_state():
    return {
        "best_fitness": None,
        "best_generation": 0,
        "stagnation_generations": 0,
    }
