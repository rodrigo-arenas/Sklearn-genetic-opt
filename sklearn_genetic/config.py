from dataclasses import dataclass, field

import numpy as np


@dataclass
class EvolutionConfig:
    """Core genetic algorithm controls."""

    population_size: int = 50
    generations: int = 80
    crossover_probability: object = 0.2
    mutation_probability: object = 0.8
    tournament_size: int = 3
    elitism: bool = True
    keep_top_k: int = 1
    criteria: str = "max"
    algorithm: str = "eaMuPlusLambda"


@dataclass
class PopulationConfig:
    """Initial population strategy."""

    initializer: str = "smart"
    warm_start_configs: list[dict] = field(default_factory=list)


@dataclass
class RuntimeConfig:
    """Execution, parallelism, and result-collection controls."""

    n_jobs: int | None = None
    pre_dispatch: str | int | None = "2*n_jobs"
    error_score: float | str = np.nan
    return_train_score: bool = False
    use_cache: bool = True
    parallel_backend: str = "auto"
    verbose: bool = True


@dataclass
class OptimizationConfig:
    """Optional quality controls used around the main GA loop."""

    local_search: bool = False
    local_search_top_k: int = 1
    local_search_steps: int = 1
    local_search_radius: float = 0.1
    diversity_control: bool = False
    diversity_threshold: float = 0.1
    diversity_stagnation_generations: int = 5
    diversity_mutation_boost: float = 2.0
    random_immigrants_fraction: float = 0.1
    adaptive_selection: bool = False
    selection_pressure_min: int = 2
    selection_pressure_max: int | None = None
    offspring_diversity_retries: int = 0
    fitness_sharing: bool = False
    sharing_radius: float = 0.2
    sharing_alpha: float = 1.0
    final_selection: bool = False
    final_selection_top_k: int = 3
    final_selection_cv: object = None
