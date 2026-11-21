"""Interface for third-party optimizers of GASearchCV and GAFeatureSelectionCV."""

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from importlib.metadata import entry_points
from numbers import Integral, Real
from typing import Literal, Protocol

from .config import EvolutionConfig, OptimizationConfig

__all__ = ["OptimizationProblem", "Optimizer"]


@dataclass(frozen=True)
class OptimizationProblem:
    """Inputs for one fit. Each proposal follows the coordinate order of the initial rows.

    For parameters, names and dimensions describe those coordinates. For features,
    both tuples are empty and each row is a nonempty binary mask. Treat categorical
    values and dimensions as read-only. The seed belongs to this invocation.
    """

    task: Literal["parameters", "features"]
    initial_population: tuple
    parameter_names: tuple
    dimensions: tuple
    max_features: int | None
    generations: int
    crossover_probability: float
    seed: int

    @property
    def max_evaluations(self):
        """Maximum number of proposals, including cache hits and initial rows."""
        return len(self.initial_population) * (self.generations + 1)


class Optimizer(Protocol):
    """One synchronous optimization run. Lower loss is better. Handles belong to one fit."""

    def __call__(
        self,
        problem: OptimizationProblem,
        evaluate: Callable[[Iterable[Sequence]], list[tuple[float, object]]],
        report: Callable[[Iterable[object]], None],
        /,
    ) -> None: ...


def _validate_controls(estimator, label):
    if estimator.callbacks:
        raise ValueError("Callbacks, including ModelCheckpoint, require optimizer_backend='deap'")
    if not isinstance(estimator.population_size, Integral) or estimator.population_size < 1:
        raise ValueError(f"{label} requires positive integer population_size")
    if not isinstance(estimator.keep_top_k, Integral) or estimator.keep_top_k < 1:
        raise ValueError(f"{label} requires positive integer keep_top_k")
    if not isinstance(estimator.generations, Integral) or estimator.generations < 0:
        raise ValueError(f"{label} requires nonnegative integer generations")
    if not isinstance(estimator.crossover_probability, Real) or not (
        0 <= estimator.crossover_probability <= 1
    ):
        raise ValueError(f"{label} requires a fixed crossover_probability in [0, 1]")

    # The legacy constructor defaults also apply when only optimizer_backend is supplied.
    for name in ("algorithm", "mutation_probability", "tournament_size", "elitism"):
        if getattr(estimator, name) != getattr(EvolutionConfig, name):
            raise ValueError(f"{name} is a DEAP-only option")
    for name in OptimizationConfig.__dataclass_fields__:
        if name == "diversity_control" or name.startswith("final_selection"):
            continue
        if getattr(estimator, name) != getattr(OptimizationConfig, name):
            raise ValueError(f"{name} is a DEAP-only option")


def resolve_optimizer(estimator) -> Optimizer | None:
    """Load only the selected engine, before checkpoint I/O or any model fits."""
    name = estimator.optimizer_backend
    if isinstance(name, str) and name == "deap":
        return None
    if isinstance(name, str) and name == "scipy":
        from ._scipy_backend import optimize, validate_scipy_backend

        _validate_controls(estimator, "SciPy")
        validate_scipy_backend(estimator)
        return optimize
    _validate_controls(estimator, "Optimizer plugins")
    if callable(name):
        return name
    if not isinstance(name, str) or not name:
        raise ValueError("optimizer_backend must be an installed backend name or a callable")
    matches = tuple(entry_points(group="sklearn_genetic.optimizers", name=name))
    if not matches:
        raise ValueError(f"Unknown optimizer_backend {name!r}; install the corresponding plugin")
    if len(matches) > 1:
        raise ValueError(f"Multiple optimizer plugins registered as {name!r}")
    optimize = matches[0].load()
    if not callable(optimize):
        raise TypeError(f"Optimizer plugin {name!r} must export a callable")
    return optimize
