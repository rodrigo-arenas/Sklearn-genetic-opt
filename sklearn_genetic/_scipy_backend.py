"""Internal adapter for SciPy's vectorized differential evolution solver."""

from collections import defaultdict
from numbers import Integral, Real

import numpy as np
from scipy import __version__ as scipy_version
from scipy.optimize import differential_evolution

from .config import EvolutionConfig, OptimizationConfig
from .space import Categorical, Continuous, Integer


class _EvaluationError(Exception):
    """Carry CV errors past SciPy's map-wrapper validation."""


def validate_scipy_backend(estimator):
    if np.lib.NumpyVersion(scipy_version) < "1.15.0":
        raise ImportError("Install sklearn-genetic-opt[scipy] for SciPy >= 1.15")
    if estimator.callbacks:
        raise ValueError("Callbacks, including ModelCheckpoint, require optimizer_backend='deap'")
    if not isinstance(estimator.population_size, Integral) or estimator.population_size < 5:
        raise ValueError("SciPy requires population_size >= 5")
    if not isinstance(estimator.generations, Integral) or estimator.generations < 0:
        raise ValueError("SciPy requires nonnegative integer generations")
    if not isinstance(estimator.crossover_probability, Real) or not (
        0 <= estimator.crossover_probability <= 1
    ):
        raise ValueError("SciPy requires a fixed crossover_probability in [0, 1]")

    # The legacy constructor defaults also apply when only optimizer_backend is supplied.
    for name in ("algorithm", "mutation_probability", "tournament_size", "elitism"):
        if getattr(estimator, name) != getattr(EvolutionConfig, name):
            raise ValueError(f"{name} is a DEAP-only option")
    for name in OptimizationConfig.__dataclass_fields__:
        if name == "diversity_control" or name.startswith("final_selection"):
            continue
        if getattr(estimator, name) != getattr(OptimizationConfig, name):
            raise ValueError(f"{name} is a DEAP-only option")

    for name, dim in getattr(estimator, "param_grid", {}).items():
        if isinstance(dim, Categorical) and dim.priors is None and len(dim.choices):
            continue
        if isinstance(dim, Integer) and (
            dim.upper == dim.lower
            or (-(2**63) <= dim.lower <= dim.upper < 2**63 and dim.upper - dim.lower < 2**52)
        ):
            continue
        if isinstance(dim, Continuous) and np.isfinite(float(dim.upper) - float(dim.lower)):
            continue
        raise ValueError(
            f"SciPy cannot encode dimension {name!r}. Use unweighted categories, "
            "fixed integers, int64 spans below 2**52 or finite continuous intervals."
        )


def _encode(dim, value):
    if isinstance(dim, Categorical):
        return (dim._choice_index(value) + 0.5) / len(dim.choices)
    if isinstance(dim, Integer):
        return (int(value) - dim.lower + 0.5) / (dim.upper - dim.lower + 1)
    if dim.lower == dim.upper:
        return 0.5
    if dim.distribution == "log-uniform" and np.log(dim.upper) != np.log(dim.lower):
        return (np.log(value) - np.log(dim.lower)) / (np.log(dim.upper) - np.log(dim.lower))
    return (value - dim.lower) / (dim.upper - dim.lower)


def _decode(dim, value):
    if isinstance(dim, Categorical):
        return dim.choices[min(int(value * len(dim.choices)), len(dim.choices) - 1)]
    if isinstance(dim, Integer):
        return min(dim.lower + int(value * (dim.upper - dim.lower + 1)), dim.upper)
    if dim.lower == dim.upper:
        return float(dim.lower)
    if dim.distribution == "log-uniform" and np.log(dim.upper) != np.log(dim.lower):
        return float(np.exp(np.log(dim.lower) + value * (np.log(dim.upper) - np.log(dim.lower))))
    return float(dim.lower + value * (dim.upper - dim.lower))


def _decode_mask(row, max_features):
    selected = row >= 0.5
    if not selected.any():
        selected[row.argmax()] = True
    elif max_features is not None and selected.sum() > max_features:
        selected[np.argsort(-row, kind="stable")[max_features:]] = False
    return selected.astype(int).tolist()


def run_scipy(
    initial, evaluate, report, *, dimensions, max_features, generations, recombination, seed
):
    """Ask for (loss, handle) pairs and report survivor handles in generation order.

    Candidates are plain lists. A handle identifies one evaluated proposal, even if
    several proposals decode to the same candidate. Only the caller interprets it.
    Equal coordinates and loss are indistinguishable in SciPy's survivor report.
    """
    if dimensions:
        encoded = [[_encode(dim, x) for dim, x in zip(dimensions, ind)] for ind in initial]
    else:
        encoded = initial
    # Keep handles for parents and trials until the solver reports its survivors.
    evaluated = defaultdict(list)
    first_batch = True

    def objective(columns):
        nonlocal first_batch
        rows = columns.T
        if first_batch:
            # Keep exact warm starts, including floats that lose precision in a round trip.
            candidates = initial
        elif dimensions:
            candidates = [[_decode(dim, x) for dim, x in zip(dimensions, row)] for row in rows]
        else:
            candidates = [_decode_mask(row, max_features) for row in rows]

        try:
            results = list(evaluate(candidates))
        except (TypeError, ValueError) as exc:
            raise _EvaluationError from exc
        losses = []
        handles = []
        for row, (loss, handle) in zip(rows, results, strict=True):
            # An all-infinite initial population makes SciPy evaluate initialization again.
            loss = loss if np.isfinite(loss) else np.finfo(float).max
            evaluated[(row.tobytes(), loss)].append(handle)
            handles.append(handle)
            losses.append(loss)
        if first_batch:
            report(handles)
            first_batch = False
        return np.asarray(losses)

    def generation_finished(intermediate_result):
        nonlocal evaluated
        keys = [
            (row.tobytes(), loss)
            for row, loss in zip(
                intermediate_result.population, intermediate_result.population_energies
            )
        ]
        # Equal coordinates and energy carry no trial ID. Use each matching
        # evaluation at most once and retain only handles for surviving copies.
        survivors = [evaluated[key].pop() for key in keys]
        evaluated = defaultdict(list)
        for key, handle in zip(keys, survivors):
            evaluated[key].append(handle)
        report(survivors)

    try:
        differential_evolution(
            objective,
            bounds=[(0.0, 1.0)] * len(initial[0]),
            init=np.asarray(encoded, dtype=float),
            maxiter=generations,
            strategy="best1bin",
            recombination=recombination,
            updating="deferred",
            vectorized=True,
            polish=False,
            tol=0,
            rng=seed,
            callback=generation_finished,
        )
    except _EvaluationError as exc:
        raise exc.__cause__ from None
