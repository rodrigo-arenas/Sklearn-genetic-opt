"""Internal adapter for SciPy's vectorized differential evolution solver."""

from collections import defaultdict
from math import copysign, inf, isfinite, sqrt
from sys import float_info

import numpy as np
from scipy import __version__ as scipy_version
from scipy.optimize import differential_evolution

from ._optimizer_checkpoint import replay_checkpoint
from ._mixed_space import decode_real, encode_real, real_coordinate_interval
from .space import Categorical, Continuous, Integer
from .utils._score_statistics import is_finite_score, score_scale_limits


class _EvaluationError(Exception):
    """Carry CV errors past SciPy's map-wrapper validation."""


def validate_scipy_backend(estimator):
    if np.lib.NumpyVersion(scipy_version) < "1.15.0":
        raise ImportError("Install sklearn-genetic-opt[scipy] for SciPy >= 1.15")
    if estimator.population_size < 5:
        raise ValueError("SciPy requires population_size >= 5")
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


def _encode(dim, value, interval):
    if isinstance(dim, Categorical):
        return (dim._choice_index(value) + 0.5) / len(dim.choices)
    if isinstance(dim, Integer):
        return (int(value) - dim.lower + 0.5) / (dim.upper - dim.lower + 1)
    return encode_real(dim, value, interval=interval)


def _decode(dim, value, interval):
    if isinstance(dim, Categorical):
        return dim.choices[min(int(value * len(dim.choices)), len(dim.choices) - 1)]
    if isinstance(dim, Integer):
        return min(dim.lower + int(value * (dim.upper - dim.lower + 1)), dim.upper)
    return decode_real(dim, value, interval=interval)


def _decode_mask(row, max_features):
    selected = row >= 0.5
    if not selected.any():
        selected[row.argmax()] = True
    elif max_features is not None and selected.sum() > max_features:
        selected[np.argsort(-row, kind="stable")[max_features:]] = False
    return selected.astype(int).tolist()


def _losses_converged(losses):
    first = losses[0]
    if not np.isfinite(first) or not np.all(losses == first):
        return False
    _, lower, upper = score_scale_limits(losses.dtype)
    scale = abs(first)
    if scale < lower or scale >= upper / sqrt(len(losses)):
        return True
    # Preserve SciPy's zero-tolerance decision for ordinary flat populations,
    # including its rounding of repeated values in previously saved searches.
    return np.std(losses) == 0


def run_scipy(
    initial,
    evaluate,
    report,
    *,
    dimensions,
    max_features,
    generations,
    recombination,
    seed,
    stop_on_convergence=True,
):
    """Ask for (loss, handle) pairs and report survivor handles in generation order.

    Candidates are plain lists. A handle identifies one evaluated proposal, even if
    several proposals decode to the same candidate. Only the caller interprets it.
    Equal coordinates and loss are indistinguishable in SciPy's survivor report.
    """
    coordinates = [
        (dim, real_coordinate_interval(dim) if type(dim) is Continuous else None)
        for dim in dimensions
    ]
    if coordinates:
        encoded = [
            [_encode(dim, x, interval) for (dim, interval), x in zip(coordinates, row)]
            for row in initial
        ]
    else:
        encoded = initial
    # Keep handles for parents and trials until the solver reports its survivors.
    evaluated = defaultdict(list)
    first_batch = True
    retry_rows = None
    caller_policy = np.geterr()

    def objective(columns):
        nonlocal first_batch, retry_rows
        rows = columns.T
        # SciPy treats an all-infinite population as uninitialized and asks
        # for its parent energies again before proposing the next trial batch.
        if retry_rows is not None:
            previous, retry_rows = retry_rows, None
            if np.array_equal(rows, previous):
                return np.full(len(rows), np.inf)
        with np.errstate(**caller_policy):
            if first_batch:
                # Preserve warm starts that lose precision in an encode/decode round trip.
                candidates = initial
            elif coordinates:
                candidates = [
                    [_decode(dim, x, interval) for (dim, interval), x in zip(coordinates, row)]
                    for row in rows
                ]
            else:
                candidates = [_decode_mask(row, max_features) for row in rows]
            try:
                results = list(evaluate(candidates))
            except (TypeError, ValueError) as exc:
                raise _EvaluationError from exc
        losses = []
        handles = []
        for row, (loss, handle) in zip(rows, results, strict=True):
            # SciPy stores energies as doubles. Index handles by the exact
            # value the solver will report, keeping infinity for invalid CV.
            solver_loss = float(loss)
            if not isfinite(solver_loss):
                solver_loss = (
                    copysign(float_info.max, solver_loss) if is_finite_score(loss) else inf
                )
            evaluated[(row.tobytes(), solver_loss)].append(handle)
            handles.append(handle)
            losses.append(solver_loss)
        if first_batch:
            if all(loss == np.inf for loss in losses):
                retry_rows = rows.copy()
            with np.errstate(**caller_policy):
                report(handles)
            first_batch = False
        return np.asarray(losses)

    def generation_finished(intermediate_result):
        nonlocal evaluated, retry_rows
        population = intermediate_result.population
        energies = intermediate_result.population_energies
        keys = [(row.tobytes(), loss) for row, loss in zip(population, energies)]
        # Equal coordinates and energy carry no trial ID. Use each matching
        # evaluation at most once and retain only handles for surviving copies.
        survivors = [evaluated[key].pop() for key in keys]
        evaluated = defaultdict(list)
        for key, handle in zip(keys, survivors):
            evaluated[key].append(handle)
        if energies.min() == np.inf:
            retry_rows = population.copy()
        with np.errstate(**caller_policy):
            replay_pending = report(survivors)
        return stop_on_convergence and not replay_pending and _losses_converged(energies)

    try:
        # SciPy still computes raw moments for its convergence diagnostics.
        # Survivor selection uses the double-precision losses and the callback makes
        # our stopping decision without those potentially overflowing moments.
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
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
                atol=-1.0,
                rng=seed,
                callback=generation_finished,
            )
    except _EvaluationError as exc:
        raise exc.__cause__ from None


@replay_checkpoint("scipy")
def optimize(problem, evaluate, report, *, stop_on_convergence=True):
    run_scipy(
        problem.initial_population,
        evaluate,
        report,
        dimensions=problem.dimensions,
        max_features=problem.max_features,
        generations=problem.generations,
        recombination=problem.crossover_probability,
        seed=problem.seed,
        stop_on_convergence=stop_on_convergence,
    )
