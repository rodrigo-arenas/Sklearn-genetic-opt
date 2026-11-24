"""CatCMA with Margin for continuous, integer and unordered coordinates."""

from math import frexp, isfinite, ldexp

import numpy as np

from ._mixed_space import MixedSpace
from ._optimizer_checkpoint import native_checkpoint, report_native
from .config import EvolutionConfig

# Leave headroom for the continuous boundary penalty's covariance and step-size factors.
_LOSS_EXPONENT_LIMIT = 500


@native_checkpoint("cmaes")
def optimize(problem, evaluate, report, *, sigma=0.25, _state=None):
    try:
        from cmaes import CatCMAwM
    except ModuleNotFoundError as exc:
        if exc.name != "cmaes":
            raise
        raise ImportError("Install sklearn-genetic-opt[cmaes] to use this optimizer") from exc
    if problem.crossover_probability != EvolutionConfig.crossover_probability:
        raise ValueError("CatCMAwM manages crossover_probability internally")
    size = len(problem.initial_population)
    if size < 4:
        raise ValueError("CatCMAwM requires population_size >= 4")
    if not 0 < sigma <= 1:
        raise ValueError("sigma must be in (0, 1]")
    space = MixedSpace(problem)
    if _state is None:
        initial = evaluate(problem.initial_population)
        losses = [loss for loss, _ in initial]
        _state = (0, None, False, losses)
        report_native(problem, report, [handle for _, handle in initial], _state)
    generation, solver, penalty_initialized, initial_losses, *saved_scale = _state
    loss_exponent = saved_scale[0] if saved_scale else 0
    if not space.axes or not problem.generations:
        return
    # Wide integer intervals would allocate a dense table proportional to their range.
    # Represent these on a unit interval, then round during decoding.
    real = [
        i
        for i, (kind, span) in enumerate(space.axes)
        if kind == "real" or (kind == "int" and span > 255)
    ]
    integer = [i for i, (kind, span) in enumerate(space.axes) if kind == "int" and span <= 255]
    cat = [i for i, (kind, _) in enumerate(space.axes) if kind in ("choice", "bit")]
    spans = np.asarray([upper for _, upper in space.axes])
    if solver is None:
        solver = _make_solver(
            CatCMAwM, problem, space, initial_losses, real, integer, cat, spans, sigma
        )
    for gen in range(generation + 1, problem.generations + 1):
        trials = [solver.ask() for _ in range(size)]
        if problem.task == "features":
            rows = [space.decode(trial.c.argmax(axis=1)) for trial in trials]
        else:
            rows = []
            for trial in trials:
                coords = [None] * len(space.axes)
                for indices, values in ((real, trial.x), (integer, trial.z)):
                    for i, value in zip(indices, values if indices else (), strict=True):
                        bounded = np.clip(value, 0, 1)
                        coords[i] = (
                            bounded if space.axes[i][0] == "real" else round(bounded * spans[i])
                        )
                for j, i in enumerate(cat):
                    coords[i] = int(trial.c[j].argmax())
                rows.append(space.decode(coords))
        results = evaluate(rows)
        losses = [loss for loss, _ in results]
        if real:
            losses, loss_exponent = _scale_continuous_losses(solver, losses, loss_exponent)
        if not real or _continuous_update_ready(losses, penalty_initialized):
            solver.tell(list(zip(trials, losses, strict=True)))
            penalty_initialized = bool(real)
        report_native(
            problem,
            report,
            [handle for _, handle in results],
            (gen, solver, penalty_initialized, None, loss_exponent),
        )


def _scale_continuous_losses(solver, losses, exponent):
    largest = max((abs(loss) for loss in losses if isfinite(loss)), default=0.0)
    required = frexp(largest)[1] - _LOSS_EXPONENT_LIMIT
    if required > exponent:
        # CatCMAwM 0.13 uses this history to price future boundary penalties.
        # Earlier spreads must be in the same units as the next call to tell.
        solver._iqhist_values[:] = [
            ldexp(spread, exponent - required) for spread in solver._iqhist_values
        ]
        exponent = required
    if exponent:
        losses = [ldexp(loss, -exponent) for loss in losses]
    return losses, exponent


def _continuous_update_ready(losses, penalty_initialized):
    losses = sorted(losses)
    size = len(losses)
    lower, upper = losses[size // 4], losses[3 * size // 4]
    # A failed lower quartile would make the solver subtract two infinities.
    if not np.isfinite(lower):
        return False
    if not np.isfinite(upper):
        return penalty_initialized and upper == np.inf
    # The first update needs a positive, representable range for the boundary penalty.
    with np.errstate(over="ignore"):
        spread = upper - lower
    return np.isfinite(spread) and (penalty_initialized or spread > 0)


def _make_solver(cls, problem, space, initial_losses, real, integer, cat, spans, sigma):
    size = len(problem.initial_population)
    elite = np.asarray(
        [
            space.encode(problem.initial_population[i])
            for i in np.argsort(initial_losses, kind="stable")[: max(1, size // 4)]
        ]
    )
    gaussian = real + integer
    mean = (elite[:, gaussian] / spans[gaussian]).mean(axis=0) if gaussian else None
    counts = [int(spans[i]) + 1 for i in cat]
    probs = None
    if problem.task == "features":
        ones = elite.sum(axis=0)
        frequencies = np.column_stack((len(elite) - ones, ones)) / len(elite)
        probs = (frequencies + 0.5) / 2
    elif cat:
        probs = np.zeros((len(cat), max(counts)))
        for j, (i, k) in enumerate(zip(cat, counts)):
            frequencies = np.bincount(elite[:, i].astype(int), minlength=k) / len(elite)
            probs[j, :k] = (frequencies + 1 / k) / 2
    return cls(
        x_space=[(0.0, 1.0)] * len(real) or None,
        z_space=[np.linspace(0, 1, int(spans[i]) + 1) for i in integer] or None,
        c_space=counts or None,
        mean=mean,
        # The upstream default has zero variance when an elite lies on a bound.
        cov=np.eye(len(gaussian)) if gaussian else None,
        sigma=sigma,
        cat_param=probs,
        population_size=size,
        seed=problem.seed,
    )
