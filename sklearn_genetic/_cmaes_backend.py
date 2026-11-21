"""CatCMA with Margin for continuous, integer and unordered coordinates."""

import numpy as np

from ._mixed_space import MixedSpace
from .config import EvolutionConfig


def optimize(problem, evaluate, report, *, sigma=0.25):
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
    initial = evaluate(problem.initial_population)
    report([handle for _, handle in initial])
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
    elite = np.asarray(
        [
            space.encode(problem.initial_population[i])
            for i in np.argsort([loss for loss, _ in initial], kind="stable")[: max(1, size // 4)]
        ]
    )
    gaussian = real + integer
    mean = (elite[:, gaussian] / spans[gaussian]).mean(axis=0) if gaussian else None
    counts = [int(spans[i]) + 1 for i in cat]
    probs = None
    if cat:
        probs = np.zeros((len(cat), max(counts)))
        for j, (i, k) in enumerate(zip(cat, counts)):
            frequencies = np.bincount(elite[:, i].astype(int), minlength=k) / len(elite)
            probs[j, :k] = (frequencies + 1 / k) / 2
    solver = CatCMAwM(
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
    penalty_initialized = False
    for _ in range(problem.generations):
        trials = [solver.ask() for _ in range(size)]
        rows = []
        for trial in trials:
            coords = [None] * len(space.axes)
            for indices, values in ((real, trial.x), (integer, trial.z)):
                for i, value in zip(indices, values if indices else (), strict=True):
                    bounded = np.clip(value, 0, 1)
                    coords[i] = bounded if space.axes[i][0] == "real" else round(bounded * spans[i])
            for j, i in enumerate(cat):
                coords[i] = int(trial.c[j].argmax())
            rows.append(space.decode(coords))
        results = evaluate(rows)
        # CatCMAwM's continuous-bound penalty needs a positive interquartile range
        # on its first update. Keep sampling on a flat landscape until it exists.
        losses = sorted(loss for loss, _ in results)
        if real and not penalty_initialized:
            spread = losses[3 * size // 4] - losses[size // 4]
            penalty_initialized = np.isfinite(spread) and spread > 0
        if not real or penalty_initialized:
            solver.tell([(trial, loss) for trial, (loss, _) in zip(trials, results, strict=True)])
        report([handle for _, handle in results])
