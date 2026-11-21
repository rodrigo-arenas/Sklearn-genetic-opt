"""Nevergrad optimizers with batched ask/tell and typed coordinates."""

import numpy as np

from ._mixed_space import MixedSpace
from ._optimizer_checkpoint import native_checkpoint, report_native
from .config import EvolutionConfig


@native_checkpoint("nevergrad")
def optimize(problem, evaluate, report, *, strategy="OnePlusOne", _state=None):
    try:
        import nevergrad as ng
    except ModuleNotFoundError as exc:
        if exc.name != "nevergrad":
            raise
        raise ImportError("Install sklearn-genetic-opt[nevergrad] to use this optimizer") from exc
    if problem.crossover_probability != EvolutionConfig.crossover_probability:
        raise ValueError("Nevergrad manages crossover_probability internally")
    space = MixedSpace(problem)
    # NGOpt's portfolio can select an incompatible metamodel in mixed spaces too.
    if strategy not in ("OnePlusOne", "TwoPointsDE"):
        raise ValueError("strategy must be OnePlusOne or TwoPointsDE")
    size = len(problem.initial_population)
    if _state is None:
        initial = evaluate(problem.initial_population)
        best_index = int(np.argmin([loss for loss, _ in initial]))
        _state = (0, None, best_index)
        report_native(problem, report, [handle for _, handle in initial], _state)
    generation, solver, best_index = _state
    if not space.axes or not problem.generations:
        return
    if solver is None:
        best = space.encode(problem.initial_population[best_index])
        solver = _make_solver(ng, problem, space, best, size, strategy)
    for gen in range(generation + 1, problem.generations + 1):
        trials = [solver.ask() for _ in range(size)]
        results = evaluate([space.decode(trial.value) for trial in trials])
        for trial, (loss, _) in zip(trials, results, strict=True):
            solver.tell(trial, loss)
        report_native(problem, report, [handle for _, handle in results], (gen, solver, None))


def _make_solver(ng, problem, space, best, size, strategy):
    if problem.task == "features":
        param = ng.p.TransitionChoice([0, 1], repetitions=len(best))
        param.value = tuple(best)
    else:
        coords = []
        for (kind, upper), value in zip(space.axes, best):
            if kind == "choice":
                coord = ng.p.Choice(range(upper + 1))
                coord.value = value
            else:
                coord = ng.p.Scalar(init=value, lower=0, upper=upper)
                if kind == "int":
                    coord.set_integer_casting()
            coords.append(coord)
        param = ng.p.Tuple(*coords)
    param.random_state.seed(problem.seed)
    return ng.optimizers.registry[strategy](
        parametrization=param, budget=size * problem.generations, num_workers=size
    )
