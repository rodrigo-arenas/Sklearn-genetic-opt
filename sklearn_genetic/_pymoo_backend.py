"""pymoo mixed-variable GA and binary GA with estimator-owned evaluations."""

import numpy as np

from ._mixed_space import MixedSpace


def optimize(problem, evaluate, report):
    try:
        from pymoo.algorithms.soo.nonconvex.ga import GA
        from pymoo.core.mixed import MixedVariableGA, MixedVariableMating
        from pymoo.core.population import Population
        from pymoo.core.problem import Problem
        from pymoo.core.repair import Repair
        from pymoo.core.variable import Choice, Integer, Real
        from pymoo.operators.crossover.ux import UX
        from pymoo.operators.mutation.bitflip import BitflipMutation
    except ModuleNotFoundError as exc:
        if exc.name != "pymoo":
            raise
        raise ImportError("Install sklearn-genetic-opt[pymoo] to use this optimizer") from exc
    space = MixedSpace(problem)
    if not space.axes or not problem.generations:
        report([handle for _, handle in evaluate(problem.initial_population)])
        return
    size = len(problem.initial_population)
    initial = [space.encode(row) for row in problem.initial_population]
    prob = problem.crossover_probability
    binary = problem.task == "features"
    if binary:
        model = Problem(n_var=len(space.axes), xl=0, xu=1, vtype=bool)
        pop = Population.new(X=np.asarray(initial, dtype=bool))
        solver = GA(
            pop_size=size,
            sampling=pop,
            crossover=UX(prob=prob),
            mutation=BitflipMutation(),
            eliminate_duplicates=False,
        )
    else:
        variables = {}
        for i, (kind, upper) in enumerate(space.axes):
            if kind == "choice":
                variables[i] = Choice(options=range(upper + 1))
            elif kind == "int":
                variables[i] = Integer(bounds=(0, upper))
            else:
                variables[i] = Real(bounds=(0, 1))
        model = Problem(vars=variables)
        pop = Population.new(X=[dict(enumerate(row)) for row in initial])
        mating = MixedVariableMating(eliminate_duplicates=None)

        # NumPy 1.x uses a 32-bit default int on Windows. Keep wide spans intact.
        class Int64Repair(Repair):
            def _do(self, problem, X, **kwargs):
                return np.rint(X).astype(np.int64)

        mating.crossover[Integer].repair = mating.mutation[Integer].repair = Int64Repair()
        for operator in mating.crossover.values():
            operator.prob.set(prob)
        solver = MixedVariableGA(
            pop_size=size, sampling=pop, mating=mating, eliminate_duplicates=False
        )
    solver.setup(model, seed=problem.seed, termination=("n_gen", problem.generations + 1))
    for gen in range(problem.generations + 1):
        trials = solver.ask()
        if gen == 0:
            rows = problem.initial_population
        elif binary:
            rows = [space.decode(x) for x in trials.get("X")]
            trials.set("X", np.asarray(rows, dtype=bool))
        else:
            rows = [space.decode([x[i] for i in variables]) for x in trials.get("X")]
        results = evaluate(rows)
        for trial, (loss, handle) in zip(trials, results, strict=True):
            trial.F = np.array([loss])
            trial.set("observation", handle)
        solver.tell(infills=trials)
        report([ind.get("observation") for ind in solver.pop])
