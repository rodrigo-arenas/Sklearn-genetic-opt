import random
from types import SimpleNamespace

import numpy as np
import pytest
from deap import base, creator, tools

from ..algorithms import eaDifferentialEvolution, eaMuCommaLambda, eaMuPlusLambda, eaSimple
from ..callbacks import ConsecutiveStopping, DeltaThreshold
from ..callbacks.base import BaseCallback
from ..evaluation import create_fit_stats
from ..optimizer_control import shared_fitness
from ..schedules.schedulers import ConstantAdapter
from ..space import Categorical, Continuous, Integer, Space


class StopOnFirstStep(BaseCallback):
    def __init__(self):
        self.started = False
        self.ended = False

    def on_start(self, estimator=None):
        self.started = True

    def on_step(self, record=None, logbook=None, estimator=None):
        return True

    def on_end(self, logbook=None, estimator=None):
        self.ended = True


@pytest.fixture
def simple_toolbox():
    if not hasattr(creator, "FitnessAlgorithmTest"):
        creator.create("FitnessAlgorithmTest", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "IndividualAlgorithmTest"):
        creator.create("IndividualAlgorithmTest", list, fitness=creator.FitnessAlgorithmTest)

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register(
        "individual",
        tools.initRepeat,
        creator.IndividualAlgorithmTest,
        toolbox.attr_bool,
        4,
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", lambda individual: (sum(individual),))
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.5)
    toolbox.register("select", tools.selBest)
    toolbox.register("map", map)
    return toolbox


@pytest.mark.parametrize(
    "algorithm, kwargs",
    [
        (eaSimple, {}),
        (eaMuPlusLambda, {"mu": 4, "lambda_": 4}),
        (eaMuCommaLambda, {"mu": 4, "lambda_": 4}),
        (eaDifferentialEvolution, {}),
    ],
)
def test_algorithm_stops_when_callback_requests_stop(simple_toolbox, algorithm, kwargs):
    random_state = random.getstate()
    numpy_state = np.random.get_state()

    try:
        random.seed(0)
        np.random.seed(0)
        population = simple_toolbox.population(n=4)
        callback = StopOnFirstStep()
        stats = tools.Statistics(lambda individual: individual.fitness.values)
        stats.register("fitness", max)

        _, logbook, n_gen = algorithm(
            population=population,
            toolbox=simple_toolbox,
            cxpb=ConstantAdapter(0.5, 0.5, 0),
            mutpb=ConstantAdapter(0.5, 0.5, 0),
            ngen=3,
            stats=stats,
            halloffame=tools.HallOfFame(1),
            callbacks=[callback],
            verbose=False,
            estimator=SimpleNamespace(elitism=False),
            **kwargs,
        )
    finally:
        random.setstate(random_state)
        np.random.set_state(numpy_state)

    assert callback.started
    assert callback.ended
    assert n_gen == 0
    assert len(logbook) == 1


@pytest.mark.parametrize("weight", [1.0, -1.0])
@pytest.mark.parametrize("all_invalid", [False, True])
def test_differential_evolution_replaces_only_improved_targets(
    simple_toolbox, monkeypatch, weight, all_invalid
):
    # Trials are prescribed to isolate the public algorithm's survival contract:
    # improvement, deterioration, NaN target, NaN trial, and exact tie respectively.
    class Fitness(base.Fitness):
        weights = (weight,)

    class Individual(list):
        def __init__(self, value):
            super().__init__([value])
            self.fitness = Fitness()

    parents = [Individual(value) for value in [1, 2, 3, 4, 5]]
    trials = [Individual(value) for value in [6, 7, 8, 9, 10]]
    scores = [1, 4, np.nan, 2, 3, 3, 2, 1, np.nan, 3]
    if all_invalid:
        scores[:5] = [np.nan] * 5
    simple_toolbox.register("evaluate", lambda ind: (weight * scores[ind[0] - 1],))
    monkeypatch.setattr("sklearn_genetic.algorithms.differential_offspring", lambda *args: trials)
    hall = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("fitness_max" if weight > 0 else "fitness_min", np.max)
    result, log, _ = eaDifferentialEvolution(
        parents,
        simple_toolbox,
        cxpb=ConstantAdapter(0.5, 0.5, 0),
        mutpb=ConstantAdapter(0, 0, 0),
        ngen=1,
        halloffame=hall,
        stats=stats,
        callbacks=[
            DeltaThreshold(threshold=0.001, generations=2, metric="fitness_best"),
            ConsecutiveStopping(generations=1, metric="fitness_best"),
        ],
        verbose=False,
        estimator=SimpleNamespace(elitism=False),
    )
    assert [ind[0] for ind in result] == ([6, 7, 8, 4, 10] if all_invalid else [6, 2, 8, 4, 5])
    assert hall[0][0] == (6 if all_invalid else 2)
    if all_invalid:
        assert np.isnan(log[0]["fitness_best"])
        assert log[1]["fitness_best"] == weight * 3
    else:
        assert log.select("fitness_best") == [weight * 4] * 2


@pytest.mark.parametrize(
    "dimension, parents, trials",
    [
        (Continuous(0, 1), [0.2, 0.4, 0.6, 0.8], [0.24, 0.04, 0, 0.04]),
        (
            Continuous(1, 1e10, distribution="log-uniform"),
            [1e2, 1e4, 1e6, 1e8],
            [10**2.4, 10**0.4, 1, 10**0.4],
        ),
        (Integer(1, 11), [3, 5, 7, 9], [3, 1, 1, 1]),
        (Integer(5, 5), [5, 5, 5, 5], [5, 5, 5, 5]),
        (Categorical([0.2, 0.4, 0.6, 0.8]), [0.2, 0.4, 0.6, 0.8], [0.4, 0.2, 0.2, 0.2]),
    ],
)
def test_differential_evolution_uses_deferred_distinct_donors(
    simple_toolbox, monkeypatch, dimension, parents, trials
):
    # For a one-dimensional space, CR=0 must still force the donor coordinate.
    # Excluding the target maps samples [0, 1, 2] to the three other parents.
    population = [creator.IndividualAlgorithmTest([x]) for x in parents]
    evaluated = []

    def evaluate(individual):
        evaluated.append(individual[0])
        return (individual[0],)

    simple_toolbox.register("evaluate", evaluate)
    monkeypatch.setattr(random, "sample", lambda values, k: list(values)[:k])
    estimator = SimpleNamespace(
        space=Space({"x": dimension}),
        _repair_individual=lambda individual: individual,
        elitism=False,
    )
    result, _, _ = eaDifferentialEvolution(
        population,
        simple_toolbox,
        cxpb=ConstantAdapter(0, 0, 0),
        mutpb=ConstantAdapter(0, 0, 0),
        ngen=1,
        halloffame=tools.HallOfFame(1),
        callbacks=[],
        verbose=False,
        estimator=estimator,
    )
    assert evaluated == pytest.approx(parents + trials)
    assert [ind[0] for ind in result] == pytest.approx(
        [max(parent, trial) for parent, trial in zip(parents, trials)]
    )


@pytest.mark.parametrize(
    "algorithm", [eaSimple, eaMuPlusLambda, eaMuCommaLambda, eaDifferentialEvolution]
)
@pytest.mark.parametrize(
    "case",
    [
        "covered",
        "incomplete",
        "continuous",
        "uncached",
        "no_archive",
        "feature_pairs",
        "feature_unbounded",
    ],
)
def test_space_exhaustion_counts_current_run_proposals(
    simple_toolbox, monkeypatch, algorithm, case
):
    feature_case = case.startswith("feature_")
    initial = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]] if feature_case else [[0]] * 4
    proposed = (
        [[1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 0]]
        if feature_case
        else [[v] for v in (1, 2, 3, 4)]
    )
    population = [creator.IndividualAlgorithmTest(values) for values in initial]

    def trials(*args):
        return [creator.IndividualAlgorithmTest(values) for values in proposed]

    monkeypatch.setattr("sklearn_genetic.algorithms.varAnd", trials)
    monkeypatch.setattr("sklearn_genetic.algorithms.varOr", trials)
    monkeypatch.setattr("sklearn_genetic.algorithms.differential_offspring", trials)
    dimension = (
        Continuous(0, 4) if case == "continuous" else Integer(0, 5 if case == "incomplete" else 4)
    )
    estimator = SimpleNamespace(
        space=Space({"x": dimension}),
        elitism=False,
        use_cache=case != "uncached",
        _individual_key=tuple,
        # Restored cache entries do not prove the current run observed them.
        fitness_cache={(value,): (float(value),) for value in range(10)},
    )
    if feature_case:
        del estimator.space
        estimator.n_features = 3
        estimator.max_features = 2 if case == "feature_pairs" else None
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("fitness_max", np.max)
    _, log, _ = algorithm(
        population,
        simple_toolbox,
        cxpb=ConstantAdapter(0.5, 0.5, 0),
        mutpb=ConstantAdapter(0.5, 0.5, 0),
        ngen=3,
        halloffame=None if case == "no_archive" else tools.HallOfFame(1),
        stats=stats,
        callbacks=[],
        verbose=False,
        estimator=estimator,
        **({"mu": 4, "lambda_": 4} if algorithm in (eaMuPlusLambda, eaMuCommaLambda) else {}),
    )
    covered = case in ("covered", "feature_pairs")
    assert len(log) == (2 if covered else 4)
    assert estimator.search_space_exhausted_ is covered
    assert log[-1]["fitness_best"] == (2 if feature_case else 4)


def test_verbose_output_uses_compact_generation_summary(simple_toolbox, capsys):
    population = simple_toolbox.population(n=4)
    stats = tools.Statistics(lambda individual: individual.fitness.values)
    stats.register("fitness", np.mean)
    stats.register("fitness_max", np.max)

    eaSimple(
        population=population,
        toolbox=simple_toolbox,
        cxpb=ConstantAdapter(0.5, 0.5, 0),
        mutpb=ConstantAdapter(0.5, 0.5, 0),
        ngen=1,
        stats=stats,
        halloffame=tools.HallOfFame(1),
        callbacks=[],
        verbose=True,
        estimator=SimpleNamespace(elitism=False),
    )

    output = capsys.readouterr().out

    assert " gen evals" in output
    assert "best" in output
    assert "unique" in output
    assert "events" in output
    assert "unique_individual_ratio" not in output
    assert "fitness_sharing_applied" not in output

    table_lines = [line for line in output.splitlines() if line.strip()]
    assert len({len(line) for line in table_lines}) == 1


def test_algorithm_records_monotonic_best_fitness(simple_toolbox):
    population = [
        creator.IndividualAlgorithmTest([1, 0, 0, 0]),
        creator.IndividualAlgorithmTest([1, 1, 0, 0]),
        creator.IndividualAlgorithmTest([0, 0, 0, 0]),
        creator.IndividualAlgorithmTest([0, 1, 0, 0]),
    ]
    stats = tools.Statistics(lambda individual: individual.fitness.values)
    stats.register("fitness", np.mean)
    stats.register("fitness_max", np.max)
    stats.register("fitness_min", np.min)

    _, logbook, _ = eaSimple(
        population=population,
        toolbox=simple_toolbox,
        cxpb=ConstantAdapter(1.0, 1.0, 0),
        mutpb=ConstantAdapter(1.0, 1.0, 0),
        ngen=4,
        stats=stats,
        halloffame=tools.HallOfFame(1),
        callbacks=[],
        verbose=False,
        estimator=SimpleNamespace(elitism=False),
    )

    best_history = logbook.select("fitness_best")

    assert all(current >= previous for previous, current in zip(best_history, best_history[1:]))


def test_diversity_control_boosts_mutation_and_adds_immigrants(simple_toolbox):
    population = [creator.IndividualAlgorithmTest([0, 0, 0, 0]) for _ in range(4)]
    estimator = SimpleNamespace(
        elitism=False,
        diversity_control=True,
        diversity_threshold=0.5,
        diversity_stagnation_generations=5,
        diversity_mutation_boost=3.0,
        random_immigrants_fraction=0.5,
        local_search=False,
        fit_stats_=create_fit_stats(),
    )

    _, logbook, _ = eaSimple(
        population=population,
        toolbox=simple_toolbox,
        cxpb=ConstantAdapter(0.1, 0.1, 0),
        mutpb=ConstantAdapter(0.2, 0.2, 0),
        ngen=1,
        stats=None,
        halloffame=tools.HallOfFame(1),
        callbacks=[],
        verbose=False,
        estimator=estimator,
    )

    assert logbook[1]["diversity_control_triggered"] is True
    assert logbook[1]["mutation_probability"] == pytest.approx(0.6)
    assert logbook[1]["random_immigrants"] == 2
    assert estimator.fit_stats_["random_immigrants"] == 2


def test_local_search_refines_hall_of_fame_candidates(simple_toolbox):
    population = simple_toolbox.population(n=4)
    estimator = SimpleNamespace(
        elitism=False,
        local_search=True,
        local_search_top_k=1,
        local_search_steps=2,
        local_search_radius=0.25,
        max_features=None,
        fit_stats_=create_fit_stats(),
    )

    _, logbook, _ = eaSimple(
        population=population,
        toolbox=simple_toolbox,
        cxpb=ConstantAdapter(0.0, 0.0, 0),
        mutpb=ConstantAdapter(0.0, 0.0, 0),
        ngen=1,
        stats=None,
        halloffame=tools.HallOfFame(1),
        callbacks=[],
        verbose=False,
        estimator=estimator,
    )

    assert logbook[-1]["local_refinements"] == 2
    assert estimator.fit_stats_["local_refinement_candidates"] == 2


@pytest.mark.parametrize("algorithm", [eaSimple, eaMuPlusLambda, eaMuCommaLambda])
@pytest.mark.parametrize("sign", [1.0, -1.0])
@pytest.mark.parametrize("kind", ["numeric", "features"])
@pytest.mark.parametrize("top_k", [1, 2])
def test_local_refinement_follows_improving_candidates(algorithm, sign, kind, top_k):
    fitness_type = type("RefinementFitness", (base.Fitness,), {"weights": (sign,)})

    class Individual(list):
        def __init__(self, values):
            super().__init__(values)
            self.fitness = fitness_type()

    toolbox = base.Toolbox()
    toolbox.register("evaluate", lambda individual: (sign * sum(individual),))
    batches = []

    def evaluate_batch(individuals):
        batches.append(len(individuals))
        return [toolbox.evaluate(individual) for individual in individuals]

    toolbox.register("evaluate_population", evaluate_batch)
    population = [
        Individual([i] if kind == "numeric" else [int(j == i) for j in range(8)])
        for i in range(top_k)
    ]
    estimator = SimpleNamespace(
        elitism=False,
        local_search=True,
        local_search_top_k=top_k,
        local_search_steps=6,
        local_search_radius=0.125,
        max_features=None,
        fit_stats_=create_fit_stats(),
    )
    if kind == "numeric":
        estimator.space = Space({"x": Integer(0, 8)})
    hof = tools.HallOfFame(top_k)
    state = random.getstate()
    random.seed(0)
    try:
        result, log, _ = algorithm(
            population,
            toolbox,
            cxpb=ConstantAdapter(0, 0, 0),
            mutpb=ConstantAdapter(0, 0, 0),
            ngen=0,
            halloffame=hof,
            callbacks=[],
            verbose=False,
            estimator=estimator,
            **({} if algorithm is eaSimple else {"mu": top_k, "lambda_": 2 * top_k}),
        )
    finally:
        random.setstate(state)
    # One-shot neighbors of these initial parents cannot reach 3.
    # Chained accepted improvements must move further.
    assert sum(hof[0]) >= 3
    assert len(result) == top_k
    assert estimator.fit_stats_["local_refinement_candidates"] == 6 * top_k
    assert batches == [top_k] * 7  # Initial population, then six refinement steps.
    assert log[-1]["fitness_best"] == sign * sum(hof[0])


def test_shared_fitness_temporarily_penalizes_crowded_niches(simple_toolbox):
    crowded_a = creator.IndividualAlgorithmTest([1, 1, 1, 1])
    crowded_b = creator.IndividualAlgorithmTest([1, 1, 1, 1])
    isolated = creator.IndividualAlgorithmTest([0, 0, 0, 0])

    for individual in [crowded_a, crowded_b, isolated]:
        individual.fitness.values = (10,)

    population = [crowded_a, crowded_b, isolated]
    estimator = SimpleNamespace(
        fitness_sharing=True,
        sharing_radius=0.25,
        sharing_alpha=1.0,
    )

    with shared_fitness(population, estimator) as sharing_record:
        selected = tools.selBest(population, 1)
        assert selected == [isolated]
        assert crowded_a.fitness.values[0] < isolated.fitness.values[0]
        assert sharing_record["fitness_sharing_applied"] is True
        assert sharing_record["max_niche_count"] == pytest.approx(2.0)

    assert crowded_a.fitness.values == (10,)
    assert crowded_b.fitness.values == (10,)
    assert isolated.fitness.values == (10,)


@pytest.mark.parametrize("weight", [1.0, -1.0])
def test_shared_fitness_keeps_nonfinite_candidate_below_finite_candidates(weight):
    fitness_type = type("Fitness", (base.Fitness,), {"weights": (weight,)})

    class Individual(list):
        def __init__(self, genes):
            super().__init__(genes)
            self.fitness = fitness_type()

    population = []
    for value, score in [(0, -weight * np.inf), (1, weight * -1e15), (2, weight * 1.0)]:
        individual = Individual([value, value, value, value])
        individual.fitness.values = (score,)
        population.append(individual)

    estimator = SimpleNamespace(
        fitness_sharing=True,
        sharing_radius=0.25,
        sharing_alpha=1.0,
    )
    original = [item.fitness.values for item in population]
    with shared_fitness(population, estimator):
        assert all(not np.isnan(item.fitness.values[0]) for item in population)
        assert tools.selBest(population, 1)[0] is population[2]
    assert [item.fitness.values for item in population] == original


@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("radius, alpha", [(0.25, 1.0), (0.8, 2.0)])
def test_shared_fitness_matches_pairwise_reference(simple_toolbox, mixed, radius, alpha):
    from ..optimizer_control import individual_distance, sharing_value

    values = (
        [[1, 0.1, ("a", 1), 2], [3, 0.5, None, 2], [1, 0.1, ("a", 1), 2], [5, 1.0, None, 2]]
        if mixed
        else [[1, 0, 0, 1], [1, 1, 0, 1], [1, 0, 0, 1], [0, 0, 1, 0]]
    )
    population = [creator.IndividualAlgorithmTest(row) for row in values]
    estimator = SimpleNamespace(fitness_sharing=True, sharing_radius=radius, sharing_alpha=alpha)
    if mixed:
        estimator.space = Space(
            {
                "depth": Integer(1, 5),
                "rate": Continuous(0.1, 1.0),
                "category": Categorical([("a", 1), None]),
                "fixed": Integer(2, 2),
            }
        )
    expected = []
    for index, individual in enumerate(population):
        individual.fitness.values = (index + 1.0,)
        count = sum(
            sharing_value(individual_distance(estimator, individual, other), radius, alpha)
            for other in population
        )
        expected.append((index + 1.0) / count)
    with shared_fitness(population, estimator):
        assert [item.fitness.values[0] for item in population] == pytest.approx(expected)
    assert [item.fitness.values[0] for item in population] == [1.0, 2.0, 3.0, 4.0]


@pytest.mark.parametrize(
    "kind",
    ["large_integer", "continuous_integer", "unsigned_integer", "nan_category", "list_category"],
)
def test_shared_fitness_preserves_dimension_equality(simple_toolbox, kind):
    if kind in {"large_integer", "continuous_integer"}:
        values = [2**53, 2**53 + 1]
        dimension = Integer(*values) if kind == "large_integer" else Continuous(*values)
    elif kind == "unsigned_integer":
        values = [np.uint64(1), np.uint64(2)]
        dimension = Integer(0, 10)
    elif kind == "nan_category":
        value = float("nan")
        values = [value, value]
        dimension = Categorical([value, "other"])
    else:
        values = [[1], [2]]
        dimension = Categorical(values)
    population = [creator.IndividualAlgorithmTest([value]) for value in values]
    for individual in population:
        individual.fitness.values = (10.0,)
    estimator = SimpleNamespace(
        space=Space({"x": dimension}), fitness_sharing=True, sharing_radius=0.5, sharing_alpha=1.0
    )
    with shared_fitness(population, estimator):
        expected = [10.0, 10.0 / 1.8] if kind == "unsigned_integer" else [10.0, 10.0]
        assert [item.fitness.values[0] for item in population] == expected


def test_shared_fitness_preserves_selection_ties(simple_toolbox):
    values = [
        [1, 0, 0, 1, 1, 0, 0],
        [1, 0, 1, 0, 1, 1, 0],
        [1, 0, 0, 1, 0, 1, 1],
        [1, 1, 0, 0, 1, 1, 0],
        [1, 0, 0, 1, 0, 0, 0],
    ]
    population = [creator.IndividualAlgorithmTest(row) for row in values]
    for individual in population:
        individual.fitness.values = (1.0,)
    estimator = SimpleNamespace(fitness_sharing=True, sharing_radius=0.8, sharing_alpha=1.0)
    with shared_fitness(population, estimator):
        assert tools.selBest(population, 1)[0] is population[2]


def test_algorithm_records_fitness_sharing_telemetry(simple_toolbox):
    population = [creator.IndividualAlgorithmTest([1, 1, 1, 1]) for _ in range(4)]
    estimator = SimpleNamespace(
        elitism=False,
        local_search=False,
        fitness_sharing=True,
        sharing_radius=0.25,
        sharing_alpha=1.0,
    )

    _, logbook, _ = eaSimple(
        population=population,
        toolbox=simple_toolbox,
        cxpb=ConstantAdapter(0.0, 0.0, 0),
        mutpb=ConstantAdapter(0.0, 0.0, 0),
        ngen=1,
        stats=None,
        halloffame=tools.HallOfFame(1),
        callbacks=[],
        verbose=False,
        estimator=estimator,
    )

    assert logbook[1]["fitness_sharing_applied"] is True
    assert logbook[1]["mean_niche_count"] >= 1.0
    assert logbook[1]["max_niche_count"] >= 1.0


@pytest.mark.parametrize(
    "algorithm,kwargs",
    [
        (eaSimple, {}),
        (eaMuPlusLambda, {"mu": 10, "lambda_": 10}),
        (eaMuCommaLambda, {"mu": 10, "lambda_": 10}),
    ],
)
@pytest.mark.parametrize("kind", ["binary", "continuous", "collapsed"])
def test_balanced_binary_population_does_not_trigger_diversity_recovery(
    simple_toolbox, algorithm, kwargs, kind
):
    masks = np.eye(5, dtype=int).tolist()
    masks += [[1 - value for value in mask] for mask in masks]
    if kind == "continuous":
        masks = [[i / 10.0] for i in range(10)]
    elif kind == "collapsed":
        masks = [[1, 0, 0, 0, 0] for _ in range(10)]
    population = [creator.IndividualAlgorithmTest(mask) for mask in masks]
    estimator = SimpleNamespace(
        elitism=False,
        diversity_control=True,
        diversity_threshold=0.25,
        diversity_stagnation_generations=5,
        diversity_mutation_boost=2,
        random_immigrants_fraction=0.1,
        local_search=False,
        fit_stats_=create_fit_stats(),
    )
    _, logbook, _ = algorithm(
        population,
        simple_toolbox,
        cxpb=ConstantAdapter(0, 0, 0),
        mutpb=ConstantAdapter(0, 0, 0),
        ngen=1,
        verbose=False,
        estimator=estimator,
        halloffame=tools.HallOfFame(1),
        **kwargs,
    )
    expected = {"binary": 5 / 9, "continuous": 1.0, "collapsed": 0.0}[kind]
    assert logbook[0]["genotype_diversity"] == pytest.approx(expected)
    assert logbook[1]["diversity_control_triggered"] is (kind == "collapsed")
    assert logbook[1]["random_immigrants"] == (1 if kind == "collapsed" else 0)


@pytest.mark.parametrize("algorithm", [eaMuPlusLambda, eaMuCommaLambda])
@pytest.mark.parametrize("direction", [1.0, -1.0])
@pytest.mark.parametrize("elitism", [True, False])
def test_survivors_and_history_keep_best_evaluated_candidate(
    simple_toolbox, algorithm, direction, elitism
):
    from itertools import cycle

    class Fitness(base.Fitness):
        weights = (direction,)

    class Individual(list):
        def __init__(self, values):
            super().__init__(values)
            self.fitness = Fitness()

    scores = cycle([direction * 10, direction * 5, 0, 0])

    def mutate(individual):
        individual[0] = next(scores)
        return (individual,)

    simple_toolbox.register("mutate", mutate)
    # A stochastic selector is allowed to miss the best candidate entirely.
    simple_toolbox.register("select", lambda population, k: tools.selWorst(population, 1) * k)
    population = [Individual([0]) for _ in range(4)]
    hof = tools.HallOfFame(2)
    stats = tools.Statistics(lambda individual: individual.fitness.values[0])
    stats.register("fitness_max", max)
    stats.register("fitness_min", min)
    population, log, _ = algorithm(
        population,
        simple_toolbox,
        mu=4,
        lambda_=4,
        cxpb=ConstantAdapter(0, 0, 0),
        mutpb=ConstantAdapter(1, 1, 0),
        ngen=2,
        stats=stats,
        halloffame=hof,
        verbose=False,
        estimator=SimpleNamespace(elitism=elitism),
    )
    assert len(population) == 4
    assert hof[0][0] == direction * 10
    assert any(ind[0] == direction * 10 for ind in population) is elitism
    assert log.select("fitness_best") == [0, direction * 10, direction * 10]
    assert log[-1]["best_generation"] == 1
    assert log[-1]["stagnation_generations"] == 1
    if elitism:
        assert all(any(ind == elite for ind in population) for elite in hof)
        archived_values = [list(elite) for elite in hof]
        for individual in population:
            individual[0] = 99
        assert [list(elite) for elite in hof] == archived_values
