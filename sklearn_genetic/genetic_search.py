import numpy as np
import random
from deap import base, creator, tools, algorithms
from sklearn.base import clone, ClassifierMixin, RegressorMixin
from sklearn.model_selection import cross_val_score
from sklearn.base import is_classifier, is_regressor
from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.utils.validation import check_array
from sklearn.metrics import check_scoring


class GASearchCV(ClassifierMixin, RegressorMixin):
    """
    Hyper parameter tuning using generic algorithms.
    """

    def __init__(self,
                 estimator,
                 cv: int = 3,
                 scoring=None,
                 population_size: int = 20,
                 generations: int = 50,
                 crossover_probability: float = 1.0,
                 mutation_probability: float = 0.1,
                 tournament_size: int = 3,
                 elitism: bool = True,
                 verbose: bool = True,
                 continuous_parameters: dict = None,
                 categorical_parameters: dict = None,
                 integer_parameters: dict = None,
                 criteria: str = 'max',
                 n_jobs: int = 1):
        """

        Parameters
        ----------
        estimator: Sklearn Classifier or Regressor
        cv: int, number of splits used for calculating cross_val_score
        scoring: string, Scoring function to use as fitness value
        population_size: int, size of the population
        crossover_probability: float, probability of crossover operation
        mutation_probability: float, probability of child mutation
        tournament_size: number of chromosomes to perform tournament selection
        elitism: bool, if true takes the |tournament_size| best solution to the next generation
        verbose: bool, if true, shows the best solution in each generation
        generations: int, number of generations to run the genetic algorithm
        continuous_parameters: dict, continuous parameters to tune, expected a list or tuple with the range (min,max) to search
        categorical_parameters: dict, categorical parameters to tune, expected a list with the possible options to choose
        integer_parameters: dict, integers parameters to tune, expected a list or tuple with the range (min,max) to search
        criteria: str, 'max' if a higher scoring metric is better, 'min' otherwise
        n_jobs: int, Number of jobs to run in parallel
        """

        self.estimator = clone(estimator)
        self.toolbox = base.Toolbox()
        self.cv = cv
        self.scoring = scoring
        self.pop_size = population_size
        self.generations = generations
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.tournament_size = tournament_size
        self.elitism = elitism
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.logbook = None
        self.history = None
        self.X = None
        self.Y = None
        self.best_params = None
        self.X_predict = None

        if not is_classifier(self.estimator) and not is_regressor(self.estimator):
            raise ValueError("{} is not a valid Sklearn classifier or regressor".format(self.estimator))

        if criteria not in ['max', 'min']:
            raise ValueError(f"Criteria must be 'max' or 'min', got {criteria} instead")
        elif criteria == 'max':
            self.criteria_sign = 1
        else:
            self.criteria_sign = -1

        if not continuous_parameters:
            self.continuous_parameters = {}
        else:
            self.continuous_parameters = continuous_parameters

        if not categorical_parameters:
            self.categorical_parameters = {}
        else:
            self.categorical_parameters = categorical_parameters

        if not integer_parameters:
            self.integer_parameters = {}
        else:
            self.integer_parameters = integer_parameters

        self.parameters = [*list(self.continuous_parameters.keys()),
                           *list(self.integer_parameters.keys()),
                           *list(self.categorical_parameters.keys())]

        self.continuous_parameters_range = (0, len(self.continuous_parameters))
        self.integer_parameters_range = (self.continuous_parameters_range[1],
                                         self.continuous_parameters_range[1] + len(self.integer_parameters))
        self.categorical_parameters_range = (self.integer_parameters_range[1],
                                             self.integer_parameters_range[1] + len(self.categorical_parameters))

    def register(self):

        creator.create("FitnessMax", base.Fitness, weights=[1.0])
        creator.create("Individual", list, fitness=creator.FitnessMax)

        attributes = []

        for key, value in self.continuous_parameters.items():
            self.toolbox.register(f"{key}", random.uniform, value[0], value[1])
            attributes.append(getattr(self.toolbox, key))

        for key, value in self.integer_parameters.items():
            self.toolbox.register(f"{key}", random.randint, value[0], value[1])
            attributes.append(getattr(self.toolbox, key))

        for key, value in self.categorical_parameters.items():
            self.toolbox.register(f"{key}", random.choice, value)
            attributes.append(getattr(self.toolbox, key))

        IND_SIZE = 1

        self.toolbox.register("individual",
                              tools.initCycle, creator.Individual,
                              tuple(attributes), n=IND_SIZE)

        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", self.mutate)
        if self.elitism:
            self.toolbox.register("select", tools.selTournament, tournsize=self.tournament_size)
        else:
            self.toolbox.register("select", tools.selRoulette)

        self.toolbox.register("evaluate", self.evaluate)

    def mutate(self, individual):
        gen = random.randrange(0, len(self.parameters))
        parameter_index = self.parameters[gen]

        if gen in range(self.continuous_parameters_range[0], self.continuous_parameters_range[1]):
            parameter = self.continuous_parameters[parameter_index]
            individual[gen] = random.uniform(parameter[0], parameter[1])
        elif gen in range(self.integer_parameters_range[0], self.integer_parameters_range[1]):
            parameter = self.integer_parameters[parameter_index]
            individual[gen] = random.randint(parameter[0], parameter[1])
        elif gen in range(self.categorical_parameters_range[0], self.categorical_parameters_range[1]):
            parameter = self.categorical_parameters[parameter_index]
            individual[gen] = random.choice(parameter)
        else:
            raise IndexError(f'mutate gen number {gen} is out of index')

        return [individual]

    def evaluate(self, individual):
        current_generation_params = {key: individual[n] for n, key in enumerate(self.parameters)}
        self.estimator.set_params(**current_generation_params)
        cv_scores = self.criteria_sign * cross_val_score(self.estimator,
                                                         self.X, self.Y,
                                                         cv=self.cv,
                                                         scoring=self.scoring,
                                                         n_jobs=self.n_jobs)

        return [np.mean(cv_scores)]

    @if_delegate_has_method(delegate='estimator')
    def fit(self, X, y):
        """
        Main method of GASearchCV, optimize the hyper parameters of the given estimator
        Parameters
        ----------
        X: training samples to learn from
        y: training labels for each X obversation

        Returns

        fitted sklearn Regressor or Classifier
        -------

        """
        scorer = check_scoring(self.estimator, scoring=self.scoring)

        self.X = X
        self.Y = y

        self.register()

        pop = self.toolbox.population(n=self.pop_size)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("fitness", np.mean)
        stats.register("fitness_std", np.std)
        stats.register("fitness_max", np.max)
        stats.register("fitness_min", np.min)

        pop, log = algorithms.eaSimple(pop, self.toolbox,
                                       cxpb=self.crossover_probability,
                                       stats=stats,
                                       mutpb=self.mutation_probability,
                                       ngen=self.generations,
                                       halloffame=hof,
                                       verbose=self.verbose)

        self.best_params = {key: hof[0][n] for n, key in enumerate(self.parameters)}
        self.logbook = log

        self.history = {"gen": log.select("gen"),
                        "fitness": log.select("fitness"),
                        "fitness_std": log.select("fitness_std"),
                        "fitness_max": log.select("fitness_max"),
                        "fitness_min": log.select("fitness_min")}

        self.estimator.set_params(**self.best_params)
        self.estimator.fit(self.X, self.Y)

        return self

    def __getitem__(self, index):
        """

        Parameters
        ----------
        index: slice required to get

        Returns
        -------
        Best solution of the iteration corresponding to the index number
        """
        if not self.best_solutions:
            raise IndexError("Make sure the model is already fitted")

        return self.best_solutions[index]

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        """
        Returns
        -------
        Iteration over the best solution found in each generation
        """
        if self.n < self.generations:
            result = self.__getitem__(self.n)
            self.n += 1
            return result
        else:
            raise StopIteration

    def __len__(self):
        """
        Returns
        -------
        Number of generations fitted
        """
        return self.generations

    @if_delegate_has_method(delegate='estimator')
    def predict(self, X):
        X = check_array(X)
        return self.estimator.predict(X)

    @if_delegate_has_method(delegate='estimator')
    def score(self, X, y):
        X = check_array(X)
        return self.estimator.score(X, y)

    @if_delegate_has_method(delegate='estimator')
    def decision_function(self, X):
        X = check_array(X)
        return self.estimator.decision_function(X)

    @if_delegate_has_method(delegate='estimator')
    def predict_proba(self, X):
        X = check_array(X)
        return self.estimator.predict_proba(X)

    @if_delegate_has_method(delegate='estimator')
    def predict_log_proba(self, X):
        X = check_array(X)
        return self.estimator.predict_log_proba(X)
