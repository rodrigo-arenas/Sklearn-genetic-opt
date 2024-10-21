
---
layout: Sklearn_genetic
title: Add Early stopping callback
author: vgauraha62
---

Distributed cloud <br>
early_stopping <br>
Across the network <br>

---


from deap import tools

class EarlyStopping:
    def __init__(self, patience=5, tolerance=1e-4):
        self.patience = patience  # Number of generations to wait for improvement
        self.tolerance = tolerance  # Minimum improvement required
        self.best_fitness = -np.inf  # Initialize with negative infinity
        self.wait = 0  # Counter for generations without improvement

    def __call__(self, population, toolbox, stats, hof):
        # Get the current best fitness
        current_fitness = stats.compile(population)[0][0]  # Assuming fitness is the first value

        # Check for improvement
        if current_fitness > self.best_fitness + self.tolerance:
            self.best_fitness = current_fitness
            self.wait = 0
        else:
            self.wait += 1

        # Trigger early stopping only, if the patience is exceeded
        if self.wait >= self.patience:
            raise tools.FitnessLimitReached("Early stopping triggered.")



 def _select_algorithm(self, pop, stats, hof):
        """
        It selects the algorithm to run from the sklearn_genetic.algorithms module
        based in the parameter self.algorithm.

        Parameters
        ----------
        pop: pop object from DEAP
        stats: stats object from DEAP
        hof: hof object from DEAP

        Returns
        -------
        pop: pop object
            The last evaluated population
        log: Logbook object
            It contains the calculated metrics {'fitness', 'fitness_std', 'fitness_max', 'fitness_min'}
            the number of generations and the number of evaluated individuals per generation
        n_gen: int
            The number of generations that the evolutionary algorithm ran
        """

        selected_algorithm = algorithms_factory.get(self.algorithm, None)
        
        ------
        # Here we create the early stopping callback
        early_stopping = EarlyStopping(patience=10, tolerance=1e-5)  
        ------


        if selected_algorithm:
            pop, log, gen = selected_algorithm(
                pop,
                self.toolbox,
                mu=self.population_size,
                lambda_=2 * self.population_size,
                cxpb=self.crossover_adapter,
                stats=stats,
                mutpb=self.mutation_adapter,
                ngen=self.generations,
                halloffame=hof,
                ------
                
                callbacks=callbacks, #change
                -------
                verbose=self.verbose,
                estimator=self,
            )

        else:
            raise ValueError(
                f"The algorithm {self.algorithm} is not supported, "
                f"please select one from {Algorithms.list()}"
            )

        return pop, log, gen
