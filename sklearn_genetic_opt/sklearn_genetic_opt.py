import numpy as np
import random
import functools
import operator
from sklearn.base import clone
from sklearn.model_selection import cross_val_score

class GASearchCV:
    
    """
    Hyper parameter tunning using generic algorithms.
    Parameters
    ----------
    estimator: Sklearn Classifier or Regressor
    cv: int, number of splits used for calculating cross_val_score
    scoring: string, Scoring function to use as fitness value
    pop_size: int, size of the population
    crossover_prob: float, probability of crossover operation
    mutation_prob: float, probability of child mutation
    tournament_size: number of chromosomes to performe tournament selection
    elitism: bool, if true takes the two best solution to the next generation
    verbose: bool, if true, shows the best solution in each generation
    generations: int, number of generations to run the genetic algorithm
    continuous_parameters: dict, continous parameters to tune, expected a list or tuple with the range (min,max) to search
    categorical_parameters: dict, categorical parameters to tune, expected a list with the posible options to choose
    int_parameters: dict, integers parameters to tune, expected a list or tuple with the range (min,max) to search
    encoding_len: enconding lenght for the continuous_parameters and int_parameters
    """
    
    
    def __init__(self,
                 estimator,
                 cv=3,
                 scoring=None,
                 pop_size=20,
                 generations=50,
                 crossover_prob=1.0,
                 mutation_prob=0.1,
                 tournament_size=3,
                 elitism= True,
                 verbose=True,
                 continuous_parameters = {},
                 categorical_parameters = {}, 
                 int_parameters = {},
                 encoding_len=10):
        
        self.estimator=clone(estimator)
        self.cv=cv
        self.scoring=scoring
        self.pop_size =  pop_size
        self.generations=generations
        self.crossover_prob=crossover_prob
        self.mutation_prob=mutation_prob
        self.tournament_size=tournament_size
        self.elitism= elitism
        self.verbose= verbose
        self.continuous_parameters = continuous_parameters
        self.categorical_parameters = categorical_parameters
        self.int_parameters = int_parameters
        self._encoding_len = encoding_len
        
        self._continous_parameters_number = len(self.continuous_parameters.keys())
        self._continous_parameters_range= (0,self._continous_parameters_number*self._encoding_len)
        self._continous_parameters_indexes= {key:(x*self._encoding_len, x*self._encoding_len +self._encoding_len)
                                                   for x,key in enumerate([*self.continuous_parameters])}
        
        self._int_parameters_number = len(self.int_parameters.keys())
        self._int_parameters_range= (self._continous_parameters_range[1],
                                     self._continous_parameters_range[1]+self._int_parameters_number*self._encoding_len)
        self._int_parameters_indexes= {key:(self._int_parameters_range[0] + x*self._encoding_len, 
                                            self._int_parameters_range[0] + x*self._encoding_len +self._encoding_len)
                                            for x,key in enumerate([*self.int_parameters])}
        
        
        self._categorical_parameters_number = len(self.categorical_parameters.keys())
        self._categorical_parameters_range= (self._int_parameters_range[1],
                                             self._int_parameters_range[1]+self._categorical_parameters_number)
        self._categorical_parameters_indexes= {key:(self._categorical_parameters_range[0] + x, 
                                                    self._categorical_parameters_range[0] + x 
                                                    +1)
                                                    for x,key in enumerate([*self.categorical_parameters])}
        
    def _get_precision(self):
        
        self._continuous_parameters_precision = {}
        self._int_parameters_precision = {}
        
        if bool(self.continuous_parameters):
            for key, value in self.continuous_parameters.items():
                self._continuous_parameters_precision[key] = round((value[1] - value[0])/(2**self._encoding_len -1),10)
                
        if bool(self.int_parameters):
            for key, value in self.int_parameters.items():
                self._int_parameters_precision[key] = round((value[1] - value[0])/(2**self._encoding_len -1),10)
        
        self._params_precision = {**self._continuous_parameters_precision,**self._int_parameters_precision}
        
        return self._params_precision
    
    
    def _initialize_population(self):
        
        self._continous_chromosomes_init= np.random.randint(2, size=(self.pop_size,self._continous_parameters_number*self._encoding_len))
        self._int_chromosomes_init= np.random.randint(2, size=(self.pop_size,self._int_parameters_number*self._encoding_len)
                                                     )
        
        self._categorical_chromosomes_init= np.empty((self.pop_size,0), int)
        if bool(self.categorical_parameters):
            self._categorical_chromosomes_init= np.transpose(np.array([np.random.randint(len(value),
                                                                       size=(self.pop_size)) 
                                                              for key, value in self.categorical_parameters.items()]))
        return np.hstack((self._continous_chromosomes_init, self._int_chromosomes_init,self._categorical_chromosomes_init))

    
    def _decode(self,chromosome):
        
        _decoded_dict= {}
        
        #Continuos variables
        for key, value in self.continuous_parameters.items():
            
            __index = self._continous_parameters_indexes[key]

            chrom = chromosome[__index[0]:__index[1]]
            decoded = round(value[0] + sum([x*(2**n) for n,x in enumerate(chrom)])*self._params_precision[key],15)
            
            _decoded_dict[key] = decoded
        
        #Integer variables
        for key_int, value_int in self.int_parameters.items():
            
            __index_int = self._int_parameters_indexes[key_int]

            chrom_int = chromosome[__index_int[0]:__index_int[1]]
            decoded_int = int(value_int[0] + sum([x*(2**n) for n,x in enumerate(chrom_int)])*self._params_precision[key_int])
            
            _decoded_dict[key_int] = decoded_int
            
        
        #categorical variables
        for key_categorical, value_categorical in self.categorical_parameters.items():
            
            __index_categorical = self._categorical_parameters_indexes[key_categorical]
    
            chrom_categorical = chromosome[__index_categorical[0]:__index_categorical[1]]
            decoded_categorical = np.array(self.categorical_parameters[key_categorical])[chrom_categorical][0]
            _decoded_dict[key_categorical] = decoded_categorical
            
        return _decoded_dict
    
    
    def _tournament(self,gen_results):
        
        _contestants = random.sample(list(gen_results.keys()),k=self.tournament_size)
        _best_score_idx= np.argmax([gen_results.get(key)["fitness"] for key in _contestants])
        
        return _contestants[_best_score_idx]
    
    def _elitism(self,gen_results):
        """
        Returns top 2 by fitness value
        """
        return sorted(gen_results.keys(), key = lambda x:gen_results[x]["fitness"], reverse = True)[:2]
         
        
    
    def _crossover(self,parent1,parent2):
        
        if random.random() < self.crossover_prob:
        
            crossover_points= random.sample(range(len(parent1)),2)
            _point1, _point2= min(crossover_points),max(crossover_points)

            
            _child1=[parent1[0:_point1],parent2[_point1:_point2],parent1[_point2:]]
            _child2=[parent2[0:_point1],parent1[_point1:_point2],parent2[_point2:]]
            
            _child1= np.array(functools.reduce(operator.iconcat, _child1, []))
            _child2= np.array(functools.reduce(operator.iconcat, _child2, []))

            return _child1,_child2
        
        return parent1, parent2
    
    def _mutation(self,child):
        
        for n in range(self._continous_parameters_range[0], 
                       self._continous_parameters_range[1] + self._int_parameters_number*self._encoding_len):
            if random.random() < self.mutation_prob:
                child[n]= 1-child[n]
        
        for key, value in self._categorical_parameters_indexes.items():
            if random.random() < self.mutation_prob:
                child[value[0]:value[1]] = np.random.randint(len(value))
        
        return child    
    
    def fit(self,X,y):
        
        self._get_precision()
        self.X= X
        self.y = y
        _current_generation_chromosomes = self._initialize_population()
        if self.elitism:
            self._child_range = int((len(_current_generation_chromosomes)/2) -2)
        self._child_range = int((len(_current_generation_chromosomes)/2))
        self._best_solutions = {}
        
        for gen in range(0,self.generations):
            self._gen_results= {}
            
            """
            Adjust the model for each chromosome and get fitness
            """
            for n_chrom,chromosome in enumerate(_current_generation_chromosomes):
                _current_generation_params=self._decode(chromosome)
                
                self.estimator.set_params(**_current_generation_params)
                
                _cv_score = cross_val_score(self.estimator, self.X, self.y, cv=self.cv,scoring=self.scoring,n_jobs=-1)
                
                self._gen_results[n_chrom]={"n_chrom":n_chrom,
                                            "params":_current_generation_params,
                                            "fitness":round(np.mean(_cv_score),4),
                                            "fitness_std":round(np.std(_cv_score),4)}
            
            
            _temp_current_generation_chromosomes = []
            
            for n_chrom in range(self._child_range): 
                
                _parent_1_idx= self._tournament(self._gen_results)
                _parent_2_idx= self._tournament(self._gen_results)

                _parent_1 = _current_generation_chromosomes[_parent_1_idx]
                _parent_2 = _current_generation_chromosomes[_parent_2_idx]

                _child_1, _child_2= self._crossover(_parent_1,_parent_2)
                _child_1, _child_2= self._mutation(_child_1), self._mutation(_child_2)
                
                _temp_current_generation_chromosomes.append(_child_1)
                _temp_current_generation_chromosomes.append(_child_2)
                
            if self.elitism:
                _elite_child_1_idx, _elite_child_2_idx= self._elitism(self._gen_results)
                
                _elite_child_1 = _current_generation_chromosomes[_elite_child_1_idx]
                _elite_child_2 = _current_generation_chromosomes[_elite_child_2_idx]
                
                _temp_current_generation_chromosomes.append(_elite_child_1)
                _temp_current_generation_chromosomes.append(_elite_child_2)
                    
            
            _current_generation_chromosomes= np.array(_temp_current_generation_chromosomes)
            
            _best_solution_idx= self._elitism(self._gen_results)[0]
            self._best_solutions[gen] = self._gen_results[_best_solution_idx]
        
            if self.verbose:
                print("n_gen:",gen,self._best_solutions[gen])
                print()
            
        self.best_params_= self._best_solutions[self.generations-1]["params"]
        
        self.estimator.set_params(**self.best_params_)
        self.estimator.fit(self.X,self.y)
        return self._best_solutions[self.generations-1]
    
    def predict(self,X_predict):
        
        self.X_predict= X_predict
        return self.estimator.predict(self.X_predict)
