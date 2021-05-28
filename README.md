[![Build Status](https://www.travis-ci.com/rodrigo-arenas/Sklearn-genetic-opt.svg?branch=master)](https://www.travis-ci.com/rodrigo-arenas/Sklearn-genetic-opt)
[![Codecov](https://codecov.io/gh/rodrigo-arenas/Sklearn-genetic-opt/branch/master/graphs/badge.svg?branch=master&service=github)](https://codecov.io/github/rodrigo-arenas/Sklearn-genetic-opt?branch=master)
[![PyPI Version](https://badge.fury.io/py/sklearn-genetic-opt.svg)](https://badge.fury.io/py/sklearn-genetic-opt)
[![Python Version](https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8%20%7C%203.9-blue)](https://www.python.org/downloads/)

# Sklearn-genetic-opt
scikit-learn models hyperparameters tuning, using evolutionary algorithms.

This is meant to be an alternative from popular methods inside scikit-learn such as Grid Search and Random Grid Search.

Sklearn-genetic-opt uses evolutionary algorithms from the deap package to find the "best" set of hyperparameters that optimizes (max or min) the cross validation scores, it can be used for both regression and classification problems.

# Usage:
Install sklearn-genetic-opt

It's advised to install sklearn-genetic using a virtual env, inside the env use:

```
pip install sklearn-genetic-opt
```

## Example

```python
from sklearn_genetic import GASearchCV
from sklearn_genetic.utils import plot_fitness_evolution
from sklearn_genetic.space import Continuous, Categorical
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


data = load_digits() 
n_samples = len(data.images)
X = data.images.reshape((n_samples, -1))
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

clf = SGDClassifier()

param_grid = {'l1_ratio': Continuous(0, 1),
              'alpha': Continuous(1e-4, 1, distribution='log-uniform'),
              'average': Categorical([True, False])}

evolved_estimator = GASearchCV(estimator=clf,
                               cv=3,
                               scoring='accuracy',
                               population_size=10,
                               generations=30,
                               tournament_size=3,
                               elitism=True,
                               crossover_probability=0.8,
                               mutation_probability=0.1,
                               param_grid=param_grid,
                               criteria='max',
                               algorithm='eaMuPlusLambda',
                               n_jobs=-1,
                               verbose=True,
                               keep_top_k=4)

# Train and optimize the estimator 
evolved_estimator.fit(X_train,y_train)
# Best parameters found
print(evolved_estimator.best_params)
# Use the model fitted with the best parameters
y_predict_ga = evolved_estimator.predict(X_test)
print(accuracy_score(y_test,y_predict_ga))

# See the evolution of the optimization per generation
plot_fitness_evolution(evolved_estimator)
plt.show()

# Saved metadata for further analysis
print("Stats achieved in each generation: ", evolved_estimator.history)
print("Parameters and cv scores in each iteration: ", evolved_estimator.logbook)
print("Best k solutions: ", evolved_estimator.hof)
```
### Result

![demo](https://github.com/rodrigo-arenas/Sklearn-genetic-opt/blob/master/demo/geneticopt.gif?raw=true)