[![Build Status](https://www.travis-ci.com/rodrigo-arenas/Sklearn-genetic.svg?branch=master)](https://www.travis-ci.com/rodrigo-arenas/Sklearn-genetic)
[![Codecov](https://codecov.io/gh/rodrigo-arenas/Sklearn-genetic/branch/main/graphs/badge.svg?branch=master&service=github)](https://codecov.io/github/rodrigo-arenas/Sklearn-genetic?branch=master)
[![PyPI Version](https://badge.fury.io/py/sklearn-genetic.svg)](https://badge.fury.io/py/sklearn-genetic)
[![Python Version](https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8%20%7C%203.9-blue)](https://www.python.org/downloads/)

# Sklearn-genetic
Sklearn models hyperparameters tuning using genetic algorithms

# Usage:
Install sklearn-genetic

It's advised to install sklearn-genetic using a virtual env, inside the env use:

```
pip install sklearn-genetic
```

## Example

```python
from sklearn_genetic import GASearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score


data = load_digits() 
y = data['target']
X = data['data'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

clf = DecisionTreeClassifier()

evolved_estimator = GASearchCV(clf,
                               cv=3,
                               scoring='accuracy',
                               population_size=16,
                               generations=30,
                               tournament_size=3,
                               elitism=True,
                               crossover_probability=0.9,
                               mutation_probability=0.05,
                               continuous_parameters={'min_weight_fraction_leaf': (0, 0.5)},
                               categorical_parameters={'criterion': ['gini', 'entropy']},
                               integer_parameters={'max_depth': (2, 20), 'max_leaf_nodes': (2, 30)},
                               encoding_length=10,
                               n_jobs=-1)
                    
evolved_estimator.fit(X_train,y_train)
print(evolved_estimator.best_params_)
y_predict_ga = evolved_estimator.predict(X_test)
print(accuracy_score(y_test,y_predict_ga))
