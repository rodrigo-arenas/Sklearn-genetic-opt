# Sklearn-genetic-opt
Repository for sklearn models parameters optimization using genetic algorithms

This is not ment to be a highly optimized implementation, it was made for demostration purposes

## Example

```python
from sklearn_genetic_opt import GASearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
import numpy as np


data = load_digits() 
label_names = data['target_names'] 
y = data['target']
X = data['data'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

clf = SGDClassifier(loss='hinge',fit_intercept=True)

evolved_estimator = GASearchCV(clf,
                    cv=3,
                    scoring='accuracy',
                    pop_size=20,
                    generations=8,
                    tournament_size=3,
                    elitism=True,
                    continuous_parameters = {'l1_ratio':(0,1), 'alpha':(1e-4,1)},
                    categorical_parameters = {'average': [True, False]},
                    int_parameters = {},
                    encoding_len=10)
                    
evolved_estimator.fit(X_train,y_train)
evolved_estimator.best_params_
y_predict_ga = evolved_estimator.predict(X_test)
accuracy_score(y_test,y_predict_ga)
