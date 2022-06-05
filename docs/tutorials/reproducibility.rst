Reproducibility
===============


One of the desirable capabilities of a package that makes several "random" choices is to be able to reproduce the results.

The usual strategy is to fix the random seed that starts generating the pseudo-random numbers.
Unfortunately, the DEAP package, which is the main dependency for all the evolutionary algorithms,
doesn't have an explicit parameter to fix this seed.

However, there is a workaround that seems to work to reproduce these results; this is:

* Set the random seed of `numpy` and `random` package, which are the underlying random numbers generators
* Use the random_state parameter In each of the scikit-learn and sklearn-genetic-opt objects that support it

In the following example, the random_state is set for the `train_test_split`, `cross-validation` generator,
each of the hyperparameters in the `param_grid`, the `RandomForestClassifier`, and at the file level.

Example:
--------
.. code:: python3

   import numpy as np
   import random
   from sklearn_genetic import GASearchCV
   from sklearn_genetic.space import Continuous, Categorical, Integer
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.model_selection import train_test_split, StratifiedKFold
   from sklearn.datasets import load_digits
   from sklearn.metrics import accuracy_score


   # Random Seed at file level
   random_seed = 54

   np.random.seed(random_seed)
   random.seed(random_seed)


   data = load_digits()
   n_samples = len(data.images)
   X = data.images.reshape((n_samples, -1))
   y = data['target']
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=random_seed)

   clf = RandomForestClassifier(random_state=random_seed)

   param_grid = {'min_weight_fraction_leaf': Continuous(0.01, 0.5, distribution='log-uniform',
                                                        random_state=random_seed),
                 'bootstrap': Categorical([True, False], random_state=random_seed),
                 'max_depth': Integer(2, 30, random_state=random_seed),
                 'max_leaf_nodes': Integer(2, 35, random_state=random_seed),
                 'n_estimators': Integer(100, 300, random_state=random_seed)}

   cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_seed)

   evolved_estimator = GASearchCV(estimator=clf,
                                  cv=cv,
                                  scoring='accuracy',
                                  population_size=8,
                                  generations=5,
                                  param_grid=param_grid,
                                  n_jobs=-1,
                                  verbose=True,
                                  keep_top_k=4)

   # Train and optimize the estimator
   evolved_estimator.fit(X_train, y_train)
   # Best parameters found
   print(evolved_estimator.best_params_)
   # Use the model fitted with the best parameters
   y_predict_ga = evolved_estimator.predict(X_test)
   print(accuracy_score(y_test, y_predict_ga))

   # Saved metadata for further analysis
   print("Stats achieved in each generation: ", evolved_estimator.history)
   print("Best k solutions: ", evolved_estimator.hof)




