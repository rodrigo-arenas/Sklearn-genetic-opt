import scipy.stats as stats
import numpy as np
import warnings
from sklearn_genetic import GASearchCV
from sklearn_genetic.space import Continuous, Categorical
from sklearn_genetic.callbacks import DeltaThreshold
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.utils.fixes import loguniform
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score


warnings.filterwarnings("ignore")

data = load_digits()
label_names = data["target_names"]
y = data["target"]
X = data["data"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

# Base classifier
clf = SGDClassifier(loss="hinge", fit_intercept=True)

# 1. Random Search

param_dist = {
    "average": [True, False],
    "l1_ratio": stats.uniform(0, 1),
    "alpha": loguniform(1e-4, 1e0),
}

n_iter_search = 30
random_search = RandomizedSearchCV(
    clf, param_distributions=param_dist, n_iter=n_iter_search, n_jobs=-1
)

random_search.fit(X_train, y_train)
accuracy = accuracy_score(y_test, random_search.predict(X_test))
print("accuracy score: ", "{:.2f}".format(accuracy))
print("random search best params: \n", random_search.best_params_)

# 2. Grid Search

param_grid = {
    "average": [True, False],
    "l1_ratio": np.linspace(0, 1, num=10),
    "alpha": np.power(10, np.arange(-4, 1, dtype=float)),
}

grid_search = GridSearchCV(clf, param_grid=param_grid, n_jobs=-1)

grid_search.fit(X_train, y_train)
accuracy = accuracy_score(y_test, grid_search.predict(X_test))
print("Accuracy Score: ", "{:.2f}".format(accuracy))
print("grid search best params: \n", grid_search.best_params_)

#  3. Genetic Algorithm

param_grid = {
    "l1_ratio": Continuous(0, 1),
    "alpha": Continuous(1e-4, 1, distribution="log-uniform"),
    "average": Categorical([True, False]),
}

callbacks = DeltaThreshold(threshold=0.0001)

evolved_estimator = GASearchCV(
    clf,
    cv=3,
    scoring="accuracy",
    population_size=12,
    generations=20,
    tournament_size=3,
    elitism=True,
    param_grid=param_grid,
    n_jobs=-1,
)

evolved_estimator.fit(X_train, y_train, callbacks=callbacks)
y_predict_ga = evolved_estimator.predict(X_test)
accuracy = accuracy_score(y_test, y_predict_ga)
print("accuracy score: ", "{:.2f}".format(accuracy))
print("genetic algorithm best params: \n", evolved_estimator.best_params_)
print(
    "genetic algorithm decision function: \n",
    evolved_estimator.decision_function(X_test),
)
