import matplotlib.pyplot as plt
from sklearn_genetic import GASearchCV
from sklearn_genetic.space import Integer, Categorical, Continuous
from sklearn_genetic.utils import plot_fitness_evolution
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score


data = load_boston()

y = data['target']
X = data['data']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

clf = DecisionTreeRegressor()

param_grid = {'ccp_alpha': Continuous(0, 1),
              'criterion': Categorical(['mse', 'mae']),
              'max_depth': Integer(2, 20),
              'min_samples_split': Integer(2, 30)}

evolved_estimator = GASearchCV(clf,
                               cv=3,
                               scoring='r2',
                               population_size=12,
                               generations=30,
                               tournament_size=3,
                               elitism=True,
                               keep_top_k=4,
                               crossover_probability=0.9,
                               mutation_probability=0.05,
                               param_grid=param_grid,
                               criteria='max',
                               algorithm='eaMuCommaLambda',
                               n_jobs=-1)

evolved_estimator.fit(X_train, y_train)
y_predict_ga = evolved_estimator.predict(X_test)
r_squared = r2_score(y_test, y_predict_ga)

print(evolved_estimator.best_params_)
print("r-squared: ", "{:.2f}".format(r_squared))

print("Best k solutions: ", evolved_estimator.hof)
plot = plot_fitness_evolution(evolved_estimator)

plt.show()

