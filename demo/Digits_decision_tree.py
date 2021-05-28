import warnings
from sklearn_genetic import GASearchCV
from sklearn_genetic.space import Categorical, Integer, Continuous
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score


warnings.filterwarnings("ignore")

data = load_digits()
label_names = data['target_names']
y = data['target']
X = data['data']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

clf = DecisionTreeClassifier()

params_grid = {'min_weight_fraction_leaf': Continuous(0, 0.5),
               'criterion': Categorical(['gini', 'entropy']),
               'max_depth': Integer(2, 20), 'max_leaf_nodes': Integer(2, 30)}

evolved_estimator = GASearchCV(clf,
                               cv=3,
                               scoring='accuracy',
                               population_size=16,
                               generations=30,
                               tournament_size=3,
                               elitism=True,
                               crossover_probability=0.9,
                               mutation_probability=0.05,
                               param_grid=params_grid,
                               algorithm='eaMuPlusLambda',
                               n_jobs=-1,
                               verbose=True)

evolved_estimator.fit(X_train, y_train)
y_predict_ga = evolved_estimator.predict(X_test)
accuracy = accuracy_score(y_test, y_predict_ga)

print(evolved_estimator.best_params)
print("accuracy score: ", "{:.2f}".format(accuracy))
