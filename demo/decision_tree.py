from sklearn_genetic import GASearchCV
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings("ignore")

data = load_digits()
label_names = data['target_names']
y = data['target']
X = data['data']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

clf = DecisionTreeClassifier()

evolved_estimator = GASearchCV(clf,
                               cv=3,
                               scoring='accuracy',
                               pop_size=16,
                               generations=30,
                               tournament_size=3,
                               elitism=True,
                               crossover_prob=0.9,
                               mutation_prob=0.05,
                               continuous_parameters={'min_weight_fraction_leaf': (0, 0.5)},
                               categorical_parameters={'criterion': ['gini', 'entropy']},
                               int_parameters={'max_depth': (2, 20), 'max_leaf_nodes': (2, 30)},
                               encoding_len=10,
                               n_jobs=-1)

evolved_estimator.fit(X_train, y_train)
y_predict_ga = evolved_estimator.predict(X_test)
accuracy = accuracy_score(y_test, y_predict_ga)

print(evolved_estimator.best_params_)
print("accuracy score: ", "{:.2f}".format(accuracy))
