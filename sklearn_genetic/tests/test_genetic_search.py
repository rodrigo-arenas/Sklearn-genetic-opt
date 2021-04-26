from sklearn_genetic import GASearchCV
from sklearn.datasets import load_digits
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted


data = load_digits()
label_names = data['target_names']
y = data['target']
X = data['data']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


def test_expected_ga_results():
    clf = SGDClassifier(loss='log', fit_intercept=True)
    generations = 8
    evolved_estimator = GASearchCV(clf,
                                   cv=3,
                                   scoring='accuracy',
                                   population_size=12,
                                   generations=generations,
                                   tournament_size=3,
                                   elitism=False,
                                   continuous_parameters={'l1_ratio': (0, 1), 'alpha': (1e-4, 1)},
                                   categorical_parameters={'average': [True, False]},
                                   verbose=False,
                                   encoding_length=10)

    evolved_estimator.fit(X_train, y_train)

    assert check_is_fitted(evolved_estimator) is None
    assert 'l1_ratio' in evolved_estimator.best_params_
    assert 'alpha' in evolved_estimator.best_params_
    assert 'average' in evolved_estimator.best_params_
    assert len(evolved_estimator._best_solutions) == generations
    assert len(evolved_estimator.predict(X_test)) == len(X_test)
    assert evolved_estimator.score(X_train, y_train) >= 0
    assert len(evolved_estimator.decision_function(X_test)) == len(X_test)
    assert len(evolved_estimator.predict_proba(X_test)) == len(X_test)
    assert len(evolved_estimator.predict_log_proba(X_test)) == len(X_test)


def test_expected_ga_no_continuous():
    clf = DecisionTreeClassifier()
    generations = 20
    evolved_estimator = GASearchCV(clf,
                                   cv=3,
                                   scoring='accuracy',
                                   population_size=15,
                                   generations=generations,
                                   tournament_size=3,
                                   elitism=True,
                                   categorical_parameters={'criterion': ['gini', 'entropy']},
                                   integer_parameters={'max_depth': (2, 20), 'max_leaf_nodes': (2, 30)},
                                   verbose=False,
                                   encoding_length=10)

    evolved_estimator.fit(X_train, y_train)

    assert check_is_fitted(evolved_estimator) is None
    assert 'criterion' in evolved_estimator.best_params_
    assert 'max_depth' in evolved_estimator.best_params_
    assert 'max_leaf_nodes' in evolved_estimator.best_params_
    assert len(evolved_estimator._best_solutions) == generations


def test_expected_ga_no_categorical():
    clf = DecisionTreeClassifier()
    generations = 20
    evolved_estimator = GASearchCV(clf,
                                   cv=3,
                                   scoring='accuracy',
                                   population_size=15,
                                   generations=generations,
                                   tournament_size=3,
                                   elitism=True,
                                   continuous_parameters={'min_weight_fraction_leaf': (0, 0.5)},
                                   integer_parameters={'max_depth': (2, 20), 'max_leaf_nodes': (2, 30)},
                                   verbose=False,
                                   encoding_length=10)

    evolved_estimator.fit(X_train, y_train)

    assert check_is_fitted(evolved_estimator) is None
    assert 'min_weight_fraction_leaf' in evolved_estimator.best_params_
    assert 'max_depth' in evolved_estimator.best_params_
    assert 'max_leaf_nodes' in evolved_estimator.best_params_
    assert len(evolved_estimator._best_solutions) == generations
