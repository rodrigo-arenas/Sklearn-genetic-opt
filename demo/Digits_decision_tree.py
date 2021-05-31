import warnings
from sklearn_genetic import GASearchCV
from sklearn_genetic.space import Categorical, Integer, Continuous
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn_genetic.callbacks import ThresholdStopping, DeltaThreshold

warnings.filterwarnings("ignore")

data = load_digits()
label_names = data["target_names"]
y = data["target"]
X = data["data"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

clf = DecisionTreeClassifier()

params_grid = {
    "min_weight_fraction_leaf": Continuous(0, 0.5),
    "criterion": Categorical(["gini", "entropy"]),
    "max_depth": Integer(2, 20),
    "max_leaf_nodes": Integer(2, 30),
}

cv = StratifiedKFold(n_splits=3, shuffle=True)

threshold_callback = ThresholdStopping(threshold=0.98, metric="fitness_max")
delta_callback = DeltaThreshold(threshold=0.001, metric="fitness")

callbacks = [threshold_callback, delta_callback]

evolved_estimator = GASearchCV(
    clf,
    cv=cv,
    scoring="accuracy",
    population_size=16,
    generations=30,
    tournament_size=3,
    elitism=True,
    crossover_probability=0.9,
    mutation_probability=0.05,
    param_grid=params_grid,
    algorithm="eaMuPlusLambda",
    n_jobs=-1,
    verbose=True,
)

evolved_estimator.fit(X_train, y_train, callbacks=callbacks)
y_predict_ga = evolved_estimator.predict(X_test)
accuracy = accuracy_score(y_test, y_predict_ga)

print(evolved_estimator.best_params_)
print("accuracy score: ", "{:.2f}".format(accuracy))
